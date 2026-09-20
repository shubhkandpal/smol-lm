"""A Llama-style decoder-only transformer.

RMSNorm (pre-norm) -> causal self-attention with rotary position embeddings ->
RMSNorm -> SwiGLU MLP, repeated n_layers times, then a final RMSNorm and an
output head tied to the token embedding. No biases anywhere.

Layer names follow Hugging Face's LlamaForCausalLM so the week-6 export is a
straight rename (q_proj/k_proj/v_proj/o_proj, gate/up/down = w1/w3/w2).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from smol_lm.config import ModelConfig


class RMSNorm(nn.Module):
    """x / rms(x) * weight, computed in float32 for stability."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_cache(
    context_len: int, head_dim: int, theta: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (context_len, head_dim), HF 'rotate_half' layout."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(context_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, head_dim); cos/sin: (T, head_dim), broadcast over B and H
    return (x * cos + rotate_half(x) * sin).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, head_dim)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    """w2(silu(w1 x) * w3 x). w1 = gate_proj, w3 = up_proj, w2 = down_proj in HF naming."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied: one matrix, used twice

        cos, sin = rope_cache(cfg.context_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual-path output projections get a smaller init so the residual
        # stream's variance doesn't grow with depth (GPT-2 / Llama practice).
        scaled_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(block.mlp.w2.weight, mean=0.0, std=scaled_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_len = idx.size(1)
        if seq_len > self.cfg.context_len:
            raise ValueError(f"sequence length {seq_len} exceeds context {self.cfg.context_len}")
        cos, sin = self.rope_cos[:seq_len], self.rope_sin[:seq_len]

        x = self.wte(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None  # only the last position, for sampling
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def count_params(self) -> dict[str, int]:
        """Embedding counted once (tied); non-embedding includes the RMSNorm weights."""
        total = sum(p.numel() for p in self.parameters())  # parameters() de-duplicates tied weights
        embedding = self.wte.weight.numel()
        norms = sum(m.weight.numel() for m in self.modules() if isinstance(m, RMSNorm))
        return {
            "embedding": embedding,
            "non_embedding": total - embedding,
            "norm": norms,
            "total": total,
        }


if __name__ == "__main__":
    import sys

    cfg = ModelConfig.from_yaml(sys.argv[1] if len(sys.argv) > 1 else "configs/xs.yaml")
    model = GPT(cfg)
    counts = model.count_params()
    print(f"config {cfg.name}: " + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in counts.items()))
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    targets = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        _, loss = model(idx, targets)
    print(f"loss at init {loss.item():.3f}  (ln vocab = {math.log(cfg.vocab_size):.3f})")
