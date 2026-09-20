import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from smol_lm.config import ModelConfig
from smol_lm.model import GPT, RMSNorm, apply_rope, rope_cache

TINY = ModelConfig(
    name="tiny", d_model=64, n_layers=2, n_heads=4, d_ff=172, vocab_size=256, context_len=64
)


@pytest.mark.parametrize("name", ["xs", "s", "m", "mplus", "l"])
def test_param_counts_match_closed_form(name: str) -> None:
    cfg = ModelConfig.from_yaml(Path("configs") / f"{name}.yaml")
    with torch.device("meta"):  # shapes only: no memory, no init cost
        model = GPT(cfg)
    counts = model.count_params()
    assert counts["embedding"] == cfg.embedding_params()
    assert counts["norm"] == (2 * cfg.n_layers + 1) * cfg.d_model
    assert counts["non_embedding"] - counts["norm"] == cfg.non_embedding_params()


def test_output_head_is_tied_to_embedding() -> None:
    model = GPT(TINY)
    assert model.lm_head.weight is model.wte.weight


def test_rmsnorm_matches_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 10, 64)
    ours = RMSNorm(64)(x)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_sdpa_matches_naive_attention() -> None:
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 16, 32
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    fast = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    naive = scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
    torch.testing.assert_close(fast, naive, atol=1e-4, rtol=1e-4)


def test_rope_depends_only_on_relative_position() -> None:
    """q·k after RoPE must be the same for (pos 3, pos 1) and (pos 12, pos 10)."""
    torch.manual_seed(0)
    cos, sin = rope_cache(32, 16)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def dot(i: int, j: int) -> float:
        qi = apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return (qi * kj).sum().item()

    assert dot(3, 1) == pytest.approx(dot(12, 10), abs=1e-4)


def test_causal_future_tokens_do_not_leak() -> None:
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 20))
    changed = idx.clone()
    changed[0, 15:] = (changed[0, 15:] + 1) % TINY.vocab_size
    with torch.no_grad():
        a, _ = model(idx, idx)
        b, _ = model(changed, changed)
    torch.testing.assert_close(a[:, :15], b[:, :15])
    assert not torch.allclose(a[:, 15:], b[:, 15:])


def test_loss_at_init_is_close_to_uniform() -> None:
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (4, 32))
    targets = torch.randint(0, TINY.vocab_size, (4, 32))  # independent of the inputs
    with torch.no_grad():
        _, loss = model(idx, targets)
    assert loss.item() == pytest.approx(math.log(TINY.vocab_size), abs=0.2)


def test_can_overfit_a_single_batch() -> None:
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (4, 33))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
