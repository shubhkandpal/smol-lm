from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = 32_768
    context_len: int = 1024

    def non_embedding_params(self) -> int:
        # attention q,k,v,o = 4·d² ; SwiGLU w1,w2,w3 = 3·d·d_ff ; no biases (RMSNorm weights ignored)
        return self.n_layers * (4 * self.d_model**2 + 3 * self.d_model * self.d_ff)

    def embedding_params(self) -> int:
        return self.vocab_size * self.d_model  # tied input/output embedding

    def total_params(self) -> int:
        return self.non_embedding_params() + self.embedding_params()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(name=raw["name"], **raw["model"])
