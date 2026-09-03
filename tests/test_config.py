from pathlib import Path

import pytest

from smol_lm.config import ModelConfig

# (non-embedding, total) from the closed form; the week-2 model must reproduce these within 0.1%
EXPECTED = {
    "xs": (4_816_896, 13_205_504),
    "s": (14_155_776, 26_738_688),
    "m": (32_112_640, 48_889_856),
    "mplus": (54_517_760, 75_489_280),
    "l": (84_934_656, 110_100_480),
}


@pytest.mark.parametrize(("name", "expected"), EXPECTED.items())
def test_param_counts(name: str, expected: tuple[int, int]) -> None:
    cfg = ModelConfig.from_yaml(Path("configs") / f"{name}.yaml")
    assert (cfg.non_embedding_params(), cfg.total_params()) == expected


@pytest.mark.parametrize("name", EXPECTED)
def test_heads_divide_width(name: str) -> None:
    cfg = ModelConfig.from_yaml(Path("configs") / f"{name}.yaml")
    assert cfg.d_model % cfg.n_heads == 0
