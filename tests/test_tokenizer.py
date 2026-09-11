from pathlib import Path

import pytest
from tokenizers import Tokenizer

TOK = Path("artifacts/tokenizer/tokenizer.json")
pytestmark = pytest.mark.skipif(not TOK.exists(), reason="tokenizer not trained yet")


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_file(str(TOK))


def test_vocab_and_specials(tok: Tokenizer) -> None:
    assert tok.get_vocab_size() == 32_768
    assert tok.token_to_id("<|endoftext|>") == 0
    assert tok.token_to_id("<|user|>") == 1
    assert tok.token_to_id("<|assistant|>") == 2
    assert tok.token_to_id("<|pad|>") == 3


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "Tabs\tand\nnewlines  and   runs of spaces.",
        "Numbers 1234567890 and dates 2026-09-04.",
        "Unicode: café, naïve, 東京, emoji 🙂🚀",
    ],
)
def test_round_trip(tok: Tokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text).ids) == text


def test_special_tokens_are_single_ids(tok: Tokenizer) -> None:
    ids = tok.encode("<|user|>hi<|assistant|>").ids
    assert ids[0] == 1
    assert ids[-1] == 2
