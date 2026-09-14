import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

TOK_PATH = Path("artifacts/tokenizer/tokenizer.json")
RAW = Path("data/raw/train_sample.jsonl")
SHARD_DIR = Path("data/shards")
MANIFEST = Path("data/MANIFEST_train.json")
EOT = 0
VOCAB = 32_768


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    if not TOK_PATH.exists():
        pytest.skip("tokenizer not trained")
    return Tokenizer.from_file(str(TOK_PATH))


@pytest.mark.skipif(not RAW.exists(), reason="raw sample not downloaded")
def test_round_trip_on_corpus(tok: Tokenizer) -> None:
    """Real corpus text survives encode->decode, and every id fits the vocabulary."""
    texts: list[str] = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) == 1000:
                break
    assert len(texts) == 1000

    encodings = tok.encode_batch(texts, add_special_tokens=False)
    for enc, original in zip(encodings, texts, strict=True):
        assert tok.decode(enc.ids, skip_special_tokens=False) == original
        assert max(enc.ids) < VOCAB


@pytest.mark.skipif(not MANIFEST.exists(), reason="shard manifest not present")
def test_shards_were_built_with_this_tokenizer() -> None:
    """The guard against training on shards from a different vocabulary."""
    m = json.loads(MANIFEST.read_text())
    digest = hashlib.sha256(TOK_PATH.read_bytes()).hexdigest()
    assert m["tokenizer_sha256"] == digest, "shards were built with a different tokenizer"
    assert m["vocab_size"] == VOCAB
    assert m["dtype"] == "uint16"
    assert m["eot_id"] == EOT
    assert m["total_tokens"] == 3_000_000_000


def test_shard_ids_within_vocab() -> None:
    shards = sorted(SHARD_DIR.glob("*.npy")) if SHARD_DIR.exists() else []
    if not shards:
        pytest.skip("no token shards downloaded locally")
    for shard in shards:
        a = np.load(shard, mmap_mode="r")
        assert a.dtype == np.uint16, shard.name
        assert a.ndim == 1 and a.size > 0, shard.name
        assert int(a.max()) < VOCAB, shard.name
        assert int(np.count_nonzero(a[:1_000_000] == EOT)) > 0, f"no EOT in {shard.name}"
