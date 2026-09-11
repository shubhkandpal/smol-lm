"""Train a byte-level BPE tokenizer on data/raw/train_sample.jsonl.

Output: artifacts/tokenizer/tokenizer.json (+ stats.json). Committed to git.
"""

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

# GPT-4 / Llama-3 style split: contractions | words (with optional leading char) |
# digit runs of <=3 | punctuation runs | newlines | whitespace. BPE merges never cross these.
GPT4_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]  # ids 0..3


def iter_texts(path: Path, max_bytes: int) -> Iterator[str]:
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            seen += len(line.encode("utf-8"))
            yield json.loads(line)["text"]
            if seen >= max_bytes:
                break


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(GPT4_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/train_sample.jsonl")
    ap.add_argument("--out", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--vocab", type=int, default=32_768)
    ap.add_argument("--max-gb", type=float, default=2.0, help="how much of the sample to use")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    t0 = time.time()
    tok.train_from_iterator(iter_texts(Path(args.input), int(args.max_gb * 1e9)), trainer=trainer)
    elapsed = time.time() - t0

    tok.save(str(out))
    stats = {
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": {t: tok.token_to_id(t) for t in SPECIAL_TOKENS},
        "trained_on_gb": args.max_gb,
        "input": args.input,
        "train_seconds": round(elapsed),
    }
    (out.parent / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

    sample = "The quick brown fox jumps over 12345 lazy dogs.\nTabs\t, accents é, emoji 🙂."
    ids = tok.encode(sample).ids
    print(f"\nsample: {len(sample)} chars -> {len(ids)} tokens")
    print("round-trip exact:", tok.decode(ids) == sample)


if __name__ == "__main__":
    main()
