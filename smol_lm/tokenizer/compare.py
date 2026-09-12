"""Compare bytes-per-token: our tokenizer vs GPT-2 (50,257) vs Llama-3 (128,256).

Measured on held-out text (validation shard), which the tokenizer never saw.
Writes artifacts/tokenizer/compression.json and prints a markdown table.
"""

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

OURS = "artifacts/tokenizer/tokenizer.json"
# Ungated mirror: meta-llama/* requires accepting a licence on the Hub.
BASELINES = {"gpt2": "gpt2", "llama3": "NousResearch/Meta-Llama-3-8B"}
D_MODEL = 768  # config L, for the embedding-cost column


def iter_batches(path: Path, max_bytes: int, batch: int = 2000) -> Iterator[list[str]]:
    buf: list[str] = []
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            n = len(text.encode("utf-8"))
            if seen + n > max_bytes:
                break
            buf.append(text)
            seen += n
            if len(buf) >= batch:
                yield buf
                buf = []
    if buf:
        yield buf


def measure(tok: Tokenizer, path: Path, max_bytes: int) -> tuple[int, int]:
    total_bytes = total_tokens = 0
    for texts in iter_batches(path, max_bytes):
        total_bytes += sum(len(t.encode("utf-8")) for t in texts)
        for enc in tok.encode_batch(texts, add_special_tokens=False):
            total_tokens += len(enc.ids)
    return total_bytes, total_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/val.jsonl")
    ap.add_argument("--mb", type=float, default=100.0)
    ap.add_argument("--out", default="artifacts/tokenizer/compression.json")
    args = ap.parse_args()

    path = Path(args.input)
    max_bytes = int(args.mb * 1e6)

    tokenizers: dict[str, Tokenizer] = {"smol-lm (ours)": Tokenizer.from_file(OURS)}
    for name, repo in BASELINES.items():
        try:
            tokenizers[name] = Tokenizer.from_pretrained(repo)
        except Exception as e:  # noqa: BLE001 - a missing baseline shouldn't kill the run
            print(f"skipping {name} ({repo}): {type(e).__name__}: {e}")

    rows = []
    byte_counts = set()
    for name, tok in tokenizers.items():
        nbytes, ntokens = measure(tok, path, max_bytes)
        byte_counts.add(nbytes)
        vocab = tok.get_vocab_size()
        rows.append(
            {
                "tokenizer": name,
                "vocab_size": vocab,
                "bytes": nbytes,
                "tokens": ntokens,
                "bytes_per_token": round(nbytes / ntokens, 3),
                "embedding_params_at_d768": vocab * D_MODEL,
            }
        )
        print(f"{name:<16} vocab {vocab:>7,}  {nbytes / ntokens:.3f} bytes/token")

    assert len(byte_counts) == 1, f"byte counts differ between tokenizers: {byte_counts}"

    base = next((r for r in rows if r["tokenizer"] == "gpt2"), rows[0])
    for r in rows:
        r["vs_gpt2_pct"] = round(100 * r["bytes_per_token"] / base["bytes_per_token"] - 100, 1)

    result = {"input": args.input, "mb_measured": round(rows[0]["bytes"] / 1e6, 1), "rows": rows}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    print("\n| Tokenizer | Vocab | Bytes/token | vs GPT-2 | Embedding params (d=768) |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['tokenizer']} | {r['vocab_size']:,} | {r['bytes_per_token']:.3f} | "
            f"{r['vs_gpt2_pct']:+.1f}% | {r['embedding_params_at_d768'] / 1e6:.1f}M |"
        )


if __name__ == "__main__":
    main()
