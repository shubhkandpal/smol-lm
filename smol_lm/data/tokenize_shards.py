"""Tokenise FineWeb-Edu into uint16 .npy shards for training.

Each source parquet is processed independently and written as
<split>_<src>_<chunk>.npy, so a dead session resumes at source-shard
granularity. Documents are separated by <|endoftext|> (id 0).

    python -m smol_lm.data.tokenize_shards --split train --out data/shards
    python -m smol_lm.data.tokenize_shards --split val   --out data/shards
"""

import argparse
import hashlib
import json
import os
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the Pool does the parallelism

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

REPO = "HuggingFaceFW/fineweb-edu"
EOT = 0  # <|endoftext|>

_tok: Tokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _tok
    _tok = Tokenizer.from_file(tokenizer_path)


def _encode(texts: list[str]) -> np.ndarray:
    ids: list[int] = []
    for enc in _tok.encode_batch(texts, add_special_tokens=False):
        ids.extend(enc.ids)
        ids.append(EOT)
    return np.asarray(ids, dtype=np.uint16)


def doc_batches(data_file: str, batch: int) -> Iterator[list[str]]:
    ds = load_dataset(REPO, data_files=[data_file], split="train", streaming=True)
    buf: list[str] = []
    for row in ds:
        buf.append(row["text"])
        if len(buf) == batch:
            yield buf
            buf = []
    if buf:
        yield buf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--manifest", default="data/MANIFEST.json")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--target-tokens", type=float, default=None)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    tok_path = Path(args.tokenizer)
    vocab = Tokenizer.from_file(str(tok_path)).get_vocab_size()
    assert vocab <= 65_536, f"vocab {vocab} does not fit in uint16"

    src = json.loads(Path(args.manifest).read_text())
    if args.split == "train":
        sources, target = src["train_shards"], int(args.target_tokens or 3e9)
    else:
        sources, target = [src["val_shard"]], int(args.target_tokens or 50e6)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_tokens = args.shard_tokens

    written: list[dict] = []
    total = 0
    t0 = time.time()

    for source in sources:
        if total >= target:
            break
        tag = Path(source).stem
        marker = out / f".done_{args.split}_{tag}"
        if marker.exists():
            for f in sorted(out.glob(f"{args.split}_{tag}_*.npy")):
                n = int(np.load(f, mmap_mode="r").shape[0])
                written.append({"file": f.name, "tokens": n})
                total += n
            print(f"[{tag}] already done ({total:,} tokens so far)", flush=True)
            continue

        buf = np.empty(shard_tokens, dtype=np.uint16)
        fill = chunk = 0
        with Pool(args.workers, initializer=_init_worker, initargs=(str(tok_path),)) as pool:
            for arr in pool.imap(_encode, doc_batches(source, args.batch), chunksize=2):
                pos = 0
                while pos < arr.size:
                    take = min(shard_tokens - fill, arr.size - pos)
                    buf[fill : fill + take] = arr[pos : pos + take]
                    fill += take
                    pos += take
                    if fill == shard_tokens:
                        name = f"{args.split}_{tag}_{chunk:03d}.npy"
                        np.save(out / name, buf)
                        written.append({"file": name, "tokens": shard_tokens})
                        total += shard_tokens
                        chunk += 1
                        fill = 0
                        mins = (time.time() - t0) / 60
                        print(f"{total / 1e9:6.3f}B tokens {mins:6.1f} min -> {name}", flush=True)
                    if total + fill >= target:
                        break
                if total + fill >= target:
                    break

        keep = min(fill, max(0, target - total))
        if keep:
            name = f"{args.split}_{tag}_{chunk:03d}.npy"
            np.save(out / name, buf[:keep])
            written.append({"file": name, "tokens": keep})
            total += keep
            print(f"{total / 1e9:6.3f}B tokens -> {name} (partial)", flush=True)
        marker.touch()

    manifest = {
        "split": args.split,
        "repo": REPO,
        "source_shards": sources,
        "tokenizer_sha256": sha256(tok_path),
        "vocab_size": vocab,
        "dtype": "uint16",
        "eot_id": EOT,
        "total_tokens": total,
        "files": written,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / f"MANIFEST_{args.split}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{args.split}: {total:,} tokens in {len(written)} files, {manifest['minutes']} min")


if __name__ == "__main__":
    main()
