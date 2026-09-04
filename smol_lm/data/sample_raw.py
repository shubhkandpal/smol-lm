"""Write a raw-text sample for tokenizer training and a held-out validation slice.

Both are streamed from FineWeb-Edu sample-10BT. The validation slice comes from the
LAST parquet shard only; the training sample (and, later, the 3B-token shards) come
from the others. data/MANIFEST.json records the split so it can never be crossed.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

REPO = "HuggingFaceFW/fineweb-edu"
SUBSET_DIR = "sample/10BT"


def shard_files() -> list[str]:
    entries = HfApi().list_repo_tree(REPO, path_in_repo=SUBSET_DIR, repo_type="dataset")
    return sorted(e.path for e in entries if e.path.endswith(".parquet"))


def write_jsonl(files: list[str], out_path: Path, target_bytes: int, label: str) -> int:
    ds = load_dataset(REPO, data_files=files, split="train", streaming=True)
    written = 0
    n_docs = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            line = json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            n_docs += 1
            if n_docs % 20_000 == 0:
                print(f"[{label}] {n_docs:>9,} docs  {written / 1e9:5.2f} GB", flush=True)
            if written >= target_bytes:
                break
    print(f"[{label}] done: {n_docs:,} docs, {written / 1e9:.2f} GB -> {out_path}")
    return n_docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gb", type=float, default=2.0, help="raw text for tokenizer training")
    ap.add_argument("--val-mb", type=float, default=220.0, help="~50M tokens at ~4.4 bytes/token")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = shard_files()
    if len(files) < 2:
        raise SystemExit(f"expected several shards under {SUBSET_DIR}, found {files}")
    train_files, val_file = files[:-1], files[-1]
    print(f"{len(files)} shards found. validation shard: {val_file}")

    n_train = write_jsonl(
        train_files, out / "train_sample.jsonl", int(args.train_gb * 1e9), "train"
    )
    n_val = write_jsonl([val_file], out / "val.jsonl", int(args.val_mb * 1e6), "val")

    manifest = {
        "repo": REPO,
        "subset": SUBSET_DIR,
        "train_shards": train_files,
        "val_shard": val_file,
        "train_sample_docs": n_train,
        "val_docs": n_val,
        "rule": "val_shard is never read for anything except validation",
    }
    Path("data/MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote data/MANIFEST.json")


if __name__ == "__main__":
    main()
