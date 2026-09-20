# Experiment log

| date | run | result | note |
|---|---|---|---|
| 2026-09-04 | sample_raw | 416,421 docs / 1.9 GB train, 45,071 docs / 210 MB val | val = shard 013 only |
| 2026-09-08 | train_bpe | vocab 32,768 on 2.0 GB, 52 s | specials 0–3; round-trip exact; 74 chars → 25 tokens |
| 2026-09-12 | compare | 4.661 B/tok vs GPT-2 4.629, Llama-3 4.769 | 100 MB held-out; 32k vocab = 25.2M embed params (23% of model) |
| 2026-09-12 | tokenize_shards | 3.000B train (33 files) + 50.0M val, 70.3 + 1.4 min | uint16 .npy, 5.7 GiB; on HF as shubh-kandpal/smol-lm-fineweb-edu-3b |
