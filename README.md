# smol-lm

A ~110M-parameter Llama-style language model trained from scratch on 2.5B tokens of
FineWeb-Edu, with a scaling-law study on a 13M–76M ladder, a loss prediction committed
*before* the final run, and instruction tuning on SmolTalk.

## Status
- [ ] Week 1 — tokenizer & data shards
- [ ] Week 2 — model & training loop
- [ ] Week 3 — parity run & LR sweep
- [ ] Weeks 4–5 — scaling ladder & prediction
- [ ] Weeks 6–7 — final pretrain & SFT
- [ ] Weeks 8–9 — ablations & release

## Results
### Tokenizer

32,768-token byte-level BPE (GPT-4-style split regex) trained on 2.0 GB of FineWeb-Edu,
measured on 100 MB of held-out text from a shard the tokenizer never saw.

| Tokenizer | Vocab | Bytes/token | vs GPT-2 | Embedding params (d=768) |
|---|---:|---:|---:|---:|
| smol-lm (ours) | 32,768 | 4.661 | +0.7% | 25.2M |
| gpt2 | 50,257 | 4.629 | +0.0% | 38.6M |
| llama3 | 128,256 | 4.769 | +3.0% | 98.5M |

Vocabulary size is not free: every token needs an embedding row. At d_model=768 our 32k vocabulary costs 25.2M parameters — 23% of the 110M model. Llama-3's 128k vocabulary would cost 98.5M, making a 183M model that is 54% lookup table, for 2.3% better compression. The comparison flatters us: it is measured in-domain, on the same corpus the tokenizer was trained on. On code or non-English text the larger vocabularies would likely win.


## Setup
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -e ".[dev]"
    pre-commit install
    pytest
