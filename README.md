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
_(tables land here as each week finishes)_

## Setup
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -e ".[dev]"
    pre-commit install
    pytest
