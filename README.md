# Spectral-LLM V0.2

Local experiment on one RTX 5070. This repository tracks the source and a ledger of Arm 0 checkpoints. The checkpoint files stay on the training machine.

An epoch here is one permanent weight file, written every 15,000 steps (5 percent of the 300,000-step run). Each file is about 134 MB, above GitHub's 100 MB file limit, so `ledger/arm0.jsonl` records the step, byte size, sha256, and the loss printed on that step. That loss is one logged step, not a mean over the interval. `runs/arm0/last.pt` is the resume file and is not published.

Hashed files under `runs/*/metrics.json` are measurements. A `literature_target` field is not a measurement. A host sha256 is not a jovin-harness receipt.

CodeRabbit reviews this repository when a pull request against `main` carries the `review-ready` label. The review pauses after one pass. Later epoch commits change `ledger/` only, which is outside the review paths.
