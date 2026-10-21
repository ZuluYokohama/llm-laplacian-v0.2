# Spectral-LLM V0.2

Local experiment on one RTX 5070. This repository tracks the source and a ledger of Arm 0 checkpoints. The checkpoint files stay on the training machine.

An epoch here is one permanent weight file, written every 15,000 steps (5 percent of the 300,000-step run). Each file is about 134 MB, above GitHub's 100 MB file limit, so `ledger/arm0.jsonl` records the step, byte size, sha256, and the loss printed on that step. That loss is one logged step, not a mean over the interval. `runs/arm0/last.pt` is the resume file and is not published.

Hashed files under `runs/*/metrics.json` are measurements. A `literature_target` field is not a measurement. A host sha256 is not a jovin-harness receipt.

Jovin watches the twelve pure reference operators in `reference/operators/`. Each one runs in a separate process inside the pinned image: no network, no mounts, no GPU, no PyTorch. A passing receipt says those tests passed on that source. It does not say the host training loop, the checkpoint resume, or the Windows process supervisor did the same thing. `src/laplacian/host_invariants.py` is the host-side check for that gap. It compares the training λ2 with `reference/operators/fiedler.py` on a path graph, and it checks that dead-unit resampling writes into the parameter. That result is marked `jovin_receipt: false`. The queue runs it on CPU and runs it again when those source files change. The live training process does not reload this file until it is started again.

CodeRabbit reviews this repository when a pull request against `main` carries the `review-ready` label. The review pauses after one pass. Later epoch commits change `ledger/` only, which is outside the review paths.
