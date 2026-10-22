# Bounded study training and explicit continuations

This is an implemented **proposed protocol**, not a frozen confirmatory experiment
or a reproduction of the original 300,000-update schedule. No command below is
scheduled by this document. CPU fixtures establish the interface and state
transfer; real-data CUDA rates and replay are separate gates. Every CUDA command
must use the approved `scripts/study_budget.py` ledger and its remaining cap.

## Protocol and pairing

The default CE pretrain is 50,000 optimizer updates with seed supplied explicitly,
microbatch 8, accumulation 8, context 256, the existing 256-dimensional five-layer
model, AdamW (betas 0.9/0.95, weight decay 0.1), clipping 1, LR 3e-4 with 2,000-step
warmup and cosine decay toward 3e-5. Original CE dead-unit resampling is retained
before update 30,000. There are no pruning pulses, normalization removal,
distillation, or embedded spectral probes.

Each completed CE parent can create two new 10,000-update branches. Both preserve
model tensors and buffers, optimizer moments and counters, local sampler RNG,
Python/NumPy/Torch CPU/CUDA RNG, training flags, and dead-unit window counters.
The branch's update counter resets to zero, but optimizer counters do not reset.
LR is constant 3e-5, warmup is zero, and all spectral weights come from schedule
position **135000 for every update**. Both branches disable dead-unit resampling:
the normal 50k parent has already passed that intervention window. Both use the
same parent state and training data stream. The only intended update difference
between the two branches is baseline versus optimized loss implementation.

The baseline source must have raw SHA-256
`604a993fc6db5cb0b86df375f6415313bab6b88600642acc3d40a65bd151e803` (the saved prior
loss implementation). It imports the **current repaired fractional operator**,
as does the optimized implementation. This compares implementation cost and
bounded trajectory behavior under one shared fractional policy; it does not
restore the prior unsafe fractional gradient or establish spectral quality gains.

The proposed core has five independently seeded CE parents and ten paired
continuations. Seed count is not a power calculation. Do not schedule these until
the real pipeline pilot, review, protocol freeze, and budget arithmetic are accepted.

## CLI

These are syntax examples; replace the manifest path with the validated token
bundle's explicit `manifest.json`. Choose a conservative reservation after the
bounded pilot, including input hashing, checkpoint writes, and teardown.

```powershell
# This workspace's venv is not installed as an editable package.
$env:PYTHONPATH=(Resolve-Path src).Path

# Fresh CE pretrain; executed only when explicitly dispatched under the ledger.
.venv/Scripts/python.exe scripts/study_budget.py run `
  --run-id core-seed0-pretrain --reserve-seconds SECONDS --cuda-device 0 `
  -- .venv/Scripts/python.exe -m laplacian.study_train train `
  --token-manifest data/TOKEN_BUNDLE/manifest.json `
  --run-id publication_study/core/seed0/pretrain --seed 0

# CPU preparation only. The local checkpoint must be trusted: it contains full
# Python/NumPy state and is deliberately loaded with weights_only=False.
.venv/Scripts/python.exe -m laplacian.study_train fork `
  --parent-checkpoint runs/publication_study/core/seed0/pretrain/last.pt `
  --run-id publication_study/core/seed0/optimized --loss-implementation optimized
.venv/Scripts/python.exe -m laplacian.study_train fork `
  --parent-checkpoint runs/publication_study/core/seed0/pretrain/last.pt `
  --run-id publication_study/core/seed0/baseline --loss-implementation baseline `
  --baseline-loss-source runs/optimization_20261021/baseline_losses.py

# Launch or ordinary resume of a prepared branch. No overrides are accepted.
.venv/Scripts/python.exe scripts/study_budget.py run `
  --run-id core-seed0-optimized-attempt1 --reserve-seconds SECONDS --cuda-device 0 `
  -- .venv/Scripts/python.exe -m laplacian.study_train train `
  --config runs/publication_study/core/seed0/optimized/study_config.json
```

Use a new budget invocation ID for a resumed attempt, while keeping the study
configuration path unchanged. Do not pass an altered step count to resume.
`fork` creates files but performs no training or CUDA initialization. New run
directories must not exist. Completed runs reject further execution. An OS lock
guards each entry-point execution against concurrent writes.

## Exploratory real-data rate pilot

Before scheduling 5×50k, a separate exploratory seed (for example 9001) can run
**50 CE updates**, followed by **10 updates in each paired continuation**, with
production microbatch/accumulation/context and the real validated token bundle.
Use `train ... --run-id publication_study/pilot/... --seed 9001 --steps 50`, then
two `fork ... --steps 10` commands and separately budgeted `train --config` calls.
The default CE warmup remains 2,000, so this pilot is a rate/integration check in
early warmup, not a proxy for a 50k-trained checkpoint. Do not reuse its seed or
artifacts as independent confirmatory evidence. No quality conclusion follows
from its training CE or tiny branch trajectories.

Measure total wrapper wall time, trainer segment time, checkpoint and validation
overhead, peak memory, finite-state success, and paired draw equality. Keep the
execution order and hardware state in the pilot receipt. A single short pilot
does not estimate hardware variance or stable steady-state throughput. Use its
rates conservatively with explicit verification/evaluation/contingency allowance;
revisit the planned matrix if projected cost exceeds the remaining ledger.

Full timing states are chosen prospectively at initial, ceil(10% of updates),
floor(50%), and final updates, with duplicates collapsed for tiny pilots.
`state_stepNNNNNNN.pt` includes optimizer and RNG state, alongside regular
`last.pt` recovery checkpoints (default every 1,000 updates and at selected
snapshots). For the core this is CE 0/5000/25000/50000 and branch 0/1000/5000/10000.
These states support later randomized/repeated matched timing; their existence
does not itself supply those measurements. Legacy periodic model-only exports
are retained as well. Storage overhead is part of the pilot measurement.

`metrics.study.timing` separates the update-loop wall time (CUDA synchronized
at both boundaries, including sampling, logging, in-loop checkpoints/exports,
and memory guards), initial snapshot time, and post-loop validation/export time.
The legacy `seconds`/`seconds_per_step` segment fields include the first final
rehash/export but omit the last validation; they must not be called pure compute
throughput. Use the wrapper's full wall time for budget projections, and matched
state benchmarks for isolated implementation timing.

## Provenance and failure behavior

`study_config.json` binds raw source hashes for all local package modules, the
token manifest and all eight output file identities, vocabulary, fractional
policy, schedule position, and complete hyperparameters. Startup and completion
call the token bundle validator, including its pinned GPT-2 tokenizer, partition
manifest, original Arrow/input records, code/dependency records, and raw hashes.
Only the manifest's training split is sampled. Historical `DATA`/`meta.json` are
never read by a study training path.

Each fork binds the parent checkpoint's raw bytes, original config and digest,
step, completion-metrics raw identity, and elapsed training time. Fork creation
checks the completion metrics' LF-normalized sidecar and its raw final-checkpoint
binding. Parent time remains separate from branch time; do not double-count it
when aggregating paired branches. Ordinary resume rejects changed configs,
legacy/incomplete state, incomplete training flags, and environment mismatch.
Restored state is necessary for replay; it does not guarantee deterministic CUDA
kernels. CUDA replay must be measured on the intended host/runtime.

Each invocation writes a new `samples-<uuid>.jsonl` with a header binding the
training stream and story index, then update/microbatch positions and sampled
token offsets. A window consists of context+1 tokens; its first context tokens
are inputs and its final context tokens are targets. Bounds match the historical
sampler, including exclusion of its last otherwise-valid start offset. Windows
can cross EOS/story boundaries **within the training split**. Intersect logged
offset intervals with the immutable story-index offsets/lengths to reconstruct
story spans. Logs describe draws, not committed optimizer updates. After a crash,
old uncommitted draws remain labeled by their original attempt and may repeat;
the checkpoint's step/generator state determines the resumed update stream.
Logs are fsynced before checkpoints and their raw hashes are included in metrics.
Checkpoint counters identify committed updates, but prior-attempt log inventories
are not sealed inside each checkpoint. Logs alone therefore do not prove an
exhaustive, tamper-evident history of every resumed update.

Final metrics include all timing-state identities, final checkpoint/weights
identities, sample-log identities, branch lineage, and budget invocation fields.
Source/data/policy drift, missing selected states, non-finite state, or incomplete
resume prevents `complete:true`. A freshness failure can leave partial checkpoint
artifacts for audit; it never authorizes altering hashes to resume. Start a new
explicit protocol/run if source or data changes are necessary.

## CPU verification

`tests/test_study_train.py` uses an explicitly injected tiny-vocabulary manifest
validator and a tiny CPU model. It does not claim production-tokenizer validation;
that is tested separately in `tests/test_study_tokens.py`. Its integration checks
cover manifest-only access, draw bounds/identity, full paired fork state transfer,
fixed penalty position, actual interrupted update replay, immutable completion,
source/data/policy drift, tampered completion/checkpoint bytes, incomplete state,
unsafe run paths, CLI override refusal, and concurrent entry-point exclusion.

Run with CUDA hidden and a fresh workspace test directory:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
.venv/Scripts/python.exe -m pytest tests/test_study_train.py tests/test_resume.py `
  tests/test_loss_optimization.py -q --basetemp cache/study-train-NEW-CPU-only
```
