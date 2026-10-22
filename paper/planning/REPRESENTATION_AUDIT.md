# Representation audit: implementation and execution contract

The audit turns a conceptual relationship into a declared representation,
prediction, control, measurement and next decision. It is a separate CPU sidecar;
the frozen training method and manuscript claims are unchanged. The representation
contracts and REP-H1 through REP-H3 are in
`research/representation_audit/contracts.json`. They distinguish example nodes,
token nodes and neuron coordinates rather than treating every graph as the same
geometric object. They deliberately supply no combined merit score.

## What is implemented

- Preparation binds the original stage and its CE continuation, seed0 admission,
  frozen config, four retained checkpoint files, training token/index identities,
  interpreter, Torch/NumPy versions, tested source closure, and verification receipt.
- Sampling uses eight deterministic microbatches of eight training windows, context
  256. SHA256-derived offsets are prospective diagnostic samples, reused unchanged
  at updates 0, 5000, 25000 and 50000. No held-out token/index file is opened.
- The worker reports all model layers: centered pooled variance, entropy effective
  rank, off-diagonal RBF affinity saturation, normalized spectrum counting nulls,
  current C1 fractional energy, positive incidence density and the frozen token
  band's activity. Selected manifold layers remain labelled `[1,4]`.
- Diagnostic inference explicitly uses evaluation mode and FP32 CPU forwards, then
  the defined FP64 spectral calculations. It does not replay training mode, restore
  an optimizer, compute task CE, or establish mixed-precision GPU fidelity.
- New immutable JSONL measurements, raw logs, window/checkpoint identities, a
  proposal-only next experiment and a final receipt preserve both success and
  failure. A failed attempt cannot be rerun or overwritten automatically.

The proposal chooses a training-only bandwidth investigation when any measured
layer/microbatch has all off-diagonal affinities at the declared near-zero or
near-one cutoff. Otherwise it proposes representation-intervention controls.
This descriptive routing is not a statistical threshold or evidence of benefit.
Each proposal states controls, endpoints, a two-hour one-thread CPU ceiling and
zero GPU hours. It never launches the proposed experiment. Runtime optimization
and pruning treatments still require separate protocols and cost proposals.

## Admission and resource boundaries

`prepare` is safe while CE training continues, once seed0's snapshots exist and
its integrity/cost admission is available. It archives tested sources first.
`run` requires all 50 CE segment receipts reconciled, five completed parents and
no open canonical reservation. An unfinished/paused/failed continuation returns
pending (exit 3), creates no attempt, and launches no worker. It does not poll or
install a hook/daemon; invoke the explicit command after CE completion.

The canonical budget lock is held throughout the CPU audit to prevent a new
budgeted GPU invocation from racing admission. No GPU reservation is created and
ledger bytes are unchanged. CUDA is hidden from the worker; Torch and numerical
libraries use one thread. On Windows a parent-owned kill-on-close Job Object
contains the child before permitting model work, sets affinity to one available
CPU, and enforces a 2 GiB **process committed-memory** limit. Memory-mapped file
pages do not imply a strict 2 GiB resident working-set guarantee. A second worker
limit provides defense in depth. Controller death kills the contained worker.
The execution timeout is 7200 seconds; timeout kills the worker and retains a
resource-exhausted receipt. Unsupported parent containment fails closed.

Only trusted local checkpoints admitted by the existing controller are loaded.
They contain Python state and use `torch.load(..., map_location='cpu', mmap=True,
weights_only=False)` after hash checks. Strict state loading and config/update
checks precede forward execution. `last.pt` is never an admissible snapshot.
Sources, checkpoints and training inputs are rechecked after analysis.

## Commands

Use an unused output directory and a completed verification receipt that binds
the current implementation sources. Preparation fails if verification is stale.

```powershell
.\.venv\Scripts\python.exe scripts/representation_audit.py prepare `
  --stage runs/publication_study/staged_20261022_001/stage.json `
  --continuation runs/publication_study/continuation_ce_20261022_001/manifest.json `
  --verification runs/publication_study/representation_audit_verification_20261022_004/receipt.json `
  --output runs/publication_study/representation_audit_20261022_001

# After successful completion of the five CE parents:
.\.venv\Scripts\python.exe scripts/representation_audit.py run `
  --manifest runs/publication_study/representation_audit_20261022_001/manifest.json

# Attach the resulting receipt without changing any scientific gate:
.\.venv\Scripts\python.exe scripts/concept_status.py `
  --map paper/planning/concept_experiment_map.json `
  --audit runs/publication_study/representation_audit_20261022_001/attempt/receipt.json `
  --output-dir runs/publication_study/concept_status_WITH_AUDIT_NEW
```

The status reporter accepts repeatable `--audit` arguments, validates raw hashes
and scope, rejects duplicate receipts/manifests, and annotates only geometry and
execution regions. Failed/resource-exhausted diagnostics remain failures. Prior
region status, RQ3, confirmatory admission and publication admission are unchanged.

## Verification and retained contrary findings

The integrated checks cover deterministic train-only sampling, checkpoint/data
lineage substitution, source drift, active and unfinished reservations, duplicate
dispatch, failure receipts, timeout, process death, memory enforcement and receipt
integration. Real tiny-model CPU integration loads four synthetic checkpoints,
executes all 32 microbatches twice, reproduces raw diagnostic outputs, checks the
same windows at every checkpoint and preserves input bytes. These synthetic
checkpoints are test fixtures, not completed CE training or a task-quality result.
Eight additional pure representation controls ran in a pinned, process-isolated
JoVin chain. The previous eight CST assumption controls retain their separate
scope. Host model execution is not described as JoVin-isolated.

Two initial verification failures are preserved rather than rewritten:

- FP32 affinity construction perturbs null modes; the C1 slope near zero can
  amplify that perturbation in a fractional energy. Constant three-point features
  produced approximately `6.7055e-5`, exceeding the old `2e-6` geometry tolerance.
  Independent reference invariance and host FP32 behavior are therefore reported
  separately; no relaxed tolerance or training-source change was introduced.
- Changed incidence support does not guarantee a change in aggregate incidence
  density. The corrected control measures support directly.

Training completion and descriptive trajectories do not establish a causal
mechanism, improved compression, inference speed, language quality or a physical
equivalence. Positive and adverse observations are equally eligible to motivate
the next bounded experiment.
