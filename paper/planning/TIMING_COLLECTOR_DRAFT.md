# Exploratory pair-cost prototype and remaining collector work

Status: exploratory implementation; the confirmatory collector remains incomplete.
This specification was first frozen locally before the initial hardware probe.
The original source snapshot and its results remain separate execution artifacts.
This document does not launch or authorize a GPU invocation.
Existing production sources, analysis and randomized schedule remain unchanged.

The first slice is one explicitly requested pair, not the proposed 96-pair
confirmatory collection. `scripts/benchmark_study_pair.py prepare` binds a
completed local CE run and one retained checkpoint, immutable baseline source,
train token bundle, pair ID, order, sampler seed, penalty step (0 or 135000),
and update counts. Defaults are three warmup and twelve measured updates;
smaller positive bounded counts are allowed for exploratory diagnostics.
Preparation uses CPU only and writes a new directory. It does not sample or
evaluate held-out examples; integrity checks hash the bound bundle files. It
does not modify a training checkpoint.

`run-pair --plan NEW_DIRECTORY/plan.json --reserve-seconds-per-worker SECONDS`
executes baseline and optimized in the requested AB/BA order through the existing
canonical budget wrapper, one fresh process per implementation. The reserve is
explicit and the existing approved 181-hour ledger remains authoritative. All
startup, validation, restoration, transfer, timing and postcheck work inside each
worker is charged. CPU preparation and controller work are reported separately.
An unfinished reservation blocks launch. There is no retry, replacement or
resume in this prototype: an existing dispatch directory prevents rerunning it.
Failed and interrupted attempts remain evidence; normal budget recovery rules
apply without refunds for uncertain work.

The original training configuration is validated unchanged. A separate recorded
benchmark override enables spectral loss at the requested fixed penalty step and
selects the implementation. This override is not a continuation configuration.
The loaded optimizer LR, moments, parameters, buffers, flags, firing counters and
global Python/NumPy/Torch CPU/CUDA RNG states are preserved. Only the local data
generator is deliberately reseeded, and its before/after identities are recorded.
A shared batch manifest retains every train-stream start offset and input/target
hash. The legacy exclusive upper bound `len(tokens)-context-1` is preserved;
stream windows may cross story/EOS boundaries. Both workers must reconstruct the
same batches and have the same full prepared-state identity.

Each timed interval includes zeroing gradients, forward/loss and finite checks,
accumulated backward, finite gradient clipping and AdamW, synchronized before and
after. There is no LR scheduling, firing accumulation, resampling, pruning,
checkpoint/export, sampling, transfers, state hashing or telemetry inside that
interval. Warmup changes the in-memory state; all warmup durations are retained.
The full process cost, rather than those intervals alone, prices this prototype.
The original LR may be small in a pilot checkpoint; it is disclosed, not silently
replaced with the proposed continuation LR.

Immutable attempt/error/success receipts bind source and input bytes before and
after execution. Pair completion also requires canonical successful budget
settlements, matching command provenance and distinct worker executions; exact
ledger-prefix snapshots remain verifiable after later ledger appends. A child
receipt alone cannot prove budget settlement. No execution advances a study run.

Every receipt is explicitly exploratory. Finiteness is measured; component and
gradient equivalence, thermal/competing-process eligibility, session independence
and confirmatory admission remain **unknown/not evaluated**. Speed does not set
those gates. `analyze_study.py` requires a different confirmatory timing kind and
also retains its unconditional collector-admission blocker. No prototype result
can be promoted by relabeling an approval field. One pair supplies no confidence
interval or generalization claim. Its measured cost still omits future eligibility
sampling, session scheduling and any additional numerical checks.

CPU acceptance covers exact batch replay, full-state fingerprint sensitivity,
loaded optimizer/RNG preservation, distinct-process dispatch and budget evidence,
phase overrides, source/input mutation, nonfinite failures, immutable attempts,
partial dispatch and pre-CUDA admission failures. Synthetic substitutes are
identified in tests; no GPU or held-out evaluation is part of verification.

Before a later confirmatory collector, freeze: optimizer/LR and sampler policies;
numeric tolerances and bound numerical receipts; session spacing/restart rules;
telemetry/eligibility and maximum within-pair gap; failure/replacement policy;
source/environment identities; and the pricing envelope within the same cap.
The four CE seed-0 states require its 50k training trajectory first. Prospective
run/step roles must precede training; actual checkpoint hashes can only be bound
after those artifacts exist. A later controller can admit states directly from
that parent's completed receipt, independently of final quality evaluation.

The full draft design remains twelve sessions by eight cells by two fresh
processes: 96 pairs, 192 workers and 2,880 updates at 3+12 each. Resume should
reconstruct already-settled evidence without rerunning workers; an interrupted
pair must follow a frozen policy rather than silently reuse favorable data.
Canonical budget serialization prevents simultaneous wrapped GPU jobs, but does
not alone prevent another job between A and B or prove session independence.
The prototype rejects intervening ledger events within its pair. A later schema
and independent review must bind actual telemetry/restart/budget evidence before
the analyzer's permanent admission blocker can be removed. No such expansion is
implemented in this slice.
