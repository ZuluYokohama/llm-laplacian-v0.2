# Proposed held-out story NLL evaluation

`scripts/evaluate_study.py` implements a **proposed, unfrozen metric** for the
bounded core study. Its default split is `development`, device is CPU, context
is 256 tokens, stride is 128, and batch size is four windows. The implementation
has been checked with synthetic CPU fixtures; no actual held-out test evaluation
or GPU evaluation is implied by those checks. This document schedules nothing.

## Target and context policy

For one story of length L, token indices are 0 through L−1. The tokenizer appends
EOS, so the last token is EOS. The first token has no preceding story context and
is never scored. Every target index 1 through L−1, including the final EOS, is
scored exactly once. A window never includes any token from another story.

The first window's target end is min(256, L−1). For every subsequent window,
advance that end by 128, clipped to L−1. A window's input start is
max(0, target_end−256); its inputs span start through target_end−1, and its
possible targets span start+1 through target_end. Score only target indices
greater than the preceding window's end. Earlier targets provide context but
contribute no NLL in that window. The model's rotary positions start at zero
inside each window, as in the existing model forward path.

For example, a 600-token story scores targets 1–256, then 257–384, 385–512,
and 513–599. Their input windows start at 0, 128, 256, and 343 respectively.
The last window is anchored at the final target, so its first newly scored
target can have more context than in a full 128-target chunk. This is a precisely
defined limited-context metric; it is not full-story likelihood or a policy that
gives every target exactly its preceding 256 tokens. Altering context or stride
changes the metric and must be recorded and frozen before comparisons.
The policy label names the proposed defaults; consumers must include the actual
`context` and `stride` receipt fields when identifying or grouping a metric.

Windows are right-padded only to the largest input length in their batch, using
the EOS ID as padding. The causal model cannot attend to padding to the right of
a valid position. Only newly scored, non-padding targets are selected before
cross-entropy. Per-token CE uses FP32 logits without autocast, TF32 is disabled,
and target losses are summed in FP64. The evaluator calls `model.eval()` and
`torch.inference_mode()`; it never updates weights, optimizer state, or norm
running buffers. Precision and runtime are recorded. Cross-device bitwise
identity is not assumed.

## Inputs and immutable receipts

Required inputs are the original `study_config.json`, an explicit trusted full
checkpoint from that run, and its explicit token-bundle `manifest.json`. The
checkpoint must be either the final `last.pt` or a prospectively saved timing
state whose **raw bytes** appear in that completed run's metrics. The metrics'
sidecar digest, config identity, checkpoint contents, complete training flags,
and bound data/source/fractional-policy identities must validate. Deserialization
uses `weights_only=False` because full checkpoints contain Python/NumPy RNG
objects; do not give this command an untrusted checkpoint.

The production token validator verifies pinned GPT-2/tokenizer policy, all split
files, upstream partition/source identities, and its dependency/code bindings.
The evaluator additionally checks the selected story index: unique identities,
contiguous token offsets, positive nonempty text plus EOS, vocabulary bounds,
final EOS, and exact story/token counts. Only `development`, `calibration`, or
`test` may be selected; the training split is not an evaluation option.

Output must be a new directory. `attempt.json` records the requested policy,
input and source observations before validation. On success, `stories.jsonl`
retains story identity, source row, token offset/length, summed NLL, valid target
count, and window count. `metrics.json` binds that file, the checkpoint, config,
manifest, training completion receipt, evaluator source, package sources,
runtime/precision, split identities, timing, and memory measurements. Inputs and
sources are rechecked after scoring, and final success JSON is promoted only
after its sidecar exists. Failure writes `error.json` with `complete:false` and
before/after observations; partial story or pending files remain audit artifacts.
Rerunning into an existing directory is refused, including failed attempts.

The principal aggregate is total summed NLL divided by total valid targets,
with its exponential reported as perplexity when representable. This is
token-weighted, not an unweighted average of story means. Per-story outputs
support later paired comparisons and uncertainty calculations; the evaluator
does not compute significance, select checkpoints, or claim quality benefits.
Training seeds, rather than stories or overlapping windows, remain the primary
independent experimental units for the paired training comparison.

## Usage and held-out discipline

Examples show syntax only and perform no work by themselves:

```powershell
# A CPU development evaluation. The script adds src to its import path.
.venv/Scripts/python.exe scripts/evaluate_study.py `
  --config runs/publication_study/core/seed0/optimized/study_config.json `
  --checkpoint runs/publication_study/core/seed0/optimized/last.pt `
  --token-manifest data/TOKEN_BUNDLE/manifest.json `
  --output runs/publication_study/evaluations/seed0-optimized-development-attempt1

# Optional CUDA path, only after review and within the existing hard budget.
.venv/Scripts/python.exe scripts/study_budget.py run `
  --run-id eval-seed0-optimized-development --reserve-seconds SECONDS --cuda-device 0 `
  -- .venv/Scripts/python.exe scripts/evaluate_study.py `
  --config runs/publication_study/core/seed0/optimized/study_config.json `
  --checkpoint runs/publication_study/core/seed0/optimized/last.pt `
  --token-manifest data/TOKEN_BUNDLE/manifest.json `
  --output runs/publication_study/evaluations/seed0-optimized-development-attempt1 `
  --device cuda:0
```

CUDA admission requires the canonical study budget environment before any CUDA
query, visible device zero, and the approved RTX5070. It checks the existing
memory guard after every batch. Wrapper wall time is the budget measure;
`scoring_seconds` excludes checkpoint loading, validation, and result writing.

An optional `--max-stories N` evaluates the first N indexed stories only for an
explicit exploratory development/calibration check. The receipt retains both
available and evaluated counts. It is forbidden with `--split test`. Full test
evaluation remains available as an explicit choice for the eventual frozen
protocol; availability is not permission to repeatedly inspect test performance.
Freeze checkpoint selection, windows, precision, split usage, and statistical
analysis before the final test pass. Do not use test scores to choose settings,
seeds, checkpoints, or which results to report. Calibration is reserved for
predeclared calibration purposes, separate from final test reporting.

## Verification scope

`tests/test_study_evaluation.py` uses a uniform-logit oracle, a hand-enumerated
context-dependent oracle, and a tiny real causal model on synthetic stories.
It covers short/uneven/long stories, EOS/first-token accounting, overlapping
window ownership, masked padding (including deliberately nonfinite padded
logits), batch-size invariance, source/input mutation, mismatched checkpoint
bytes, completion sidecars, nonfinite scored logits, immutable output, atomic
success promotion, and budget admission before CUDA queries. The integration
fixture injects a tiny token validator and constructs a synthetic bound checkpoint;
it is not a claim that a production held-out model or corpus was evaluated.

```powershell
$env:CUDA_VISIBLE_DEVICES=''
.venv/Scripts/python.exe -m pytest tests/test_study_evaluation.py -q `
  --basetemp cache/study-evaluation-NEW-CPU-only
```
