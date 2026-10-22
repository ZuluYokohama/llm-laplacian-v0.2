# Spectral-LLM V0.2

Local experiment on one RTX 5070. This repository tracks the source and a ledger of Arm 0 checkpoints. The checkpoint files stay on the training machine.

An epoch here is one permanent weight file, written every 15,000 steps (5 percent of the 300,000-step run). Each file is about 134 MB, above GitHub's 100 MB file limit, so `ledger/arm0.jsonl` records the step, byte size, sha256, and the loss printed on that step. That loss is one logged step, not a mean over the interval. `runs/arm0/last.pt` is the resume file and is not published.

Hashed files under `runs/*/metrics.json` are measurements. A `literature_target` field is not a measurement. A host sha256 is not a jovin-harness receipt.

Jovin watches the twelve pure reference operators in `reference/operators/`. Each one runs in a separate process inside the pinned image: no network, no mounts, no GPU, no PyTorch. A passing receipt says those tests passed on that source. It does not say the host training loop, the checkpoint resume, or the Windows process supervisor did the same thing. `src/laplacian/host_invariants.py` is the host-side check for that gap. It compares the training λ2 with `reference/operators/fiedler.py` on a path graph, and it checks that dead-unit resampling writes into the parameter. That result is marked `jovin_receipt: false`. The queue runs it on CPU and runs it again when those source files change. The live training process does not reload this file until it is started again.

CodeRabbit is configured to review a non-draft pull request against `main` when it carries the `review-ready` label. The review pauses after one pass. Later epoch commits change `ledger/` only, which is outside the review paths. This configuration is not a claim that an external review has been completed.

## Method and study scope

Training skips penalties whose schedule weights are zero. Hypergraph and trace
contractions use smaller intermediates while preserving their respective
algebraic objectives. The fractional matrix function now uses an explicit C1
cutoff policy and a stable first-order Fréchet derivative. That repair is a
methodological change from the original uncapped fractional objective. Both
the archived-loss comparison and the optimized implementation import the same
current fractional operator; they compare loss implementations under that
shared policy. Historical timings collected before the repair do not establish
the repaired method's runtime or training quality.

The [fractional policy](paper/planning/FRACTIONAL_POLICY.md) specifies the
forward function, derivative and limits. The [publication plan](paper/PUBLICATION_PLAN.md)
and [draft analysis plan](paper/planning/ANALYSIS_DRAFT.md) describe a proposed
paired study. Confirmatory model-quality results and a frozen final analysis
are not supplied by this source change. Short probes measure execution cost,
not long-run convergence or sparse inference acceleration.

## Verification and local setup

Python 3.12 and the relevant Torch, NumPy, PyArrow, tokenizers and pytest
dependencies are required; `pyproject.toml` does not install them automatically.
The [observed version list](paper/planning/environment_20261021/requirements-observed.txt)
and [hashed Windows CPython 3.12 wheel list](paper/planning/environment_20261021/requirements-windows-cp312-hashed.txt)
describe the local reproduction environment. They do not supply wheels, the
Python/Windows runtime, corpus data or checkpoints. The small environment and
wheel inventories are capture metadata, not external replication evidence.

For example, these synthetic checks run on CPU:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv\Scripts\python.exe -m pytest `
  tests/test_fractional_gradients.py tests/test_reproducibility.py `
  tests/test_resume.py tests/test_study_budget.py tests/test_study_train.py `
  tests/test_study_evaluation.py -q --basetemp cache/pr-cpu-check-NEW
```

Use a new temporary directory. The broader token tests use generated Arrow
fixtures; their additional real-tokenizer smoke requires the pinned GPT-2
tokenizer bytes. Set `LAPLACIAN_TOKENIZER_JSON` to a local copy as described in
[study data](paper/planning/STUDY_DATA.md), and report any skipped smoke check.
The [reproducibility notes](paper/planning/REPRODUCIBILITY.md) distinguish exact
resume checks from legacy checkpoints and cross-device guarantees.

Run the correctness gates without starting the experiment queue:

```powershell
.\.venv\Scripts\python.exe scripts/run_queue.py --gates-only
```

The four gates fingerprint their source and test contents. Old receipts without
fingerprints, changed files, and added or deleted tests require a new check.
An edit during verification prevents a successful attestation. Reference output
files do not invalidate their own inputs. Completed training measurements keep
their historical meaning and are not restarted by this mode. Set
`JOVIN_HARNESS_EXE` to override harness discovery; the installed local path is
still supported. The real isolated reference gate needs the local Jovin and
Docker setup. CPU tests, host GPU checks and isolated Jovin receipts establish
different things; none substitutes for the other.

## Budgeted local experiments

All new study CUDA work must run through `scripts/study_budget.py`. The public
[decision example](paper/planning/decisions.example.json) has `approved: false`;
the machine-specific execution grant and live ledger remain local. These
examples assume an actual authorized local decision, enough remaining budget,
the approved RTX 5070 and the required local inputs. They illustrate commands
and do not schedule runs. See [budget accounting](paper/planning/BUDGET_ACCOUNTING.md)
for reservations, process-tree containment, recovery and accounting limits.

After clearing a CPU-only `CUDA_VISIBLE_DEVICES` override, compare the archived
loss implementation with the current loss on local historical data:

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts/study_budget.py run `
  --run-id optimization-comparison-NEW --reserve-seconds 900 --cuda-device 0 `
  -- .\.venv\Scripts\python.exe scripts/benchmark_optimization.py `
  --baseline-source runs/optimization_20261021/baseline_losses.py `
  --output-dir runs/optimization_comparison_new `
  --checkpoint runs/arm0/weights_final.pt `
  --steps 5 --warmup 2 --seeds 0 1 2 --accum 8
```

Use a new output directory for each run. The baseline source is trusted local
Python code and is imported by the benchmark. The optional checkpoint is read
without modifying it; omit it to compare freshly initialized models. The runner
uses matched validation token windows and measures forward, backward, clipping,
and AdamW updates after warmup. It checks finite losses and parameter gradients,
records numerical differences, and refuses a completed result when its
equivalence checks fail or sources change during execution. It records source,
checkpoint, dataset, and batch hashes plus dependency versions. These are host
GPU measurements, separate from Jovin's isolated reference receipts. This
historical-data comparison is not a final held-out quality evaluation.

For a prospectively prepared study checkpoint, a bounded update-rate probe can
instead use the bound training token bundle:

```powershell
.\.venv\Scripts\python.exe scripts/study_budget.py run `
  --run-id study-rate-NEW --reserve-seconds 180 --cuda-device 0 `
  -- .\.venv\Scripts\python.exe scripts/benchmark_study_rate.py `
  --checkpoint runs/publication_study/YOUR_RUN/last.pt `
  --output runs/publication_study/rate-probe-NEW `
  --warmup 3 --steps 10 --max-seconds 120
```

Replace the checkpoint path and choose fresh run/output names. Reservation
time includes startup, input validation and cleanup; a short measured update
loop does not define the entire charge. The [study training contract](paper/planning/STUDY_TRAIN.md)
describes parent/branch state transfer and sampling provenance. The
[proposed evaluation policy](paper/planning/STUDY_EVALUATION.md) defines
story-isolated NLL and held-out discipline. Freeze the final protocol before
confirmatory comparisons; BF16 equivalence uses explicit tolerances rather
than a bitwise-identity claim.
