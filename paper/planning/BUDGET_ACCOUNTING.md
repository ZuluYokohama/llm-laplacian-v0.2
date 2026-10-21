# Publication-study budget accounting

The approved core ceiling is 181 single-GPU hours. `scripts/study_budget.py`
does not schedule work. Every future CUDA check, pilot, benchmark, training,
evaluation, or GPU reproduction in this study must be invoked through it.
Historical experiments completed before approval are not retroactively charged.

## Approval and admission

The public source includes `paper/planning/decisions.example.json` with
`approved: false`. The actual machine's execution grant and budget ledger are
local artifacts and are not shipped with the source. A fresh clone therefore
has no GPU authorization. Record the independently authorized local study's
actual decision before using the wrapper; never overwrite an existing grant
or reset its ledger to initialize another allowance.

The wrapper reads `paper/planning/decisions.json` and requires
`recommended_core_gpu_hours.approved: true` and a positive numeric `hard_cap`
no greater than 181. Approval must bind `recommended_core_gpu_hours.ledger` to
one canonical ledger path; repository-relative paths resolve against the project
root. It records the approval file's raw SHA-256 for each
reservation. There is no command-line budget-increase option. A revised study
budget requires a new explicit user decision and accounting review.

The authoritative default ledger is
`runs/publication_study/budget.jsonl`. The `--decisions` and `--ledger` overrides
exist for isolated CPU tests and separately approved studies. An override that
differs from the ledger explicitly named in that approval is rejected. Do not reset, replace, or
delete the production ledger or its run directory.

```powershell
.\.venv\Scripts\python.exe scripts/study_budget.py status

# Example syntax only; this document does not launch the command.
.\.venv\Scripts\python.exe scripts/study_budget.py run `
  --run-id g0-cuda-regression-001 --reserve-seconds 900 --cuda-device 0 `
  -- .\.venv\Scripts\python.exe -m pytest tests/test_loss_optimization.py -q
```

Choose a conservative reservation for each bounded command. The example reserves
15 minutes, including startup and cleanup. A command is rejected before launch
if its full reservation exceeds the remaining allowance, its run ID exists,
approval is missing, the ledger is corrupt, another invocation holds the lock,
or any earlier reservation remains unresolved. One CUDA device is visible to
the child. An explicitly empty `--cuda-device` is used in CPU subprocess tests;
on Windows, relying on an inherited empty environment variable is insufficient.

## Time charged

The unit is elapsed wall-clock seconds during a serial, single-visible-GPU
invocation. CPU startup, synchronization, and cleanup inside that invocation
also count. This conservatively exceeds pure GPU-active time; it is not a
utilization-weighted or energy measurement.

Before launch, the wrapper appends and fsyncs the full reservation. Hash-linked
JSONL events include monotonic sequence numbers, approval identity, unique run
identity, reservation, cutoff, and artifact directory. The OS lock remains held
for the entire command, so concurrent invocations cannot overbook the allowance.

- A successful command with a reaped, contained process tree and completed log
  capture is charged measured elapsed time, releasing the unused reservation.
- Failure, timeout, interruption, uncertain termination, and crash recovery keep
  the entire reservation. If observed elapsed time exceeds it, the larger value
  is charged and the overrun is recorded.
- Crashes leave the durable reservation charged. The next invocation refuses
  new work until explicit recovery; no uncertainty earns a refund.
- Missing ledgers beside existing run artifacts and incomplete/corrupt ledger
  records fail closed. They require audit, not automated repair or deletion.

The child uses a shared-host monotonic cutoff, including launch overhead, so
wall-clock changes cannot extend its timeout. Its cutoff precedes the reservation deadline by 20% of the reservation,
capped at five seconds. A second supervisor timeout leaves additional cleanup
time inside the reservation. Admission and accounting enforce the approved cap;
OS suspension or pathological termination delays cannot be made into an absolute
physical-time guarantee. Any observed overrun is charged rather than hidden.

## Process containment and recovery

On Windows the dedicated supervisor establishes a Job Object with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and no breakaway permission before launching
the study command. Descendants inherit membership. The supervisor retains the
sole non-inheritable job handle; normal exit, forced termination, or supervisor
crash causes Windows to terminate contained descendants. Failure to establish
the job prevents child launch.

On POSIX, the supervisor and descendants share a new process group. The
supervisor terminates that group after recording its outcome; the accounting
parent also terminates the group during cleanup. This path supports commands
that do not deliberately daemonize or create independent sessions. Recovery
refuses a remaining group instead of guessing that it is safe to kill a reused
process identity. Windows containment is the tested production path here.

If the accounting wrapper dies, the supervisor still enforces the original
cutoff. After the reservation deadline, inspect the retained logs and use:

```powershell
.\.venv\Scripts\python.exe scripts/study_budget.py recover --run-id RUN_ID
```

Recovery obtains the OS lock, validates the chain and deadline, checks that the
recorded supervisor is gone, and appends a full-charge abandoned outcome. A live
or inaccessible PID, remaining POSIX group, or missing containment identity is
uncertain and blocks recovery. Do not bypass this by removing lock files or
editing previous events. OS locks release automatically when their owner dies.

## Artifacts and privacy

Each immutable run directory contains `provenance.json`, `supervisor.json`,
`result.json` when available, and separate child stdout/stderr and supervisor
logs. Provenance records the wrapper hash, Python/package versions, platform,
working directory, selected CUDA device, redacted command, and a small explicit
environment allowlist. It never dumps the full environment or PATH. Child
processes receive `STUDY_BUDGET_RUN_ID`, `STUDY_BUDGET_LEDGER`, and
`STUDY_BUDGET_DEADLINE_UNIX` so scientific receipts can link to the accounting run.

Credential-like argument values and inline command bodies are omitted from
provenance. Known credential values in the environment/credential arguments are
redacted from stdout/stderr. Commands should consume credentials through their
normal secure environment and must not print secrets: arbitrary transformed
secrets cannot be reliably detected by a log filter. Actual command arguments
are sent to the supervisor through a pipe, not saved in a plaintext command file.

A budget outcome establishes process/accounting status, not scientific validity.
Scientific metric hashes, numerical gates, data/model provenance and Jovin
receipts remain separate and must be linked explicitly by the experiment runner.

## CPU verification

`tests/test_study_budget.py` launches only short Python CPU commands in isolated
temporary ledgers. Tests exercise success, full-charge failure, timeout and
grandchild termination, cap rejection, approval rejection, immutable run IDs,
hash corruption, incomplete writes, missing-ledger reset prevention, concurrent
admission, accounting-wrapper crash, explicit recovery, and credential redaction.
These are not GPU experiments and do not spend the production study budget.
