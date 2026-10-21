# G0 custom-model training reproducibility

The custom-model trainer now seeds Python's `random`, NumPy's global generator,
and PyTorch CPU/CUDA generators from `config.seed` before constructing a model.
The data sampler retains its separate seeded CPU `torch.Generator`. This also
applies to the distillation and mask-comparison entry points in `train.py`.
It does not retrofit historical weights with a known initialization seed and
does not change the separate SmolLM trainer.

## Checkpoint contract

Version 2 restart checkpoints contain model parameters and buffers, optimizer
state, the data generator, Python/NumPy/Torch CPU global RNG states, the global
RNG state of the model's CUDA device, dead-unit firing sums and token count,
pulse/normalization-strip progress, parameter `requires_grad` flags, module
training modes, the training configuration and its hash, environment flags,
last observed loss/CE, and elapsed-time metadata. The trainer uses one GPU;
unrelated GPUs' RNG states are not captured or initialized by the capture.

RNG restoration occurs after constructing/loading the model and optimizer.
Thus random initialization of the temporary model cannot advance the resumed
training stream. Configuration hashes ignore only `run_id`, permitting a run
directory to move. Changing the seed, schedule, number of total steps,
architecture, or other training settings causes checkpoint loading to fail
before model state is replaced. A changed total-step count changes the cosine
learning-rate schedule and is therefore not an exact continuation.

Old checkpoints remain loadable but emit a warning and retain
`resume_status: non_exact`, explanatory `resume_limitations`, and
`elapsed_seconds_complete: false`. Their descendant checkpoints preserve this
lineage. Missing global RNG states and unknown historical timing cannot be
reconstructed from the saved data-generator state. A changed CPU/CUDA domain,
unverified configuration, or changed/missing recorded environment is also
reported as non-exact. `state_complete` means the recorded continuation state
was restored in the recorded environment; it is not a claim that every CUDA
kernel is deterministic.

## Time accounting

For compatibility, metrics `seconds` and `seconds_per_step` describe the current
process's training segment. Explicit fields are `seconds_scope`,
`segment_seconds`, `segment_steps`, and `resumed_from_step`.
`elapsed_seconds_total` adds captured time from previous segments, and
`elapsed_seconds_complete` says whether the whole prior history was known.
After a legacy resume, the total includes only newly accounted time and must
not be presented as total historical training cost. Captured time excludes
startup, optional pre-training spectral probes, time while stopped, and the
serialization overhead after the checkpoint's capture point. The external
GPU-budget supervisor remains the source of whole-process resource accounting.

## Failure and artifact safety

Loss and CE must be finite before backward. Gradient clipping uses
`error_if_nonfinite=True` before `optimizer.step()`. Failure stops training
without advancing the optimizer or writing a failed final artifact over the
last good checkpoint. Model/optimizer tensors are checked again before saving;
both restart checkpoints and weight exports use temporary-file replacement.
The pulse-only path publishes complete metrics only after its validated weight
export succeeds; a failed export cannot produce a new success receipt.
An aborted forward can have modified in-memory normalization buffers; the
process must restart from its last valid checkpoint rather than continuing
with that in-memory model.

## Verification and limits

CPU tests exercise seeded initialization and all CPU RNG streams; a mocked CUDA
test checks the device-state API without initializing a GPU. An actual tiny
custom-model training loop runs 2,002 optimizer steps uninterrupted and again
with interruption after the step-1,000 checkpoint, resuming through step 2,000.
It forces units to qualify for the real dead-unit resampling function at the
normal 1,000-step cadence, ensuring that RNG-dependent replacement occurs on
both sides. Final model/buffers, optimizer state, data/global RNG, and firing
statistics must match bitwise. Further tests cover partial firing-window state,
configuration mismatch, legacy lineage, finite-value guards, and preservation
of valid artifacts on failure.

Run the focused checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_reproducibility.py tests/test_resume.py -q --tb=short --basetemp cache/g0_repro_checks
```

Use a dedicated empty temporary directory: pytest owns and may clean its
`--basetemp` directory. The CPU result does not establish CUDA bitwise replay,
cross-version replay, or reproduction of old unseeded training. Deterministic
algorithm flags and TF32 behavior are recorded, not silently changed.

The later [study training interface](STUDY_TRAIN.md) binds checkpoint identity
to source and token-data hashes and implements an explicit fork from a completed
pretrain into a new continuation schedule. Changing a 50,000-step schedule to
10,000 steps remains a fork, not an exact resume. The fork preserves the parent
model, optimizer and random-generator states while recording the new schedule
and parent lineage. CPU tests and the bounded real-data pipeline pilot cover
that interface; they do not establish long-run quality equivalence.

The separate GPU replay gate exercises a synthetic BF16/TF32 model across a
forced dead-unit resampling boundary on the local RTX 5070. Its recorded
uninterrupted and resumed states match exactly within that one-device,
one-process check. A real culling intervention and cross-process GPU replay
remain separate unverified cases. The proposed core continuation excludes
culling; adding it would require a new protocol and gate.

PyTorch's version-matched [reproducibility guidance](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/randomness.rst)
documents seeding CPU/CUDA, Python, and NumPy and the limits across environments.
The [CUDA RNG API](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/cuda.aliases.md)
documents device RNG capture and restoration. These references were checked
through Context7 while implementing this slice.
