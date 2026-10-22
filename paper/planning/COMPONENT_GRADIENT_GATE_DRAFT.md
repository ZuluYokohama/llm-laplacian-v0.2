# Individual-component gradient diagnostic — prospective draft

This is an exploratory engineering diagnostic. Its initial source snapshot was
frozen before hardware execution; subsequent results remain separate artifacts.
It does not freeze or promote
the confirmatory study, establish quality, or demonstrate runtime improvement.
Policy identifier: `component-unweighted-bf16-v1-draft`.

## Question and inputs

For the same model state, training batches and repaired fractional policy, does
each active baseline penalty agree numerically with the optimized penalty in
its **unweighted scalar and complete parameter-gradient vector**? CE and other
penalties never enter that component's backward call. The old implementation
may compute other terms, but their tensors are not added to the scalar.

Default phase 135000 selects manifold, fractional, hypergraph and Dirichlet.
Optional phase 0 selects only Dirichlet. Schedule magnitudes are recorded solely
to define activity; none rescales the tested scalar or gradient. One accumulated
batch comprises eight microbatches of eight sequences, context 256. Scalars and
gradients are their arithmetic means. This tests that accumulated objective, not
every individual example gradient. No optimizer update, clipping, resampling,
checkpoint write, evaluation or CE backward occurs.

The production model is the existing five-layer, width-256, M=4/k=0 Transformer,
GPT-2 vocabulary 50,257, selected layers 2/5 and fractional exponent 0.5.
BF16 autocast and TF32 enabled follow the production configuration; the repaired
matrix function retains its existing precision policy. Both implementations
must import the identical repaired `fractional_matrix` object. Exact trace and
contraction identities concern real arithmetic; these finite-precision tests
cannot establish bitwise or universal equivalence.

Inputs are a trusted local full checkpoint from a **completed CE study run**:
either its metrics-bound `last.pt`, or a declared `state_stepNNNNNNN.pt` in its
completion receipt. This admits the seed-9002 `pilot_v2/ce_002` integration input
and later prospectively chosen production CE snapshots. Completion does not
make the tiny pilot a mature training state. Config, metrics, sidecars, step,
checkpoint bytes, token bundle and current source policy are checked before
unpickling; hashes do not make arbitrary pickle content safe.

Only the validated training partition is sampled, using the restored local
sampler RNG. Offsets, input/target tensor hashes and sampler hashes are recorded.
Each implementation/component starts with the same parameter/buffer state,
training/gradient flags, and Python/NumPy/Torch CPU/CUDA RNG. Training RMSNorm
legitimately updates running buffers during forward: baseline/candidate final
buffers must match, parameters must remain unchanged, and the next pass resets
the original buffers. No test/development/calibration model scoring occurs.

## Fixed prospective BF16 acceptance policy

The following thresholds are set **before hardware output** and are not CLI
overrides. They retain the prior engineering scale for BF16 comparisons, while
separating every term and strengthening tiny-gradient/local-tensor protection.
They are not justified as statistically negligible changes to training quality.

| Check, separately for every active component | Required result |
|---|---|
| Scalar value | `abs(candidate-reference) <= 1e-5 + 0.005*abs(reference)` |
| Nonzero reference gradient vector | `||candidate-reference||_2 / ||reference||_2 <= 0.025`, with **no epsilon or absolute floor** |
| Exactly zero reference gradient vector | Exactly zero candidate norm; no relative ratio is defined |
| Each parameter tensor | Maximum absolute error `<= 1e-5 + 0.025*max(abs(reference_tensor))`; an exactly zero reference tensor requires exactly zero candidate tensor |
| Connectivity, shape and finiteness | Identical parameter names/shapes and `None` versus tensor semantics; every scalar/gradient finite |

All criteria must pass. Gradient norms/errors are accumulated in float64.
Tiny positive reference norms (below `1e-5`) are flagged but retain the relative
cap: erasing even an arbitrarily small nonzero gradient fails. No positive
absolute allowance can turn that erasure into a pass. Zero-gradient passes are
marked as lacking nonzero-gradient coverage; they do not certify behavior away
from that state. The receipt reports all component outcomes and per-parameter
errors, not only a pooled favorable metric.

A failure retains its immutable output and fixed thresholds. Any later policy
revision requires a new version, rationale and receipt and must disclose prior
observations; do not silently relax this policy after GPU results. Agreement
with the baseline is implementation fidelity, not an independent proof that
the baseline's scientific objective or stopped-gradient heuristic is correct.

## Execution and artifact boundary

`scripts/check_component_gradients.py` requires a nonblank budget run ID, the
canonical ledger and approved visible RTX 5070/BF16 before any CUDA query. The
cooperative limit defaults to 120 seconds and cannot interrupt an in-flight
CUDA call; the budget wrapper supplies the hard process deadline. The output
directory must be new. Failed and successful directories cannot be reused.

The script binds sources before/after fresh imports, copies the exact sources,
and rechecks source/archive/input identities after execution. Config and metrics
bytes are captured before parsing; JSON admission and exact checkpoint hashing
precede full-state load. Final bundle validation rechecks its input closure.
Generated sampling logs and tensor identities are bound to the receipt. This is
host provenance, not a Jovin process-isolated attestation or independently
verified in-memory bytecode provenance. Future source changes can invalidate
old configurations; do not patch their hashes to make admission pass.

Illustrative wrapper payload, to be launched only by the study coordinator with
a conservative reservation (this document schedules nothing):

```powershell
.venv/Scripts/python.exe scripts/check_component_gradients.py `
  --checkpoint runs/publication_study/pilot_v2/ce_002/last.pt `
  --output runs/publication_study/COMPONENT_NEW_OUTPUT `
  --phase 135000 --max-seconds 120
```

## CPU contract verification

The new test file uses tiny injected nonproduction inputs to exercise full
receipt flow, source/config/data mutation, immutable failure, checkpoint
admission before unpickling, matching state/RNG resets, and no optimizer update.
Single-component sign/reweight/erasure injections preserve forward values and
make other components enormous, ensuring they cannot conceal the damaged
gradient. A nonzero gradient near `1e-30` is explicitly erased and rejected.
CPU fakes check budget/device admission without querying actual CUDA. At the
initial freeze, the production GPU gate and mature-checkpoint coverage were
unexecuted; the CPU results alone do not establish either.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_component_gradients.py -q
```

CPU result on October 21, 2026: **38 passed, 1 warning in 5.85 seconds**.
The warning is the existing culling tensor-to-scalar conversion exercised by
the baseline. These injected CPU fixtures do not admit a production GPU run or
establish that the BF16 gate will pass on the selected checkpoint.
