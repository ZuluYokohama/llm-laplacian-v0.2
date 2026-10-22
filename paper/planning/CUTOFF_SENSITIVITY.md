# Bounded fractional-cutoff sensitivity evidence

Date: October 21, 2026. Policy: `c1-quadratic-cap-frechet-v1`; default cutoff
\(\delta=10^{-4}\). This supplements
[`FRACTIONAL_POLICY.md`](FRACTIONAL_POLICY.md). It does not establish that the
cutoff is statistically optimal or harmless for language-model quality.

## Actual-checkpoint pilot, interpreted narrowly

The existing host diagnostic
[`g0_cutoff_sensitivity_001/metrics.json`](../../runs/publication_study/g0_cutoff_sensitivity_001/metrics.json)
records exponent 0.5, three sampled batches, and cutoffs \(10^{-5},10^{-4},10^{-3}\)
at the same historical Arm 0 checkpoint. It uses BF16/TF32 production settings,
reloads the initial model state before each case, and computes fractional-only
parameter gradients. It performs no optimizer updates. All nine cases were
finite. Maximum relative component-value change from the default was
\(3.1997\times10^{-5}\) (0.00320%); maximum relative parameter-gradient
L2 change was 0.0031104 (0.311%).

These host hashes identify artifact bytes; they are not process-isolated
Jovin-chain receipts. The linked diagnostic artifacts are local and are not
distributed in PR #2.

Each of the fifteen inspected 8-by-8 graphs had only one eigenvalue below
\(10^{-3}\), at roundoff scale, with minimum eigenvalues across graphs from
approximately \(-3.12\times10^{-8}\) to \(3.55\times10^{-8}\). Thus this
pilot principally probes the computed null mode. It does not test true
positive modes inside the transition interval, nearly disconnected graphs,
or all possible training states. There is no same-cutoff A/A noise control in
this pilot, so its small gradient differences should not all be attributed to
cutoff choice independently of mixed-precision arithmetic. This is engineering
evidence from an exploratory checkpoint, not new held-out quality evidence.

## CPU analytic-spectrum supplement

Command, run on CPU only:

```powershell
.\.venv\Scripts\python.exe scripts/check_fractional_cutoff_cpu.py --output runs/publication_study/g0_cutoff_cpu_001
```

The output directory is created with `exist_ok=False`. The diagnostic copies
its own source, the operator, runtime writer and policy document to
`source_snapshot/`, records raw-byte SHA-256 hashes, verifies each snapshot,
and checks source hashes again after computation. A computation failure produces
an incomplete metrics receipt; a failed output is not promoted or overwritten.
The metrics sidecar uses the project's documented LF-normalized text convention;
source hashes use raw bytes.

Receipt/failure checks in `tests/test_fractional_cutoff_cpu.py` also passed
on CPU: **2 passed**. They rerun the 144-case sweep, verify archived source
bytes and the sidecar, reject reuse without modifying the existing receipt,
and inject an operator exception to check incomplete status and restoration
of the prior CPU thread setting.

The receipt is
[`g0_cutoff_cpu_001/metrics.json`](../../runs/publication_study/g0_cutoff_cpu_001/metrics.json).
Its raw-byte SHA-256 is
`49e573400a5709355b5e972e8958fdaae80ae1f6ef016b29d6522e975bd7ae41`;
its LF-normalized sidecar is
`a771ee121addc73d1f74b9b66ae2c3eb39ec140b7400bfaad48025e92d6d4d2e`.
This is a host diagnostic with file-identity hashes, not a Jovin-chain receipt.
The linked artifact is local and is not distributed in PR #2.

The supplement uses float64, one deterministic 4-by-4 orthogonal basis, a
fixed 4-by-3 feature matrix and a symmetric unit-Frobenius perturbation, all
recorded in the receipt. Twelve spectral fixtures cover exact zero, repeated
positive eigenvalues, two-dimensional null space, a small positive gap,
repetition inside the default transition, a tolerated negative-roundoff mode,
each cutoff join, and three spectral scales. Crossing four exponents
\(\{.25,.5,.75,1\}\) with three cutoffs gives **144 cases; all passed**.

The forward and matrix-gradient oracle uses the known spectrum and basis,
an independently expressed scalar polynomial, and explicit divided
differences with the analytic derivative on equal eigenvalues. A separate
central directional finite difference uses NumPy's dense eigensolver on
perturbed matrices, rather than the production autograd implementation.
Both oracles intentionally implement the declared smoothed function; they
test its implementation, not agreement with an undefined derivative of
\(x^\alpha\) at zero. The finite-difference step is \(10^{-5}\delta\).

| Check across 144 cases | Maximum observed error | Predeclared criterion |
|---|---:|---|
| Matrix forward, absolute entry | \(2.40\times10^{-12}\) | `atol=2e-9, rtol=1e-9` |
| Matrix gradient, absolute entry | \(3.01\times10^{-7}\) | `atol=2e-7, rtol=1e-8` |
| Directional difference, scaled | \(1.15\times10^{-5}\) | \(5\times10^{-5}\) |

The gradient criterion is elementwise absolute-plus-relative; the maximum
absolute error can exceed its absolute component while passing the combined
criterion. Directional error divides the absolute difference by
\(\max(1,|\text{analytic directional derivative}|)\). For PSD fixtures, the
diagnostic also checks the declared unsmoothed-forward energy gap bound
\(0\leq E_{\rm power}-E_{\rm cap}\leq\frac12\delta^\alpha\|Z\|_F^2\).
This is a bound on forward energy, not on its relative gradient error.

## Interpretation and freeze recommendation

Cutoff sensitivity can be large even when the forward value is unchanged.
At \(L=0\), all tested cutoffs give zero energy, while the matrix gradient is
\(\tfrac12(2-\alpha)\delta^{\alpha-1}ZZ^T\). For \(\alpha=.5\), reducing
the default cutoff tenfold multiplies the gradient norm by \(\sqrt{10}\),
a relative change of 216.23%; increasing it tenfold reduces the norm by
68.38%. At \(\alpha=.25\), the corresponding smaller-cutoff relative change
is 462.34%. These expected differences are changes to the declared method,
not implementation failures. They also occur in positive repeated/small-gap
fixtures. A fixed absolute cutoff deliberately breaks unregularized power
scaling when eigenvalues cross its transition interval.

The CPU matrix-gradient results cannot be translated directly into model
parameter-gradient changes. Synthetic spectra need not be realizable by the
model's Gaussian affinity graph, and the graph-construction Jacobian can
attenuate or eliminate matrix-gradient directions. One basis, feature matrix
and perturbation direction are finite tests, not a uniform numerical proof.
The analytic function and its C1 derivative definition provide the method;
these independent computational constructions check representative boundary
cases.

Retaining \(\delta=10^{-4}\) is defensible for the bounded core **as an
explicitly fixed engineering policy**, shared by baseline and optimized
variants. Its justification is a finite derivative, agreement above the
declared band, analytic boundary checks and the bounded checkpoint diagnostic.
It is not justified by a claim of uniformly small sensitivity. Freeze this
identity before confirmatory training and retain the negative-spectrum
rejection and nonfinite guards. A policy change after inspecting confirmatory
results would define a new method and require new comparisons. A full cutoff
training search is not established or requested by this diagnostic.

If subsequent runs show material low-positive spectral mass, instability or
unexpected gradient sensitivity, the minimal follow-up is a same-checkpoint,
same-cutoff A/A comparison plus an FP32 diagnostic to separate arithmetic
variation from policy effects, followed by the already specified training
gates. The present results alone do not authorize a new GPU search or broaden
the approved 181 GPU-hour core budget.
