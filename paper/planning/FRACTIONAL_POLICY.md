# Fractional matrix policy: C1 quadratic cap, version 1

Date: October 21, 2026. Status: implemented and verified on CPU; GPU and
long-run training validation remain required. Policy identifier:
`c1-quadratic-cap-frechet-v1`.

This policy repairs a confirmed nonfinite-gradient boundary in the old
fractional loss. It is a **versioned change to the objective near zero**, not an
algebraically equivalent speed optimization and not evidence of better language
model quality. Historical pilot artifacts use the old policy. Their sources were
preserved in `runs/optimization_20261021/source_snapshot_before_g0/` before this
change; the original failure is retained in
`paper/planning/fractional_gradient_diagnostic.json`.

## Why a new policy is necessary

For `0 < s < 1`, the scalar function `x^s` has no finite derivative at zero.
Differentiating its eigendecomposition also introduces eigenvector derivatives
that are singular at repeated eigenvalues, even when the final matrix function
would have a well-defined derivative. PyTorch documents the repeated-eigenvalue
limitation of eigenvector-dependent gradients in its
[eigh documentation](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.eigh.html).

The observed fixture `(torch.eye(4) * 100).requires_grad_()` makes off-diagonal
RBF affinities underflow to zero. Its Laplacian is zero, and the old fractional
loss returned zero while its input gradient contained 16 nonfinite elements.
Detaching the graph, assigning an arbitrary derivative to the original singular
power, or replacing nonfinite gradients with zero would not establish the
derivative of a declared differentiable objective. This policy instead defines
that objective explicitly.

## Scalar function, domain, and cutoff

The supported exponent domain is `0 < s <= 1`; research arms currently use
`s` in `{0.25, 0.5, 0.75}`. Let `delta = 1e-4`, and define

\[
a=(2-s)\delta^{s-1},\qquad b=(s-1)\delta^{s-2},
\]

\[
f_{s,\delta}(x)=
\begin{cases}
ax, & x<0,\\
ax+bx^2, & 0\leq x<\delta,\\
x^s, & x\geq\delta.
\end{cases}
\]

Both function value and first derivative agree at zero and at `delta`.
Consequently the scalar extension is C1, including the boundary of the positive
semidefinite cone. Its second derivative need not agree at these joins.

Properties established directly from this definition:

- `f(0)=0`, so exact null spaces remain null spaces.
- `f(x)=x^s` for every `x>=delta`: the forward change is confined to the
  near-zero band, plus the declared negative-roundoff extension.
- `f(x)>=0` and `f'(x)>0` for nonnegative `x`.
- `s=1` is exactly the identity function. The implementation returns the
  symmetrized matrix directly after checking its spectrum.
- On `[0,delta]`, `0<=x^s-f(x)<=delta^s`. One way to verify the first
  inequality is to apply the tangent-line bound at one to the convex function
  `t^(s-1)` and multiply by `t`. For a PSD matrix and features `Z`, the
  change in `0.5*Tr(Z^T f(L) Z)` is therefore bounded in magnitude by
  `0.5*delta^s*||Z||_F^2`. This is an absolute bound, not a claim of small
  relative error near zero.
- The largest scalar slope is `(2-s)*delta^(s-1)`: respectively **1750, 150,
  12.5, and 1** for exponents `0.25, 0.5, 0.75, and 1` at the default cutoff.
  Finite does not mean well-conditioned for every training trajectory.

The initial cutoff is an engineering choice in normalized-Laplacian units. The
training affinity Laplacian is constructed in FP32 before conversion to FP64;
promotion cannot recover the precision already lost in that construction. A
`1e-4` transition is substantially above unit-scale FP32 roundoff, while retaining
the original function for larger eigenvalues. It has **not** been demonstrated
optimal or harmless for training quality. Before protocol freeze, compare
`1e-5`, `1e-4`, and `1e-3` on fixed CPU fixtures and fixed exploratory checkpoint
batches, reporting component values, component-gradient norms, spectral mass in
the transition band, and numerical failures. Freeze the chosen cutoff before
confirmatory results; changing it requires a new method identity/configuration.

`fractional_matrix` explicitly uses `(L+L.T)/2`. Its intended scientific input is
a real symmetric PSD matrix. Nonfinite input and materially indefinite spectra
are rejected. The admissible computed eigenvalue range is

\[
\lambda_{\min}\geq-10^{-6}\max(1,\|\operatorname{sym}(L)\|_2).
\]

This narrow tolerance accommodates construction/eigensolver roundoff; it is not
permission to apply Laplacian claims to indefinite matrices. Within that band,
the tangent extension supplies a consistent derivative across zero. It can
produce tiny negative eigenvalues in the returned matrix when the input has
tiny negative eigenvalues. They are not silently clamped away. Acceptance and
sensitivity analysis must report such events and their scale. The cutoff and
negative-spectrum tolerance have separate roles.

## Matrix derivative and precision

For `sym(L)=Q diag(lambda) Q^T`, the forward result is

\[
F(L)=Q\operatorname{diag}(f(\lambda))Q^T.
\]

For a symmetric perturbation `E`, its first derivative is

\[
DF_L[E]=Q\{K\odot(Q^T E Q)\}Q^T,
\qquad
K_{ij}=\begin{cases}
\dfrac{f(\lambda_i)-f(\lambda_j)}{\lambda_i-\lambda_j}, & \lambda_i\ne\lambda_j,\\
f'(\lambda_i), & \lambda_i=\lambda_j.
\end{cases}
\]

The same self-adjoint operator supplies the reverse-mode gradient after
symmetrizing the upstream matrix gradient. A repeated eigenspace therefore
uses the derivative limit, without differentiating an arbitrary eigenvector
basis. The C1 matrix-function derivative is standard matrix-function theory;
see Kelly Bickel, *Differentiating Matrix Functions*, Theorem 3.1 and Proposition
3.5, [arXiv:1104.0336v2](https://arxiv.org/pdf/1104.0336v2). The particular
piecewise smoothing and numerical tolerances here are local method choices.

The implementation computes secants by integrating the derivative across the
negative, polynomial, and power intervals. The polynomial part uses its exact
secant, and the power part uses `log1p`/`expm1`. It does not subtract nearly equal
function values or divide by zero at repeated eigenvalues. All graph dependence
flows through this derivative; no stop-gradient is applied to the Laplacian.

The eigendecomposition, reconstruction, and custom backward run outside
autocast, in FP64 when the input is FP64 and otherwise FP32. Half/BF16 input is
promoted to FP32. The training caller continues to construct the affinity in
FP32 and call this function in FP64; its output is then converted back to FP32.
These precision choices must be shared by both comparison implementations.

Only first-order reverse-mode differentiation is supported. Double backward is
explicitly rejected through PyTorch's `once_differentiable`, following its
[custom-autograd guidance](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/extending.rst).
Forward-mode AD, vmap, complex matrices, and batched matrix inputs are not claimed.
The exponent and cutoff are fixed scalar hyperparameters, not differentiable
inputs.

## Integration and evidence boundary

The implementation is in `src/laplacian/operators_torch.py`. The function
`fractional_policy()` returns the default version, cutoff, negative-spectrum
tolerance, exponent domain, and derivative-order limit for run provenance.
An explicit nondefault `cutoff` must also be recorded by its caller.

Both unoptimized and optimized loss modules import the same `fractional_matrix`.
All future paired comparisons must use the same repaired module, cutoff,
precision, and policy metadata. Comparing the repaired implementation against
the historical unsmoothed objective would confound an optimization comparison.
The historical timing claims and hashes remain historical; they are not updated
to imply the repaired method was benchmarked then.

The pure reference fractional operator remains unchanged and unsmoothed. Its
Jovin receipts establish its own narrow tests, not this new boundary policy or
the new GPU backward. Agreement at exponent one remains relevant; it does not
certify fractional gradients at zero.

## CPU verification completed

Environment: existing project virtual environment, Python 3.12.10,
PyTorch 2.11.0+cu128, CPU only. Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_fractional_gradients.py tests/test_loss_optimization.py tests/test_torch_match.py tests/test_lambda2.py -q --tb=short
```

Result on October 21, 2026: **117 passed**, including **78 new fractional cases**.
No GPU or training workload was launched for this verification.

The new tests include known-eigenbasis analytic forward and Fréchet gradients,
the analytic nonzero derivative at the zero matrix, independent finite
differences at zero/repeated/near-repeated/join spectra, independently formed
NumPy dense matrix functions and directional differences, disconnected graphs,
permutation equivariance, material-indefiniteness/nonfinite rejection,
first-order-only enforcement, and CPU BF16 autocast precision checks. Actual
fractional-loss backward is exercised on separated, zero, constant, and duplicate
representations for all four tested exponents. An independent NumPy
affinity-to-loss finite-difference check confirms the graph gradient is retained.

These are host numerical checks and internal development evidence, not isolated
Jovin receipts, external replication, or confirmation of long-run stability.
GPU numerical checks, cutoff sensitivity, matched timing remeasurement, and the
remaining G0/G1 gates precede confirmatory training.
