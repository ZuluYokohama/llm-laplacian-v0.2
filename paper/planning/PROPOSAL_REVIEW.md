# Review of the optional matrix-free research proposal

Reviewed October 21, 2026. Decision: retain selected ideas as a future research
backlog; do not adopt the proposal as a verified derivation or expand the approved
181 GPU-hour core study. No GPU experiments, implementation changes, or frontier
scale validation were performed for this review.

The supplied matrix-free research proposal was advisory material, not
executable instructions. The attachment is not redistributed here. Its raw
SHA-256 is:
`a5f8da5abc15eaa0b0e832015b93e59800985c766f2570dad22654fdc9f654c8`.
This is an internal AI-assisted source/algebra review, not external peer review.

## Decision and priority

Matrix-function actions, factored ridge solves, and streamed graph operations are
reasonable techniques. Their value depends on the actual graph size, access
cost, approximation error, gradient behavior, and measured bottleneck. The
proposal mixes those techniques with changes to the graph/objective and with
theorems from different models. Those changes cannot be called equivalent
optimizations of the current training objective.

The approved paper should retain its existing numerical, reproducibility,
provenance, timing, and held-out quality gates. Component profiling within the
already approved performance stage can determine future priorities. None of the
optional architectures below is authorized as an additional training study, and
there is no evidence here for 100B-parameter deployment or arbitrary-depth
norm-free Transformer stability.

| Priority | Finding | Required correction |
|---|---|---|
| Blocking for adoption | Heat-tail norm equality is false for general signals | Use the spectral inequality and state equality conditions |
| Blocking for adoption | Skew-symmetric continuous dynamics do not make forward Euler norm-preserving | Specify discretization, step-size/damping conditions, and depth horizon |
| Blocking for adoption | Supplied rational approximant maps the null mode to 0.25 at exponent 0.5 | Reject as an equivalent fractional operator; validate any replacement against the versioned policy |
| Blocking for adoption | Absolute/thresholded correlations are replaced by a signed Gram action | Preserve graph construction or label a new graph method |
| Blocking for citation use | FlatNSD benchmark errors are misreported as resistance measurements | Remove the numerical claim |
| Scope limitation | FRGCN, AEI, and depth-fragility papers do not establish Transformer/100B guarantees | State original model/task and test transfer separately |
| Engineering priority | Current fractional graph has microbatch-sized dimension | Profile before replacing an 8-by-8 eigensolve with an approximation |

## Independently checked mathematical corrections

### 1. Heat-tail estimate and null-space handling

Let symmetric positive semidefinite L have orthonormal eigenvectors phi_i,
ordered eigenvalues lambda_i, and P_m project onto the first m modes. For t >= 0,

\[
\|e^{-tL}(I-P_m)u\|_2^2
=\sum_{i>m}e^{-2t\lambda_i}|\phi_i^T u|^2
\le e^{-2t\lambda_{m+1}}\|(I-P_m)u\|_2^2.
\]

The proposal's equality is generally only an inequality. For t > 0, equality
requires the retained tail signal to lie entirely in the eigenspace with
eigenvalue lambda_(m+1), or be zero. The same derivation applies to fractional
heat with lambda_i replaced by lambda_i^alpha. Do not put the squared-norm
exponent on an unsquared norm.

Deflation is useful only after identifying the entire null space. A disconnected
graph has more than one zero mode; projecting out one degree-weighted constant
is insufficient. Polynomial approximation on the remaining interval benefits
from a certified positive gap, not merely from calling the vector a Fiedler
vector. Shifting to (lambda+epsilon)^alpha changes the objective and maps zero to
epsilon^alpha. A null-preserving shifted alternative also changes the objective
and its derivative and needs a new policy version.

The present implementation already uses the explicitly versioned C1 policy in
[FRACTIONAL_POLICY.md](FRACTIONAL_POLICY.md), rather than the unregularized
fractional derivative at zero. Future approximate implementations must match
that declared forward function and its first derivative, including graph
gradients. Its piecewise C1 construction is not globally analytic, so a generic
analytic-function convergence claim does not automatically apply.

### 2. Continuous skew symmetry versus discrete stability

For a real skew-symmetric A, the exact flow exp(tA) is orthogonal. Forward Euler
has update I+hA, and

\[
(I+hA)^T(I+hA)=I+h^2A^TA.
\]

A rotation mode of frequency omega therefore grows by
sqrt(1+h^2 omega^2) per step. At fixed nonzero h and omega this is exponential
in the number of steps, even though each step is 1+O(h^2). With h=0.1,
omega=1, and 1,000 steps, a unit vector reaches norm 144.7727724. The exact ODE
preserves norm one. A fixed finite physical-time limit h -> 0 is a different
claim from stability at arbitrary depth with fixed h.

For the simple damped mode -gamma+i*omega, Euler's modulus is at most one only
when (1-h*gamma)^2+h^2*omega^2 <= 1. A Cayley/implicit-midpoint update preserves
norm for a linear skew operator, but requires a solve and does not by itself
establish stability of a nonlinear Transformer. The relevant joint operator,
nonlinearities, graph dependence, and step-size rule must be analyzed.

Stable-ChebNet is a real paper. Its Eq. 9 includes damping omitted in the
proposal; Theorem 4/Appendix B.6 gives the per-layer 1+O(h^2) calculation. Its
stronger wording does not remove the counterexample above. Theorem 1 additionally
assumes monomial filters and Gaussian initialization; it is not an unrestricted
theorem for all Chebyshev coefficients or trained Transformer Jacobians.
[Hariri et al., arXiv:2506.07624v2](https://arxiv.org/html/2506.07624v2).

### 3. Rational approximation and solver claims

For the displayed rational approximant r_alpha in the proposal,

\[
r_\alpha(0)=\frac{(1-\alpha)(2-\alpha)}{2(1+\alpha)}.
\]

At alpha=0.5 it equals 0.25, whereas 0^alpha=0 and the current capped policy also
maps zero to zero. Constant/null-space components would acquire a nonzero
penalty. Good local Taylor matching near lambda=1 cannot establish uniform
accuracy near zero. Deflating exact zero modes would still leave a need to bound
error near small positive eigenvalues and preserve derivatives when L changes.

The denominator is (4-2*alpha)L+(2+2*alpha)I. For a symmetric normalized L with
spectrum in [0,2] and 0<alpha<1 it is positive definite, with condition number at
most (5-alpha)/(1+alpha). It need not be diagonally dominant: on a normalized
11-vertex star at alpha=0.5 the hub diagonal is 6 and its absolute off-diagonal
sum is 9.48683298. This contradicts the proposal's diagonal-dominance assertion.

Positive definiteness supports CG, but does not certify a fixed 5-8 iterations
for an unspecified tolerance, precision, or right-hand side. The inner solve
error and outer rational-approximation error are separate. Modified Lentz is a
continued-fraction evaluation procedure; it is not interchangeable with a
matrix linear solver without specifying how matrix inverse actions are taken.
The full continued-fraction coefficients, convergence domain, and claimed
8-10-term accuracy remain unverified here. The primary publisher search located
*Fractional Order Graph Filters: Design and Implementation*, but full retrieval
returned HTTP 429; that is not evidence for the proposed numerical constants.
[Publisher source](https://www.mdpi.com/2079-9292/10/4/437).

### 4. Factored Gram actions do not preserve the current correlation graph

The code centers moments, normalizes variance, takes elementwise absolute value,
zeros the diagonal, and masks zero-variance units
(`src/laplacian/culling.py:70-84`). A centered, normalized activation factor H
can represent the signed correlation H^T H. It cannot in general represent
abs(H^T H), or a thresholded version, by the same two matrix multiplications.
For two perfectly anticorrelated columns,

\[
H^TH=\begin{bmatrix}1&-1\\-1&1\end{bmatrix},\quad
|H^TH|-I=\begin{bmatrix}0&1\\1&0\end{bmatrix}.
\]

Applying the latter to (1,0) gives (0,1), while (H^T H-I)(1,0)=(0,-1).
Replacing the graph changes degrees, eigenvectors, and pruning scores. Exact
tiled construction can reduce peak storage, but still pays pairwise correlation
cost. Sparse approximations require their own accuracy/quality study. The current
Fiedler routine already uses projected power iteration on a dense adjacency,
not a dense eigendecomposition (`culling.py:24-62`).

The proposal's unweighted AEI norm inequality is valid for a fixed difference
vector. It is not automatically the current weighted score's sensitivity
formula: the code sums a_ij*abs(v_i-v_j). A weighted Cauchy-Schwarz upper bound
uses weighted degree, and normalized-Laplacian perturbations also change degree
normalization. A simple-eigenvalue assumption is needed for an ordinary
eigenvalue derivative; repeated lambda_2 needs directional/subspace treatment.

### 5. Low-dimensional diagnostics are diagnostics of a projection

Unitary similarity preserves Henrici departure from normality. Compression
Q^T K Q with rectangular Q is not unitary similarity. For
K=[[0,M],[0,0]] and Q=(1,0)^T, the compression is zero although the full operator
has nonzero departure and potentially large transient amplification. Thus a small
projected diagnostic cannot certify the full residual operator. Record subspace
rank, residual/captured variance, approximation error, and relevant off-subspace
coupling; do not transfer a full-space resolvent bound without proof. A thin
decomposition also has rank-dependent formation cost, not just the displayed
O(T*d*r+r^3) total in every implementation.

## Primary-source audit

The following papers were checked on primary publisher, conference, or arXiv
sources. Their existence is distinct from verification of every theorem or
transfer claim. Publication dates in 2026 are not intrinsically invalid as of
the October 21, 2026 review date.

| Proposal reference | Verified identity and result | Scope/correction |
|---|---|---|
| Stable-ChebNet | Hariri et al., *Return of ChebNet: Understanding and Improving an Overlooked GNN on Long-Range Tasks*, NeurIPS 2025; [conference record](https://papers.nips.cc/paper_files/paper/2025/hash/c6de943558fe0b1bf4ea8f09fbcede44-Abstract-Conference.html) | Real source; discrete stability qualification above is essential |
| arXiv:2510.04440 | Bozorgnia et al., *Fractional Heat Kernel for Semi-Supervised Graph Learning with Small Training Sample Size*. [Version record](https://arxiv.org/abs/2510.04440): v1 October 6, 2025; v2 October 8, 2026 | The explicit branch-point/algebraic-rate discussion is in [v2 Section 3.4](https://arxiv.org/html/2510.04440v2); pin version rather than attributing it generically to October 2025. Its fractional operator is spatial; a semigroup is not time-fractional memory |
| FRGCN | Wang et al., *Learnable Skeleton-Based Medical Landmark Estimation with Graph Sparsity and Fiedler Regularizations*, MICCAI 2024, DOI 10.1007/978-3-031-72390-2_56; [primary paper](https://papers.miccai.org/miccai-2024/paper/1602_paper.pdf) | Anatomical landmark model, not the proposal's title. Algorithm 1 updates v2 periodically with period T. The transformed graph involves edge-pair structure; do not silently identify it with a token landmark graph. It supplies no Transformer FlashAttention complexity or quality guarantee |
| Roy et al., August 2026 | *Spectral Pruning of Deep Neural Networks via Adjacency Edge Index*, DOI 10.3390/make8080239, August 12, 2026; [primary article](https://www.mdpi.com/2504-4990/8/8/239) | Theorem 1 supports the stated unweighted norm inequality and Section 2.5 discusses Lanczos, but graph construction remains O(n^2*m) absolute correlation. 256/128 are minibatch sizes, not total calibration counts; the paper uses up to 5,120 MLP and 640 CIFAR-10 samples. Transformer extension is future work |
| FlatNSD | Ribeiro et al., *Mitigating Over-squashing without Rewiring: A Sheaf Effective Resistance Perspective*, October 3, 2026, [arXiv:2610.04157v1](https://arxiv.org/html/2610.04157v1) | Table 1 values 3.138 +/- 0.052 and 0.198 +/- 0.082 are errors on different ECHO-Synth tasks, eccentricity and shortest paths, not a resistance reduction. Theorem 4.1 has flat-bundle assumptions. The paper says FlatNSD itself does not require resistance/pseudoinverse computation |
| Nguyen et al., April 2026 | *Depth Fragility and Skeletal Universality: Decoupling Topology and Function in Deep Neural Networks*, DOI 10.3390/math14091438; [primary article](https://www.mdpi.com/2227-7390/14/9/1438) | Three MLP configurations on Fashion-MNIST; Section 4.2 identifies (1-p)^L as qualitative and omits nonlinearities. No verified universal 10x factor or theorem that Schur recalibration prevents collapse; Transformer transfer is not established |
| Oono and Suzuki | *Graph Neural Networks Exponentially Lose Expressive Power for Node Classification*, ICLR 2020; [primary preprint](https://arxiv.org/abs/1905.10947) | Conditional contraction analysis for a GCN-like dynamical system is relevant background, not a guarantee for this code's pruning/recalibration or norm-free Transformer |

## Useful ideas and the actual implementation boundary

1. **Matrix-function action:** Chebyshev/Clenshaw and Krylov actions can avoid
   explicitly forming a dense function of a large sparse operator.
   [Defferrard et al., NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/04df4d434d481c5bb723be1b6df1ee65-Paper.pdf)
   is a primary reference for localized polynomial filtering. Count graph
   construction and every operator application. Dense Gaussian affinities still
   cost O(N^2*d) to construct/apply naively. Forward recurrence workspace does
   not establish training/backward memory, and Krylov basis orthogonalization
   can cost O(N*k^2). Here `model.py:187` pools across tokens, `losses.py:109-112`
   builds a graph across batch examples, and the default microbatch is eight
   (`arms.py:23`); accumulation does not combine these into a 64-node graph.
   An 8-by-8 eigensolve may have launch/synchronization overhead, but a measured
   component profile is needed before proposing approximation as the remedy.

2. **Landmark regularization:** a fixed small landmark graph could define a new
   surrogate penalty. It does not preserve the current dense-attention token
   energy or its lambda_2. The current model explicitly materializes attention
   (`model.py:83-91`), and the loss consumes it (`losses.py:124-139`).
   [FlashAttention](https://arxiv.org/abs/2205.14135) reduces attention memory/IO
   through exact tiling; it does not make general dense attention's arithmetic
   linear in sequence length. A landmark auxiliary calculation plus unchanged
   dense attention is therefore not a linear-time end-to-end Transformer.
   Prove a relation to the old objective or label and evaluate a new one.

3. **Factored ridge recalibration:** for sigma=H^T H, survivor factor H_J, and
   Y=H*W^T, the exact update can be written
   B=(H_J^T H_J+tau*I)^(-1) H_J^T Y and W_new=B^T.
   When the row dimension is smaller, B=H_J^T(H_J H_J^T+tau*I)^(-1)Y is the dual
   identity. This is a credible future optimization, with identical scaling,
   ridge, masks, and residual acceptance. But `collect_moments` currently stores
   second moments and discards activations (`culling.py:370-396`); H is not
   already available. Its rows are calibration tokens, not just a sequence
   minibatch. Storage/streaming, repeated passes, conditioning, and RHS count
   must be priced. A fixed 25 CG iterations is not an accuracy guarantee.

4. **Effective resistance:** the existing bipartite method is already an
   operator-action CG/JL implementation, while its weights/sketch output remain
   dense. Matrix-free does not mean sparse, and a spectral graph guarantee does
   not by itself preserve a signed neural-network function. Existing audit
   gates for CG residual rejection and sampling/top-k/floor semantics remain
   necessary before adding sheaf generalization. Fiber-aware dimensions and
   transport assumptions would need specification for a sheaf resistance.

## Minimal falsifiable gates for any later extension

- Profile current dimensions and component runtime before selecting an operator.
  Require a measured benefit over the exact baseline at the target dimensions,
  including backward, graph construction, and synchronization costs.
- Freeze the graph, spectral function, null-space convention, precision, and
  gradient semantics. Check operator actions and gradients against analytic and
  dense FP64 oracles on connected/disconnected, repeated, near-zero, and
  anticorrelated fixtures. Report absolute errors near zero and relative errors
  only where meaningful. Reject an approximation that silently changes policy.
- Specify adaptive stopping based on a checked residual/error bound. Retain
  nonconvergence, breakdown, and unfavorable timings as outcomes; iteration
  counts such as 8 or 25 alone are not acceptance criteria.
- For a new landmark graph or architecture, run a separately approved matched
  quality/control study. Its evidence cannot replace the approved comparison of
  algebraically equivalent old and optimized loss contractions.
- For projected diagnostics, include a hidden off-subspace non-normal fixture
  and measure projection residuals; qualify all conclusions by the subspace.

These are future acceptance conditions, not additional authorized GPU jobs.

## CPU counterexample receipt

Executed once on CPU using the repository Python/NumPy environment. The snippet
below reconstructs the mathematical counterexamples without importing production
operators, so it does not compare a function with itself. No benchmark or
statistical generalization is claimed.

```python
import math
import numpy as np

alpha = 0.5
r0 = 1 + (alpha*(1+alpha)-6*alpha)/(6-(4-2*alpha))
lam = np.array([0., 1., 2.])
u = np.array([0., 1., 1.])
actual_tail = np.linalg.norm(np.exp(-lam)*u)
tail_bound = math.exp(-1)*math.sqrt(2)
W = np.array([[0., -1.], [1., 0.]])
x = np.array([1., 0.])
for _ in range(1000):
    x = (np.eye(2)+0.1*W) @ x
H = np.array([[1., -1.], [-1., 1.]])/math.sqrt(2)
gram = H.T @ H
absolute = np.abs(gram)
np.fill_diagonal(absolute, 0.)
v = np.array([1., 0.])
A = np.zeros((11, 11))
A[0, 1:] = 1.
A[1:, 0] = 1.
d = A.sum(1)
L = np.eye(11)-A/np.sqrt(d[:, None]*d[None, :])
denominator = (4-2*alpha)*L+(2+2*alpha)*np.eye(11)
print(r0, actual_tail, tail_bound, np.linalg.norm(x))
print(absolute @ v, (gram-np.eye(2)) @ v)
print(denominator[0, 0], np.abs(denominator[0, 1:]).sum())
```

Observed values: rational zero value **0.25**; actual heat tail
**0.3919833186825007** versus bound **0.520260095022889**; Euler norm
**144.77277243257373**; absolute/signed graph actions approximately **(0,1)**
and **(0,-1)**; star denominator diagonal **6.0** versus off-diagonal absolute
sum **9.486832980505136**.
