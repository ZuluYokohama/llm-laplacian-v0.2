# Implemented spectral objectives and computational scope

Date: October 21, 2026. This is a methods specification of the current code,
not a claim that the penalties improve language-model quality. Equations below
describe real-arithmetic objectives; mixed-precision evaluations and reordered
contractions need numerical comparisons. The fractional boundary policy is an
explicit objective change shared by both comparison implementations.

Source anchors: [`losses.py`](../../src/laplacian/losses.py),
[`operators_torch.py`](../../src/laplacian/operators_torch.py),
[`model.py`](../../src/laplacian/model.py),
[`train.py`](../../src/laplacian/train.py),
[`schedule.py`](../../src/laplacian/schedule.py), and
[`culling.py`](../../src/laplacian/culling.py). A run's archived, hashed sources
and configuration, rather than a later working tree, define its actual method.

## Tensors and training unit

Let \(B\) be microbatch size, \(T\) sequence length, \(D\) residual-stream
width, \(J\) number of layers, \(H\) attention heads, and \(N=mD\) MLP width.
At each layer \(\ell\), the model returns:

| Symbol | Code output | Shape | Definition |
|---|---|---|---|
| \(U\) | `embedded` | \(B\times T\times D\) | Token embeddings before the blocks |
| \(X_\ell\) | `stream[ell]` | \(B\times T\times D\) | Residual stream after the block |
| \(Z_\ell\) | `pooled[ell]` | \(B\times D\) | Mean of \(X_\ell\) over token positions |
| \(F_\ell^\mathrm{raw}\) | `hidden[ell]` | \(B\times T\times N\) | ReLU MLP activations after applicable gates and masks |
| \(W_\ell\) | `attn[ell]` | \(B\times H\times T\times T\) | Causal attention softmax weights |

The main M=4 configuration uses \(J=5,D=256,H=8,N=1024,B=8,T=256\).
The vocabulary size follows the tokenizer; the GPT-2 store uses 50,257 tokens.
The small hardware resume gate instead uses vocabulary 128 and context 16 and
does not establish full-context throughput or quality. Eight accumulated
microbatches form an optimizer update in the default configuration. Each
microbatch builds its **own 8-by-8 pooled-representation graph**. Accumulation
averages eight losses/gradients; it does not build a 64-by-64 graph.

For logits \(o_{bt}\) and next-token targets \(y_{bt}\), the implemented loss is

\[
\mathcal L=\frac1{BT}\sum_{b,t}\operatorname{CE}(o_{bt},y_{bt})
 +\gamma_M M+\gamma_F F+\gamma_H H_\mathrm{hyp}+\gamma_D D_\mathrm{band}.
\]

The current target stream has no padding mask in `forward_loss`. There is no
implicit layer, neuron, or event normalization beyond the factors specified
below. Consequently the unweighted penalties have different scales and
different dependencies on batch size, sequence length, and width.

## Pooled graph and manifold penalty

For \(Z\in\mathbb R^{B\times D}\), with \(\tau=1\) by default, define

\[
A(Z)_{ij}=\exp\!\left[-\frac{\|z_i-z_j\|_2^2}{2\tau^2}\right],\quad
d_i=\sum_j A_{ij},\quad L(Z)=I-D_d^{-1/2}A D_d^{-1/2}.
\]

Self-loops \(A_{ii}=1\) are retained. For finite representations these ensure
positive degrees, even when all off-diagonal affinities underflow to zero.
The code has a zero-degree fallback, although it is unnecessary for this
finite RBF construction. Distances, affinity entries and normalization are
formed from FP32 tensors.

Let \(S\) be the selected layer indices and \(K=|S|\). The default is the
second and fifth layers, represented by zero-based indices `[1, 4]`.

\[
M=\frac1{K^2}\sum_{s\in S}\sum_{t\in S}
       \frac{\operatorname{Tr}(Z_t^T L(Z_s)Z_t)}{B^2}.
\]

All ordered layer pairs, including self-pairs, contribute. Gradients flow
through both the graph-building representation and the target representation.
`_quadratic` uses the identity
\(\operatorname{Tr}(Z^TLZ)=\sum_{i,d} Z_{id}(LZ)_{id}\), avoiding a
\(D\times D\) output. This contraction is an algebraic identity. A claim that
minimizing it preserves semantics or improves generalization is an empirical
hypothesis, not a consequence of that identity.

## Fractional penalty and boundary policy

The fractional term uses **all layers**, independently of \(S\):

\[
F=\frac12\sum_{\ell=1}^J
        \operatorname{Tr}\{Z_\ell^T f_{\alpha,\delta}(L(Z_\ell))Z_\ell\}.
\]

For \(0<\alpha\leq1\), policy `c1-quadratic-cap-frechet-v1` sets
\(\delta=10^{-4}\), \(a=(2-\alpha)\delta^{\alpha-1}\),
\(b=(\alpha-1)\delta^{\alpha-2}\), and

\[
f_{\alpha,\delta}(x)=
\begin{cases}
ax,&x<0,\\
ax+bx^2,&0\leq x<\delta,\\
x^\alpha,&x\geq\delta.
\end{cases}
\]

The matrix is symmetrized before eigendecomposition. Materially negative
eigenvalues, below \(-10^{-6}\max(1,\|\operatorname{sym}L\|_2)\), cause
failure. The negative branch is only a roundoff extension. The function is C1,
preserves exact null spaces, and agrees with the unsmoothed power above the
cutoff. Its first derivative is the Fréchet divided-difference derivative,
including the scalar derivative limit at repeated eigenvalues. The graph
gradient is retained. Only first-order reverse-mode differentiation is supported;
the exponent and cutoff are fixed hyperparameters.

The training caller promotes the already constructed FP32 Laplacian to FP64
for the matrix function, then converts its result to FP32. Spectral
reconstruction and its custom backward disable autocast. Promotion does not
recover accuracy lost during the original affinity construction. Other matrix
multiplications in the loss can still be autocast under BF16 training even
when their operands were explicitly converted to FP32.

The complete definition, derivative, error bound and initial CPU tests are in
[`FRACTIONAL_POLICY.md`](FRACTIONAL_POLICY.md). The default cutoff is an
engineering choice in normalized-Laplacian units, not an optimized statistical
hyperparameter. The subsequent bounded sensitivity evidence and its limits are
in [`CUTOFF_SENSITIVITY.md`](CUTOFF_SENSITIVITY.md).

## Hypergraph penalty, degree-scaled features, and contraction proof

For each layer, flatten batch and time into \(E=BT\) events and transpose:
\(F_\ell\in\mathbb R^{N\times E}\). Neurons are vertices; events are
hyperedges. Define the binary incidence matrix and degrees by

\[
C_{ve}=\mathbf1[F_{ve}>0],\quad
d_v=\sum_e C_{ve},\quad d_e=\max(\sum_v C_{ve},1),
\quad V_{vv}=\begin{cases}d_v^{-1/2}&d_v>0\\0&d_v=0.\end{cases}
\]

All hyperedge weights are one. Set \(S=VC\), \(X=VF_\ell\), and
\(D_e=\operatorname{diag}(d_e)\). The implemented per-layer scalar is

\[
h_\ell=\left[\|X\|_F^2-
            \operatorname{Tr}(X^TS D_e^{-1}S^TX)\right]_+,
\qquad H_\mathrm{hyp}=\sum_{\ell=1}^J h_\ell.
\]

In particular, with \(L_h=I-S D_e^{-1}S^T\), the unclamped energy equals
\(\operatorname{Tr}(F_\ell^T V L_h V F_\ell)\). The extra vertex-degree
scaling of the **features** is part of the existing objective. It is not the
unscaled feature energy \(\operatorname{Tr}(F_\ell^T L_hF_\ell)\).
Isolated vertices have zero rows in \(X\), so their identity diagonal in
\(L_h\) has no effect on this energy. Empty events have zero incidence and
zero activations; clamping their edge degree avoids division by zero.

To prove the optimized contraction, let \(Y=S^TX\in\mathbb R^{E\times E}\).
Then, by expanding the trace,

\[
\operatorname{Tr}(X^TS D_e^{-1}S^TX)
 =\operatorname{Tr}(Y^TD_e^{-1}Y)
 =\sum_{e,r}\frac{Y_{er}^2}{d_e}.
\]

Alternatively form \(P=(S D_e^{-1})S^T\in\mathbb R^{N\times N}\):

\[
\operatorname{Tr}(X^TPX)=\sum_{v,e}X_{ve}(PX)_{ve}.
\]

The code chooses the \(N\times N\) form when \(N<E\), and the
\(E\times E\) form otherwise. The equality is exact in real arithmetic;
floating-point association can change the rounded value and gradient. For
nonnegative incidence the normalized hypergraph Laplacian is PSD: applying
Cauchy-Schwarz within each event gives
\(\sum_e(\sum_v C_{ve}u_v)^2/d_e\leq\sum_v d_vu_v^2\).
The unclamped scalar is therefore nonnegative in exact arithmetic. The final
`clamp_min(0)` handles cancellation but has a nonsmooth boundary.

The incidence test `F > 0` has no autograd derivative. Gradients differentiate
the feature expression with incidence held fixed on each activation region.
This is not a differentiable learned incidence construction at changes of
support. Nonnegativity or the contraction identity does not prove that this
penalty induces a desirable linguistic topology.

## Token energy and detached spectral band heuristic

For each layer and example, average attention over heads, symmetrize, remove
the diagonal and clamp negative entries:

\[
\bar W_{\ell b}=H^{-1}\sum_h W_{\ell bh},\qquad
A_{\ell b}=\left[\tfrac12(\bar W_{\ell b}+\bar W_{\ell b}^T)
              -\operatorname{diag}(\bar W_{\ell b})\right]_+.
\]

Write \(d=A\mathbf1\), \(L_c=\operatorname{diag}(d)-A\), and
\(q=\max(\#\{(i,j):i<j,A_{ij}>0\},1)\). The denominator counts positive
undirected edges; it is **not** their total weight. The token energy is

\[
e(X,A)=\frac{\operatorname{Tr}(X^TL_cX)}q
 =\frac{\sum_i d_i\|x_i\|^2-\sum_{i,j}A_{ij}\langle x_i,x_j\rangle}q
 =\frac{\sum_{i<j}A_{ij}\|x_i-x_j\|^2}q.
\]

The code evaluates the middle form, avoiding a \(D\times D\) result.
This is the combinatorial-Laplacian energy, not a normalized-Laplacian energy.
For each layer define

\[
\bar e_\ell=B^{-1}\sum_b e(X_{\ell b},A_{\ell b}),\qquad
\bar e^0_\ell=B^{-1}\sum_b e(U_b,A_{\ell b}),\qquad
\sigma^2=\max(\operatorname{Var}_{b,t,d}(U),10^{-8}),
\]

where variance has population normalization (`unbiased=False`). The base
energy uses the same layer-specific attention graph as the stream energy.
Let \(\widehat\lambda_\ell\) be the nonnegative, detached scalar estimate
from `fiedler_from_adjacency` on **only the first example** \(A_{\ell,0}\).
With defaults \(c_1=1,c_2=4\), set

\[
e_\ell^{\min}=c_1\sigma^2\widehat\lambda_\ell,\qquad
e_\ell^{\max}=\max(c_2\bar e^0_\ell,e_\ell^{\min}+10^{-4}),
\]

\[
D_\mathrm{band}=\sum_{\ell=1}^J
  [e_\ell^{\min}-\bar e_\ell]_+^2+
  [\bar e_\ell-e_\ell^{\max}]_+^2.
\]

Both variance and the base/stream energies retain their ordinary gradients.
The spectral estimate runs under `no_grad` and is reconstructed from a Python
float; its derivative with respect to attention is zero in this training rule.
This gives an explicitly stopped-gradient optimization rule, not the full
gradient of the displayed scalar when every intermediate is regarded as a
differentiable function of parameters.

The estimator projects out one degree-weighted constant vector. Twelve power
steps estimate a spectral shift, followed by up to thirty projected iterations
on `shift * I - L_norm` and a Rayleigh quotient. The shift estimate is not a
certified upper bound. Fixed iteration counts, deterministic starts and possible
small eigengaps do not establish convergence to a particular eigenpair. No
residual/eigengap acceptance gate is applied here. The all-zero adjacency
returns zero. With disconnected graphs, additional null vectors remain after
projecting out one vector; a second eigenvalue can be zero. The code docstring
“smallest nonzero eigenvalue” is therefore stronger than the implemented
guarantee and should not be used as a manuscript claim.

The band is a heuristic that combines an approximate normalized-graph
eigenvalue from one example with batch-mean combinatorial energies and a global
embedding variance. It is not an established Poincaré inequality for these
quantities. It provides no theorem that oversmoothing, collapse, or quality
degradation is prevented. Selecting the first example also makes the scalar
penalty potentially sensitive to batch permutation. The returned
`token_energies` diagnostics likewise contain only each layer's first-example
energy, not \(\bar e_\ell\).

## Schedule and comparison identity

The original schedule has phase I before update 30,000 (geometric weights zero,
Dirichlet weight \(10^{-4}\)); phase II ramps to
\((\gamma_M,\gamma_F,\gamma_H,\gamma_D)=(.05,.02,.01,.001)\) by 240,000;
phase III decays the first three to zero by 300,000 and immediately returns the
Dirichlet weight to \(10^{-4}\). At schedule position 135,000 the weights are
\((.025,.01,.005,.00055)\). The bounded continuation study can pin that
position through `study.penalty_step`; it is a distinct continuation protocol,
not a reproduction of the original 300,000-update schedule. A nonspectral arm
has all four weights zero; a `manifold_constant` override activates only that
term.

Passing explicit weights allows the optimized implementation to skip terms
whose weight is zero. On inputs where the corresponding original calculations
are finite, this preserves the weighted objective and gradient. It also avoids
executing otherwise unused computations that could fail; no equivalence claim
is made for an undefined/nonfinite original evaluation. Omitting `weights`
still computes all terms for diagnostics. An explicitly skipped diagnostic is
returned as zero, and `token_energies` is empty when the band is skipped.
Comparisons must share the fractional policy and numerical settings; they must
not treat the repaired method as identical to the historical unsmoothed one.

## Dense computational bounds and scientific boundary

The following are leading-order per-layer or per-pair dense arithmetic bounds,
not measured wall-clock guarantees or full training peak-memory predictions.

| Component | Leading work | Principal loss-side matrices |
|---|---|---|
| Pooled RBF graph | \(O(B^2D)\) | \(B\times B\) |
| One trace quadratic | \(O(B^2D)\) | \(B\times D\), no \(D\times D\) result |
| Manifold term | \(O(KB^2D+K^2B^2D)\) | Selected pooled graphs and features |
| Fractional eigensolve and energy | \(O(B^3+B^2D)\) per layer | \(B\times B\) eigensystem |
| Hypergraph energy | \(O(NE\min(N,E))\) per layer | \(N\times E\) inputs and a \(\min(N,E)^2\) square intermediate |
| Token/base energy | \(O(BT^2D)\) per layer | \(B\times T\times T\) adjacency and token features |
| Detached spectral estimate | \(O((12+30)T^2)\) per layer | First-example \(T\times T\) adjacency and vectors |

At the main M=4 dimensions, the hypergraph contraction uses a 1,024-by-1,024
square intermediate in place of a 2,048-by-2,048 event matrix. The fractional
eigendecompositions are only 8-by-8. Autograd retains additional intermediates;
the table is not a proof of a particular peak GPU-memory reduction. The model
still materializes dense head attention, with storage \(O(BHT^2)\), and the
vocabulary logits, MLPs, optimizer, backward pass, synchronization and data
handling also contribute costs. A tiny fractional graph therefore does not
establish an asymptotic bottleneck requiring an approximate matrix-function
front end.

These definitions and contraction proofs establish what is computed. CPU
oracles check bounded numerical cases; GPU pilots check bounded implementation
behavior. Neither supplies held-out language-quality results, a learned
topology theorem, a universal speedup, external replication, or a validated
large-model scaling claim. Those require the separately specified experimental
gates and evidence.
