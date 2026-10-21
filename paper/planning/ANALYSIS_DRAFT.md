# Draft analysis protocol for the bounded core study

Date: October 21, 2026. **DRAFT — not a frozen confirmatory protocol.**
This document makes the analysis choices in
[`PUBLICATION_PLAN.md`](../PUBLICATION_PLAN.md), G2/G3, concrete for review.
It authorizes no experiment, budget extension, or final-test inspection. No
confirmatory scientific data were analyzed to prepare it. Existing numerical,
timing and historical-checkpoint pilots remain exploratory. Before execution,
freeze this document, executable analysis code, seed lists, randomization,
source/input hashes and run identities together. Changes then require a dated
amendment identifying whether any confirmatory output has been observed.

The three research questions remain numerical fidelity on a stated domain,
equal-work timing on one RTX 5070, and bounded held-out CE noninferiority.
The study does not compare spectral regularization against CE-only training
as a quality intervention, establish improved language ability, or validate
sparse inference acceleration. The approved core budget remains at most
181 GPU-hours; this draft does not spend contingency or schedule extra seeds.

## 1. Analysis populations and immutable identities

All comparisons use the same repaired fractional policy
`c1-quadratic-cap-frechet-v1`, cutoff \(10^{-4}\), exponent 0.5, graph
definitions and precision settings. Baseline and optimized source identities
must be frozen separately. The numerical acceptance gates are prerequisites
for useful-speedup and implementation-comparison claims; aggregate CE or timing
cannot excuse a failed component-gradient gate.

The quality experiment has five independently initialized CE parents trained
for 50,000 updates. Each produces two 10,000-update continuations, baseline and
optimized, with identical starting model, buffers, optimizer, RNG and data
stream, fixed `gammas(135000)`, and the continuation rules in
[`STUDY_TRAIN.md`](STUDY_TRAIN.md). Parent states are shared within a pair,
not between distinct seeds. A proposed seed list is `[0, 1, 2, 3, 4]`, subject
to checking that these exact new runs have not already been used for exploratory
selection. Any pilot seed, including the proposed 9001, is excluded from this
five-seed analysis. The chosen list must be frozen, not selected by outcomes.

The primary model is each continuation's final committed update 10,000.
Development results cannot select an earlier checkpoint for the primary
endpoint. The newly frozen final-test story partition is disjoint from training,
development and calibration, and is not used for checkpoint, cutoff, tolerance
or protocol selection. Historical checkpoints and historical validation data
do not become untouched test assets by changing their labels.

Every analysis row must bind the run/configuration, parent lineage, checkpoint,
source, fractional policy, token bundle, evaluation policy and raw result
identities. A recomputed hash alone does not validate a scientific claim:
semantic checks must confirm the intended state, work and evaluation population.

## 2. Quality endpoint: token-weighted CE per model

Let \(v\in\{O,B\}\) denote optimized or baseline, \(i=1,\ldots,5\) the
paired training seed, and \(j\) a final-test story. Save summed negative
log-likelihood \(N_{ivj}\) and valid-target count \(n_j\) for every story.
The per-model cross-entropy, in **nats per valid target token**, is

\[
C_{iv}=\frac{\sum_j N_{ivj}}{\sum_j n_j}.
\]

This is token-weighted CE. Do not average the per-story CE values without
their token counts, average minibatch means of unequal size, or average
perplexities as a substitute. The same ordered story identities, target-token
counts and scoring policy must hold for all ten models; any discrepancy fails
the primary endpoint's input checks.

The proposed evaluator policy is story-isolated context 256, stride 128,
each eligible target scored exactly once, padding and context-only targets
excluded. Its current convention scores positions 1 through the final EOS of
each story, not the story's first token, because there is no preceding BOS
context. No scoring window crosses into an adjacent story. Freeze evaluation
batch size, FP32/TF32 settings, EOS convention, window construction and the
evaluator source before final-test execution. This definition describes the
proposed evaluator; its implementation and fixture tests must pass before it
can support this endpoint.

Define the paired differences

\[
d_i=C_{iO}-C_{iB},\qquad
\bar d=\frac15\sum_i d_i,\qquad
s_d^2=\frac14\sum_i(d_i-\bar d)^2.
\]

Each independently trained seed pair has equal weight in \(\bar d\).
Tokens and stories are not additional independent training replications.
The inferential target is the mean optimized-minus-baseline CE difference over
the declared training-seed mechanism, **conditional on this fixed test corpus,
architecture, training protocol and numerical environment**. It is not a
confidence statement over all future corpora or hardware.

## 3. Noninferiority hypothesis, assumptions and decision rule

The proposed practical margin is \(m=0.01\) nats/token:

\[
H_0:\mu_d\geq m,\qquad H_1:\mu_d<m.
\]

Report the upper one-sided nominal 95% paired Student-t bound

\[
U_{.95}=\bar d+t_{.95,4}\frac{s_d}{\sqrt5}.
\]

The primary CE noninferiority gate passes only if all five valid pairs are
available and **\(U_{.95}<0.01\)**. Report all ten CEs, all five differences,
\(\bar d\), \(s_d\), the upper bound, and the ordinary two-sided 95%
interval \(\bar d\pm t_{.975,4}s_d/\sqrt5\). Label the one-sided bound
separately; it is not the upper end of the two-sided 95% interval.

The t bound is exact under independent, identically distributed normal paired
differences with unknown finite variance. Here it is a **model-based nominal
interval**: independent seeds support replication, but five observations do
not establish normality, rule out heavy tails, or guarantee adequate power.
The common corpus and within-pair parent are deliberate conditioning and
pairing, not extra independent samples. Preserve all seed differences and
describe visible asymmetry or outliers. Do not delete an outlier, replace a
seed, switch to a favorable estimator, or claim a small-sample bootstrap
automatically fixes these assumptions after observing results.

The margin corresponds to a paired perplexity ratio \(\exp(.01)\), about
1.01005. Exponentiating the mean CE difference gives the geometric mean of
the paired perplexity ratios, not a bound on every seed's ratio or on the
arithmetic mean of perplexities. This mathematical conversion does not
scientifically justify the margin; its practical justification remains a
freeze requirement. Passing establishes only degradation within that declared
margin under these assumptions. It does not establish exact equality or
quality superiority. If the upper bound reaches/exceeds the margin, report
noninferiority as **not established**, including when uncertainty is simply
too large. A lower bound above the margin is a distinct adverse finding and
must be retained.

If \(s_d=0\), report that observed degeneracy explicitly. The zero-width
formula supplies no distribution-free evidence that unseen seed variation is
absent. Do not use it to imply a guarantee across all seeds or environments.

## 4. Timing design: eight cells and 96 paired blocks

The planned matrix contains four prospectively selected M=4 states crossed
with two schedule regimes, twelve paired process blocks per cell. The schedule
regimes are phase I (`gammas(0)`) and all-active fixed midpoint
(`gammas(135000)`). Freeze exact checkpoint/optimizer/RNG identities before
timing. A concrete proposed state set is initialization and CE updates 5,000,
25,000 and 50,000 from one prospectively selected parent seed, e.g. seed 0.
These would be states along a CE trajectory, not four independently trained
models or four states along a spectral trajectory. Replacing this set requires
an explicit pre-execution choice; no state is selected for a favorable speedup.

Use **twelve sessions**, each containing one pair for each of the eight cells.
For each cell, assign six pairs to AB and six to BA using a frozen randomization
seed; A is baseline and B optimized. Randomize cell order within each session.
Pair implementations in separate fresh processes with identical frozen initial
state and matched batches/seed within each pair. This is balanced **AB/BA**
ordering, not a four-period A-B-B-A run: the latter would change the planned
number of implementations executed and the budget. Session spacing, restart
conditions and the randomization ledger must be fixed before timing.

Each implementation executes three warmup and twelve measured optimizer
updates. The total is
\(8\times12\times2\times(3+12)=2{,}880\) updates, including warmup.
Microbatch 8, accumulation 8, context 256 and all numerical/optimizer settings
are identical between implementations. Each pair starts from the same full
state; each implementation then performs its own warmup trajectory. Numerical
drift is recorded under the prespecified tolerances, not silently removed.

The proposed primary time is synchronized host wall time for the measured
forward/loss/backward, accumulation, gradient clipping, finite guards and AdamW
update. Match the exact included operations in the frozen benchmark script.
Prepare and transfer matched data before timing. Loading states, process
startup, data transfer, external telemetry, checkpoint IO and teardown are
outside this isolated timing endpoint and must be reported separately when
discussing total training cost. Synchronize CUDA at both timing boundaries;
record all twelve update durations and state what any per-update timing calls
include. This endpoint is neither a CUDA-kernel-only measurement nor the full
study wrapper's billed wall time.

Record temperatures, clocks, power, CPU load, competing activity, GPU/runtime
identity and process order. Use fixed warmup and eligibility rules. Ordinary
slow blocks remain observations; do not trim high times, discard unfavorable
orders, or extend warmup after seeing a result. A numerical/source/data failure
invalidates the claimed matched comparison and is retained as a failed attempt.

## 5. Timing estimator, uncertainty and useful-speedup claim

For cell \(c=1,\ldots,8\), session \(s=1,\ldots,12\), and implementation
\(v\), let \(T_{csv}\) be the arithmetic mean of its twelve measured update
times in **seconds per optimizer update**. This equals the timed-update sum
divided by twelve; it is not the within-block median. Define

\[
r_{cs}=\log(T_{csA}/T_{csB}).
\]

Positive \(r\) means optimized is faster. Report per-cell raw block times,
mean log ratio \(\bar r_c\), and \(\exp(\bar r_c)\), the geometric mean
paired speedup. For each cell, report the pointwise two-sided nominal 95%
interval

\[
\exp\left[\bar r_c\pm t_{.975,11}
                 \frac{s_{r,c}}{\sqrt{12}}\right].
\]

The single primary timing aggregate weights the **eight cells equally on the
log scale**. First compute each session's mean

\[
q_s=\frac18\sum_{c=1}^{8}r_{cs},\qquad
\bar q=\frac1{12}\sum_s q_s,\qquad S=\exp(\bar q).
\]

Let \(s_q\) be the sample standard deviation of the twelve \(q_s\).
The aggregate two-sided nominal 95% interval is
\(\exp[\bar q\pm t_{.975,11}s_q/\sqrt{12}]\). The prespecified **lower
one-sided nominal 95% bound** used for the useful-speedup criterion is

\[
L_{.95}=\exp\left[\bar q-t_{.95,11}\frac{s_q}{\sqrt{12}}\right].
\]

The useful-speedup gate passes only when all numerical gates and all planned
timing input checks pass and **\(L_{.95}>1.05\)**. This makes the confidence
level and tail explicit; changing to the lower endpoint of a two-sided 95%
interval would be a different, more conservative decision rule. Freeze this
choice before execution. The claim is an equal-cell geometric mean paired
speedup exceeding 1.05 on this matrix, not a 5% reduction in total project time,
not a hardware-independent claim, and not a claim that every cell exceeds 1.05.
The cell weights do not represent the frequency of phases in a 300k training
schedule. Any workload-weighted time projection is separate and descriptive.

These intervals assume independent, approximately normal session-level log
effects for their stated conditional population. Within-session cell effects
can be dependent: forming \(q_s\) before computing uncertainty preserves
that dependence. The 96 cell pairs, the twelve updates per implementation,
and individual microbatches must not be pooled as 96, 1,152 or more independent
aggregate observations. Separate processes do not themselves guarantee
independent hardware conditions. If sessions have time drift or carryover,
nominal t coverage can be inaccurate; report the order and session plots and
retain this limitation. No post hoc regression or block-bootstrap choice
replaces the primary rule. The twelve-session design is small and conditional
on one host and the chosen checkpoint states.

## 6. Multiplicity and permitted interpretations

There is one primary timing aggregate and one primary quality noninferiority
endpoint. Each has its declared one-sided nominal 95% bound. The primary
combined statement, “useful equal-cell speedup with CE noninferiority under
this protocol,” requires **both** gates plus numerical/provenance gates.
This conjunction is an intersection-union decision: under either component
null, passing both cannot be more likely than incorrectly passing that
component's 5% test, subject to its assumptions. It is not permission to claim
a familywise 95% guarantee for arbitrarily selected individual conclusions.

The eight pointwise cell intervals are secondary, descriptive intervals and
are labeled **not multiplicity-adjusted**. They are not eight separately
confirmed favorable results. If individual-cell useful-speedup declarations
are desired at freeze, use eight simultaneous conservative one-sided bounds
with Bonferroni tail probability \(.05/8\): replace \(t_{.95,11}\) by
\(t_{1-.05/8,11}\) in each cell's lower-bound formula. Declare only cells
whose adjusted bound exceeds 1.05, and show every cell. Do not select the best
cell first or retroactively promote a secondary aggregation to primary.

Memory, energy, per-update distributions, order effects, per-story diagnostics
and any component timing are secondary descriptions unless separately
prespecified. No p-value or interval computed on these is silently added to
the confirmatory claim family. Numerical oracle fixtures and stress tests are
correctness evidence with stated domains, not independent quality-study seeds.

## 7. Fixed sample size, failures and audit outputs

Collect the prespecified five quality pairs and 96 timing pairs. Do not peek
at accumulating quality differences or timing confidence bounds to decide
whether to stop or add runs. Operational monitoring may inspect integrity,
finite-state checks, completion and budget use; it does not authorize outcome
selection. The hard budget and nonfinite safety guards take priority over
completion. Stopping for those reasons produces an incomplete study, not a
successful endpoint.

Resume an interrupted run only from its last valid checkpoint under the same
frozen method/configuration, retaining the complete attempt history and costs.
An irrecoverable training failure cannot be dropped or replaced by a new seed
while keeping the five-seed claim. A failed/incomplete timing block cannot be
replaced after its time is inspected to improve the result. This draft
authorizes no replacement attempts beyond ordinary same-run recovery; if a
bounded infrastructure-retry rule is needed, define its trigger, maximum
attempts and accounting before confirmatory collection. Missing required
pairs prevent the corresponding primary gate from passing. Complete subsets
may be reported descriptively with their missingness and reduced scope.

Freeze analysis code and test it on generated data, including unequal story
lengths, sign conventions, known t intervals, session dependence, absent pairs,
nonfinite records and equality at the decision thresholds. These are future
implementation gates; this document does not assert they have already passed.
The release must retain raw per-story NLL/counts, per-update timings, pairing
and order ledgers, failed-attempt records, input/source hashes, generated tables
and figures, and exact analysis commands. Report every prespecified primary
result regardless of direction or significance.

## 8. Items that remain to freeze

- Exact five seed identities and confirmation of no exploratory reuse of their
  new-run outcomes; final parent/branch hyperparameters and required numerical
  tolerances.
- Scientific justification of the 0.01-nat margin and the 1.05 timing threshold;
  neither threshold is established by choosing it in this draft.
- Exact four checkpoint states, all eight timing cells, matched batch source,
  twelve-session definition/spacing, AB/BA randomization seed and full order.
- Evaluation batch size, precision, context/stride/EOS rules, final-test
  partition hash, final checkpoint rules and audited evaluator implementation.
- Exact benchmark timing boundary, environment, telemetry, eligibility rules,
  any bounded infrastructure-retry policy, and conservative budget projection.
- Executable analysis implementation, generated-data tests, frozen protocol
  hash and dated approval record before confirmatory outputs are opened.

If the budget or prerequisites cannot support this design, revise it
prospectively and narrow the claims. Do not present a smaller, outcome-selected
subset as though it satisfied this draft's confirmatory design.
