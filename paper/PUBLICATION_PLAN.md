# Publication development plan

Status: core study approved, October 21, 2026, with a hard cap of **181 GPU-hours**.
The user selected the proposed core scope after reviewing the full budget.
Numerical and reproducibility repairs precede confirmatory experiment scheduling.
The initial approval preceded new GPU work. Subsequent bounded numerical and
pipeline checks have run under that cap; the confirmatory study and external
submission remain pending. This document records the plan and original audit,
not a completed experiment report.

Publication route: **arXiv preprint first, then peer review**.
Human author: **Blake A. Jones**. Affiliation, correspondence address, ORCID if
desired, publication license, and account/endorsement status remain unspecified.

The objective is a useful, reproducible contribution with falsifiable claims.
No finite experiment or harness can provide zero possibility of refutation or
guarantee acceptance. Contrary findings will be retained and claims narrowed
when required by the evidence.

## 1. Preserve the original work and establish the evidence boundary

The public repository is
[ZuluYokohama/llm-laplacian-v0.2](https://github.com/ZuluYokohama/llm-laplacian-v0.2).
The current public main commit observed through GitHub is
`5083118a36ddfc35aeba20c8a3c9855b3f08da12`; its tree matches local HEAD
`08612a6bcea8500238d8bf75f1000c87a349da52`, tree
`4823948c99975a7a7a724c3e857adc4d77a47e9c`. The subsequent optimization is in the
local working tree and must be captured in its own reproducible release.
Neither the public merge SHA nor local HEAD alone contains that optimization.

The following table records the **historical audit before the G0 repairs**.
Its speedups and gradient comparisons use the original fractional policy. They
do not validate the later C1 cutoff policy, current source, or future training
quality. Later method and reproducibility changes are described in the linked
planning documents and require their own bound receipts.

| Historical claim | Evidence status at original audit | Publication treatment |
|---|---|---|
| Optimized M=4 loss executes faster on this RTX 5070 | Verified within six short paired cases; median speedups 1.355x and 1.136x | Pilot result; strengthen independent execution replication and uncertainty |
| Arithmetic reassociation preserves tested values/gradients | Verified on stated tests/tolerances; BF16 full-model gradient L2 gap up to 0.543% | State domain, precision, tolerances, and numerical differences |
| Twelve reference modules pass process-isolated tests | Verified by the recorded Jovin receipts | Describe exact source/tests/image/limits; do not generalize to the GPU pipeline |
| Arm 0 CE is 1.531456 on its recorded evaluation | Historical measurement on one checkpoint and 1,024 sampled windows | One-run observation; not a seed-averaged finding |
| Arm 0 trained all 300k updates in 14.49 hours | Unsupported: recorded elapsed time covers the final 152k-update segment | Correct the interpretation; total historical time is unknown from this metric |
| All spectral losses have reliable finite gradients | Failed on a newly reproduced fractional-loss fixture | Mandatory repair before new confirmatory spectral training |
| Configured seeds and resume fully reproduce training | Not established; initialization/global RNG state is not controlled/restored | Mandatory RNG and resume upgrade |
| Spectral pruning beats random/magnitude/activation-aware baselines | Not established; the Qwen pilot has AEI worse than its random mask | Retain negative pilot and uncertainty limits |
| Sparse masks accelerate actual inference | Not established | Requires compact export or supported kernels plus latency measurements |
| A submission-ready paper exists | Not verified; no manuscript/bibliography/PDF exists in paper/ | Build the manuscript and evidence package after the research gates |

Primary local sources: [optimization report](../runs/optimization_20261021/REPORT.md),
[paired metrics](../runs/optimization_20261021/paired_accum8/metrics.json),
[Arm 0](../runs/arm0/metrics.json), [evaluation](../runs/eval_arm0/metrics.json),
[Qwen pilot](../runs/frozen_qwen3_0p6b/metrics.json),
[fractional diagnostic](planning/fractional_gradient_diagnostic.json).
The benchmark source hashes matched the source at that original planning audit;
later repairs changed it. Historical files remain unchanged; corrections are
annotations, not rewritten experimental history. These local evidence files are
not all distributed with the source-review PR; that PR supplies methods and
test code, while a curated evidence release is a later publication gate.

## 2. Recommended first paper and explicit exclusions

Working title: **Auditable Optimization of Spectral Losses in Small Language
Models: Numerical Fidelity, Runtime, and Reproducibility**.

The recommended first paper studies three questions:

- RQ1: Under an explicitly specified mathematical objective, does the optimized
  implementation preserve component values and gradients on the supported domain?
- RQ2: Does it reduce equal-work training-step time under replicated, controlled
  execution on the RTX 5070, and which changes account for the difference?
- RQ3: Under a declared bounded training protocol, is held-out CE degradation
  within a preregistered practically acceptable margin?

This scope does not claim a new graph-sparsification theorem, better general
language ability, the superiority of spectral regularization, improved
interpretability, broad hardware portability, or sparse inference acceleration.
Those questions require the separately budgeted extensions below. The research
contribution must explain what other researchers learn, not just that local
tests passed. Novelty and related-work assessment remain required.

## 3. Work packages and completion gates

### G0 — Mathematical specification and reproducible execution

**Required before any confirmatory run.**

1. Specify the actual hypergraph objective on degree-scaled features and the
   heuristic Dirichlet band. Separate algebraic identities, numerical
   approximations, and empirical hypotheses. Do not silently replace the current
   objective with a more conventional one while calling the results equivalent.
2. Resolve the fractional-gradient failure. The CPU fixture
   `(torch.eye(4) * 100).requires_grad_()` with exponent 0.5 and fractional-only
   weight returns loss zero and 16 nonfinite gradient elements. Define behavior
   at zero/repeated eigenvalues before implementing a remedy. A smoothing or
   gradient-policy change is a versioned method change, not merely faster code.
   Apply any shared safety correction to both future comparison implementations;
   keep the original pre-correction timing artifacts separate.
3. Add analytic and independent dense-oracle checks for isolated/disconnected
   graphs, repeated/near-zero eigenvalues, scale changes, permutations, and
   gradients away from nondifferentiable boundaries. Test individual penalty
   gradients so cross-entropy cannot hide an error. Include FP32 and BF16.
4. Seed initialization, minibatch selection, resampling, and pruning explicitly.
   Capture/restore Python, NumPy, Torch CPU and relevant CUDA RNG state wherever
   used. Test uninterrupted versus resumed execution across resampling and
   culling boundaries. Specify deterministic-mode limits rather than promising
   bitwise reproducibility on every platform.
5. Reject nonfinite loss/gradients before optimizer updates and successful
   checkpoint promotion. Add failure-recovery tests.
6. Record cumulative updates, tokens, and elapsed segments across resumes;
   distinguish active training time, evaluation, checkpoint IO and downtime.

Acceptance: all new negative/regression cases pass under the documented
numerical policy; a resumed seeded run reproduces its uninterrupted comparator
within the declared mode; failures cannot generate success receipts.

### G1 — Research harness and artifact provenance

- Freeze code, dependencies, CUDA/driver/runtime, data/model revisions and input
  hashes. Supply an installable environment and accessible selected checkpoints
  or a fully specified regeneration route. Choose code/artifact licenses with
  the author; do not infer licensing from a public GitHub URL.
- Give each run an immutable identity containing experiment/config/source/input
  hashes. Record planned, running, passed, failed, OOM, and interrupted outcomes.
  Preserve failed trials. Never overwrite evidence to make a run look successful.
- Extend source freshness to environment/config/checkpoint/test identity.
  Normalize receipt exports as valid JSON records/JSONL instead of concatenated
  JSON documents. Specify raw-byte versus LF-normalized hashes explicitly and
  retain compatibility with historical metrics.
- Export the tested sources, test commands, image digest, limits, receipt chain,
  stdout/stderr, host-to-reference checks, and numeric acceptance results.
  Verify the export without relying solely on the original machine's state.
- Replace the paper binder's marker/number-denylist logic with an explicit
  claim-to-artifact manifest. A missing citation source currently passes because
  an empty path resolves to the repository directory; fix this and require
  actual retrieved, readable sources. Download PDFs as intact bytes and verify
  metadata/claims rather than treating truncated UTF-8 as a paper.
- Mutation tests must reject altered source, missing inputs, corrupted metrics,
  incomplete runs, broken chains, invalid numerical output, and unsupported
  manuscript references. Independent replication must be labeled separately
  from internal AI-assisted reviews.

Acceptance: a clean environment can execute the documented small reproduction
and regenerate a figure/table from immutable raw artifacts. Tampering and
partial completion produce explicit failure.

### G2 — Protocol freeze and timing replication

Freeze the primary endpoints, tolerances, randomization, data splits, seed list,
maximum run counts, analysis code, and stopping rules before confirmatory output.
Archive exploratory results separately. If G0 changes the objective, freeze a new
protocol version and remeasure both implementations.

Timing matrix: four M=4 states (initialization, early/middle/final checkpoints),
two schedule regimes, and twelve independent paired process/session blocks per
cell. Balance AB and BA six/six within each cell; randomize phase/session order.
Proposed block: three warmup plus twelve measured updates for each implementation.
This is 96 paired blocks, 2,880 total optimizer updates before component probes.
Use identical batches, state, optimizer settings, precision and work per pair.
Record GPU temperature/clocks/power, driver, CPU load and competing activity.
Observe a fixed warmup rule; do not remove unfavorable blocks after seeing times.

Report paired log-runtime ratios and confidence intervals at the independent
block/session level. Updates within a process are correlated observations, not
independent experimental units. Retain per-block distributions. The proposed
useful-speedup criterion is a lower 95% interval bound above 1.05x, conditional
on numerical gates; otherwise report the measured result as inconclusive or
configuration-specific. Final thresholds must be justified before execution.

### G3 — Bounded training validation for the first paper

Train five independently initialized CE models for 50,000 updates. From each
result, branch into an unoptimized and optimized implementation for 10,000
additional updates with fixed mid-phase spectral weights (`gammas(135000)`).
Within each pair, use the same model/optimizer/buffer/RNG starting state, token
order, and hyperparameters. Different pretraining seeds are genuine independent
training seeds. This is five shared pretrains plus ten paired continuations.

This **50k+10k fixed-weight protocol is a new bounded experiment**. It does not
replicate the original 300k schedule, whose first 30k steps have zero manifold,
fractional, and hypergraph weights. Shortened original-schedule runs must not be
presented as evidence about terms that never became active.

Rebuild splits at story level before tokenization. New training, development,
calibration, and final evaluation partitions must be disjoint. Historical Arm 0
and historical validation are exploratory assets. Disclose their earlier
exposure; do not claim a historically used validation set is untouched. Final
quality conclusions use newly trained models and a frozen held-out story split
excluded from their training and all model selection. Log sampled story IDs and
token spans so overlaps can be audited.

Save summed per-story NLL and valid-target-token counts; exclude padding.
Report every seed's paired CE difference, the aggregate and its uncertainty.
The proposed noninferiority margin is 0.01 nats/token (about 1% perplexity), to
be scientifically justified and frozen before results. Five seeds are a budgeted
starting design, not a power guarantee. If the interval is too wide, report
inconclusive; do not continue sampling until significance appears. Any new
confirmatory study needs its own prospectively fixed protocol.

### G4 — Manuscript, independent audit, and release

Produce LaTeX source, a verified BibTeX bibliography, generated tables/figures,
appendices, an artifact README, a claim-to-evidence map, and a readable PDF.
Every quantitative claim must identify its run/configuration, estimator,
uncertainty, and scope. Every external citation must be checked against a
retrieved primary source. The paper must retain numerical failures, unfavorable
pruning pilots, and failed/OOM experiments where relevant.

Suggested manuscript structure: research questions; related work; exact
objectives and numerical policies; optimized contractions and complexity;
verification architecture and its limits; preregistered experiment design;
runtime/numerical/quality results; threats to validity; limitations; tool-use
disclosure; reproduction appendix. Claims of quality improvement or compression
are excluded unless the extension gates have actually passed.

Acceptance: all plotted data regenerate from the release; clean compilation and
page-by-page visual inspection pass; a source/derivation audit and an independent
reproduction attempt have documented outcomes. Subagent review alone is not
external peer review. Blake reviews and takes responsibility for the final work.

### G5 — arXiv and subsequent peer review

Prepare title/abstract/categories, source archive, PDF, ancillary artifacts,
author metadata, tool disclosure, and distribution-license selection. Verify the
uploaded compilation preview before finalizing. arXiv expects author submission
and may require endorsement; the author must make the required attestations.
Record an actual submission identifier/status only after the service provides it.
Submission, moderation approval, and peer-review acceptance are distinct states.
See [arXiv submission guidance](https://info.arxiv.org/help/submit/index.html) and
[the submission agreement](https://info.arxiv.org/help/policies/submission_agreement.html).

After the preprint, select a suitable peer-reviewed venue based on the completed
contribution. TMLR is a candidate, not a guaranteed destination: it evaluates
supported claims and useful, clear findings, and permits preprints. A TMLR
submission would need its template, anonymized manuscript/artifacts, author
profiles, and relevant disclosures. Do not link an identifying public artifact
from an anonymous submission. Recheck current venue rules at submission time.
Sources: [acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html),
[author guide](https://jmlr.org/tmlr/author-guide.html),
[editorial policies](https://jmlr.org/tmlr/editorial-policies.html).

## 4. Full proposed compute budget

All figures are **planning envelopes**, not measured end-to-end forecasts or
spending commitments. They assume one RTX 5070, microbatch eight, accumulation
eight, context 256. G0/G1 engineering and human review time are additional to
GPU time. No cloud GPU spending is included. Rates must be remeasured after
numerical repairs; a projected cap overrun stops scheduling and requires a new
budget decision before additional work.

| Package | Run matrix / allowance | Proposed GPU-hours |
|---|---|---:|
| A: replicated performance | G2 matrix, short component/rate probes | 4-hour capped planning allowance |
| B: independent pretraining | 5 x 50,000 CE updates at 0.35–0.60 s/update | 24.3–41.7 |
| C: matched continuations | 5 x 10,000 x (old 2.40–3.65 + optimized 2.10–2.90 s/update) | 62.5–91.0 |
| D: evaluation/verification | Held-out CE, numerical/resume GPU checks, artifact reproduction | 8-hour allowance |
| E: contingency | 25% of A+B+C+D, including failures/overhead | 24.7–36.2 |
| **Recommended first-paper total** | **A+B+C+D+E** | **124–181** |

Arithmetic: `(5*50000*r_plain + 5*10000*(r_old+r_new))/3600`
plus 12 hours of allowances, multiplied by 1.25. This is approximately 5.2–7.5
days of uninterrupted GPU occupation; actual calendar time includes engineering,
idle periods, writing, author review, and submission processing.

The plain rate is especially provisional: the historical 0.343 s/update applies
to a final resumed segment, not a fresh end-to-end run. The prior 96.76-day queue
ETA mixes unlike smoke workloads and is not used as a publication budget.

### Separately priced broader commitments

| Extension | Scope | Provisional total, including allowances |
|---|---|---:|
| Original 300k quality comparison | 5 seeds x CE versus full spectral x 300k updates | 1,398–1,982 GPU-hours (58–83 continuous days) |
| Full original six-arm ablation | 5 seeds x CE, each of four penalties alone, all penalties x 300k | 6,188–8,536 GPU-hours (258–356 days) |
| Pruning and compact inference | Matched rankings/recalibration/export/latency study | Not yet responsibly priced; separate component pilot required |
| Additional hardware or SmolLM | Cross-device/model generalization | Not included; profile and price separately |

The original-protocol comparison uses a schedule-weighted spectral planning
rate of roughly 2.09–2.86 s/update and plain 0.35–0.60 s/update, then 10% for
evaluation/IO and 25% contingency. The six-arm estimate conservatively charges
each spectral arm at the full spectral rate; single-term pilots may reduce it.
These extension totals are alternatives, not automatically additive mandatory
steps for the recommended first paper. Shared runs may only be reused when
their exact configurations/provenance satisfy both protocols.

A future pruning matrix should include the unpruned control; random, magnitude,
activation-aware and AEI rankings; 5/10/20% neuron removal; five calibration draws;
matched recalibration on/off and recovery budgets. Cache shared statistics.
Effective-resistance edge sampling and compact inference are separate factors.
Before that study, fix/audit deterministic top-k/floor importance weights,
CG residual acceptance, and mask-to-compact equivalence. Do not price it from
the single historical ten-minute composite pulse.

Approved decision: the **core paper scope with a hard ceiling of 181 GPU-hours**,
with G0/G1 correctness gates and the four-hour rate/performance stage first.
Approval was received on October 21, 2026; extensions remain outside this scope.
The lower-cost alternative is the four-hour performance stage alone, producing
a narrower execution/verification study without training-quality conclusions.

## 5. Authorship, AI assistance, and software credit

List Blake A. Jones as author. Grok, Opus, Astra, and JoVinHarness belong in an
assistance/software disclosure, not the human author list. AI tools cannot take
responsibility for the final scholarly work; TMLR explicitly excludes LLM
authorship while allowing assistive use. See its
[LLM policy](https://jmlr.org/tmlr/editorial-policies.html).

Create a provenance ledger of each tool's exact product/model/version when
known, dates, role, affected artifacts, and human verification. The user's report
establishes that these tools should be investigated/credited; it does not
establish their precise versions or every contribution. Do not infer that the
current assistant is a particular named model from a nickname.

- The current chat documents code optimization, harness changes, experiment
  execution, evidence review, and planning assistance.
- Grok, Opus, and Astra roles/versions must be reconciled with saved histories;
  missing details should be stated as unavailable rather than invented.
- JoVinHarness is execution/verification infrastructure. Cite its software
  version, pinned image/runner hashes, limits and receipt scope. It is not a
  coauthor, autonomous scientific authority, or substitute for replication.
- The final disclosure states the assistance actually used and verification
  actually completed. Human approval/accountability is not asserted before it
  happens. Other qualifying human contributors can be added if identified.

## 6. Decisions still required

1. Budget selection is complete: the core study is approved up to 181 GPU-hours.
   Stop and reprice if the repaired implementation cannot fit this cap; unused
   contingency does not authorize a broader study.
2. Supply affiliation (or confirm independent-researcher status), corresponding
   email, optional ORCID, and arXiv account/endorsement status before submission.
3. Select compatible code/checkpoint/manuscript distribution terms before release.
4. Review the final manuscript and author attestations after all artifacts are
   concrete. The final peer-review venue can be chosen after the preprint is ready.

Use the [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist)
as an additional audit of claims, reproducibility, uncertainty, compute, asset
provenance and disclosure. Answer it truthfully; a checklist is not a certificate
of publication merit.
