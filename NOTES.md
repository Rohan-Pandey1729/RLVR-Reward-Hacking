# Notes for the author

**Paper: "Wrong the Same Way: Universality, Not Error Rate, Governs Reward Hacking in RLVR"**

Working notes on the paper: what the claim is, why each design choice was made, what a
reviewer is likely to push on, and what to do before submitting.

---

## The claim in one paragraph

At its KL-regularised optimum, RLVR is a *filter*: `pi*(y|x) ∝ pi_ref(y|x) exp(r(x,y)/beta)`
multiplies every accepted response by the same factor, so the odds between a correct answer and a
verifier false positive are frozen at the base model's. The post-RL reward-hacking rate should
therefore be inherited, not learned. Real training departs from that baseline, and this paper says
what governs the departure: writing the exact drift of `log pi(y|x)` as a sum over *all* prompts
of `Cov[reward, tangent kernel]`, a leak accepted on a fraction `u` of prompts has leading-order
drift with the sign of `u - Vbar`, where `Vbar` is the policy's own verifier pass rate. Above the
threshold parameter sharing manufactures hacking; below it, sharing *suppresses* the leak below
the base rate, because on the `1-u` prompts that do not accept it, it takes negative advantage.
The verifier's error rate appears nowhere. (The *sign* result and the suppression are what the
experiments establish; the numerical threshold itself is not verified pointwise, see below.) For a policy whose parameters are not shared across
prompts the cross-prompt kernel is identically zero and the dynamics cannot depend on `u` at all,
which is what makes the tabular arm a control rather than just a baseline.

## Why this is not already published

Two independent prior-art sweeps were run before committing (both are summarised below in
"discarded ideas"). The pieces that already exist, and are cited as such:

- The filter theorem itself: Korbak et al. 2022; and in the RLVR diversity literature,
  Gai et al. 2511.19942 (Prop 3.2), Kruszewski et al. 2512.05962, Dymetman 2605.02375.
  **We claim none of it**; it is the null model.
- Rich-get-richer among tied responses under softmax PG: Mei et al. 2020; Lochab et al. 2605.00365
  (Thm 4.3). Used as a control, attributed.
- "Post-RL hacking is inherited, not learned" as an *empirical* finding: Zhang 2607.11022.
  Our theorem explains it.
- "Pattern, not rate" for systematic verifier error: Egashira et al. (COLM 2026) 2605.02909.
  Universality is a specific, measurable parameterisation of "pattern" with a predicted sign.

What is new: the drift decomposition applied to the accepted level set, universality as the
controlling quantity, the two-sidedness (sharing *suppresses* an idiosyncratic leak below the base
rate), the tabular-vs-shared isolation of function approximation, and the hardening-budget
consequence.

## Design decisions, and why

**Why a two-digit response space?** Because 100 responses × 64 prompts can be enumerated, so
`pi(.|x)` is computed exactly at every logged step, so the reported hack fractions, log-odds and
accuracies are not Monte-Carlo estimates (the two gradient probes subsample prompts, and say so).
That is worth more than realism at this scale: the effect being measured is a
composition *inside* the accepted set, which sampling estimates poorly.

**Why matched tabular/transformer pairs?** Corollary 1 says a per-prompt policy's dynamics
*cannot* depend on universality, exactly rather than approximately, for any learning rate and any
number of steps. So any `u`-dependence observed in the transformer is attributable to parameter
sharing and nothing else. The tabular policy is initialised to reproduce the transformer's base
distribution to 1e-4, and both get the identical learning-rate grid.

**Why report drifts rather than absolute hack fractions?** The SFT target is identical across
conditions, but the realised base model is itself a shared-parameter network and fits a universal
leak map slightly better, so base hack fraction varies a little across `u` (and rises with `k`).
Drift from each run's own base is exactly zero under the filter baseline and immune to that.
Absolute `rho` is plotted alongside the base curve so both are visible.

**Why select the learning rate by verifier reward?** It is the criterion a practitioner has;
they cannot see true accuracy. It also avoids the accusation that the operating point was chosen
to produce the effect. The full grid is in the appendix figure.

**Why the randomised-block control?** With prompts enumerated first-operand-major, a contiguous
block of size m ≥ 8 is exactly "first operand = a", i.e. one input token. That would confound
universality with representational simplicity. The whole sweep is repeated with block membership
permuted and leak strings drawn independently per (m, k, seed).

## Anticipated objections

1. **"The theorem is standard."** Yes, and it is presented as the null model with three
   attributions in the first two pages. The contribution is the drift analysis and the threshold.

2. **"Toy environment."** Conceded in Limitations. The defence is that exact enumeration is what
   makes the measurement possible, and the mechanism (cross-prompt gradient aggregation) is
   architecture-independent. The falsifiable content is the *sign* and the *controlling quantity*,
   not the numerical boundary.

3. **"Tabular vs transformer confounds sharing with capacity/geometry."** The logical structure
   handles this: Corollary 1 is exact, so the per-prompt arm is flat in `u` by construction. The
   comparison is about the *slope* in `u`, not the level, and the paper says so explicitly. The
   per-prompt arm's own departure from the filter baseline (Adam's per-coordinate normalisation)
   is reported honestly rather than hidden.

4. **"Leaks are injected via SFT contamination, not discovered."** Conceded; matches Khalifa et
   al.'s finding that ~1% contamination suffices. Leaks a base model discovers on its own may
   have a different universality distribution. That is stated as a limitation, and it is the
   obvious follow-up.

5. **"`u > Vbar` is only leading order."** Stated as such throughout, with the O(1/N) corrections
   derived in the appendix and the validity condition `N·kappa/kappa_s >> 1` checked numerically
   against the measured coherence.

6. **"Correct answers aren't perfectly prompt-specific here (2×6 and 3×4 both give 12)."**
   True, `u_g ≈ 0.037`; reported explicitly. Definition 3 makes coherence an empirical property
   of computation rather than of string identity, and the measured cross-prompt gradient cosine
   for correct answers is near zero while the leak's is not.


## Results as they landed (all verified against the raw jsonl independently)

| sweep | shared-parameter policy | per-prompt policy |
|---|---|---|
| exact PG, Adam, contiguous blocks (5 seeds) | rho 0.019 -> 0.200, acc 0.980 -> 0.799 | rho 0.304 -> 0.304 |
| exact PG, Adam, permuted blocks (3 seeds)   | rho 0.021 -> 0.302, acc 0.979 -> 0.697 | rho 0.301 -> 0.288 |
| **plain SGD** (5 seeds)                     | **rho 0.056 -> 1.000, acc 0.924 -> 0.000** | rho 0.052 -> 0.054 |
| sampled GRPO, G=8 (3 seeds)                 | drho -0.103 -> +0.543, crossing u=0.24 | flat |

(left column = u = 1/64, right = u = 1, at an identical verifier false-positive rate throughout.)

Other headline numbers:

- **Filter invariance verified exactly**: mirror descent and the KL-regularised optimum leave the
  per-prompt hack fraction invariant to 8.6e-9 and the log-odds to
  9.6e-8, i.e. floating-point zero. Meanwhile the *pooled* hack fraction moves
  by +0.057 under pure filtering, which is a trap worth knowing about.
- **Corollary 1 verified exactly**: the per-prompt policy's cross-prompt tangent kernel and
  cross-prompt drift are *identically* 0.0 at every universality; the transformer's kernel is
  0.098 of its own self-kernel.
- **Coherence** (the assumption Prop. 2 needs) rises +0.02 -> +0.29
  with u for the leak, and stays at +0.03 for correct answers.
- **Factorial**: 59% of the variance in log-odds drift is attributable to
  universality, 11% to a false-positive rate varied eightfold, and the sign of
  the error-rate effect is *negative*.
- **Verifier hardening**: two patches with identical audit improvement differ by
  40 ± 16 accuracy points (0.984 vs
  0.583); the per-prompt control puts 11 of those points in
  the base model and 29 in RL.

### What did NOT work, and is reported as such

- The **u > Vbar threshold could not be verified pointwise**. At initialisation a common transient
  dominates and the cross-prompt drift is positive at every u; after 60 RL steps the transient is
  gone but Vbar has already reached 0.993, so the predicted threshold sits at ~1 and
  everything is (correctly, but uninformatively) suppressed. The paper says this in a paragraph
  headed "What we could not verify" rather than reporting a fit.
- The **u=1 endpoint is heterogeneous across seeds** (final hack fractions
  0.02, 0.06, 0.14, 0.22, 0.56), and the amplification is concentrated: 20% of
  prompts flip entirely while the rest still suppress. That is why Δρ crosses zero and ΔΛ does not.
  Both are reported, with the reason.
- The **N-independence prediction now IS tested** (E7). At u = 1 the two prompt-set sizes agree
  (-0.004 +- 0.084 at N=16 versus +0.010 +- 0.045 at N=64),
  while at the same block size m = 16 (u = 1 for the small set, u = 0.25 for the large one) they
  differ by -0.079 against -0.004. So the controlling quantity is u, not m. Across the
  5 matched values of u the largest gap is 0.065, about the size of the O(1/N) correction the
  appendix flags as non-negligible at N=16.
- The **KL sweep at u=1 is within noise** and used a single fixed learning rate; only the u=1/N side
  is clean. Stated.

### Verification done

Two independent adversarial reviews were run before writing (one on the code, one on the
derivations) and one on the finished draft. They caught, and the final version fixes: a confound
where contiguous leak blocks coincided with "first operand = a" (fixed by the permuted-block
sweep); a base-model mismatch across conditions (fixed by reporting drifts, and eliminated
entirely in the permuted design, base spread 0.0011); a k-dependent
contamination artefact in the factorial (fixed by keeping the SFT uniform floor off the leak
strings); an inconsistent treatment of the diagonal kernel term in Prop. 2; and an over-strong
"amplified if and only if" that is now stated as a leading-order sign.

## Before submitting

- [x] **Official `neurips_2026.sty` is in place** (from `Formatting_Instructions_For_NeurIPS_2026`,
      unmodified). `checklist.tex` is in `paper/` if the workshop wants it.
- [ ] Set `\workshoptitle{...}` to the actual workshop name (currently a placeholder); the style
      file warns if it is empty and the proceedings footnote comes out incomplete.
- [ ] Fill in the author block (it is currently the anonymous template; most 2026 workshops are
      double-blind, so leave it anonymous for submission).
- [ ] Check the target workshop's page limit. Main content is currently ~8.5 pages, which fits the
      common 9-page long-paper limit. For a 4-page venue (e.g. MATH-AI), cut Section 5.4–5.5 and
      the related-work paragraphs on reward-hacking theory, and move Figures 3–4 to the appendix.
- [ ] Add the NeurIPS checklist if the venue requires it (`checklist.tex` ships with the template).
- [ ] Push `src/` and `results/` to a public repo and put the link in the reproducibility section.

## Candidate venues (verified 2026-08-22)

| Workshop | City | Deadline (AoE) | Pages | Fit |
|---|---|---|---|---|
| Transitioning from Pre-Training to Post-Training | Sydney | **Aug 29** | 4–5 short / 9 long | best fit, it is a post-training paper |
| Can We Trust the Judge? (JUDGe) | Atlanta | **Aug 29** | 6 full / 4 short | strong, verifier reliability is the topic |
| Who Verifies the Agents? | Sydney | **Aug 29** | 4–9 | strong |
| Foundations of Language Model Security (FLMSec) | Paris | **Aug 27** | 8 | plausible, reward hacking as a security property |
| 6th MATH-AI | Atlanta | **Sep 25** | 4 (5 camera-ready) | needs a heavy cut, but most time |

NeurIPS 2026 runs in Sydney (Dec 6–12), Paris and Atlanta (Dec 9–13); workshop acceptance
notification is Sep 29 AoE for all of them. Workshop papers are non-archival.

## Ideas that were discarded (and why), in case they are useful later

1. **"GRPO's std-normalisation amplifies verifier false positives on hard prompts."**
   Dead. S-GRPO (2508.05928) states it in almost identical terms, derives the same closed forms
   `A_correct = sqrt((G-k)/k)`, and proposes an MSE-optimal reweighting. Also overlapping:
   Dr. GRPO (2503.20783), 2607.00152, 2510.18924.

2. **"i.i.d. verifier noise preserves the optimum; response-correlated error does not."**
   Dead. Everitt et al. (IJCAI 2017) formalised state-dependent reward corruption with a stronger
   no-free-lunch theorem; Egashira et al. (COLM 2026) demonstrated the RLVR case empirically;
   Fluri et al. (2406.15753) proved the "low training error does not bound regret" corollary;
   Cai et al. (2510.00915) name content-dependence as a limitation of their own noise model.
