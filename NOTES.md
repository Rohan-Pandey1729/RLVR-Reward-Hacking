# Notes for the authors

**Paper: "Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward
Hacking in RLVR"** (NeurIPS 2026 workshop *Transitioning from Pre-Training to Post-Training*,
camera-ready).

Working notes: what the claim is now, what changed between the accepted draft and the
camera-ready and why, what each reviewer asked for and where it is answered, what the data
say at ten seeds, what did not work, and what the authors still have to do by hand.

---

## The claim in one paragraph

At its KL-regularised optimum RLVR is a *filter*: `pi*(y|x) ∝ pi_ref(y|x) exp(r(x,y)/beta)`
multiplies every accepted response by the same factor, so the odds between a correct answer
and a verifier false positive are frozen at the base model's and the post-RL hack rate is
inherited (Prop. 1, known). First-order training on a shared-parameter model leaves that
baseline in *both* directions. Writing the drift of `log pi(h|x)` as a sum over all prompts of
`Cov[reward, tangent kernel]` (Lemma 1, known: Ren & Sutherland 2025, Tomihari 2026), a wrong
answer `h` is pushed up by every prompt that accepts it and pushed down, through the baseline,
by every prompt that rejects it, each in proportion to the model's own mass on `h` there.
For a coherently represented `h` the sign of the cross-prompt term is that of `u_eff - Vbar`
(Prop. 2), where `u_eff` is the kernel- and policy-weighted share of the model's mass on `h`
that sits on prompts which accept it and `Vbar` is the policy's own pass rate. The verifier's
error rate appears nowhere, and neither does universality `u = m/N` by itself: what decides
the sign is whether the model is *wrong the same way the verifier is*. A per-prompt policy has
no cross-prompt term at all (Cor. 1), which is what makes it a control rather than a baseline.

## What changed for the camera-ready, and why

The accepted draft said two things the data do not support, and the reviewers found both.

1. **"Sharing at RL time is not sufficient; a two-stage mechanism is needed."** This came from
   summarising the frozen-base controls (E9/E9b) by a linear slope in `log2 m`. Read per
   universality value, the shared model's curve is a *step*: at every `u < 1` it suppresses the
   accepted string far more than the per-prompt policy (paired within-seed difference −0.05 to
   −0.07, bootstrap CIs excluding zero at all six conditions), and at `u = 1` both arms amplify
   it, with the shared model amplifying as strongly as in the headline sweep at the next learning
   rate (+0.44 vs +0.60). A line fitted through a flat negative run that jumps at the last point
   has a small slope, which is how the wrong conclusion arose. The claim is withdrawn and the
   correction is stated in the paper (Section 6.3, contribution 2).
2. **"A property of the verifier's errors decides the outcome."** Not quite: the blind base and
   the headline base differ in *where* their mass on the leak sits (`u_eff` = `u` for the blind
   base; 0.09 / 0.23 / 0.68 / 1.00 at `u` = 1/64, 1/16, 1/4, 1 for the headline base), and that is
   what Eq. (2) weights by. The paper is now framed around this alignment rather than around
   universality as a property of the verifier alone.

Everything else is strengthening: every three-seed experiment was re-run at ten seeds from
regenerated base models, a learning-rate-robustness figure was added, a direct probe of
Prop. 2 was added, the missing citations were added, and the prose was rewritten in the plain
register of the two example papers.

3. **Final camera-ready pass (2026-11-06).** The main text now carries five figures instead of
   three: an overview schematic of the environment and the mechanism (Fig. 1), the headline
   sweep (Fig. 2), the slope at every learning rate (Fig. 3), a two-row mechanism figure with the
   cross-prompt term, coherence, per-prompt copies, both frozen-base controls and `u_eff` at the
   base (Fig. 4), and a two-row practice figure with GRPO, SGD, the KL sweep and verifier
   hardening (Fig. 5). The appendix keeps the probe, the factorial and the Adam grid. The
   abstract was restored to the structure of the reviewed abstract (same opening, the filter null
   model, the decomposition, the controls, the hardening consequence) with the withdrawn
   two-stage sentence replaced by the mass-placement reading and the step-size caveat. Prose was
   tightened to fit nine content pages with the extra figures; no number changed.

### Reviewer points and where they are answered

| Reviewer point | What we did | Where |
|---|---|---|
| pm4V-1, d6TQ-2, meta: fixed-base controls contradict "verifier property decides"; identify the representation difference before claiming "two-stage" | Per-`u` paired re-analysis of E9/E9b; two-stage claim withdrawn; `u_eff` measured for both bases; representation compared directly at `u = 1` (coherence 0.255 vs 0.294, mass on `h` 0.0151 vs 0.0355) | §6.3, Fig. 4 (bottom), App. C |
| pm4V-2: low-`u` suppression and the missing `u = 1` amplification depend on the learning-rate rule; the rise with `u` is the trustworthy result | New figure: universality slope at every LR for Adam, GRPO, SGD, both arms (positive with CI excluding zero at 10 of 11 settings; per-prompt slope ≈ 0 everywhere); both verifier-saturating Adam LRs reported (`u = 1`: +0.028 at 1e-3, +0.600 with 10/10 seeds at 3e-3); step-size caveat in abstract, contributions, limitations | §6.1, Fig. 3, App. D Table 2 |
| pm4V-3, d6TQ-1: raw-`u` threshold mispredicts the crossings; separate instantaneous prediction from final drift | Prop. 2 restated in terms of `u_eff`; raw-`u` threshold removed from abstract and contributions; new probe E11 compares the coherent-channel prediction with the exact cross-prompt log-odds drift along early SGD steps and reports the residual | §4, App. C, Fig. 6 |
| pm4V-4, d6TQ-3: the reconciliation of the two literatures is argued, not measured | Stated as a hypothesis; says what would test it (per-prompt hack fraction on released data; cross-prompt gradient cosine of accepted wrong responses on a real model); Egashira's relative-error leaks explained through coherence rather than string identity | §7 |
| pm4V-5: Lemma 1 is known (Tomihari; Ren & Sutherland); amplification half is coherent gradients (Chatterjee); cite Echo Chamber | All cited; Lemma 1 labelled "known"; contribution framed around the RL-specific push-down through the baseline and its two-sidedness | §2, §4 |
| pm4V-6, d6TQ-4, d6TQ-Q2: three-seed claims; E6 used a different operating rule and no test; "29 points manufactured by RL" assumes additivity; were both patches run from the same checkpoint? | E2, E3, E5, E6, E7 re-run at ten seeds; E6 from the identical checkpoint per seed (base accuracy identical by construction), standard operating rule, both LRs stored and reported, paired within-seed gaps with bootstrap CIs; the decomposition is stated as "available without sharing" vs "requires sharing if the effects add" | §6.5, Fig. 5 (bottom), App. B |
| pm4V-7: the "wrong sign" audit claim is a design artefact (base hack mass fixed at `eps` for every `k`) | Removed; the `k` trend is described as a property of the design | §6, App. B, App. D |
| d6TQ-Q3: kernel entries can be negative; when is `u_eff` in [0, 1]? | Assumption stated; measured: 13% of same-string entries negative at the base models, 9% of the weight | §4, App. C |
| q735, meta: toy scale, one architecture | Limitations paragraph; future-work pointer to the measurable quantity on real models | §7 |
| Camera-ready | `\usepackage[final,dblblindworkshop]{neurips_2026}`, workshop title footnote, author block from OpenReview, 9 content pages, no overfull boxes | `paper/main.tex` |

## Design decisions, and why

**Why one panel size?** Every data figure is built by `panel_grid()` in `src/analysis.py`: three
identical panels to a row on an 11 in canvas, so a one-row figure is 3:1, a two-row figure about
3:2, and default 10pt text prints at 5pt. These are the proportions of the plots in
an earlier paper. A panel keeps its size in every figure; a cell with no panel holds the
legend. Only `fig0_overview` and `fig6_field` sit outside the grid, and the two heat maps of
`fig8_appendix` keep the row height but are wider. The larger figures were paid for by moving the
"RLVR as filtering" and "Verifier error" paragraphs, verbatim, to Appendix F.

**Why a two-digit response space?** Because 100 responses × 64 prompts can be enumerated, so
`pi(.|x)` is computed exactly at every logged step and no reported hack fraction, log-odds or
accuracy is a Monte-Carlo estimate. The effect being measured is a composition *inside* the
accepted set, which sampling estimates poorly.

**Why matched tabular/transformer pairs?** Corollary 1 says a per-prompt policy's dynamics
*cannot* depend on universality, exactly, for any learning rate and any number of steps. Any
`u`-dependence in the transformer is therefore attributable to sharing. The tabular policy is
initialised to reproduce the transformer's base distribution to 1e-4 and gets the same grid.

**Why also per-prompt copies of the transformer (E10)?** Tabular vs transformer confounds
parameterisation with sharing. Copies of the identical network, one per prompt, remove sharing
with the architecture held fixed; the slope vanishes (+0.0019 vs +0.0210 per doubling, paired
difference +0.019, CI [+0.008, +0.033]).

**Why frozen-base controls (E9/E9b)?** In the sweeps the supervised target and the verifier
change together, so verifier universality and the universality of the base model's own errors
coincide by construction. Freezing one checkpoint per seed and varying only the verifier's mask
separates them. E9 fixes the static false-positive rate, E9b fixes the accepted string's
per-prompt base mass.

**Why report drifts rather than absolute hack fractions?** The realised base model is a
shared-parameter network that fits a universal leak slightly better, so base hack fraction
varies a little across `u`. Drift from each run's own base is exactly zero under the filter
baseline, immune to that.

**Why select the learning rate by verifier reward, and why also report every learning rate?**
It is the criterion a practitioner has. The rule (smallest LR within 1e-3 of the best final
verifier reward) is conservative with respect to our hypothesis, and reviewer pm4V was right
that it decides how the low-`u` end looks; hence Fig. 3 and Table 2 report everything.

**Why the randomised-block control (E2)?** With prompts enumerated first-operand-major, a
contiguous block of size ≥ 8 is "first operand = a", one input token. The sweep is repeated with
block membership permuted. The 3e-2 learning rate was dropped from E2 because it diverges and
is never selected by the rule.

**Why the threshold probe (E11)?** Prop. 2 is a statement about one channel of the exact
drift, at leading order, under homogeneity assumptions the base models violate. The probe
computes the exact cross-prompt log-odds drift and the coherent-channel prediction per prompt
along early SGD steps and reports sign agreement and the residual, so the reader can see how
much of the drift the proposition actually describes.

## Results as they landed (ten seeds everywhere unless noted)

| sweep | shared-parameter policy | per-prompt policy |
|---|---|---|
| exact PG, Adam, contiguous blocks (E1) | `rho` 0.025 → 0.160 ± 0.050; `drho` −0.104 → +0.028 ± 0.050; acc 0.974 → 0.840 | slope +0.0028 per doubling (flat) |
| same, Adam 3e-3 (also saturates the verifier) | `drho` −0.098 → +0.600 ± 0.084, 10/10 seeds positive at `u = 1`, 10/10 negative at `u = 1/64` | flat |
| permuted blocks (E2) | `drho` −0.108 → +0.130 ± 0.100; slope +0.027 | slope −0.0001 |
| sampled GRPO, G = 8 (E4) | `drho` −0.103 → +0.663 ± 0.079, crossing `u ≈ 0.27` | flat |
| plain SGD (E8) | `rho` 0.051 → 1.000, acc 0.926 → 0.000, crossing `u ≈ 0.07` | `rho` 0.053 → 0.055 |
| per-prompt copies of the transformer (E10) | slope +0.0019 (CI [−0.0008, +0.0050]) vs +0.0210 shared, paired diff +0.019 (CI [+0.008, +0.033]) | — |
| frozen blind base, fixed FPR (E9) | paired shared − per-prompt: −0.049 at `u = 1/64`, −0.069 at `u = 1/2` (CIs exclude zero); at `u = 1` +0.014 vs +0.046, not distinguishable; +0.439 at `u = 1` at the next LR | — |
| frozen blind base, fixed mass (E9b) | −0.056 at `u = 1/2` (CI [−0.064, −0.048]); +0.427 at `u = 1` at the next LR | — |
| factorial `u × k`, both eightfold (E3) | 43% of the variance in `dLambda` from universality, 12% from the false-positive rate; averaged over `k`, `dLambda` −7.19 → −4.15 across `u`; averaged over `u`, −5.54 → −6.85 across `k` | — |
| prompt-set size `N ∈ {16, 64}` (E7) | at `u = 1`: −0.024 ± 0.053 (N = 16) vs +0.021 ± 0.028 (N = 64); same `m = 16`: −0.024 vs −0.081; matched-`u` gaps ≤ 0.016 at the three smallest `u`, ≤ 0.045 overall | — |
| KL sweep (E5) | at `u = 1/64`, `beta` 0 → 0.3 takes `drho` −0.104 → 0.000 and accuracy 0.978 → 0.821 | — |
| verifier hardening (E6), same checkpoint, paired | gap (universal patch − idiosyncratic patch) 8 ± 2 points at 1e-3 (CI [4, 13], 10/10), 38 ± 9 at 3e-3 (CI [22, 55], 9/10); verifier reward ≥ 0.973 everywhere | gap 6 (CI [5, 8]) |

Other headline numbers:

- **Filter invariance verified exactly**: mirror descent and the KL optimum leave the per-prompt
  hack fraction invariant to 8.6e-9 and the log-odds to 9.6e-8; the *pooled* share moves by
  +0.057 under pure filtering (the trap in pooled metrics).
- **Corollary 1 verified exactly**: the per-prompt policy's cross-prompt kernel and drift are
  identically 0 at every `u`; the transformer's cross-prompt kernel is 0.098 of its self-kernel.
- **Coherence** of the leak's per-prompt gradients rises +0.02 → +0.29 with `u`; correct answers
  stay at +0.03.
- **Threshold probe (E11, 3 seeds, 28 conditions, 672 prompt-level points)**: the
  coherent-channel prediction has the sign of the exact cross-prompt drift in 93% of condition
  means and 70% of prompt-level points; the homogeneous raw-`u` formula manages 71% / 53%
  (chance per prompt). Median |residual| / |exact| = 0.9, so the channel is directional, not a
  bound. 13% of same-string kernel entries are negative (9% of the weight).

### What did NOT work, and is reported as such

- **The `u = 1` endpoint under Adam at the selected operating point is inside its own error bar.**
  Final hack fractions span 0.02–0.56; 16% of prompts flip at `u = 1` vs 2% at `u = 1/64`, so
  `drho` approaches zero while `dLambda` stays negative. The decisive sign change is carried by
  GRPO, SGD and Adam at 3e-3. All of this is in §6 and §6.1.
- **The hardening gap at the conservative operating point is small.** The old 40 ± 16 (three
  seeds) came from an effectively higher learning rate for one arm. With the standard rule and
  the same checkpoint it is 8 ± 2 points, of which the per-prompt policy already shows 6. The
  paper reports both learning rates and says "up to 38 points".
- **Prop. 2 is one channel.** The residual (everything the coherent same-string channel does not
  capture, in particular the shared machinery that raises correct answers) is the same order as
  the prediction. The paper says so in §4, §7 and App. C rather than fitting a threshold.
- **The factorial's `k` effect is a design property.** Base hack mass is fixed at `eps` for every
  `k`, so more accepted strings spread the same mass thinner; the mild decrease with `k` says
  nothing general about audits.

### Verification done

Every number in the paper is a macro written by `src/figures.py` into `paper/numbers.tex` (and
`paper/lrtable.tex`); nothing is typed by hand. The build was checked for unresolved macros
(`??`), TeX errors and overfull boxes. Two adversarial reviews of the original code and
derivations preceded the first version; this round re-derived Prop. 2 in the `u_eff` form,
checked the E6 code path (both patches from one checkpoint per seed, both learning rates stored),
and verified the four new citations against arXiv. The final pass re-ran `figures.py` and
confirmed that every macro value is unchanged (three macros were added: the median `u = 1` seed,
the count of verifier-saturating Adam learning rates, and `N kappa / kappa_s` at low `u`), that
no `
ef` is unresolved, that the build has no overfull boxes, and that every number quoted in
the main text matches `results/numbers.json`.

## Before submitting

- [x] **Author block.** Names and affiliations are taken from OpenReview (Rohan Pandey, UW;
      Michael Li, CMU; Ray Chen, Independent). All three are marked equal contribution and
      Ray Chen's email is set. Check author order.
- [ ] **Title.** Unchanged from the accepted version; change it in `\title{}` if the new framing
      warrants (the subtitle already says "both suppresses and amplifies").
- [ ] **LLM-use sentence** at the end of §7 ("LLMs assisted with experiment design and manuscript
      preparation; every number comes from the released code"): keep, edit or remove.
- [ ] **Rebuild once** with `PYTHON=/path/to/python bash build.sh` after any edit and confirm
      content still ends on page 9 (there are about four lines of slack).
- [ ] Push `src/` and `results/` to a public repo and put the link in App. E.
- [ ] `paper/checklist.tex` ships with the template if the workshop wants it.
- [x] Official `neurips_2026.sty`, `final` option, `\workshoptitle{}` set, nine content pages.
- [x] Citations verified (2026-08-23 pass, plus the four new entries on 2026-10-31).

## Ideas that were discarded (and why)

1. **"GRPO's std-normalisation amplifies verifier false positives on hard prompts."** Dead:
   S-GRPO (2508.05928) states it and derives the same closed forms; also Dr. GRPO (2503.20783),
   2607.00152, 2510.18924.
2. **"i.i.d. verifier noise preserves the optimum; response-correlated error does not."** Dead:
   Everitt et al. (IJCAI 2017), Egashira et al. (COLM 2026), Fluri et al. (2406.15753), Cai et al.
   (2510.00915).
3. **"Two-stage mechanism: the base model must encode the acceptance pattern before sharing can
   act."** Withdrawn in this revision; see above. The base model matters through where it puts
   its mass, which `u_eff` already captures.
