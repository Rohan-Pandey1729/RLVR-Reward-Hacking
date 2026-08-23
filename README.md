# Wrong the Same Way

### Universality, Not Error Rate, Governs Reward Hacking in RLVR

Code and data for the paper. Everything runs on CPU; no GPU is required. The whole
pipeline reproduces on two cores.

## What this measures

RLVR at its KL-regularised optimum is a *filter*: it reweights the verifier-accepted set as a
whole and leaves the composition of that set at the base model's, so the post-RL rate of verifier
false positives should be inherited. This repository tests when and why real training departs from
that baseline, and shows the controlling quantity is the **universality** of a leak, the fraction
of prompts on which the same wrong response is accepted, rather than the verifier's error rate.

## Layout

```
src/env.py        task, response space, and the leaky verifier (universality m, error rate k)
src/policies.py   two policy classes over the identical distribution family:
                  TabularPolicy      -- free per-prompt parameters (no sharing)
                  TransformerPolicy  -- 2-layer decoder, all prompts share all weights
src/base.py       SFT construction of the base model and the matched tabular initialisation
src/rlvr.py       objectives (exact PG, sampled GRPO, exact mirror descent) + exact measurement
src/runner.py     cached base models, matched-pair runner, gradient-coherence probe
src/theory.py     numerical verification of Prop. 1 and Lemma 1, and the drift/kernel measurements
src/exp2..exp5    the four experiments
src/figures.py    regenerates every figure and paper/numbers.tex
paper/            LaTeX source
```

## Reproducing

```bash
pip install torch numpy matplotlib
cd src
python3 exp2_universality.py --nshards 1 --opt adam   # headline universality sweep
python3 exp3_factorial.py                             # universality x false-positive rate
python3 theory.py                                     # propositions + mechanism
python3 exp5_practice.py --part a                     # sampled GRPO
python3 exp5_practice.py --part b                     # KL sweep
python3 exp5_practice.py --part c                     # verifier-hardening budget
python3 exp4_scaling.py                               # prompt-set size (u versus m)
python3 exp2_universality.py --opt sgd                # SGD robustness
python3 figures.py                                    # figures + numbers.tex
```

`--shard i --nshards n` splits any experiment across `n` worker processes.
Base models are cached in `cache/`, so re-runs are cheap.

## Measurement

The response space is small enough to enumerate, so `pi(.|x)` is computed **exactly** for every
prompt at every logged step. No metric reported in the paper is a Monte-Carlo estimate; the only
stochasticity is across seeds (transformer init, leak-string draw, and rollout sampling in the
GRPO condition). `src/figures.py` writes every number quoted in the paper into
`paper/numbers.tex`, so nothing is transcribed by hand.

## LaTeX

`paper/neurips_2026.sty` is the official NeurIPS 2026 style file, unmodified, taken from
`Formatting_Instructions_For_NeurIPS_2026`. `paper/checklist.tex` ships with it and is included
in case the target workshop requires the checklist; it is not currently `\input` by the paper.

Build with `bash build.sh`, which regenerates every figure and `paper/numbers.tex` from the raw
results and then runs pdflatex/bibtex. The paper is set with
`\usepackage[dblblindworkshop]{neurips_2026}` and
`\workshoptitle{Transitioning from Pre-Training to Post-Training}`, the NeurIPS 2026 workshop it
targets. Content ends on page 9, inside that workshop's 9-page long-paper limit.
