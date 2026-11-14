# Wrong the Same Way

### Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR

Code and data for the paper (NeurIPS 2026 workshop *Transitioning from Pre-Training to
Post-Training*, camera-ready). Everything runs on CPU; no GPU is required.

## What this measures

RLVR at its KL-regularised optimum is a *filter*: it reweights the verifier-accepted set as a
whole and leaves the composition of that set at the base model's, so the post-RL rate of verifier
false positives should be inherited. First-order training on a shared-parameter model departs
from that baseline in both directions. A wrong answer is pushed up by every prompt that accepts it
and pushed down, through the baseline, by every prompt that rejects it, each in proportion to the
model's own probability of producing it there. The quantity that sets the sign is therefore the
share of the model's mass on a wrong answer that sits on prompts which accept it (`u_eff` in the
paper), not the verifier's error rate and not universality by itself. This repository sweeps
universality at a fixed error rate, removes sharing with the architecture held fixed, freezes the
base model and varies only the verifier, and measures the drift decomposition directly.

## Layout

```
src/env.py              task, response space, the leaky verifier (universality m, error rate k),
                        and the mask-only verifier of the frozen-base controls
src/policies.py         two policy classes over the identical distribution family:
                        TabularPolicy      -- free per-prompt parameters (no sharing)
                        TransformerPolicy  -- 2-layer decoder, all prompts share all weights
src/base.py             SFT construction of the base model and the matched tabular initialisation
src/rlvr.py             objectives (exact PG, sampled GRPO, exact mirror descent) + exact measurement
src/runner.py           cached base models, matched-pair runner, gradient-coherence probe
src/theory.py           numerical verification of Prop. 1 and Lemma 1, drift/kernel measurements
src/exp2_universality.py   E1 (Adam) and E8 (SGD): the headline sweep
src/exp6_blockperm.py      E2: permuted block membership
src/exp3_factorial.py      E3: universality x error rate
src/exp5_practice.py       E4 (sampled GRPO), E5 (KL sweep), E6 (verifier hardening)
src/exp4_scaling.py        E7: prompt-set size
src/exp9_maskonly.py       E9 / E9b: frozen base, mask-only controls
src/exp10_copies.py        E10: per-prompt copies of the same transformer
src/exp12_threshold.py     E11: u_eff at the base; predicted vs exact drift along SGD
src/mechfig.py          the overview schematic (environment and mechanism, Figure 1), drawn by figures.py
src/figures.py          regenerates every figure, paper/numbers.tex and paper/lrtable.tex
figures/                the figures, as PNG files written by figures.py:
                        fig0_overview, fig1_universality, fig2_lr, fig3_mechanism and
                        fig4_practice in the main text; fig5_probe, fig6_field,
                        fig7_factorial and fig8_appendix in the appendix
paper/                  LaTeX source
results/                raw result files (jsonl / json) read by figures.py
requirements.txt        the Python dependencies, pinned to the versions the paper was built with
```

## Reproducing

Use Python 3.11 in a virtual environment (the CPU build of `torch` is enough):

```bash
python -m venv .venv
source .venv/bin/activate                         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd src
python exp2_universality.py --opt adam           # E1, ten seeds, five learning rates
python exp2_universality.py --opt sgd            # E8
python exp6_blockperm.py                          # E2
python exp3_factorial.py                          # E3
python exp5_practice.py --part a                  # E4 sampled GRPO
python exp5_practice.py --part b                  # E5 KL sweep
python exp5_practice.py --part c                  # E6 verifier hardening (both learning rates stored)
python exp4_scaling.py                            # E7
python exp9_maskonly.py                           # E9  (add --mode constant_mass for E9b)
python exp10_copies.py                            # E10
python exp12_threshold.py --seed 0                # E11 (repeat for seeds 1 and 2)
python theory.py                                  # propositions + mechanism probes
python figures.py                                 # figures, numbers.tex, lrtable.tex
```

Every script takes `--shard i --nshards n` to split its jobs across `n` worker processes, and
`--seeds 0,1,...` to choose seeds; scripts skip cells that already exist in `results/`. Base
models are cached in `cache/` (not versioned), so re-runs are cheap. On one core of a laptop CPU
(Intel Core i7-1360P), one base model takes 30 to 100 s to build and one 400-step run of both
policies 20 to 60 s.

Results are bit-for-bit reproducible only at the torch thread count that produced them, for base
models and RL runs alike. The experiment scripts run single-threaded unless given `--threads`,
and everything in `results/` was produced single-threaded wherever we checked, except two seed-0
base models built with two threads: the headline configuration's base at `m=64`, which E5, E7
and E11 load from `cache/`, and the blind base, which E11 loads. Rebuilt single-threaded, these
two come out slightly different, and so do the results that use them. New result records also
store the thread counts of the run and of its base model (`threads`, `base_threads`).

## Measurement

The response space is small enough to enumerate, so `pi(.|x)` is computed **exactly** for every
prompt at every logged step. No metric reported in the paper is a Monte-Carlo estimate; the only
stochasticity is across seeds (transformer init, leak-string draw, and rollout sampling in the
GRPO condition). `src/figures.py` writes every number quoted in the paper into
`paper/numbers.tex` (and the learning-rate table into `paper/lrtable.tex`), so nothing is
transcribed by hand. The two-row figures (`fig3_mechanism`, `fig4_practice`) are assembled
from the same panel functions as the appendix figures, so a panel is drawn once.

## LaTeX

`paper/neurips_2026.sty` is the official NeurIPS 2026 style file, unmodified. The paper is set
with `\usepackage[final,dblblindworkshop]{neurips_2026}` and
`\workshoptitle{Transitioning from Pre-Training to Post-Training}`, so the camera-ready carries
author names and the workshop footnote. Build with `bash build.sh`, which regenerates every figure
and number from the raw results and then runs pdflatex/bibtex. It uses `.venv` when that exists;
set `PYTHON=/path/to/python` to use another interpreter. Content
ends on page 9, inside the workshop's 9-page long-paper limit, which excludes references and
appendices.
