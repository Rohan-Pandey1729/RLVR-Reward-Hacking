"""E11: raw kernel scales, gradient norms, and audit-distribution false-positive rates.

Two things the paper asserts and previously did not measure.

(1) Definition 3 is stated as a kernel model K = K_coh + R.  A cosine equals kappa/kappa_s only
    when the per-prompt gradient norms are equal, so we report the raw cross-prompt and self
    kernel entries and the spread of the gradient norms rather than a cosine alone.

(2) The static false-positive rate k/(|Y|-1) presumes an audit that samples wrong responses
    uniformly.  Real audits sample from a model.  We therefore also report the false-positive
    mass under the base policy and under the trained policy, which is the exposure that
    actually matters operationally.
"""
from __future__ import annotations

import json, os

import numpy as np
import torch

from runner import get_base, gradient_coherence
from rlvr import measure, train
from base import matched_tabular

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MS = [1, 2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2]


def main():
    torch.set_num_threads(2)
    rows = []
    for m in MS:
        for s in SEEDS:
            prompts, ver, lam, pol, lp0 = get_base(m=m, k=1, seed=s)
            base = measure(lp0, ver)
            coh = gradient_coherence(pol, ver, max_prompts=12, seed=s)
            tr = train(pol, ver, algo="pg", norm="mean", steps=400, lr=1e-3, seed=s,
                       log_every=400, opt_name="adam")
            fin = tr[-1]
            rows.append(dict(
                m=m, u=ver.u, seed=s,
                fpr_uniform=ver.static_false_positive_rate(),
                fpr_base_policy=base["hack_mass"],        # base-policy-weighted exposure
                fpr_trained_policy=fin["hack_mass"],      # current-policy-weighted exposure
                **coh))
            r = rows[-1]
            print(f"m={m} seed={s} | uniform={r['fpr_uniform']:.4f} "
                  f"base={r['fpr_base_policy']:.4f} trained={r['fpr_trained_policy']:.4f} "
                  f"| kappa={r['hack_kappa_raw']:.4g} kappa_s={r['hack_kappa_s_raw']:.4g} "
                  f"gnorm_spread={r['hack_gnorm_spread']:.2f}", flush=True)
    with open(os.path.join(OUT, "exp11_kernel.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
