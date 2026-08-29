"""E10: per-prompt weight copies of the *same* architecture.

The tabular arm in E1 is a valid test of Corollary 1, which is exact, but it is not literally
a "differ only in parameter sharing" comparison: a free per-prompt softmax table and a
transformer have different parameterisations, different optimisation geometry and different
effective step sizes.  A reviewer can therefore attribute part of the gap to architecture.

E10 removes that objection.  Every arm here is the identical transformer architecture,
identically initialised from the same base checkpoint, trained with the same optimiser, the
same learning-rate grid and the same number of steps.  The only difference is that the
per-prompt arm holds N independent copies of the weights, one per prompt, each trained on its
own prompt alone.  Parameter sharing is then the single manipulated variable, and
Corollary 1 predicts the copies carry no dependence on universality at all.

Copies are independent by construction (copy i only ever sees prompt i), so training them in
one loop is exactly equivalent to training them separately.
"""
from __future__ import annotations

import argparse, copy, glob, json, os, time

import numpy as np
import torch

from runner import get_base
from rlvr import measure

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

MS = [1, 4, 16, 64]
LRS = [3e-4, 1e-3, 3e-3]


def train_copies(base_pol, ver, *, steps, lr, seed, log_every=100):
    """One independent copy of the base transformer per prompt, each on its own prompt."""
    n = ver.n
    torch.manual_seed(seed)
    copies = [copy.deepcopy(base_pol) for _ in range(n)]
    opts = [torch.optim.Adam(c.parameters(), lr=lr) for c in copies]
    accept = torch.as_tensor(ver.accept, dtype=torch.float32)
    traj = []
    for t in range(steps + 1):
        if t % log_every == 0 or t == steps:
            with torch.no_grad():
                lp = torch.stack([copies[i].full_logprobs(prompt_idx=[i])[0] for i in range(n)])
            rec = measure(lp, ver, full=(t == steps))
            rec["step"] = t
            traj.append(rec)
        if t == steps:
            break
        for i in range(n):
            lp_i = copies[i].full_logprobs(prompt_idx=[i])[0]
            p = lp_i.exp()
            r = accept[i]
            V = (p * r).sum()
            # exact-expectation policy gradient with the exact mean baseline, per prompt
            loss = -((p.detach() * (r - V.detach())) * lp_i).sum()
            opts[i].zero_grad(); loss.backward(); opts[i].step()
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    seeds = [int(x) for x in a.seeds.split(",")]
    jobs = [(m, s, lr) for m in MS for s in seeds for lr in LRS][a.shard::a.nshards]
    path = os.path.join(OUT, f"exp10_shard{a.shard}.jsonl")
    done = set()
    for q in glob.glob(os.path.join(OUT, "exp10_shard*.jsonl")):
        for line in open(q):
            c = json.loads(line)["config"]
            if c.get("steps") == a.steps:
                done.add((c["m"], c["seed"], c["lr_neural"]))
    f = open(path, "a"); t0 = time.time()
    for i, (m, s, lr) in enumerate(jobs):
        if (m, s, lr) in done:
            continue
        prompts, ver, lam, pol, lp0 = get_base(m=m, k=1, seed=s)
        base = measure(lp0, ver, full=True)
        traj = train_copies(pol, ver, steps=a.steps, lr=lr, seed=s)
        rec = {"config": dict(m=m, k=1, seed=s, N=len(prompts), u=ver.u,
                              static_fpr=ver.static_false_positive_rate(),
                              algo="pg", norm="mean", steps=a.steps, lr_neural=lr,
                              lr_tabular=lr, opt="adam", beta=0.0, G=8, eps=0.03,
                              n_operands=8, exp="copies", code_version=5),
               "base": base, "copies": traj}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"[{i+1}/{len(jobs)}] m={m} u={ver.u:.3f} seed={s} lr={lr:g} "
              f"base_rho={base['mean_hack_frac']:.4f} -> copies_rho={traj[-1]['mean_hack_frac']:.4f} "
              f"acc={traj[-1]['acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
