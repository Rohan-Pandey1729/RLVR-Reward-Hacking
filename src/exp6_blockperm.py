"""E6: the universality sweep with RANDOMISED block membership.

With prompts enumerated first-operand-major, a contiguous block of size m>=8 is exactly
"first operand = a" -- a property the transformer can read off a single input token. That
confounds universality with how simple the prompt-to-leak map is to represent. Here block
membership is a random permutation of prompts, and leak strings are drawn independently per
(seed, m, k) rather than nested across m. Universality m, the number of accepted wrong
strings per prompt, and the static false-positive rate are unchanged.
"""
from __future__ import annotations

import argparse, json, os, time

import torch

from runner import run_cell

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

MS = [1, 2, 4, 8, 16, 32, 64]
SEEDS = list(range(10))   # camera-ready: ten seeds, as in the headline sweep
LRS = [3e-4, 1e-3, 3e-3]   # camera-ready: 3e-2 dropped; it diverges in every seed and is never selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    seeds = [int(x) for x in a.seeds.split(",")]
    bases = [(m, s) for m in MS for s in seeds][a.shard::a.nshards]
    jobs = [(m, s, lr) for (m, s) in bases for lr in LRS]
    path = os.path.join(OUT, f"exp6_shard{a.shard}.jsonl")
    done = set()
    import glob as _glob
    for q in _glob.glob(os.path.join(OUT, "exp6_shard*.jsonl")):
        for line in open(q):
            c = json.loads(line)["config"]
            if c.get("steps") == a.steps:  # a cell run at another budget is not this cell
                done.add((c["m"], c["seed"], c["lr_neural"]))
    f = open(path, "a"); t0 = time.time()
    for i, (m, s, lr) in enumerate(jobs):
        if (m, s, lr) in done:
            continue
        r = run_cell(m=m, k=1, seed=s, algo="pg", norm="mean", steps=a.steps,
                     lr_neural=lr, lr_tabular=lr, opt_name="adam", log_every=25,
                     block_perm=True)
        f.write(json.dumps(r) + "\n"); f.flush()
        print(f"[{i+1}/{len(jobs)}] m={m} seed={s} lr={lr:g} base={r['base']['mean_hack_frac']:.4f} "
              f"| N rho={r['neural'][-1]['mean_hack_frac']:.4f} acc={r['neural'][-1]['acc']:.3f} "
              f"lky={r['neural'][-1]['leaky_reward']:.3f} "
              f"| T rho={r['tabular'][-1]['mean_hack_frac']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
