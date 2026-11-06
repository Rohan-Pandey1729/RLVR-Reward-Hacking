"""E4: is the controlling quantity the *number* of prompts sharing a leak (m) or the
*fraction* (u = m / N)?

Sweeping m at three different prompt-set sizes N in {16, 36, 64} separates them: plotted
against u the curves should not coincide, plotted against m they should.
"""
from __future__ import annotations

import argparse, json, os, time

import torch

from runner import run_cell

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

GRID = {4: [1, 2, 4, 8, 16], 8: [1, 2, 4, 8, 16, 32, 64]}
SEEDS = list(range(10))   # camera-ready: ten seeds
LRS = [3e-4, 1e-3, 3e-3]


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
    bases = [(no, m, s) for no, ms in GRID.items() for m in ms for s in seeds][a.shard::a.nshards]
    jobs = [(no, m, s, lr) for (no, m, s) in bases for lr in LRS]
    path = os.path.join(OUT, f"exp4_shard{a.shard}.jsonl")
    done = set()
    import glob as _glob
    for q in _glob.glob(os.path.join(OUT, "exp4_shard*.jsonl")):
        for line in open(q):
            c = json.loads(line)["config"]
            if c.get("steps") == a.steps:  # a cell run at another budget is not this cell
                done.add((c["n_operands"], c["m"], c["seed"], c["lr_neural"]))
    f = open(path, "a"); t0 = time.time()
    for i, (no, m, s, lr) in enumerate(jobs):
        if (no, m, s, lr) in done:
            continue
        r = run_cell(m=m, k=1, seed=s, n_operands=no, algo="pg", norm="mean",
                     steps=a.steps, lr_neural=lr, lr_tabular=lr,
                     opt_name="adam", log_every=25)
        f.write(json.dumps(r) + "\n"); f.flush()
        print(f"[{i+1}/{len(jobs)}] N={r['config']['N']} m={m} u={r['config']['u']:.3f} "
              f"seed={s} lr={lr:g} base={r['base']['mean_hack_frac']:.4f} "
              f"N_rho={r['neural'][-1]['mean_hack_frac']:.4f} "
              f"T_rho={r['tabular'][-1]['mean_hack_frac']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
