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
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    jobs = [(no, m, s) for no, ms in GRID.items() for m in ms for s in SEEDS]
    jobs = jobs[a.shard::a.nshards]
    path = os.path.join(OUT, f"exp4_shard{a.shard}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            c = json.loads(line)["config"]
            done.add((c["n_operands"], c["m"], c["seed"]))
    f = open(path, "a"); t0 = time.time()
    for i, (no, m, s) in enumerate(jobs):
        if (no, m, s) in done:
            continue
        r = run_cell(m=m, k=1, seed=s, n_operands=no, algo="pg", norm="mean",
                     steps=a.steps, lr_neural=1e-3, lr_tabular=3e-2,
                     opt_name="adam", log_every=25)
        f.write(json.dumps(r) + "\n"); f.flush()
        print(f"[{i+1}/{len(jobs)}] N={r['config']['N']} m={m} u={r['config']['u']:.3f} "
              f"seed={s} base_hf={r['base']['hack_frac']:.4f} "
              f"neural_hf={r['neural'][-1]['hack_frac']:.4f} "
              f"tab_hf={r['tabular'][-1]['hack_frac']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
