"""E2 (headline): sweep leak universality at fixed verifier false-positive rate.

For every block size m the verifier accepts exactly one wrong string per prompt, so its
static false-positive rate -- what any fixed audit set measures -- is identical across all
conditions.  The only thing that changes is whether the accepted-wrong string is shared
across m prompts or is idiosyncratic to each.

Matched pairs: the tabular policy is initialised to the *identical* distribution as the
base transformer, so the two policies differ only in whether parameters are shared across
prompts.  The *same* learning-rate grid is applied to both, and the operating point is
chosen post hoc by the criterion a practitioner would use -- the learning rate that
maximises the verifier's own reward.
"""
from __future__ import annotations

import argparse, json, os, time

import torch

from runner import run_cell

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

MS = [1, 2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4]
GRIDS = {"adam": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "sgd": [0.3, 1.0, 3.0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--opt", type=str, default="adam")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    # Shard by (m, seed) so every learning rate for a given base model lands on the same
    # worker; the expensive SFT is then built exactly once.
    bases = [(m, s) for m in MS for s in SEEDS][a.shard::a.nshards]
    jobs = [(m, s, lr) for (m, s) in bases for lr in GRIDS[a.opt]]
    path = os.path.join(OUT, f"exp2_{a.opt}_shard{a.shard}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            c = json.loads(line)["config"]
            done.add((c["m"], c["seed"], c["lr_neural"]))
    f = open(path, "a"); t0 = time.time()
    for i, (m, s, lr) in enumerate(jobs):
        if (m, s, lr) in done:
            continue
        r = run_cell(m=m, k=1, seed=s, algo="pg", norm="mean", steps=a.steps,
                     lr_neural=lr, lr_tabular=lr, opt_name=a.opt, log_every=25)
        f.write(json.dumps(r) + "\n"); f.flush()
        print(f"[{i+1}/{len(jobs)}] m={m} seed={s} lr={lr:g} base={r['base']['hack_frac']:.4f} "
              f"| N hf={r['neural'][-1]['hack_frac']:.4f} acc={r['neural'][-1]['acc']:.3f} "
              f"lky={r['neural'][-1]['leaky_reward']:.3f} "
              f"| T hf={r['tabular'][-1]['hack_frac']:.4f} acc={r['tabular'][-1]['acc']:.3f} "
              f"lky={r['tabular'][-1]['leaky_reward']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
