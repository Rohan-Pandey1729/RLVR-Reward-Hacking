"""E3: crossed factorial -- leak universality m  x  verifier false-positive rate k.

k is the number of wrong strings the verifier accepts per prompt, i.e. exactly the
quantity a static verifier audit estimates.  m is how many prompts share a leak.
The SFT target keeps its uniform component off the leak strings, so total base hack mass is
exactly eps for every k and the base model's P(hack | rewarded) is matched across every cell;
only the verifier changes.

Prediction: the post-RL hack fraction of a shared-parameter policy moves with m and is
close to flat in k.
"""
from __future__ import annotations

import argparse, json, os, time

import torch

from runner import run_cell

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

MS = [8, 16, 32, 64]
KS = [1, 2, 4, 8]
SEEDS = list(range(10))   # camera-ready: ten seeds


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
    jobs = [(m, k, s) for m in MS for k in KS for s in seeds][a.shard::a.nshards]
    path = os.path.join(OUT, f"exp3cf_shard{a.shard}.jsonl")
    done = set()
    import glob as _glob
    for q in _glob.glob(os.path.join(OUT, "exp3cf_shard*.jsonl")):
        for line in open(q):
            c = json.loads(line)["config"]
            if c.get("steps") == a.steps:  # a cell run at another budget is not this cell
                done.add((c["m"], c["k"], c["seed"]))
    f = open(path, "a"); t0 = time.time()
    for i, (m, k, s) in enumerate(jobs):
        if (m, k, s) in done:
            continue
        r = run_cell(m=m, k=k, seed=s, algo="pg", norm="mean", steps=a.steps,
                     lr_neural=1e-3, lr_tabular=3e-2, opt_name="adam", log_every=25,
                     clean_floor=True)
        f.write(json.dumps(r) + "\n"); f.flush()
        print(f"[{i+1}/{len(jobs)}] m={m} k={k} seed={s} fpr={r['config']['static_fpr']:.4f} "
              f"base_hf={r['base']['hack_frac']:.4f} "
              f"neural_hf={r['neural'][-1]['hack_frac']:.4f} "
              f"neural_acc={r['neural'][-1]['acc']:.3f} "
              f"tab_hf={r['tabular'][-1]['hack_frac']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
