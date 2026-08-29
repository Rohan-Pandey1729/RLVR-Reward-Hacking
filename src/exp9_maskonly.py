"""E9: the same-checkpoint, mask-only causal control.

E1 varies the verifier's leak structure, which changes the SFT target, which changes the
realised base model.  Because the base is a shared transformer it fits a universal leak map
better than an idiosyncratic one, so the leak's cross-prompt gradient coherence grows about
seventeenfold across the E1 sweep.  Proposition 2's leading term is kappa * p_h * (u - Vbar),
so E1 cannot separate universality from coherence, from base probability, or from how easy
the leak is to represent.

E9 separates them.  One frozen checkpoint per seed serves every condition: the base is
contaminated with a shared string h and a per-prompt string s_x at every prompt, identically
in all conditions, so the initial policy, p_h, p_{s_x} and the initial tangent kernel are
bit-identical across the sweep.  The verifier then accepts h on m prompts and s_x on the
other N - m, which keeps exactly N accepted (prompt, wrong-response) pairs and therefore an
exactly constant static false-positive rate.

The only quantity that varies is which string each prompt's acceptance routes through.
"""
from __future__ import annotations

import argparse, glob, json, os, time

import torch

from runner import run_cell_maskonly

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

MS = [1, 2, 4, 8, 16, 32, 64]
LRS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--mode", type=str, default="constant_fpr",
                    choices=["constant_fpr", "constant_mass"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    seeds = [int(x) for x in a.seeds.split(",")]
    bases = [(m, s) for m in MS for s in seeds][a.shard::a.nshards]
    jobs = [(m, s, lr) for (m, s) in bases for lr in LRS]
    tag = "" if a.mode == "constant_fpr" else "b"
    path = os.path.join(OUT, f"exp9{tag}_shard{a.shard}.jsonl")
    done = set()
    for q in glob.glob(os.path.join(OUT, f"exp9{tag}_shard*.jsonl")):
        for line in open(q):
            c = json.loads(line)["config"]
            if c.get("steps") == a.steps:
                done.add((c["m"], c["seed"], c["lr_neural"]))
    f = open(path, "a"); t0 = time.time()
    for i, (m, s, lr) in enumerate(jobs):
        if (m, s, lr) in done:
            continue
        r = run_cell_maskonly(m=m, seed=s, steps=a.steps, lr_neural=lr, lr_tabular=lr,
                              opt_name="adam", log_every=100, mode=a.mode)
        f.write(json.dumps(r) + "\n"); f.flush()
        b, n = r["base"], r["neural"][-1]
        print(f"[{i+1}/{len(jobs)}] m={m} u={r['config']['u']:.3f} seed={s} lr={lr:g} "
              f"| base p_h={b['p_h_mean']:.4f} -> neural inU={n['p_h_in_U']:.4f} "
              f"outU={n['p_h_out_U']:.4f} rho={n['mean_hack_frac']:.4f} acc={n['acc']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    f.close()


if __name__ == "__main__":
    main()
