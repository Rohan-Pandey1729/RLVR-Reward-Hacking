"""E5: the phenomenon under the algorithm people actually run, and two interventions.

E5a  Sampled GRPO (group size G, group-standard-deviation normalisation) across the
     universality sweep -- the effect is not an artifact of exact-expectation gradients.
E5b  KL-to-reference sweep.  Proposition 1 says the KL-regularised optimum inherits the
     base model's hack odds exactly, so beta should pin the hack fraction near its base
     value; we measure the price in verifier reward and in true accuracy.
E5c  Verifier hardening with a fixed audit budget.  A verifier leaks through one universal
     string AND one idiosyncratic string per prompt.  Patching either halves the measured
     static false-positive rate by exactly the same amount.  We compare the two patches.
"""
from __future__ import annotations

import argparse, copy, json, os, time

import numpy as np
import torch

from env import make_env, build_prompts, gold_indices, non_gold_pool, N_RESP
from base import build_base, matched_tabular, competence_profile
from policies import TransformerPolicy
from rlvr import measure, train
from runner import get_base, run_cell

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(10))   # camera-ready: ten seeds in every part


# ---------------------------------------------------------------- mixed-leak verifier
class MixedVerifier:
    """Per prompt the verifier accepts: gold, one shared universal string, and one
    prompt-specific string.  `mode` selects which leaks remain after patching.

      'none'    : both leak families present            (FP mass 2 per prompt)
      'patchU'  : universal leak removed                (FP mass 1 per prompt)
      'patchI'  : all idiosyncratic leaks removed       (FP mass 1 per prompt)

    'patchU' and 'patchI' remove exactly the same number of accepted (prompt, wrong-string)
    pairs, so any static audit rates them as equally improved.
    """

    def __init__(self, prompts, seed=0, mode="none"):
        self.prompts = prompts
        self.n = len(prompts)
        self.gold = gold_indices(prompts)
        pool = non_gold_pool(prompts)
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(pool)
        self.universal = int(perm[0])
        self.idio = np.array(perm[1:1 + self.n], dtype=np.int64)
        assert len(np.unique(self.idio)) == self.n
        self.mode = mode

        cols = []
        if mode in ("none", "patchI"):
            cols.append(np.full(self.n, self.universal, dtype=np.int64))
        if mode in ("none", "patchU"):
            cols.append(self.idio)
        self.hack = np.stack(cols, axis=1)
        self.k = self.hack.shape[1]
        self.m = self.n if mode == "patchI" else (1 if mode == "patchU" else None)
        self.u = (self.m / self.n) if self.m else None

        self.accept = np.zeros((self.n, N_RESP))
        self.accept[np.arange(self.n), self.gold] = 1.0
        self.is_hack = np.zeros((self.n, N_RESP))
        for j in range(self.k):
            self.accept[np.arange(self.n), self.hack[:, j]] = 1.0
            self.is_hack[np.arange(self.n), self.hack[:, j]] = 1.0
        self.is_gold = np.zeros((self.n, N_RESP))
        self.is_gold[np.arange(self.n), self.gold] = 1.0

    def static_false_positive_rate(self):
        return float(self.is_hack.sum() / (self.n * (N_RESP - 1)))


def e5c(steps=400, out=None, seeds=(0, 1, 2), lrs=(1e-3, 3e-3)):
    """Base model contaminated with BOTH leak families; then deploy each patched verifier.

    Both patches remove the same number of accepted (prompt, wrong-response) pairs, so any
    static audit rates them as equally improved.  Within a seed every patch starts from the
    *same* checkpoint, so the base policy, and hence base accuracy, is identical across
    patches and any difference in final accuracy arises during RL.

    Every learning rate is stored for both arms.  The operating point is chosen at analysis
    time with the same rule as the main sweeps (the smallest learning rate whose final
    verifier reward is within 1e-3 of the best), rather than a strict argmax here, so the
    hardening experiment uses one rule with the rest of the paper."""
    prompts = build_prompts(8)
    lam = competence_profile(len(prompts))
    rows = []
    for seed in seeds:
        v_none = MixedVerifier(prompts, seed=seed, mode="none")
        pol, lp0, _ = build_base(prompts, v_none, eps=0.03, lam=lam, sft_steps=600, seed=seed)
        for mode in ["none", "patchU", "patchI"]:
            v = MixedVerifier(prompts, seed=seed, mode=mode)
            b = measure(lp0, v, full=True)
            runs = {}
            for lr in lrs:
                hn = train(copy.deepcopy(pol), v, algo="pg", norm="mean", steps=steps,
                           lr=lr, opt_name="adam", seed=seed, log_every=50)
                ht = train(matched_tabular(lp0, len(prompts)), v, algo="pg", norm="mean",
                           steps=steps, lr=lr * 30, opt_name="adam", seed=seed, log_every=50)
                runs[f"{lr:g}"] = dict(lr_neural=lr, lr_tabular=lr * 30, neural=hn, tabular=ht)
            rows.append(dict(seed=seed, mode=mode, static_fpr=v.static_false_positive_rate(),
                             base=b, runs=runs, steps=steps, code_version=6))
            summ = " ".join(f"lr={k}: N acc={r['neural'][-1]['acc']:.3f} "
                            f"lky={r['neural'][-1]['leaky_reward']:.3f} "
                            f"T acc={r['tabular'][-1]['acc']:.3f}" for k, r in runs.items())
            print(f"  seed={seed} mode={mode:<7} fpr={v.static_false_positive_rate():.4f} "
                  f"base_acc={b['acc']:.3f} base_rho={b['mean_hack_frac']:.4f} | {summ}",
                  flush=True)
            if out:
                json.dump(rows, open(out, "w"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=str, required=True, choices=["a", "b", "c"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--tag", type=str, default="", help="suffix for the part-c output file")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.part == "a":
        _sd = [int(x) for x in a.seeds.split(",")]
        bases = [(m, s) for m in [1, 2, 4, 8, 16, 32, 64] for s in _sd][a.shard::a.nshards]
        jobs = [(m, s, lr) for (m, s) in bases for lr in [3e-4, 1e-3, 3e-3]]
        path = os.path.join(OUT, f"exp5a_shard{a.shard}.jsonl")
        # Read every shard: shard assignment depends on the seed list.
        done = set()
        import glob as _glob
        for q in _glob.glob(os.path.join(OUT, "exp5a_shard*.jsonl")):
            for line in open(q):
                c = json.loads(line)["config"]
                # Include the step budget: a cell run at a different budget is not the same cell.
                if c.get("steps") == a.steps:
                    if c.get("steps") == a.steps:  # a cell run at another budget is not this cell
                        done.add((c["m"], c["seed"], c["lr_neural"]))
        f = open(path, "a"); t0 = time.time()
        for i, (m, s, lr) in enumerate(jobs):
            if (m, s, lr) in done:
                continue
            r = run_cell(m=m, k=1, seed=s, algo="grpo", norm="std", G=8, steps=a.steps,
                         lr_neural=lr, lr_tabular=lr, opt_name="adam", log_every=25)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{i+1}/{len(jobs)}] GRPO m={m} seed={s} base_hf={r['base']['hack_frac']:.4f} "
                  f"neural_hf={r['neural'][-1]['hack_frac']:.4f} "
                  f"neural_acc={r['neural'][-1]['acc']:.3f} "
                  f"tab_hf={r['tabular'][-1]['hack_frac']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        f.close()

    elif a.part == "b":
        _sd = [int(x) for x in a.seeds.split(",")]
        jobs = [(m, b, s) for m in [1, 64] for b in [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]
                for s in _sd][a.shard::a.nshards]
        path = os.path.join(OUT, f"exp5b_shard{a.shard}.jsonl")
        done = set()
        import glob as _glob
        for q in _glob.glob(os.path.join(OUT, "exp5b_shard*.jsonl")):
            for line in open(q):
                c = json.loads(line)["config"]
                if c.get("steps") == a.steps:
                    done.add((c["m"], c["beta"], c["seed"]))
        f = open(path, "a"); t0 = time.time()
        for i, (m, beta, s) in enumerate(jobs):
            if (m, beta, s) in done:
                continue
            r = run_cell(m=m, k=1, seed=s, algo="pg", norm="mean", steps=a.steps,
                         lr_neural=1e-3, lr_tabular=3e-2, opt_name="adam", beta=beta,
                         log_every=25)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{i+1}/{len(jobs)}] m={m} beta={beta} seed={s} "
                  f"base_hf={r['base']['hack_frac']:.4f} "
                  f"neural_hf={r['neural'][-1]['hack_frac']:.4f} "
                  f"neural_acc={r['neural'][-1]['acc']:.3f} "
                  f"lky={r['neural'][-1]['leaky_reward']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        f.close()

    else:
        _sd = [int(x) for x in a.seeds.split(",")]
        e5c(steps=a.steps, seeds=_sd, out=os.path.join(OUT, f"exp5c{a.tag}.json"))


if __name__ == "__main__":
    main()
