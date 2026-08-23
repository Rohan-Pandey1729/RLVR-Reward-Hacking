"""Numerical verification of the propositions.

T1  Null model, optimum form.  The KL-regularised optimum pi* ~ pi_ref * exp(r/beta)
    leaves the odds between any two equally-rewarded responses unchanged, for every beta.

T2  Null model, dynamics.  Exact mirror descent pi_{t+1} ~ pi_t * exp(eta r) leaves the
    pooled hack fraction P(hack | rewarded) invariant at every step.

T3  Aggregation.  The instantaneous drift of log pi(h|x) under exact-gradient flow is
    d/dt log pi(y|x) = (1/N) sum_{x'} Cov_{y'~pi(.|x')} [ r(x',y'), K((x,y),(x',y')) ],
    with K the empirical tangent kernel.  For a policy with per-prompt parameters K
    vanishes off the diagonal, so the drift cannot depend on how many prompts share a
    leak.  For a shared-parameter policy the m accepted prompts contribute coherently and
    the drift grows linearly in m.  We verify: (a) the identity itself, to numerical
    precision; (b) the linear-in-m scaling of the initial drift for the transformer and
    its absence for the tabular policy.
"""
from __future__ import annotations

import json, os

import numpy as np
import torch

from env import make_env
from base import matched_tabular
from runner import get_base, gradient_coherence
from rlvr import measure, train_mirror, exact_pg_loss

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
BURN = 60      # RL steps before measuring the drift decomposition in T3c
CAP = 24       # prompts sampled for the drift decomposition (of N)


# --------------------------------------------------------------------------- T1
def t1_kl_optimum(lp0, ver, betas=(2.0, 1.0, 0.5, 0.2, 0.1, 0.05)):
    r = torch.as_tensor(ver.accept, dtype=torch.float64)
    lp = lp0.double()
    rows = []
    base = measure(lp, ver)
    for b in betas:
        star = lp + r / b
        star = star - torch.logsumexp(star, -1, keepdim=True)
        m = measure(star, ver)
        rows.append(dict(beta=b, hack_frac=m["mean_hack_frac"], acc=m["acc"],
                         leaky_reward=m["leaky_reward"],
                         log_odds=m["mean_log_odds"],
                         abs_dev=abs(m["mean_hack_frac"] - base["mean_hack_frac"]),
                         lo_dev=abs(m["mean_log_odds"] - base["mean_log_odds"])))
    return base, rows


# --------------------------------------------------------------------------- T2
def t2_mirror(lp0, ver, steps=400, eta=0.05):
    h = train_mirror(lp0.double(), ver, steps=steps, eta=eta, log_every=max(1, steps // 8))
    dev = max(abs(x["mean_hack_frac"] - h[0]["mean_hack_frac"]) for x in h)
    lo_dev = max(abs(x["mean_log_odds"] - h[0]["mean_log_odds"]) for x in h)
    return h, dev, lo_dev


# --------------------------------------------------------------------------- T3
def _flat(gs):
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def drift_identity_check(policy, ver, n_probe=6, seed=0):
    """Compare  d/dt log pi(y|x) = <grad log pi(y|x), grad J>  against a finite-difference
    step of exact-gradient ascent on J.  Verifies the drift decomposition numerically."""
    r = torch.as_tensor(ver.accept, dtype=torch.float32)
    params = [p for p in policy.parameters() if p.requires_grad]

    lp = policy.full_logprobs()
    loss = exact_pg_loss(lp, r, norm="mean")
    gJ = _flat(torch.autograd.grad(-loss, params, retain_graph=True, create_graph=False))

    rng = np.random.default_rng(seed)
    probes = [(int(i), int(ver.hack[i, 0])) for i in rng.permutation(ver.n)[:n_probe]]
    analytic, numeric = [], []
    eps = 1e-4
    for (i, y) in probes:
        lp = policy.full_logprobs()
        g = _flat(torch.autograd.grad(lp[i, y], params, retain_graph=True))
        analytic.append(float(g @ gJ))
        # finite difference along +eps * gJ
        with torch.no_grad():
            off = 0
            for p in params:
                n = p.numel()
                p.add_(eps * gJ[off:off + n].view_as(p)); off += n
            lp_p = policy.full_logprobs()[i, y].item()
            off = 0
            for p in params:
                n = p.numel()
                p.sub_(2 * eps * gJ[off:off + n].view_as(p)); off += n
            lp_m = policy.full_logprobs()[i, y].item()
            off = 0
            for p in params:
                n = p.numel()
                p.add_(eps * gJ[off:off + n].view_as(p)); off += n
        numeric.append((lp_p - lp_m) / (2 * eps))
    a, b = np.array(analytic), np.array(numeric)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return dict(analytic=a.tolist(), numeric=b.tolist(),
                max_rel_err=float((np.abs(a - b) / scale).max()))


def cross_prompt_kernel(policy, ver, n_pairs=8, seed=0):
    """Empirical tangent kernel K((x,h),(x',h)) for x != x' -- the quantity
    Corollary 2 says vanishes for a policy with per-prompt parameters."""
    rng = np.random.default_rng(seed)
    params = [p for p in policy.parameters() if p.requires_grad]
    idx = rng.permutation(ver.n)[: 2 * n_pairs]
    vals, self_vals = [], []
    for a_, b_ in zip(idx[:n_pairs], idx[n_pairs:]):
        lp = policy.full_logprobs()
        ga = _flat(torch.autograd.grad(lp[int(a_), int(ver.hack[a_, 0])], params,
                                       retain_graph=True))
        gb = _flat(torch.autograd.grad(lp[int(b_), int(ver.hack[b_, 0])], params,
                                       retain_graph=True))
        vals.append(float(ga @ gb)); self_vals.append(float(ga @ ga))
    return dict(cross=float(np.mean(np.abs(vals))), self=float(np.mean(self_vals)),
                ratio=float(np.mean(np.abs(vals)) / max(np.mean(self_vals), 1e-30)))


def drift_decomposition(policy, ver, cap=None):
    """Split the exact drift of log pi(y|x) into its own-prompt and cross-prompt parts.

    Lemma 1 writes d/dt log pi(y|x) as a sum over ALL prompts.  The x'=x term is the
    ordinary within-prompt reinforcement, present for any policy; the x' != x terms are the
    contribution of parameter sharing and are the only part that can depend on how many
    prompts share a leak.  Corollary 2 says the cross-prompt part vanishes identically for a
    policy with per-prompt parameters, and Proposition 3 predicts that for a shared-parameter
    policy it is proportional to (u - Vbar).  We compute both exactly by autograd.
    """
    r = torch.as_tensor(ver.accept, dtype=torch.float32)
    params = [p for p in policy.parameters() if p.requires_grad]
    lp = policy.full_logprobs()
    with torch.no_grad():
        pr = lp.exp()
        V = (pr * r).sum(-1, keepdim=True)
        W = pr * (r - V)                                     # (N, |Y|) advantage weights
    n = ver.n
    gJ = _flat(torch.autograd.grad((W * lp).sum(), params, retain_graph=True)) / n

    idx = (range(n) if cap is None
           else np.random.default_rng(0).permutation(n)[:min(cap, n)].tolist())
    out = {k: [] for k in ("hack_total", "hack_own", "hack_cross",
                           "gold_total", "gold_own", "gold_cross")}
    for i in idx:
        lp = policy.full_logprobs()
        g_own = _flat(torch.autograd.grad((W[i] * lp[i]).sum(), params, retain_graph=True)) / n
        gh = _flat(torch.autograd.grad(lp[i, int(ver.hack[i, 0])], params, retain_graph=True))
        gg = _flat(torch.autograd.grad(lp[i, int(ver.gold[i])], params, retain_graph=True))
        ht, ho = float(gh @ gJ), float(gh @ g_own)
        gt, go = float(gg @ gJ), float(gg @ g_own)
        out["hack_total"].append(ht); out["hack_own"].append(ho); out["hack_cross"].append(ht - ho)
        out["gold_total"].append(gt); out["gold_own"].append(go); out["gold_cross"].append(gt - go)
    return {k: float(np.mean(v)) for k, v in out.items()}


def initial_drift(policy, ver):
    """Mean initial growth rate of log pi(leak|x) and log pi(gold|x) under gradient flow."""
    r = torch.as_tensor(ver.accept, dtype=torch.float32)
    params = [p for p in policy.parameters() if p.requires_grad]
    lp = policy.full_logprobs()
    gJ = _flat(torch.autograd.grad(-exact_pg_loss(lp, r, norm="mean"), params,
                                   retain_graph=True))
    dh, dg = [], []
    for i in range(ver.n):
        lp = policy.full_logprobs()
        gh = _flat(torch.autograd.grad(lp[i, int(ver.hack[i, 0])], params, retain_graph=True))
        gg = _flat(torch.autograd.grad(lp[i, int(ver.gold[i])], params, retain_graph=True))
        dh.append(float(gh @ gJ)); dg.append(float(gg @ gJ))
    return dict(drift_hack=float(np.mean(dh)), drift_gold=float(np.mean(dg)),
                drift_logodds=float(np.mean(dh) - np.mean(dg)))


def main():
    res = {}
    # T1 / T2 use a single representative base model
    prompts, ver, lam, pol, lp0 = get_base(m=64, k=1, seed=0)
    base, rows = t1_kl_optimum(lp0, ver)
    res["T1"] = dict(base_hack_frac=base["mean_hack_frac"], rows=rows,
                     max_abs_dev=max(r["abs_dev"] for r in rows))
    h, dev, lo_dev = t2_mirror(lp0, ver)
    res["T2"] = dict(traj=h, max_abs_dev=dev, max_logodds_dev=lo_dev)

    # T3a: identity check on both policy classes
    res["T3a"] = {}
    res["T3a"]["transformer"] = drift_identity_check(pol, ver)
    res["T3a"]["tabular"] = drift_identity_check(matched_tabular(lp0, len(prompts)), ver)
    res["T3a"]["kernel_transformer"] = cross_prompt_kernel(pol, ver)
    res["T3a"]["kernel_tabular"] = cross_prompt_kernel(matched_tabular(lp0, len(prompts)), ver)

    # T3b: initial drift as a function of m
    res["T3b"] = []
    for m in [1, 2, 4, 8, 16, 32, 64]:
        for seed in range(3):
            prompts, ver_m, lam, pol_m, lp0_m = get_base(m=m, k=1, seed=seed)
            tab = matched_tabular(lp0_m, len(prompts))
            dcn = drift_decomposition(pol_m, ver_m, cap=CAP)
            dct = drift_decomposition(tab, ver_m, cap=CAP)
            coh = gradient_coherence(pol_m, ver_m, max_prompts=12, seed=seed)
            res["T3b"].append(dict(m=m, seed=seed, coherence=coh,
                                   decomp_neural=dcn, decomp_tabular=dct,
                                   base_leaky=measure(lp0_m, ver_m)["leaky_reward"]))
            print(f"m={m:>3} seed={seed}  cross(hack)={dcn['hack_cross']:+.4f} "
                  f"own(hack)={dcn['hack_own']:+.4f} | tab cross={dct['hack_cross']:+.2e} "
                  f"| coh={coh['hack_coh']:+.3f}/{coh['gold_coh']:+.3f}", flush=True)

    # T3c: the same decomposition after a short burn-in.  At initialisation every logit is
    # being pushed up together and the transient dominates; Proposition 3 describes the
    # differential dynamics once that has passed.
    from rlvr import train as _train
    res["T3c"] = []
    for m in [1, 2, 4, 8, 16, 32, 64]:
        for seed in range(3):
            prompts, ver_m, lam, pol_m, lp0_m = get_base(m=m, k=1, seed=seed)
            _train(pol_m, ver_m, algo="pg", norm="mean", steps=BURN, lr=1e-3,
                   opt_name="adam", seed=seed, log_every=10 ** 9)
            dc = drift_decomposition(pol_m, ver_m, cap=CAP)
            mm = measure(pol_m.full_logprobs().detach(), ver_m)
            res["T3c"].append(dict(m=m, seed=seed, decomp=dc, leaky=mm["leaky_reward"],
                                   rho=mm["mean_hack_frac"], burn=BURN))
            print(f"burn m={m:>3} seed={seed}  cross(hack)={dc['hack_cross']:+.4f} "
                  f"cross(gold)={dc['gold_cross']:+.4f} "
                  f"cross(log-odds)={dc['hack_cross']-dc['gold_cross']:+.4f} "
                  f"Vbar={mm['leaky_reward']:.3f}", flush=True)

    with open(os.path.join(OUT, "theory.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nT1 max |rho - base| over beta grid:", res["T1"]["max_abs_dev"],
          " max |Lambda - base|:", max(r["lo_dev"] for r in res["T1"]["rows"]))
    print("T2 max |rho - base| over mirror descent:", res["T2"]["max_abs_dev"],
          " max |Lambda - base|:", res["T2"]["max_logodds_dev"])
    print("T3a cross/self kernel  transformer:", res["T3a"]["kernel_transformer"]["ratio"],
          " tabular:", res["T3a"]["kernel_tabular"]["ratio"])
    print("T3a max relative error (transformer):", res["T3a"]["transformer"]["max_rel_err"])
    print("T3a max relative error (tabular):    ", res["T3a"]["tabular"]["max_rel_err"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
