"""E11: the threshold, measured where it can be seen (added for the camera-ready).

Proposition 2 gives the sign of the cross-prompt drift of a leak as that of u_eff - Vbar,
where u_eff is the kernel- and policy-weighted acceptance fraction of Eq. (2).  It equals the
raw universality u only when the policy's mass on the leak and the same-string kernel are
the same at every prompt.  Reviewers pointed out that, read with raw u, the proposition
predicts suppression wherever u < Vbar >= 0.44, which is not what the sweeps show.  This
script makes three measurements that the accepted version did not.

(1) u_eff at the base.  For the headline bases (E1) and the frozen blind bases (E9) we
    compute, for each accepting prompt, the fraction of the policy's kernel-weighted mass on
    the leak that sits on prompts which accept it.  In E1 the supervised stage put most of
    the leak's mass on the prompts that accept it, so u_eff is well above u; in E9 the mass
    on the shared string is the same everywhere, so u_eff is close to u.

(2) Predicted versus exact instantaneous drift along plain SGD, before the verifier pass rate
    saturates.  At several steps we compute by autograd the exact cross-prompt part of the
    drift of the within-prompt log-odds (Lemma 1), the coherent-kernel prediction built
    from the measured same-string kernel entries and per-prompt masses (the heterogeneous
    form of Eq. (3)), the homogeneous leading-order formula kappa * p_h * (u - Vbar), its 1/N
    correction, and the residual.

(3) The representation difference between an E1 base at u = 1 and an E9 base: the same-string
    kernel coherence of h, the mass on h, and the exact initial cross-prompt drift of h.

Every number here comes from base models regenerated with the released code; see the
reproducibility appendix for how they relate to the archived checkpoints.
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import torch

from env import build_prompts, MaskOnlyVerifier
from runner import get_base, get_base_maskonly
from rlvr import measure, exact_pg_loss

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

MS = [1, 2, 4, 8, 16, 32, 64]
PROBE_STEPS = [0, 5, 10, 15, 20, 30, 40, 60, 80, 120]
CAP = 12          # accepting prompts probed per condition
SGD_LR = 0.3      # the operating point E8 selects


def _flat(gs):
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def _params(policy):
    return [p for p in policy.parameters() if p.requires_grad]


def string_gradients(policy, y, N):
    """grad_theta log pi(y | x') for every prompt x', as an (N, P) matrix.

    Each row needs only its own prompt's ten enumeration sequences, so the forward and
    backward passes are restricted to one prompt at a time; this is about sixty times
    cheaper than differentiating through the full enumeration batch N times."""
    params = _params(policy)
    rows = []
    for xp in range(N):
        lp = policy.full_logprobs(prompt_idx=[xp])[0]
        rows.append(_flat(torch.autograd.grad(lp[y], params)))
    return torch.stack(rows)


def probe(policy, ver, accept, is_hack, cap=CAP, seed=0):
    """One probe point: exact and predicted cross-prompt log-odds drift, u_eff, kernel scales.

    `accept[x, y]` is the verifier's acceptance mask; `is_hack[x, y]` marks the leak(s) of
    each prompt.  Only prompts that accept at least one leak are probed."""
    N = ver.n
    r = torch.as_tensor(accept, dtype=torch.float32)
    params = _params(policy)
    lp = policy.full_logprobs()
    with torch.no_grad():
        p = lp.exp()
        V = (p * r).sum(-1)                               # (N,) per-prompt pass rate
        W = p * (r - V[:, None])                          # advantage weights
    gJ = _flat(torch.autograd.grad((W * lp).sum(), params)) / N
    Vbar = float(V.mean())
    Vn = V.numpy()

    accepting = [x for x in range(N) if is_hack[x].sum() > 0]
    rng = np.random.default_rng(seed)
    idx = [int(i) for i in rng.permutation(accepting)[:cap]]
    recs = []
    for x in idx:
        h = int(np.flatnonzero(is_hack[x])[0])
        g = int(ver.gold[x])
        lpx = policy.full_logprobs(prompt_idx=[x])[0]
        g_own = _flat(torch.autograd.grad((W[x] * lpx).sum(), params)) / N
        Gh = string_gradients(policy, h, N)               # (N, P)
        Gg = string_gradients(policy, g, N)
        gh, gg = Gh[x], Gg[x]
        # exact drift (Lemma 1) split into own-prompt and cross-prompt parts
        tot_h, own_h = float(gh @ gJ), float(gh @ g_own)
        tot_g, own_g = float(gg @ gJ), float(gg @ g_own)
        exact_cross_lo = (tot_h - own_h) - (tot_g - own_g)
        # same-string kernel rows and per-prompt masses
        Kh = (Gh @ gh).numpy(); Kg = (Gg @ gg).numpy()
        ph = p[:, h].numpy(); pg = p[:, g].numpy()
        rh = accept[:, h]; rg = accept[:, g]
        others = np.array([xp for xp in range(N) if xp != x])
        # coherent prediction, heterogeneous: the y' = h (resp. y' = g) terms of Lemma 1
        pred_h = float(np.sum(Kh[others] * ph[others] * (rh[others] - Vn[others])) / N)
        pred_g = float(np.sum(Kg[others] * pg[others] * (rg[others] - Vn[others])) / N)
        coh_pred = pred_h - pred_g
        # homogeneous leading-order formula and its 1/N correction (Eq. 3)
        kappa = float(Kh[others].mean()); kappa_s = float(Kh[x])
        kappa_g = float(Kg[others].mean()); kappa_sg = float(Kg[x])
        inU = others[rh[others] > 0]
        outU = others[rh[others] == 0]
        p_h = float(ph[rh > 0].mean())                   # the leak's probability where accepted
        p_g = float(pg[x])
        u = float(rh.mean())
        lead = kappa * p_h * (u - Vbar)
        corr = (1 - Vbar) / N * ((kappa_s - kappa) * p_h - kappa_sg * p_g)
        # policy-weighted acceptance fraction (Eq. 2), three readings
        w = Kh * ph
        w_cross = w.copy(); w_cross[x] = 0.0
        ueff_kernel = float(w[rh > 0].sum() / w.sum()) if w.sum() != 0 else float("nan")
        ueff_cross = (float(w_cross[rh > 0].sum() / w_cross.sum())
                      if w_cross.sum() != 0 else float("nan"))
        ueff_mass = float(ph[rh > 0].sum() / ph.sum())
        # coherence of the same string across prompts, and the sign of the kernel entries
        nh = np.linalg.norm(Gh.numpy(), axis=1); cos_h = Kh / (nh * nh[x] + 1e-30)
        ng = np.linalg.norm(Gg.numpy(), axis=1); cos_g = Kg / (ng * ng[x] + 1e-30)
        recs.append(dict(
            x=x, h=h, u=u, p_h=p_h,
            p_h_out=float(ph[rh == 0].mean()) if (rh == 0).any() else 0.0,
            p_g=p_g, Vbar=Vbar, V_x=float(Vn[x]),
            exact_cross_lo=exact_cross_lo, exact_cross_h=tot_h - own_h,
            exact_cross_g=tot_g - own_g, own_lo=own_h - own_g,
            coh_pred=coh_pred, coh_pred_h=pred_h, coh_pred_g=pred_g,
            lead=lead, corr=corr, resid=exact_cross_lo - coh_pred,
            kappa=kappa, kappa_s=kappa_s, kappa_g=kappa_g, kappa_sg=kappa_sg,
            kappa_inU=float(Kh[inU].mean()) if len(inU) else float("nan"),
            kappa_outU=float(Kh[outU].mean()) if len(outU) else float("nan"),
            ueff_kernel=ueff_kernel, ueff_cross=ueff_cross, ueff_mass=ueff_mass,
            coh_h=float(cos_h[others].mean()), coh_g=float(cos_g[others].mean()),
            coh_h_inU=float(cos_h[inU].mean()) if len(inU) else float("nan"),
            frac_neg_K=float((Kh[others] < 0).mean()),
            neg_weight_share=float(np.abs(w_cross[w_cross < 0]).sum()
                                   / (np.abs(w_cross).sum() + 1e-30)),
        ))
    return dict(Vbar=Vbar, probes=recs)


def sgd_trajectory(policy, ver, lr, probe_steps, seed, cap):
    """Plain SGD on the exact policy gradient, probing at the requested steps."""
    r = torch.as_tensor(ver.accept, dtype=torch.float32)
    opt = torch.optim.SGD(policy.parameters(), lr=lr)
    out = []
    for t in range(max(probe_steps) + 1):
        if t in probe_steps:
            pr = probe(policy, ver, ver.accept, ver.is_hack, cap=cap, seed=seed)
            with torch.no_grad():
                mm = measure(policy.full_logprobs(), ver)
            pr.update(step=t, rho=mm["mean_hack_frac"], acc=mm["acc"], lky=mm["leaky_reward"])
            out.append(pr)
        lp = policy.full_logprobs()
        loss = exact_pg_loss(lp, r, norm="mean")
        opt.zero_grad(); loss.backward(); opt.step()
    return out


def _summary(b, key):
    return float(np.mean([q[key] for q in b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ms", type=str, default=",".join(map(str, MS)))
    ap.add_argument("--probe-steps", type=str, default=",".join(map(str, PROBE_STEPS)))
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ms = [int(x) for x in a.ms.split(",")]
    steps = [int(x) for x in a.probe_steps.split(",")]
    res = {"seed": a.seed, "E1": [], "E9": [], "sgd_lr": SGD_LR, "probe_steps": steps,
           "cap": a.cap}
    t0 = time.time()

    # (1) + (2): headline bases, probed at the base and along SGD
    for m in ms:
        prompts, ver, lam, pol, lp0 = get_base(m=m, k=1, seed=a.seed)
        traj = sgd_trajectory(pol, ver, SGD_LR, steps, a.seed, a.cap)
        res["E1"].append(dict(m=m, u=ver.u, traj=traj))
        b = traj[0]["probes"]
        print(f"E1 m={m:>2} seed={a.seed} base: Vbar={traj[0]['Vbar']:.3f} "
              f"ueff(kernel)={_summary(b,'ueff_kernel'):.3f} ueff(mass)={_summary(b,'ueff_mass'):.3f} "
              f"exact cross lo={_summary(b,'exact_cross_lo'):+.3f} coh pred={_summary(b,'coh_pred'):+.3f} "
              f"lead={_summary(b,'lead'):+.3f} | coh_h={_summary(b,'coh_h'):+.3f} "
              f"negK={_summary(b,'frac_neg_K'):.2f} ({time.time()-t0:.0f}s)", flush=True)
        for pr in traj[1:]:
            b = pr["probes"]
            print(f"      step={pr['step']:>3} Vbar={pr['Vbar']:.3f} rho={pr['rho']:.3f} "
                  f"exact={_summary(b,'exact_cross_lo'):+.4f} coh={_summary(b,'coh_pred'):+.4f} "
                  f"lead={_summary(b,'lead'):+.4f} ueff={_summary(b,'ueff_cross'):.3f}", flush=True)

    # (1) + (3): the frozen blind base, every acceptance mask, probed at the base only
    prompts, lam, pol9, lp9 = get_base_maskonly(seed=a.seed)
    for m in ms:
        ver9 = MaskOnlyVerifier(prompts, m, seed=a.seed, mode="constant_mass")
        pr = probe(pol9, ver9, ver9.accept, ver9.is_hack, cap=a.cap, seed=a.seed)
        with torch.no_grad():
            mm = measure(lp9, ver9)
        pr.update(m=m, u=ver9.u, rho=mm["mean_hack_frac"], acc=mm["acc"], lky=mm["leaky_reward"])
        res["E9"].append(pr)
        b = pr["probes"]
        print(f"E9 m={m:>2} seed={a.seed} base: Vbar={pr['Vbar']:.3f} "
              f"ueff(kernel)={_summary(b,'ueff_kernel'):.3f} ueff(mass)={_summary(b,'ueff_mass'):.3f} "
              f"exact cross lo={_summary(b,'exact_cross_lo'):+.3f} coh pred={_summary(b,'coh_pred'):+.3f} "
              f"coh_h={_summary(b,'coh_h'):+.3f} p_h={_summary(b,'p_h'):.4f}", flush=True)

    with open(os.path.join(OUT, f"exp12{a.tag}_seed{a.seed}.json"), "w") as f:
        json.dump(res, f)
    print(f"wrote exp12{a.tag}_seed{a.seed}.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
