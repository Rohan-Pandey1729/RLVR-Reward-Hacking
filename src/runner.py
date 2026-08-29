"""Experiment runner: cached base models, matched policy pairs, metric trajectories."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time

import numpy as np
import torch

from env import make_env, build_prompts, MaskOnlyVerifier
from base import build_base, matched_tabular, competence_profile
from policies import TransformerPolicy
from rlvr import measure, train, train_mirror

CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE, exist_ok=True)


def _key(**kw) -> str:
    s = json.dumps(kw, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def get_base(*, m, k, seed, n_operands=8, eps=0.03, sft_steps=600, d=64, layers=2,
             heads=4, lr=3e-3, lam_seed=12345, block_perm=False, clean_floor=False):
    """Build (or load) the base transformer and the identically-initialised tabular policy."""
    key = _key(m=m, k=k, seed=seed, n_operands=n_operands, eps=eps, sft_steps=sft_steps,
               d=d, layers=layers, heads=heads, lr=lr, lam_seed=lam_seed, v=3,
               **({'bp': 1} if block_perm else {}),
               **({'cf': 1} if clean_floor else {}))
    path = os.path.join(CACHE, f"base_{key}.pt")
    prompts, ver = make_env(m, k, seed=seed, n_operands=n_operands,
                            block_perm=block_perm)
    lam = competence_profile(len(prompts), seed=lam_seed)
    if os.path.exists(path):
        blob = torch.load(path, weights_only=False)
        pol = TransformerPolicy(prompts, d=d, layers=layers, heads=heads, seed=seed)
        pol.load_state_dict(blob["sd"])
        lp0 = blob["lp0"]
    else:
        pol, lp0, _ = build_base(prompts, ver, eps=eps, lam=lam, sft_steps=sft_steps,
                                 lr=lr, seed=seed, d=d, layers=layers, heads=heads,
                                 clean_floor=clean_floor)
        tmp = path + f".tmp{os.getpid()}"
        torch.save({"sd": pol.state_dict(), "lp0": lp0}, tmp)
        os.replace(tmp, path)
    return prompts, ver, lam, pol, lp0


def run_cell(*, m, k, seed, n_operands=8, algo="pg", norm="mean", steps=600,
             lr_neural=1e-3, lr_tabular=1e-2, opt_name="adam", beta=0.0, G=8,
             eps=0.03, sft_steps=600, log_every=25, d=64, layers=2, heads=4,
             block_perm=False, clean_floor=False):
    """Run the matched neural / tabular pair for one experimental cell."""
    prompts, ver, lam, pol, lp0 = get_base(m=m, k=k, seed=seed, n_operands=n_operands,
                                           eps=eps, sft_steps=sft_steps, d=d,
                                           layers=layers, heads=heads,
                                           block_perm=block_perm,
                                           clean_floor=clean_floor)
    base = measure(lp0, ver, full=True)
    out = {"config": dict(m=m, k=k, seed=seed, N=len(prompts), u=ver.u,
                          static_fpr=ver.static_false_positive_rate(),
                          algo=algo, norm=norm, steps=steps, lr_neural=lr_neural,
                          lr_tabular=lr_tabular, opt=opt_name, beta=beta, G=G,
                          eps=eps, n_operands=n_operands, block_perm=block_perm,
                          clean_floor=clean_floor,
                          sft_steps=sft_steps, d=d, layers=layers, heads=heads,
                          code_version=4),
           "base": base}
    out["neural"] = train(copy.deepcopy(pol), ver, algo=algo, norm=norm, steps=steps,
                          lr=lr_neural, beta=beta, G=G, seed=seed,
                          log_every=log_every, opt_name=opt_name)
    out["tabular"] = train(matched_tabular(lp0, len(prompts)), ver, algo=algo, norm=norm,
                           steps=steps, lr=lr_tabular, beta=beta, G=G, seed=seed,
                           log_every=log_every, opt_name=opt_name)
    return out


# --------------------------------------------------------------------------------------
# Mechanism probe: coherence of the cross-prompt gradient signal
# --------------------------------------------------------------------------------------
def _flat_grad(policy, scalar):
    policy.zero_grad(set_to_none=True)
    gs = torch.autograd.grad(scalar, [p for p in policy.parameters() if p.requires_grad],
                             retain_graph=True, allow_unused=True)
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def gradient_coherence(policy, ver, max_prompts: int = 16, seed: int = 0):
    """Mean pairwise cosine similarity of per-prompt gradients of log pi(y|x).

    Computed separately for
      * the leak response  h(x)          -> 'hack_coh'
      * the correct response gold(x)     -> 'gold_coh'
    A universal leak is the *same* response for every prompt, so its per-prompt gradients
    point in a common direction and add coherently; prompt-specific correct answers do not.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(ver.n)[:max_prompts]
    lp = policy.full_logprobs()
    gh, gg = [], []
    for i in idx:
        gh.append(_flat_grad(policy, lp[i, int(ver.hack[i, 0])]))
        gg.append(_flat_grad(policy, lp[i, int(ver.gold[i])]))
    def mean_cos(vs):
        M = torch.stack(vs)
        M = M / (M.norm(dim=1, keepdim=True) + 1e-12)
        C = M @ M.T
        n = len(vs)
        off = (C.sum() - C.diag().sum()) / (n * (n - 1))
        return float(off)
    def scales(vs):
        M = torch.stack(vs)
        n = len(vs)
        K = (M @ M.T)                       # raw kernel entries, not cosines
        norms = M.norm(dim=1)
        off = (K.sum() - K.diag().sum()) / (n * (n - 1))
        return dict(kappa_raw=float(off), kappa_s_raw=float(K.diag().mean()),
                    gnorm_mean=float(norms.mean()), gnorm_min=float(norms.min()),
                    gnorm_max=float(norms.max()),
                    gnorm_spread=float(norms.max() / (norms.min() + 1e-12)))
    out = dict(hack_coh=mean_cos(gh), gold_coh=mean_cos(gg))
    for name, vs in (("hack", gh), ("gold", gg)):
        for kk, vv in scales(vs).items():
            out[f"{name}_{kk}"] = vv
    return out


# --------------------------------------------------------------------------------------
# E9: same-checkpoint, mask-only causal control
def get_base_maskonly(*, seed, n_operands=8, eps=0.03, sft_steps=600, d=64, layers=2,
                      heads=4, lr=3e-3, lam_seed=12345):
    """One frozen base per seed, shared by every universality condition.

    The cache key deliberately excludes m: the whole point of E9 is that a single
    checkpoint serves the entire sweep, so the initial policy, the probability on each leak
    string and the initial tangent kernel are bit-identical across conditions.
    """
    key = _key(seed=seed, n_operands=n_operands, eps=eps, sft_steps=sft_steps, d=d,
               layers=layers, heads=heads, lr=lr, lam_seed=lam_seed, kind="maskonly", v=1)
    path = os.path.join(CACHE, f"base_{key}.pt")
    prompts = build_prompts(n_operands)
    ver_any = MaskOnlyVerifier(prompts, 1, seed=seed)   # only used for gold + contamination
    lam = competence_profile(len(prompts), seed=lam_seed)
    if os.path.exists(path):
        blob = torch.load(path, weights_only=False)
        pol = TransformerPolicy(prompts, d=d, layers=layers, heads=heads, seed=seed)
        pol.load_state_dict(blob["sd"])
        lp0 = blob["lp0"]
    else:
        pol, lp0, _ = build_base(prompts, ver_any, eps=eps, lam=lam, sft_steps=sft_steps,
                                 lr=lr, seed=seed, d=d, layers=layers, heads=heads,
                                 clean_floor=True, contam=ver_any.base_contam)
        tmp = path + f".tmp{os.getpid()}"
        torch.save({"sd": pol.state_dict(), "lp0": lp0}, tmp)
        os.replace(tmp, path)
    return prompts, lam, pol, lp0


def _shared_string_stats(lp, ver):
    """Probability mass on the shared string h, split by whether the prompt accepts it."""
    import numpy as _np
    p = torch.exp(lp).detach().numpy()
    ph = p[:, ver.shared]
    pg = p[_np.arange(ver.n), ver.gold]
    inU, outU = ver.U, ~ver.U
    out = {"p_h_mean": float(ph.mean()),
           "p_h_in_U": float(ph[inU].mean()) if inU.any() else float("nan"),
           "p_h_out_U": float(ph[outU].mean()) if outU.any() else float("nan"),
           "p_gold_mean": float(pg.mean())}
    with _np.errstate(divide="ignore"):
        lo = _np.log(_np.clip(ph, 1e-30, None)) - _np.log(_np.clip(pg, 1e-30, None))
    out["logodds_h_in_U"] = float(lo[inU].mean()) if inU.any() else float("nan")
    out["logodds_h_out_U"] = float(lo[outU].mean()) if outU.any() else float("nan")
    out["logodds_h_mean"] = float(lo.mean())
    return out


def run_cell_maskonly(*, m, seed, n_operands=8, algo="pg", norm="mean", steps=400,
                      lr_neural=1e-3, lr_tabular=1e-3, opt_name="adam", beta=0.0, G=8,
                      eps=0.03, sft_steps=600, log_every=25, d=64, layers=2, heads=4,
                      mode="constant_fpr"):
    """One E9 cell.  Base, leak identities and static FPR are identical across every m."""
    prompts, lam, pol, lp0 = get_base_maskonly(seed=seed, n_operands=n_operands, eps=eps,
                                               sft_steps=sft_steps, d=d, layers=layers,
                                               heads=heads)
    ver = MaskOnlyVerifier(prompts, m, seed=seed, mode=mode)
    base = measure(lp0, ver, full=True)
    base.update(_shared_string_stats(lp0, ver))
    out = {"config": dict(m=m, k=1, seed=seed, N=len(prompts), u=ver.u,
                          static_fpr=ver.static_false_positive_rate(),
                          algo=algo, norm=norm, steps=steps, lr_neural=lr_neural,
                          lr_tabular=lr_tabular, opt=opt_name, beta=beta, G=G, eps=eps,
                          n_operands=n_operands, sft_steps=sft_steps, d=d, layers=layers,
                          heads=heads, exp="maskonly", mode=mode, code_version=5),
           "base": base}
    for arm, policy, lr in (("neural", copy.deepcopy(pol), lr_neural),
                            ("tabular", matched_tabular(lp0, len(prompts)), lr_tabular)):
        traj = train(policy, ver, algo=algo, norm=norm, steps=steps, lr=lr, beta=beta,
                     G=G, seed=seed, log_every=log_every, opt_name=opt_name)
        with torch.no_grad():
            traj[-1].update(_shared_string_stats(policy.full_logprobs(), ver))
        out[arm] = traj
    return out
