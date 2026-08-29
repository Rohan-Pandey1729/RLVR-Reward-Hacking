"""Construction of the base (pre-RL) policy, and the matched tabular initialisation.

The base model is produced by supervised fine-tuning on an exact target distribution

    q(y|x) = (1-eps) * [ lam_x * 1[y = gold(x)] + (1-lam_x) * Uniform(Y) ]
             + eps   * Uniform(H(x))

* lam_x in [0,1] is a per-prompt *competence* level.  It gives the base model a realistic
  spread of solve rates, from prompts it has essentially mastered to prompts on which the
  correct answer is a rare event -- the regime in which reward hacking actually matters.
* eps is a small contamination rate: a fraction of demonstrations are verifier-exploiting
  responses.  ~1% contamination is enough to seed hacking in practice
  (Khalifa et al., 2026).

Competence is assigned to prompts through a dedicated permutation so that it is
*uncorrelated* with the leak-block structure; otherwise universality would be confounded
with difficulty.
"""
from __future__ import annotations

import numpy as np
import torch

from env import N_RESP
from policies import TransformerPolicy, TabularPolicy
from rlvr import measure


def competence_profile(n: int, lo: float = 0.0, hi: float = 0.9, seed: int = 12345):
    """A fixed, leak-structure-independent spread of per-prompt competence."""
    grid = np.linspace(lo, hi, n)
    rng = np.random.default_rng(seed)
    return grid[rng.permutation(n)]


def target_distribution(ver, eps: float, lam: np.ndarray,
                        clean_floor: bool = False, contam=None) -> torch.Tensor:
    """Exact SFT target.

    With `clean_floor=False` the uniform component is spread over the whole response space,
    including the k leak strings, so the base model's hack mass grows with k.  With
    `clean_floor=True` the uniform component is spread only over responses that are neither
    the correct answer nor a leak, so total base hack mass is exactly eps for every k --
    which is what the universality-vs-error-rate factorial needs.
    """
    n = ver.n
    # `contam` overrides which strings the SFT stage demonstrates.  E9 passes the union of
    # both leak families so that the supervised target, and hence the base checkpoint, is
    # identical in every universality condition; the verifier's acceptance mask is then the
    # only thing that varies.  Everywhere else it defaults to the verifier's own leak set.
    contam = ver.hack if contam is None else contam
    kc = contam.shape[1]
    q = np.zeros((n, N_RESP))
    if clean_floor:
        free = np.ones((n, N_RESP), dtype=bool)
        free[np.arange(n), ver.gold] = False
        for j in range(kc):
            free[np.arange(n), contam[:, j]] = False
        q += free * ((1.0 - eps) * (1.0 - lam)[:, None] / free.sum(-1, keepdims=True))
    else:
        q += (1.0 - eps) * (1.0 - lam)[:, None] / N_RESP
    q[np.arange(n), ver.gold] += (1.0 - eps) * lam
    for j in range(kc):
        q[np.arange(n), contam[:, j]] += eps / kc
    q /= q.sum(-1, keepdims=True)
    return torch.as_tensor(q, dtype=torch.float32)


def sft(policy, q, steps: int = 600, lr: float = 3e-3, seed: int = 0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        lp = policy.full_logprobs()
        loss = -(q * lp).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return policy


def build_base(prompts, ver, *, eps: float, lam: np.ndarray, sft_steps: int = 600,
               lr: float = 3e-3, seed: int = 0, d: int = 64, layers: int = 2, heads: int = 4,
               clean_floor: bool = False, contam=None):
    q = target_distribution(ver, eps, lam, clean_floor=clean_floor, contam=contam)
    pol = TransformerPolicy(prompts, d=d, layers=layers, heads=heads, seed=seed)
    sft(pol, q, steps=sft_steps, lr=lr, seed=seed)
    with torch.no_grad():
        lp = pol.full_logprobs().detach().clone()
    return pol, lp, q


def calibrate_eps(prompts, ver, *, lam, target_hack_frac: float, sft_steps: int = 600,
                  lr: float = 3e-3, seed: int = 0, lo: float = 3e-4, hi: float = 0.3,
                  iters: int = 10, verbose: bool = False, **kw):
    """Geometric bisection on the SFT contamination rate to equalise the base hack fraction."""
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        _, lp, _ = build_base(prompts, ver, eps=mid, lam=lam, sft_steps=sft_steps,
                              lr=lr, seed=seed, **kw)
        hf = measure(lp, ver)["hack_frac"]
        if verbose:
            print(f"    eps={mid:.5f} -> base hack_frac={hf:.5f}")
        if hf < target_hack_frac:
            lo = mid
        else:
            hi = mid
    eps = float(np.sqrt(lo * hi))
    pol, lp, q = build_base(prompts, ver, eps=eps, lam=lam, sft_steps=sft_steps,
                            lr=lr, seed=seed, **kw)
    return eps, pol, lp, q


def matched_tabular(lp0: torch.Tensor, n_prompts: int) -> TabularPolicy:
    """Tabular policy initialised to the *identical* distribution as the base transformer."""
    tab = TabularPolicy(n_prompts)
    tab.set_from_logprobs(lp0)
    with torch.no_grad():
        err = (tab.full_logprobs() - lp0).abs().max().item()
    assert err < 1e-4, f"tabular init mismatch {err}"
    return tab


def build_base_maskonly_calibrated(prompts, ver, *, eps, lam, sft_steps=600, lr=3e-3,
                                   seed=0, d=64, layers=2, heads=4, rounds=8, tol=2e-4,
                                   damping=0.5, verbose=False):
    """Base for E9 with the two leak families equalised in *realised* probability.

    Giving the shared string h and the per-prompt strings s_x equal mass in the SFT *target*
    is not enough.  The base is itself a shared-parameter transformer, so it fits one string
    demonstrated at every prompt better than 64 distinct ones: at equal target mass the
    realised p_h is about 50% larger than the realised p_{s_x}.  Since E9 varies which of the
    two the verifier accepts, that difference would reappear as a base-mass gradient along u,
    and rich-get-richer alone would then produce a universality slope even for per-prompt
    parameters.

    We therefore solve a small fixed point on the SFT target: reweight the two families per
    prompt until the realised probabilities match, keeping their sum at eps.  The returned
    base has p_h(x) = p_{s_x}(x) to `tol` for every prompt, so whichever string the verifier
    accepts, it starts from the same mass.
    """
    n = ver.n
    sh, idio = ver.shared, ver.idio
    w = np.full((n, 2), 0.5)                      # split of eps between (h, s_x), per prompt
    pol = lp = None
    for it in range(rounds):
        contam_w = np.stack([w[:, 0], w[:, 1]], axis=1)
        q = target_distribution(ver, eps, lam, clean_floor=True, contam=ver.base_contam)
        q = q.numpy().copy()
        # rewrite the contamination rows with the current per-prompt split
        q[np.arange(n), sh] = 0.0
        q[np.arange(n), idio] = 0.0
        q /= q.sum(-1, keepdims=True)
        q *= (1.0 - eps)
        q[np.arange(n), sh] += eps * contam_w[:, 0]
        q[np.arange(n), idio] += eps * contam_w[:, 1]
        q /= q.sum(-1, keepdims=True)
        pol = TransformerPolicy(prompts, d=d, layers=layers, heads=heads, seed=seed)
        sft(pol, torch.as_tensor(q, dtype=torch.float32), steps=sft_steps, lr=lr, seed=seed)
        with torch.no_grad():
            lp = pol.full_logprobs().detach().clone()
        p = lp.exp().numpy()
        ph, ps = p[np.arange(n), sh], p[np.arange(n), idio]
        gap = float(abs(ph.mean() - ps.mean()))
        gap_p95 = float(np.percentile(np.abs(ph - ps), 95))
        if verbose:
            print(f"    round {it}: |mean gap| = {gap:.2e}  p95 per-prompt = {gap_p95:.2e}  "
                  f"mean p_h={ph.mean():.5f} p_s={ps.mean():.5f}")
        if gap < tol:
            break
        # damped multiplicative update in log space; the undamped version overshoots and
        # oscillates because the shared transformer cannot fit an arbitrary per-prompt target
        ratio = np.clip(ps / np.clip(ph, 1e-12, None), 0.2, 5.0) ** damping
        w[:, 0] *= ratio
        w /= w.sum(-1, keepdims=True)
    return pol, lp, gap, gap_p95
