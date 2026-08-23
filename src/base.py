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
                        clean_floor: bool = False) -> torch.Tensor:
    """Exact SFT target.

    With `clean_floor=False` the uniform component is spread over the whole response space,
    including the k leak strings, so the base model's hack mass grows with k.  With
    `clean_floor=True` the uniform component is spread only over responses that are neither
    the correct answer nor a leak, so total base hack mass is exactly eps for every k --
    which is what the universality-vs-error-rate factorial needs.
    """
    n = ver.n
    q = np.zeros((n, N_RESP))
    if clean_floor:
        free = np.ones((n, N_RESP), dtype=bool)
        free[np.arange(n), ver.gold] = False
        for j in range(ver.k):
            free[np.arange(n), ver.hack[:, j]] = False
        q += free * ((1.0 - eps) * (1.0 - lam)[:, None] / free.sum(-1, keepdims=True))
    else:
        q += (1.0 - eps) * (1.0 - lam)[:, None] / N_RESP
    q[np.arange(n), ver.gold] += (1.0 - eps) * lam
    for j in range(ver.k):
        q[np.arange(n), ver.hack[:, j]] += eps / ver.k
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
               clean_floor: bool = False):
    q = target_distribution(ver, eps, lam, clean_floor=clean_floor)
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
