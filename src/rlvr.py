"""RLVR objectives, exact measurement, and training loops."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from env import N_RESP

EPS = 1e-8


# --------------------------------------------------------------------------------------
# Exact measurement (no Monte-Carlo error: the whole response space is enumerated)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def measure(logprobs: torch.Tensor, ver, full: bool = False) -> dict:
    """Exact metrics.  Because the response space is enumerated, every quantity below is
    the true value under the policy, not a sample estimate.

    The primary statistic is `mean_log_odds`: the prompt-averaged log-odds of a leak
    against the correct answer,  (1/N) sum_x [ log pi(H(x)|x) - log pi(gold(x)|x) ].
    Proposition 1 makes its *change* from the base model exactly zero for KL-regularised
    / multiplicative updates, so the drift measures departure from the filter baseline.
    `hack_frac` is the pooled P(hack | rewarded) a practitioner would see; it is a
    reward-mass-weighted average of the per-prompt fractions and therefore also moves
    when the accepted mass is redistributed across prompts.
    """
    p = logprobs.exp()                                         # (N, |Y|)
    is_hack = torch.as_tensor(ver.is_hack, dtype=p.dtype)
    is_gold = torch.as_tensor(ver.is_gold, dtype=p.dtype)
    hack = (p * is_hack).sum(-1)                               # (N,)
    gold = (p * is_gold).sum(-1)                               # (N,)
    H, G = hack.sum().item(), gold.sum().item()
    lo = (torch.log(hack + EPS) - torch.log(gold + EPS))       # (N,)
    per_frac = hack / (hack + gold + EPS)
    out = dict(
        hack_frac=H / max(H + G, EPS),          # pooled P(hack | rewarded)
        mean_hack_frac=float(per_frac.mean()),  # prompt-averaged P(hack | rewarded)
        mean_log_odds=float(lo.mean()),         # primary: prompt-averaged log-odds
        hack_mass=hack.mean().item(),           # E_x pi(hack | x)
        acc=gold.mean().item(),                 # E_x pi(gold | x)  == true accuracy
        leaky_reward=(hack + gold).mean().item(),
        log_odds=float(np.log(max(H, EPS)) - np.log(max(G, EPS))),
        entropy=float(-(p * logprobs).sum(-1).mean()),
    )
    if full:
        out["per_prompt_log_odds"] = lo.tolist()
        out["per_prompt_gold"] = gold.tolist()
        out["per_prompt_hack"] = hack.tolist()
    return out


# --------------------------------------------------------------------------------------
# Advantages
# --------------------------------------------------------------------------------------
def exact_advantage(p: torch.Tensor, r: torch.Tensor, norm: str):
    """Exact-expectation advantage.  p, r: (N, 100)."""
    mean = (p * r).sum(-1, keepdim=True)
    a = r - mean
    if norm == "std":
        var = (p * (r - mean) ** 2).sum(-1, keepdim=True)
        a = a / (var.sqrt() + 1e-4)
    return a


def exact_pg_loss(logprobs, r, norm="mean", ref_logprobs=None, beta=0.0):
    """Loss whose gradient equals -grad E_x E_{y~pi}[r] (+ beta * grad KL(pi || ref))."""
    p = logprobs.exp()
    a = exact_advantage(p, r, norm)
    loss = -((p * a).detach() * logprobs).sum(-1).mean()
    if beta > 0 and ref_logprobs is not None:
        kl = (p * (logprobs - ref_logprobs)).sum(-1).mean()
        loss = loss + beta * kl
    return loss


def grpo_loss(logprobs, r, G=8, norm="std", ref_logprobs=None, beta=0.0, gen=None):
    """Sampled group-relative loss (one inner epoch, so the PPO ratio is 1)."""
    with torch.no_grad():
        p = logprobs.exp()
        idx = torch.multinomial(p, G, replacement=True, generator=gen)   # (N, G)
        rew = torch.gather(r, 1, idx)                                    # (N, G)
        mean = rew.mean(dim=1, keepdim=True)
        adv = rew - mean
        if norm == "std":
            std = rew.std(dim=1, unbiased=False, keepdim=True)
            adv = adv / (std + 1e-4)
    lp = torch.gather(logprobs, 1, idx)
    loss = -(adv * lp).mean()
    if beta > 0 and ref_logprobs is not None:
        kl = (logprobs.exp() * (logprobs - ref_logprobs)).sum(-1).mean()
        loss = loss + beta * kl
    return loss


# --------------------------------------------------------------------------------------
# Exact KL-regularised mirror descent -- the "null model" reference dynamics
# --------------------------------------------------------------------------------------
@torch.no_grad()
def mirror_descent(logprobs, r, eta):
    """pi_{t+1}(y|x)  proportional to  pi_t(y|x) * exp(eta * r(x,y)).

    This is the exact multiplicative / natural-gradient update.  It provably leaves the
    odds between any two responses with equal reward unchanged.
    """
    new = logprobs + eta * r
    return new - torch.logsumexp(new, dim=-1, keepdim=True)


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------
def train(policy, ver, *, algo="pg", norm="mean", steps=300, lr=0.05, beta=0.0,
          G=8, seed=0, log_every=1, opt_name="sgd"):
    """Run RLVR and return the exact metric trajectory."""
    r = torch.as_tensor(ver.accept, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    with torch.no_grad():
        ref_lp = policy.full_logprobs().detach().clone()

    if opt_name == "sgd":
        opt = torch.optim.SGD(policy.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(policy.parameters(), lr=lr)

    hist = []
    for t in range(steps + 1):
        lp = policy.full_logprobs()
        if t % log_every == 0 or t == steps:
            m = measure(lp.detach(), ver, full=(t == 0 or t == steps))
            m["step"] = t
            hist.append(m)
        if t == steps:
            break
        if algo == "pg":
            loss = exact_pg_loss(lp, r, norm=norm, ref_logprobs=ref_lp, beta=beta)
        elif algo == "grpo":
            loss = grpo_loss(lp, r, G=G, norm=norm, ref_logprobs=ref_lp, beta=beta, gen=gen)
        else:
            raise ValueError(algo)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return hist


def train_mirror(logprobs0, ver, *, steps=300, eta=0.05, log_every=1):
    """Exact mirror descent on the full distribution (tabular only)."""
    r = torch.as_tensor(ver.accept, dtype=torch.float32)
    lp = logprobs0.clone()
    hist = []
    for t in range(steps + 1):
        if t % log_every == 0 or t == steps:
            m = measure(lp, ver, full=(t == 0 or t == steps)); m["step"] = t
            hist.append(m)
        if t == steps:
            break
        lp = mirror_descent(lp, r, eta)
    return hist
