"""
Two policy parameterisations over the same response space, differing ONLY in whether
parameters are shared across prompts.

Both expose  full_logprobs() -> (N, 100) differentiable log pi(y | x), obtained from the
identical autoregressive factorisation  pi(y|x) = pi(y1|x) * pi(y2|x, y1).

TabularPolicy : free per-prompt parameters.  The factorisation is *complete* (10 free
                first-token logits + 10x10 free second-token logits per prompt parameterise
                the entire 100-dimensional simplex), so this is the fully non-parametric,
                zero-sharing limit.  N * 110 = 7040 parameters.

TransformerPolicy : a small decoder-only transformer.  Every prompt is scored by the same
                weights, so gradients from different prompts are aggregated in parameter
                space.  This is the only difference from TabularPolicy.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import N_RESP, N_PROMPTS

VOCAB = 11          # digits 0..9 plus BOS=10
BOS = 10


# --------------------------------------------------------------------------------------
# Tabular
# --------------------------------------------------------------------------------------
class TabularPolicy(nn.Module):
    """Per-prompt free parameters; no sharing whatsoever."""

    def __init__(self, n_prompts: int = N_PROMPTS):
        super().__init__()
        self.n = n_prompts
        self.w1 = nn.Parameter(torch.zeros(n_prompts, 10))
        self.w2 = nn.Parameter(torch.zeros(n_prompts, 10, 10))

    def full_logprobs(self) -> torch.Tensor:
        lp1 = F.log_softmax(self.w1, dim=-1)              # (N, 10)
        lp2 = F.log_softmax(self.w2, dim=-1)              # (N, 10, 10)
        lp = lp1.unsqueeze(-1) + lp2                      # (N, y1, y2)
        return lp.reshape(self.n, N_RESP)

    @torch.no_grad()
    def set_from_logprobs(self, target_lp: torch.Tensor):
        """Set parameters so that full_logprobs() reproduces target_lp exactly."""
        lp = target_lp.reshape(self.n, 10, 10)
        lp1 = torch.logsumexp(lp, dim=-1)                 # (N, 10)
        self.w1.copy_(lp1)
        self.w2.copy_(lp - lp1.unsqueeze(-1))


# --------------------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerPolicy(nn.Module):
    """Decoder-only transformer over [BOS, a, b, y1, y2]; all prompts share all weights."""

    def __init__(self, prompts, d: int = 64, layers: int = 2, heads: int = 4, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.n = len(prompts)
        self.d = d
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(4, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)
        with torch.no_grad():
            self.tok.weight.normal_(0, 0.02, generator=g)
            self.pos.weight.normal_(0, 0.02, generator=g)

        # Fixed enumeration batch: for every prompt and every first digit y1,
        # the sequence [BOS, a, b, y1].  Position 2 gives logits for y1, position 3 for y2.
        seqs = []
        for (a, b) in prompts:
            for y1 in range(10):
                seqs.append([BOS, a, b, y1])
        self.register_buffer("enum_seq", torch.tensor(seqs, dtype=torch.long))  # (N*10, 4)
        mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
        self.register_buffer("causal", mask)

    def _forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        m = self.causal[:T, :T]
        for blk in self.blocks:
            x = blk(x, m)
        return self.head(self.lnf(x))                     # (B, T, 10)

    def full_logprobs(self, prompt_idx=None) -> torch.Tensor:
        """Exact log pi(y|x) for every response.

        `prompt_idx` restricts the forward pass to a subset of prompts.  The per-prompt
        weight-copy control (E10) trains one copy of this architecture per prompt, and each
        copy only ever needs its own prompt, so enumerating all N would waste a factor of N.
        """
        if prompt_idx is None:
            seq, n = self.enum_seq, self.n
        else:
            pi = torch.as_tensor(prompt_idx, dtype=torch.long).reshape(-1)
            rows = (pi[:, None] * 10 + torch.arange(10)).reshape(-1)
            seq, n = self.enum_seq[rows], len(pi)
        out = self._forward(seq)                          # (n*10, 4, 10)
        out = out.reshape(n, 10, 4, 10)
        # first-token logits are identical across the y1 dimension (prefix [BOS,a,b]);
        # take index 0 to avoid redundancy.
        lg1 = out[:, 0, 2, :]                             # (N, 10)
        lg2 = out[:, :, 3, :]                             # (N, y1, 10)
        lp1 = F.log_softmax(lg1, dim=-1)
        lp2 = F.log_softmax(lg2, dim=-1)
        lp = lp1.unsqueeze(-1) + lp2
        return lp.reshape(-1, N_RESP)


def make_policy(kind: str, prompts, seed: int = 0, **kw):
    if kind == "tabular":
        return TabularPolicy(len(prompts))
    if kind == "transformer":
        return TransformerPolicy(prompts, seed=seed, **kw)
    raise ValueError(kind)
