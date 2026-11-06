"""The mechanism, drawn (Corollary 1).

With shared parameters every prompt's gradient lands on the same weights, so a leak accepted on
some prompts and rejected on others is pushed in both directions at once and the net depends on
how many accept it.  With per-prompt parameters the cross-prompt arrows do not exist at all.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from analysis import FIG_W, save

# Accepting and rejecting a leak are not policies, so they stay off the colours that mean
# "shared" and "per-prompt" in every other figure, and each box also carries a sign.
C_ACCEPT = "C2"
C_REJECT = "C4"
ACCEPTS = [True, True, False, True, False, False, True, False]
XS = np.linspace(0.06, 0.94, len(ACCEPTS))
Y_PROMPT, Y_THETA = 0.86, 0.47


def box(ax, x, y, w, h, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.02", **kw))


def arrow(ax, start, end, color):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=1.3,
                                 color=color, shrinkA=0, shrinkB=0))


def draw(ax, shared):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, accepts in zip(XS, ACCEPTS):
        box(ax, x, Y_PROMPT, 0.082, 0.15, fc=C_ACCEPT if accepts else C_REJECT, ec="none")
        ax.text(x, Y_PROMPT, "+" if accepts else "\N{MINUS SIGN}", ha="center", va="center",
                color="w")

    if shared:
        ax.set_title("Shared parameters")
        box(ax, 0.5, Y_THETA, 0.5, 0.17, fc="w", ec="k", lw=1.0)
        ax.text(0.5, Y_THETA, r"One shared $\theta$", ha="center", va="center")
        for x, accepts in zip(XS, ACCEPTS):
            arrow(ax, (x, Y_PROMPT - 0.09), (0.5 + 0.42 * (x - 0.5), Y_THETA + 0.1),
                  C_ACCEPT if accepts else C_REJECT)
        ax.text(0.5, 0.22, r"$\frac{d}{dt}\log\pi_\theta(h\mid x)"
                          r"=\frac{1}{N}\sum_{x'}\mathrm{Cov}[r,K_\theta]$",
                ha="center", va="center")
        note = r"The sum runs over every prompt $x'$"
    else:
        ax.set_title("Per-prompt parameters")
        for x, accepts in zip(XS, ACCEPTS):
            box(ax, x, Y_THETA, 0.082, 0.17, fc="w", ec="k", lw=1.0)
            ax.text(x, Y_THETA, r"$\theta_x$", ha="center", va="center")
            arrow(ax, (x, Y_PROMPT - 0.09), (x, Y_THETA + 0.1),
                  C_ACCEPT if accepts else C_REJECT)
        ax.text(0.5, 0.22, r"$K_\theta((x,y),(x',y'))=0$ for $x'\neq x$", ha="center", va="center")
        note = "The cross-prompt term is identically zero"
    ax.text(0.5, 0.03, note, ha="center", va="center")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.45), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    draw(axes[0], shared=True)
    draw(axes[1], shared=False)
    fig.legend(handles=[Patch(color=C_ACCEPT, label=r"Prompt accepts the leak $h$ (pushes it up)"),
                        Patch(color=C_REJECT, label=r"Prompt rejects $h$ (pushes it down)")],
               loc="outside lower center", ncol=2, frameon=False)
    save(fig, "fig0_mechanism")


if __name__ == "__main__":
    main()
