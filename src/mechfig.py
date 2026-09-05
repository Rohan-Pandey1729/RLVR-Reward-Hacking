"""Figure 1: the mechanism, drawn.

Corollary 1 is the paper's cleanest theoretical statement and it is currently only
available to a reader who parses two equations.  It is a picture: with shared parameters
every prompt's gradient lands on the same weights, so a leak accepted on some prompts and
rejected on others is pushed in both directions at once and the net depends on how many
accept it.  With per-prompt parameters the cross-prompt arrows do not exist at all.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from analysis import FIG, C_SHARED, C_TAB, C_NULL

INK   = "#1a1a19"
MUTED = "#6b6a66"
FAINT = "#c0bdb5"


def _prompt_row(ax, y, accepts, w=0.062, h=0.075):
    """One row of prompt boxes; accepting prompts carry the leak's colour."""
    xs = np.linspace(0.085, 0.915, len(accepts))
    for x, acc in zip(xs, accepts):
        col = C_SHARED if acc else C_TAB
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.016",
                                    fc=col, ec="white", lw=1.1, alpha=0.92, zorder=3))
    return xs


def _theta(ax, cx, cy, w, h, label, sub=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="white", ec=INK, lw=1.5, zorder=4))
    ax.text(cx, cy + (0.012 if sub else 0), label, ha="center", va="center",
            fontsize=10.2, color=INK, zorder=5)
    if sub:
        ax.text(cx, cy - 0.030, sub, ha="center", va="center",
                fontsize=7.9, color=MUTED, zorder=5)


def _arrow(ax, p0, p1, col, lw=1.5, rad=0.0, alpha=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=lw, color=col, alpha=alpha, ls=ls,
                                 shrinkA=1.5, shrinkB=2.5, zorder=2))


def main():
    accepts = [True, True, False, True, False, False, True, False]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.38))

    for ax, shared in zip(axes, (True, False)):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        xs = _prompt_row(ax, 0.775, accepts)
        ax.text(0.5, 0.888, "Prompts", ha="center", va="center",
                fontsize=9.1, color=MUTED)

        if shared:
            _theta(ax, 0.5, 0.500, 0.40, 0.115, r"One Shared $\theta$")
            for x, acc in zip(xs, accepts):
                col = C_SHARED if acc else C_TAB
                _arrow(ax, (x, 0.735), (0.5 + (x - 0.5) * 0.42, 0.560), col,
                       lw=1.6, rad=0.16 * np.sign(0.5 - x), alpha=0.85)
            ax.text(0.5, 0.290, r"$\dfrac{d}{dt}\log\pi_\theta(h\mid x)$"
                                "\n"
                                r"$=\frac{1}{N}\sum_{x'}\mathrm{Cov}[\,r,\,K_\theta\,]$",
                    ha="center", va="center", fontsize=8.9, color=INK)
            ax.text(0.5, 0.045, "Every prompt reaches every other: the sum runs\n"
                                "over all $x'$, so the net push on $h$ depends\n"
                                "on how many prompts accept it",
                    ha="center", va="center", fontsize=8.4, color=MUTED)
            ax.set_title("Shared Parameters", fontsize=10.3, color=INK, pad=5)
        else:
            for x in xs:
                _theta(ax, x, 0.500, 0.060, 0.085, "")
                ax.text(x, 0.500, r"$\theta_x$", ha="center", va="center",
                        fontsize=7.7, color=INK, zorder=5)
            for x, acc in zip(xs, accepts):
                col = C_SHARED if acc else C_TAB
                _arrow(ax, (x, 0.735), (x, 0.550), col, lw=1.6, alpha=0.85)
            # the channel that does not exist, drawn as struck through
            for i in range(len(xs) - 1):
                a, b = xs[i] + 0.033, xs[i + 1] - 0.033
                ax.plot([a, b], [0.500, 0.500], ls=(0, (2, 2.6)), lw=1.0,
                        color=FAINT, zorder=1)
                mx = 0.5 * (a + b)
                ax.plot([mx - 0.008, mx + 0.008], [0.492, 0.508], lw=1.0,
                        color=FAINT, zorder=1)
                ax.plot([mx - 0.008, mx + 0.008], [0.508, 0.492], lw=1.0,
                        color=FAINT, zorder=1)
            ax.text(0.5, 0.290, r"$K_\theta\left((x,y),(x',y')\right)=0$"
                                "\n"
                                r"for $x'\neq x$",
                    ha="center", va="center", fontsize=8.9, color=INK)
            ax.text(0.5, 0.030, "No prompt reaches any other:\n"
                                "the cross-prompt term is identically\n"
                                "zero, whatever the leak's breadth",
                    ha="center", va="center", fontsize=8.4, color=MUTED)
            ax.set_title("Per-Prompt Parameters", fontsize=10.3, color=INK, pad=5)

    # one legend, placed in empty space, never over a mark
    axes[0].scatter([], [], marker="s", s=26, color=C_SHARED, label=r"Accepts the Leak $h$")
    axes[0].scatter([], [], marker="s", s=26, color=C_TAB, label=r"Rejects $h$")
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.03, 1.035), fontsize=8.2,
                   frameon=False, handletextpad=0.4, labelspacing=0.22, ncol=2,
                   columnspacing=1.2)

    fig.tight_layout(pad=0.35, w_pad=1.6)
    out = os.path.join(FIG, "fig0_mechanism.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
