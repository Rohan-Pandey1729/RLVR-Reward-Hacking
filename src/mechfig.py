"""The overview schematic: the environment's universality knob, and the mechanism (Corollary 1).

Left: the same 64 prompts at three universality levels, coloured by whether the verifier accepts
the shared wrong answer h on that prompt.  Every prompt accepts exactly one wrong string in every
condition, so the static false-positive rate is identical across the three grids.

Middle and right: with shared parameters every prompt's gradient lands on the same weights, so a
leak accepted on some prompts and rejected on others is pushed in both directions at once and
the net depends on how many accept it and on how much mass the model puts on it there.  With
per-prompt parameters the cross-prompt arrows do not exist at all.

The axes are placed by hand (no layout engine): the panels mix image axes, free-form drawings
and figure-level text, which constrained layout does not resolve reliably.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from analysis import FIG_W, save, u_label

# Accepting and rejecting a leak are not policies, so they stay off the colours that mean
# "shared" and "per-prompt" in every other figure, and each box also carries a sign.
C_ACCEPT = "C2"
C_REJECT = "C4"
ACCEPTS = [True, True, False, True, False, False, True, False]
XS = np.linspace(0.06, 0.94, len(ACCEPTS))
Y_TITLE, Y_PROMPT, Y_THETA, Y_EQ, Y_NOTE = 0.985, 0.81, 0.51, 0.30, 0.0
FIG_H = FIG_W / 2.99
TEXT, HEAD = 10, 12                               # the sizes of axis text and of panel titles


def box(ax, x, y, w, h, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.02", **kw))


def arrow(ax, start, end, color):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, lw=1.7,
                                 color=color, shrinkA=0, shrinkB=0))


def environment(fig):
    """Three 8x8 prompt grids: which prompts accept the shared wrong answer h."""
    cmap = ListedColormap([C_REJECT, C_ACCEPT])
    w = 0.105                                  # of the figure width
    h = w * FIG_W / FIG_H                       # the same length in inches, so cells are square
    lefts = (0.045, 0.165, 0.285)
    for left, m in zip(lefts, (1, 16, 64)):
        ax = fig.add_axes([left, 0.44, w, h])
        grid = np.zeros(64)
        grid[:m] = 1                            # contiguous blocks, first-operand-major
        ax.imshow(grid.reshape(8, 8), cmap=cmap, vmin=0, vmax=1, alpha=0.9)
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="w", lw=0.9)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(rf"$u = {u_label(m / 64)}$" + "\n" + rf"$m = {m}$", fontsize=TEXT, pad=4)
        if left == lefts[0]:
            ax.set_ylabel(r"first operand $a$", fontsize=TEXT)
        if left == lefts[1]:
            ax.set_xlabel(r"second operand $b$", fontsize=TEXT)
    cx = lefts[1] + w / 2
    fig.text(cx, 0.975, r"Prompts that accept the wrong answer $h$", ha="center", va="top",
             fontsize=HEAD)
    fig.text(cx, 0.115, "Each of the $N=64$ prompts accepts its\n"
                        "correct answer and exactly one wrong string,\n"
                        "so the static false-positive rate is fixed",
             ha="center", va="bottom", fontsize=TEXT)


def mechanism(ax, shared):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, accepts in zip(XS, ACCEPTS):
        box(ax, x, Y_PROMPT, 0.082, 0.15, fc=C_ACCEPT if accepts else C_REJECT, ec="none")
        ax.text(x, Y_PROMPT, "+" if accepts else "\N{MINUS SIGN}", ha="center", va="center",
                color="w")

    if shared:
        ax.text(0.5, Y_TITLE, "Shared parameters", ha="center", va="top", fontsize=HEAD)
        box(ax, 0.5, Y_THETA, 0.5, 0.17, fc="w", ec="k", lw=1.2)
        ax.text(0.5, Y_THETA, r"One shared $\theta$", ha="center", va="center")
        for x, accepts in zip(XS, ACCEPTS):
            arrow(ax, (x, Y_PROMPT - 0.09), (0.5 + 0.42 * (x - 0.5), Y_THETA + 0.1),
                  C_ACCEPT if accepts else C_REJECT)
        ax.text(0.5, Y_EQ, r"$\frac{d}{dt}\log\pi_\theta(h\mid x)"
                           r"=\frac{1}{N}\sum_{x'}\mathrm{Cov}[r,K_\theta]$",
                ha="center", va="center", fontsize=TEXT)
        note = ("Every prompt moves $h$ at $x$:\n"
                "accepting prompts push it up,\n"
                "rejecting prompts push it down")
    else:
        ax.text(0.5, Y_TITLE, "Per-prompt parameters", ha="center", va="top", fontsize=HEAD)
        for x, accepts in zip(XS, ACCEPTS):
            box(ax, x, Y_THETA, 0.082, 0.17, fc="w", ec="k", lw=1.2)
            ax.text(x, Y_THETA, r"$\theta_x$", ha="center", va="center")
            arrow(ax, (x, Y_PROMPT - 0.09), (x, Y_THETA + 0.1),
                  C_ACCEPT if accepts else C_REJECT)
        ax.text(0.5, Y_EQ, r"$K_\theta((x,h),(x',y'))=0$ for $x'\neq x$",
                ha="center", va="center", fontsize=TEXT)
        note = ("No cross-prompt term:\n"
                "the dynamics cannot see\n"
                "how many prompts accept $h$")
    ax.text(0.5, Y_NOTE, note, ha="center", va="bottom", fontsize=TEXT)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    environment(fig)
    mechanism(fig.add_axes([0.425, 0.115, 0.275, 0.875]), shared=True)
    mechanism(fig.add_axes([0.72, 0.115, 0.275, 0.875]), shared=False)
    fig.legend(handles=[Patch(color=C_ACCEPT, label=r"Prompt accepts $h$ (pushes it up)"),
                        Patch(color=C_REJECT, label=r"Prompt rejects $h$ (pushes it down)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               borderaxespad=0.2)
    save(fig, "fig0_overview")


if __name__ == "__main__":
    main()
