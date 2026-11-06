"""Shared paths, figure conventions, plotting helpers and result loading."""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# Figures keep matplotlib's default style.  The canvas is twice the paper's 5.5in \linewidth and
# LaTeX scales it down, so default 10pt text prints at 5pt, titles at 6pt, and DPI comes to 600
# dots per inch on the page.  The printed height of a figure is half its canvas height, which is
# what the page limit needs.
FIG_W = 11.0
DPI = 300

# Every data figure is a grid of identical panels, three to a row, placed by panel_grid().  The
# numbers are inches of canvas.  A one-row figure comes to 3:1 overall and a two-row figure to
# about 3:2, with text kept small relative to the data.
MARGIN_L, MARGIN_R, GAP_X = 0.80, 0.06, 0.80      # room for tick labels and a y label
PANEL_W = (FIG_W - MARGIN_L - MARGIN_R - 2 * GAP_X) / 3
PANEL_H = 2.54
TITLE_H, LABEL_H, BAND_H = 0.30, 0.50, 0.33       # above a panel, below it, and a legend band
DOT = 9                                           # area of a seed dot, in points squared

# Colour roles, from matplotlib's default cycle.  Warm hues are shared-parameter runs and cool
# hues per-prompt ones.  These four stay distinct under simulated protanopia and deuteranopia;
# the default green and purple do not (green collapses onto orange, purple onto blue), and
# every series also carries its own marker.
C_SHARED = "C1"      # shared-parameter (transformer) policy
C_PERM = "C3"        # shared parameters, randomised block membership
C_TAB = "C0"         # per-prompt (tabular) policy
C_COPIES = "C9"      # per-prompt copies of the transformer
C_ALT = "k"          # a second condition of one arm, where no colour role applies
C_NULL = "C7"        # filter baseline / base model reference, never a data series
M_SHARED, M_PERM, M_TAB, M_COPIES = "o", "^", "s", "D"
BASELINE = dict(color=C_NULL, ls=":", lw=1.4, zorder=1)
_SEED_DOTS = []      # seed_dots() calls waiting for their axes' final limits


def save(fig, name):
    """Write figures/<name>.png.  matplotlib's PNGs carry an alpha channel that pdfLaTeX would
    embed as a separate soft mask; these figures are opaque, so it is dropped."""
    spread_seed_dots(fig)
    path = os.path.join(FIG, f"{name}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    Image.open(path).convert("RGB").save(path, dpi=(DPI, DPI), optimize=True)


def panel_grid(rows=1, top_band=True, bottom_band=False, pad_bottom=0.0):
    """The one layout of every data figure: `rows` rows of three identical panels, placed by hand
    so that a panel has the same size and shape in every figure, whatever its labels.  The bands
    are the room for top_legend() and bottom_legend().  Returns the figure and the axes, as a row
    for one row and as an array of rows otherwise."""
    pitch = TITLE_H + PANEL_H + LABEL_H
    height = rows * pitch + BAND_H * (top_band + bottom_band) + pad_bottom
    fig = plt.figure(figsize=(FIG_W, height))
    axes = np.empty((rows, 3), dtype=object)
    for i, j in np.ndindex(rows, 3):
        left = MARGIN_L + j * (PANEL_W + GAP_X)
        bottom = height - BAND_H * top_band - i * pitch - TITLE_H - PANEL_H
        axes[i, j] = fig.add_axes([left / FIG_W, bottom / height, PANEL_W / FIG_W, PANEL_H / height])
    return fig, axes[0] if rows == 1 else axes


def top_legend(fig, handles=None, labels=None, ncol=None, **kw):
    """One legend for the whole figure, in the band panel_grid() keeps above the panels."""
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1), frameon=False,
                      borderaxespad=0.1, ncol=ncol or len(handles), **kw)


def bottom_legend(fig, handles, labels, ncol=None, **kw):
    """A second legend, in the band panel_grid() keeps below the panels."""
    return fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), frameon=False,
                      borderaxespad=0.1, ncol=ncol or len(handles), **kw)


def u_label(u):
    return "1" if u == 1 else f"1/{round(1 / u)}"


def u_axis(ax, lo=-6, step=2, label=r"Leak universality $u$"):
    """Universality on a log2 axis.  The data are plotted at log2(u) on a linear axis, so
    series can be dodged by a constant offset, and the ticks read as the fractions in the text."""
    major = list(range(0, lo - 1, -step))[::-1]
    ax.set_xticks(major, [u_label(2.0 ** k) for k in major])
    ax.set_xticks(range(lo, 1), minor=True)
    ax.set_xlim(lo - 0.85, 0.85)
    ax.set_xlabel(label)


def swarm(values, dy, dx):
    """Horizontal offsets that stop dots of similar value from overlapping (a deterministic
    beeswarm).  `dy` and `dx` are one dot's height and width in data units.  Per-seed values
    often coincide exactly here, because a metric averaged over 64 prompts moves in steps."""
    v = np.asarray(values, float)
    placed, out = [], np.zeros(len(v))
    for i in np.argsort(v, kind="stable"):
        for k in range(2 * len(v) + 1):
            off = (k + 1) // 2 * (1 if k % 2 else -1) * dx
            if all(((off - o) / dx) ** 2 + ((v[i] - y) / dy) ** 2 >= 1.0 for o, y in placed):
                break
        placed.append((off, v[i]))
        out[i] = off
    return out


def seed_dots(ax, x, values, color, s=DOT, marker="o", alpha=0.55):
    """One small dot per seed at position x.  save() spreads them sideways once the axes have
    their final limits, so that none hide each other at any panel size."""
    v = np.asarray(values, float)
    dots = ax.scatter(np.full(len(v), float(x)), v, s=s, color=color, marker=marker, alpha=alpha,
                      linewidths=0, zorder=2)
    _SEED_DOTS.append((ax, dots, float(x), v, s))


def spread_seed_dots(fig):
    """Apply swarm() to every seed_dots() call on this figure, with one dot's size in data units
    read from the finished axes."""
    for ax, dots, x, v, s in [d for d in _SEED_DOTS if d[0].figure is fig]:
        box = ax.get_position()
        size = np.sqrt(s) / 72                       # a dot's diameter, in inches
        dx = size * np.ptp(ax.get_xlim()) / (box.width * fig.get_figwidth())
        dy = size * np.ptp(ax.get_ylim()) / (box.height * fig.get_figheight())
        dots.set_offsets(np.column_stack([x + swarm(v, dy, dx), v]))
    _SEED_DOTS[:] = [d for d in _SEED_DOTS if d[0].figure is not fig]


def mean_line(ax, x, mu, color, marker, label=None, yerr=None, ms=6, ls="-"):
    """Means joined by a thin line, drawn over the seed dots with a white edge to lift them."""
    ax.errorbar(x, mu, yerr=yerr, color=color, marker=marker, ms=ms, ls=ls, lw=1.6, mec="white",
                mew=0.6, elinewidth=1.2, label=label, zorder=4)


def u_colors(us):
    """Ordered levels of u take steps of matplotlib's default colormap, stopping short of its
    palest yellow.  A value of u has the same colour in every figure."""
    return {u: plt.cm.viridis(0.8 * (1 + np.log2(u) / 6)) for u in us}


def heat(ax, values, norm, cmap, fmt="{:.1f}", **text):
    """A matrix of coloured cells, each printed with its value in ink that contrasts with it."""
    im = ax.imshow(values, origin="lower", aspect="auto", norm=norm, cmap=cmap)
    ax.tick_params(length=0)
    for (i, j), v in np.ndenumerate(values):
        r, g, b, _ = plt.get_cmap(cmap)(norm(v))
        ink = "k" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.42 else "w"
        label = fmt.format(v)
        if not label.strip("-0."):      # a signed zero is a rounding artefact
            label = label.lstrip("-")
        ax.text(j, i, label.replace("-", "\N{MINUS SIGN}"), ha="center", va="center", color=ink,
                **text)
    return im


def load(pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
