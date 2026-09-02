"""Shared plotting style, palette and result loading."""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# Colour-blind-safe categorical slots, checked with the palette validator:
# all inside the lightness band, above the chroma floor, worst adjacent-pair CVD
# separation dE 9.1 (protan) / 27.0 (tritan), normal-vision dE 22.9.
C_SHARED = "#eb6834"     # shared-parameter (transformer) policy
C_TAB = "#2a78d6"        # per-prompt (tabular) policy
C_ALT = "#1baf7a"
C_NULL = "#6b6a66"       # filter baseline / base model reference

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.6,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.6,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.0,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "legend.frameon": False,
})


def load(pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
