"""Regenerate `phase1_results.png` - the whole Phase 1 measurement record.

Run it with:

    uv run --with matplotlib python Stats/make_figure.py

matplotlib is deliberately NOT a project dependency: this script is run by hand
when the numbers change, and `spiyweb` itself installs with none.

Every value below is a SEALED Phase 1 result, copied from `RESULTS.md`, which is
the source of truth and carries the confidence intervals, the provenance of each
group and the limits that go with them. They are hard-coded rather than read
from `data/` on purpose: the sealed artifacts are hundreds of megabytes and are
not in the repository, so a figure that needed them could never be regenerated
by anyone who cloned it.

Palette: the data-viz reference instance. Categorical slots 1-3 (blue / orange /
aqua) for the three systems, the blue-red diverging pair for signed deltas.
Validated all-pairs in light mode: worst CVD dE 9.2, worst normal-vision dE
24.0. Aqua sits below 3:1 on this surface, so every bar carries a visible value
label - that is the documented relief, not decoration.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --- palette ---------------------------------------------------------------
SPIYWEB = "#2a78d6"
TOPK = "#eb6834"
ITER = "#1baf7a"
POS = "#2a78d6"
NEG = "#e34948"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LBL_S = "SPIYWEB"
LBL_T = "top-k — BASELINE"
LBL_I = "iterative — BASELINE"

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": INK2,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# --- sealed data (see RESULTS.md) ------------------------------------------
SETS = [
    "MuSiQue s42\n(tuning)",
    "MuSiQue s123\n(confirm)",
    "2Wiki\n(cross)",
    "HotpotQA\n(untouched)",
]
WEB = [0.5094, 0.5073, 0.7130, 0.6228]
TOP = [0.3090, 0.3046, 0.4682, 0.5623]
ITR = [0.4631, 0.4420, 0.6867, 0.6428]

D_ITER = [0.0463, 0.0653, 0.0262, -0.0200]
D_ITER_LO = [0.0299, 0.0478, 0.0164, -0.0317]
D_ITER_HI = [0.0624, 0.0826, 0.0367, -0.0086]

# support recall / novelty at k=5; S@5 = .65*recall + .35*novelty exactly.
REC_W = [0.6641, 0.6590, 0.9640, 0.9105]
NOV_W = [0.2222, 0.2255, 0.2467, 0.0885]
REC_I = [0.6218, 0.5979, 0.9377, 0.9440]
NOV_I = [0.1683, 0.1524, 0.2205, 0.0835]

HOPS = [
    "MuSiQue\n2-hop\nn=537",
    "MuSiQue\n3-hop\nn=307",
    "MuSiQue\n4-hop\nn=156",
    "2Wiki\n2-hop\nn=784",
    "2Wiki\n4-hop\nn=216",
    "HotpotQA\n2-hop\nn=1000",
]
H_WEB = [0.589, 0.476, 0.303, 0.707, 0.733, 0.623]
H_TOP = [0.367, 0.281, 0.164, 0.502, 0.344, 0.562]
H_ITR = [0.539, 0.426, 0.274, 0.677, 0.723, 0.643]

ABL = [
    ("coloured multi-seed vs plain web", 0.1994),
    ("proposition layer, plain path", 0.0790),
    ("distinct-passage window", 0.0118),
    ("distinct_sources seed rule", 0.0074),
    ("duplicate suppression ON", -0.0019),
    ("index-time NLI + subject filter", -0.0038),
    ("index-time NLI, unfiltered", -0.0187),
    ("proposition layer, coloured path", -0.0524),
]

RESCUE = [
    ("#5 colour count to profile\n(HotpotQA)", 0.0000),
    ("#5 colour count to profile\n(confirmation set)", 0.0005),
    ("#5 colour count to profile\n(tuning set)", 0.0025),
]


def _grid(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(16.5, 13.5), dpi=200)
gs = fig.add_gridspec(
    3, 2, hspace=0.46, wspace=0.22, left=0.075, right=0.975, top=0.885, bottom=0.055
)

# --- 1. the gate -----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(SETS))
h = 0.26
ax.barh(y + h, WEB, h * 0.92, color=SPIYWEB, label=LBL_S, zorder=3)
ax.barh(y, ITR, h * 0.92, color=ITER, label=LBL_I, zorder=3)
ax.barh(y - h, TOP, h * 0.92, color=TOPK, label=LBL_T, zorder=3)
for i in range(len(SETS)):
    for val, off in ((WEB[i], h), (ITR[i], 0), (TOP[i], -h)):
        ax.text(
            val + 0.012, y[i] + off, f"{val:.4f}", va="center", fontsize=8, color=INK2
        )
ax.set_yticks(y, SETS, fontsize=8.5)
ax.set_xlim(0, 0.88)
ax.set_ylim(-0.55, 4.05)
ax.set_xlabel("S@5  =  0.65 · support recall@5  +  0.35 · Novelty@5")
ax.set_title("1 · The gate — every sealed run, n=1000 each")
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.annotate(
    "the only bar where SPIYWEB\nis behind a baseline",
    xy=(0.6350, 3.02),
    xytext=(0.520, 3.62),
    fontsize=8,
    color=NEG,
    ha="left",
    arrowprops=dict(
        arrowstyle="->", color=NEG, lw=1.2, connectionstyle="arc3,rad=-0.25"
    ),
)
_grid(ax)

# --- 2. the decisive delta -------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
err = [
    [d - lo for d, lo in zip(D_ITER, D_ITER_LO, strict=True)],
    [hi - d for d, hi in zip(D_ITER, D_ITER_HI, strict=True)],
]
ax.barh(y, D_ITER, 0.5, color=[POS if d > 0 else NEG for d in D_ITER], zorder=3)
ax.errorbar(
    D_ITER, y, xerr=err, fmt="none", ecolor=INK2, elinewidth=1.4, capsize=4, zorder=4
)
for i, d in enumerate(D_ITER):
    anchor = D_ITER_HI[i] if d > 0 else D_ITER_LO[i]
    ax.text(
        anchor + (0.005 if d > 0 else -0.005),
        y[i],
        f"{d:+.4f}",
        va="center",
        ha="left" if d > 0 else "right",
        fontsize=9,
        color=INK,
        fontweight="bold",
    )
ax.axvline(0, color=INK, lw=1.1, zorder=5)
ax.set_yticks(y, SETS, fontsize=8.5)
ax.set_xlim(-0.062, 0.108)
ax.set_ylim(-1.45, 3.55)
ax.set_xlabel("S@5 difference vs the iterative baseline  (95% paired-bootstrap CI)")
ax.set_title("2 · Against the harder baseline — this panel is the verdict")
ax.text(
    -0.060,
    -1.30,
    "gate NOT passed: the HotpotQA interval excludes zero (P=.001),\n"
    "and five pre-registered rescue rounds all came back negative",
    fontsize=8.5,
    color=NEG,
    va="bottom",
    ha="left",
)
_grid(ax)

# --- 3. depth --------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
x = np.arange(len(HOPS))
w = 0.26
ax.bar(x - w, H_TOP, w * 0.92, color=TOPK, label=LBL_T, zorder=3)
ax.bar(x, H_ITR, w * 0.92, color=ITER, label=LBL_I, zorder=3)
ax.bar(x + w, H_WEB, w * 0.92, color=SPIYWEB, label=LBL_S, zorder=3)
for i in range(len(HOPS)):
    ax.text(
        x[i] + w,
        H_WEB[i] + 0.012,
        f"{H_WEB[i]:.3f}",
        ha="center",
        fontsize=7.5,
        color=INK2,
    )
ax.set_xticks(x, HOPS, fontsize=7.5)
ax.set_ylim(0, 0.92)
ax.set_ylabel("S@5")
ax.set_title("3 · The advantage is depth-dependent — and HotpotQA is 100% 2-hop")
ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3)
_grid(ax, axis="y")

# --- 4. decomposition ------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
x = np.arange(len(SETS))
w = 0.34
for off, rec, nov, name in (
    (-w / 2 - 0.01, REC_W, NOV_W, LBL_S),
    (w / 2 + 0.01, REC_I, NOV_I, LBL_I),
):
    base = [0.65 * r for r in rec]
    top = [0.35 * n for n in nov]
    color = SPIYWEB if name == LBL_S else ITER
    ax.bar(x + off, base, w, color=color, zorder=3)
    ax.bar(
        x + off,
        top,
        w,
        bottom=base,
        color=color,
        alpha=0.42,
        hatch=None if name == LBL_S else "///",
        edgecolor=SURFACE,
        linewidth=1.4,
        zorder=3,
    )
    for i in range(len(SETS)):
        ax.text(
            x[i] + off,
            base[i] + top[i] + 0.012,
            f"{base[i] + top[i]:.3f}",
            ha="center",
            fontsize=7.5,
            color=INK2,
        )
ax.set_xticks(x, SETS, fontsize=8.5)
ax.set_ylim(0, 1.06)
ax.set_ylabel("contribution to S@5")
ax.set_title("4 · What the score is made of — solid 0.65 · recall, pale 0.35 · novelty")
ax.legend(
    handles=[Patch(facecolor=SPIYWEB, label=LBL_S), Patch(facecolor=ITER, label=LBL_I)],
    loc="upper left",
    frameon=False,
    fontsize=8.5,
)
ax.text(
    0.985,
    0.985,
    "on HotpotQA novelty collapses to .088\n"
    "(vs .222 / .247 on the deeper sets):\n"
    "there is nothing left to spread to",
    transform=ax.transAxes,
    fontsize=8,
    color=NEG,
    ha="right",
    va="top",
)
_grid(ax, axis="y")

# --- 5. ablations ----------------------------------------------------------
ax = fig.add_subplot(gs[2, 0])
labels = [a[0] for a in ABL][::-1]
vals = [a[1] for a in ABL][::-1]
yb = np.arange(len(vals))
ax.barh(yb, vals, 0.6, color=[POS if v > 0 else NEG for v in vals], zorder=3)
for i, v in enumerate(vals):
    ax.text(
        v + (0.004 if v > 0 else -0.004),
        yb[i],
        f"{v:+.4f}",
        va="center",
        ha="left" if v > 0 else "right",
        fontsize=8,
        color=INK2,
    )
ax.axvline(0, color=INK, lw=1.1, zorder=5)
ax.set_yticks(yb, labels, fontsize=8)
ax.set_xlim(-0.09, 0.25)
ax.set_xlabel("S@5 change when the mechanism is switched on")
ax.set_title("5 · Every mechanism, measured — four of eight are negative")
ax.text(
    0.985,
    0.05,
    "shipped defaults are OFF for dedup, NLI, mass,\n"
    "propositions and the learned layer — the sealed\n"
    "numbers above were produced without them",
    transform=ax.transAxes,
    fontsize=8,
    color=INK2,
    ha="right",
    va="bottom",
)
_grid(ax)

# --- 6. rescue rounds ------------------------------------------------------
ax = fig.add_subplot(gs[2, 1])
rl = [r[0] for r in RESCUE]
rv = [r[1] for r in RESCUE]
yr = np.arange(len(rv))
ax.barh(yr, rv, 0.5, color=[POS if v > 0 else MUTED for v in rv], zorder=3)
for i, v in enumerate(rv):
    ax.text(
        max(v, 0) + 0.00008, yr[i], f"{v:+.4f}", va="center", fontsize=8.5, color=INK2
    )
ax.axvline(0, color=INK, lw=1.1, zorder=5)
ax.set_yticks(yr, rl, fontsize=8)
ax.set_ylim(-2.6, 2.5)
ax.set_xlim(-0.0002, 0.0040)
ax.set_xlabel("S@5 change — the only rescue round that moved anything at all")
ax.set_title("6 · Rescue round #5 — won on the tuning set, zero where it mattered")
ax.text(
    0.02,
    0.04,
    "Rounds #1-#4 (confidence gate x2, blending, novelty-free slots) were\n"
    "negative outright, and all four worked in the ranking layer. #5 was the\n"
    "first to move inside propagation, and it changed HotpotQA by exactly\n"
    "nothing: not one question's window moved. It was not shipped.\n\n"
    "Contradiction detection, measured on WikiContradict: 31.6% of annotated\n"
    "pairs, 9.5% end to end, 0% of same-passage pairs — the mechanism is\n"
    "sound, the detector is not yet a working feature.",
    transform=ax.transAxes,
    fontsize=8,
    color=INK2,
    va="bottom",
)
_grid(ax)

# --- frame -----------------------------------------------------------------
fig.suptitle(
    "SPIYWEB — Phase 1 measurement record",
    fontsize=17,
    fontweight="bold",
    x=0.075,
    ha="left",
    y=0.968,
)
fig.text(
    0.075,
    0.930,
    "Graph-based retrieval by spreading activation, measured against two "
    "baselines on four multi-hop datasets. The mechanism works; the gate was "
    "not passed.",
    fontsize=10.5,
    color=INK2,
    ha="left",
)
fig.text(
    0.075,
    0.906,
    "Tuning on MuSiQue seed 42 only; the winning configuration was then applied "
    "unchanged to the other three sets. Paired bootstrap, 10,000 resamples. "
    "Full table with provenance and limits: RESULTS.md",
    fontsize=8.5,
    color=MUTED,
    ha="left",
)

fig.savefig(
    "Stats/phase1_results.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35
)
print("wrote Stats/phase1_results.png")
