"""Graphical abstract (AIH): design flow and pooled results.

Reads CANONICAL_NUMBERS_R1.json; no value is typed into this script. Arial only, no product names
or logos, short labels only. Distinct from Figures 1-3: pooled bars rather than per-model estimates.
Writes graphical_abstract_r1.png at 3,300 x 1,680 pixels (11.0 x 5.6 in at 300 dpi; the AIH
minimum is 1,100 x 560 pixels).

Run: python graphical_abstract_r1.py [path/to/CANONICAL_NUMBERS_R1.json]
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from r1_style import CANON  # noqa: E402

OUT = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else CANON
canon = json.load(open(src))
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

pool = canon["c1c"]["pooled"]
om = {a: 100 * pool["arms"][a]["omit_mean"] for a in ("bare", "guard", "append", "rewrite")}
catch = 100 * pool["catch"]["p"]
fpr = 100 * pool["fpr"]["p"]
c2 = canon["c2"]["arms"]
cm = {a: 100 * c2[a]["p"] for a in ("bare", "guard", "append", "rewrite")}
n_cases = canon["cohort_flow"]["c1c_n"]

W, H = 11.0, 5.6
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, W)
bg.set_ylim(0, H)
bg.axis("off")
GREY, PURPLE, ORANGE, SKY, GREEN = "#555555", "#CC79A7", "#E69F00", "#56B4E9", "#009E73"


def panel(x, y, w, h, color, title):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", linewidth=2,
                                edgecolor=color, facecolor="white"))
    bg.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=17, fontweight="bold")


def arrow(x0, x1, y):
    bg.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=22, linewidth=2.5,
                                 color=GREY))


def bars(rect, labels, values, colors, unit="%"):
    ax = fig.add_axes(rect)
    xs = range(len(values))
    ax.bar(xs, values, color=colors, width=0.62)
    for x, v in zip(xs, values):
        ax.text(x, v + max(values) * 0.04, f"{v:.0f}{unit}", ha="center", va="bottom", fontsize=14,
                fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylim(0, max(values) * 1.32 if max(values) > 0 else 1)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return ax


bg.text(W / 2, H - 0.25, "Where verification findings act, and what the clinician receives", ha="center",
        va="top", fontsize=21, fontweight="bold")

panel(0.25, 0.45, 2.25, 4.25, GREY, "Same cases")
bg.text(1.375, 3.65, f"{n_cases}", ha="center", va="center", fontsize=40, fontweight="bold", color=GREY)
bg.text(1.375, 3.0, "clinical cases", ha="center", va="center", fontsize=15)
bg.text(1.375, 2.2, "3", ha="center", va="center", fontsize=40, fontweight="bold", color=GREY)
bg.text(1.375, 1.55, "frontier LLMs", ha="center", va="center", fontsize=15)
bg.text(1.375, 0.95, "120 commission probes", ha="center", va="center", fontsize=13)
arrow(2.55, 2.95, 2.6)

panel(3.0, 0.45, 2.45, 4.25, PURPLE, "Guardrail prompt")
bars([0.300, 0.27, 0.19, 0.37], ["Bare", "Guardrail"], [om["bare"], om["guard"]], [GREY, PURPLE])
bg.text(4.225, 0.85, "required actions omitted", ha="center", va="center", fontsize=13)
arrow(5.5, 5.9, 2.6)

panel(5.95, 0.45, 2.45, 4.25, ORANGE, "Findings appended")
bg.text(7.175, 3.55, f"{catch:.0f}%", ha="center", va="center", fontsize=40, fontweight="bold", color=ORANGE)
bg.text(7.175, 2.85, "of omitted actions\nraised by the findings", ha="center", va="center", fontsize=14)
bg.text(7.175, 1.75, f"{fpr:.0f}%", ha="center", va="center", fontsize=30, fontweight="bold", color=GREY)
bg.text(7.175, 1.15, "of complete responses\nflagged anyway", ha="center", va="center", fontsize=13)
arrow(8.45, 8.85, 2.6)

panel(8.9, 0.45, 1.85, 4.25, SKY, "Rewrite")
bars([0.833, 0.27, 0.135, 0.37], ["Append", "Rewrite"], [om["append"], om["rewrite"]], [ORANGE, SKY])
bg.text(9.825, 0.85, "required actions\nomitted", ha="center", va="center", fontsize=13)

fig.savefig(OUT / "graphical_abstract_r1.png", dpi=300)
print(f"wrote {OUT / 'graphical_abstract_r1.png'} ({int(W * 300)} x {int(H * 300)} px) from {src}")
