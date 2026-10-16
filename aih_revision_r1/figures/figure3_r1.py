"""Figure 3: commission in Cohort 2 (by failure class and configuration) and Cohort 3 (by
prompt-directive composition).

Reads CANONICAL_NUMBERS_R1.json (written by packaging/anchor/aih_r1_analysis.py); no value is
typed into this script. Commission is a probe-run whose median judge score is 0. Proportions
pool the three models; whiskers are Wilson 95% CIs. One marker shape per series.

Run: python figure3_r1.py [path/to/CANONICAL_NUMBERS_R1.json]
"""
import json
import sys
from pathlib import Path

from r1_style import (ARM_LABEL, ARM_STYLE, CANON, CLASS_LABEL, CLASS_ORDER, CLASS_STYLE, PD_LABEL,
                      PD_ORDER, apply_style, panel_letter)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else CANON
canon = json.load(open(src))
c2, c3 = canon["c2"], canon["c3"]
ARMS2 = ("bare", "guard", "append", "rewrite")
OFF4 = (-0.27, -0.09, 0.09, 0.27)
OFF5 = (-0.30, -0.15, 0.0, 0.15, 0.30)


def point(ax, x, v, st, label=None, filled=True):
    m = 100.0 * v["p"]
    lo, hi = 100.0 * v["ci"][0], 100.0 * v["ci"][1]
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt=st["marker"], ms=st["ms"], color=st["color"],
                mfc=st["color"] if filled else "white", mec=st["color"], mew=1.3, elinewidth=1.3,
                capsize=2.5, label=label, zorder=3, clip_on=False)


fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.9, 6.8), gridspec_kw={"hspace": 0.78})

classes2 = [c for c in CLASS_ORDER if c in c2["by_class"]]
for j, a in enumerate(ARMS2):
    for i, cl in enumerate(classes2):
        point(ax_a, i + OFF4[j], c2["by_class"][cl]["arms"][a], ARM_STYLE[a],
              label=ARM_LABEL[a] if i == 0 else None)
n2 = [c2["by_class"][cl]["arms"]["bare"]["n"] for cl in classes2]
ax_a.set_xticks(range(len(classes2)))
ax_a.set_xticklabels([f"{CLASS_LABEL[cl]}\n(n = {n})" for cl, n in zip(classes2, n2)])
ax_a.set_xlim(-0.55, len(classes2) - 0.45)
ax_a.set_ylim(bottom=-1.5)
ax_a.set_ylabel("Probe-runs with commission (%)")
ax_a.set_title("Cohort 2: configuration and verification placement", loc="left", pad=26)
ax_a.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
for i in range(1, len(classes2)):
    ax_a.axvline(i - 0.5, color="#d0d0d0", linewidth=0.8, zorder=0)
ax_a.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3,
            columnspacing=1.2)
panel_letter(ax_a, "A", y=1.13)

classes3 = [c for c in CLASS_ORDER if c in c3["by_class"]]
for j, cl in enumerate(classes3):
    for i, pdk in enumerate(PD_ORDER):
        point(ax_b, i + OFF5[j], c3["by_class"][cl]["arms"][pdk], CLASS_STYLE[cl],
              label=CLASS_LABEL[cl].replace("\n", " ") if i == 0 else None)
pooled_style = dict(color="#111111", marker="D", ms=6.5)
for i, pdk in enumerate(PD_ORDER):
    point(ax_b, i + OFF5[4], c3["arms"][pdk], pooled_style, label="All classes" if i == 0 else None,
          filled=False)
ax_b.set_xticks(range(len(PD_ORDER)))
ax_b.set_xticklabels([PD_LABEL[p] for p in PD_ORDER])
ax_b.set_xlabel("Directive removed from the guardrail prompt")
ax_b.set_xlim(-0.55, len(PD_ORDER) - 0.45)
ax_b.set_ylim(bottom=-1.5)
ax_b.set_ylabel("Probe-runs with commission (%)")
ax_b.set_title("Cohort 3: prompt-directive composition", loc="left", pad=40)
ax_b.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
for i in range(1, len(PD_ORDER)):
    ax_b.axvline(i - 0.5, color="#d0d0d0", linewidth=0.8, zorder=0)
ax_b.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3,
            columnspacing=1.0)
panel_letter(ax_b, "B", y=1.2)

fig.savefig(OUT / "figure3_r1.png", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "figure3_r1.pdf", bbox_inches="tight", pad_inches=0.05)
print(f"wrote {OUT / 'figure3_r1.png'} from {src}")
