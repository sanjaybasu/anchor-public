"""Graphical abstract (AIH): design, omission and commission by delivered version, and alert burden.

Reads CANONICAL_NUMBERS_R1.json; no value is typed into this script. Arial only, no product names or logos.
Distinct from Figures 1-3: pooled estimates on one axis rather than per-model estimates. Writes
graphical_abstract_r1.png at 3,300 x 1,680 pixels (11.0 x 5.6 in at 300 dpi; the AIH minimum is 1,100 x 560).

Run: python graphical_abstract_r1.py [path/to/CANONICAL_NUMBERS_R1.json]
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from r1_style import CANON  # noqa: E402

OUT = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else CANON
canon = json.load(open(src))
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

v2 = canon["v2"]
pool = v2["pooled"]
c2 = canon["c2"]
n_cases = canon["cohort_flow"]["c1c_n"]
ROWS = (("guard", "First response", "#555555"),
        ("regen", "Self-revision", "#009E73"),
        ("v2_rewrite", "Revision with alerts", "#0072B2"),
        ("rewrite", "Revision with findings", "#D55E00"),
        ("v2_append", "Response and alerts", "#0072B2"),
        ("append", "Response and findings", "#D55E00"))

W, H = 11.0, 5.6
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, W)
bg.set_ylim(0, H)
bg.axis("off")


def panel(x, y, w, h, color, title):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", linewidth=2,
                                edgecolor=color, facecolor="white"))
    bg.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=16, fontweight="bold")


# Left: design
panel(0.2, 0.25, 2.3, 5.1, "#555555", "Study")
lines = [(f"{n_cases}", "clinical cases", 28), (f"{c2['n_probes']}", "commission probes", 28),
         ("3", "frontier LLMs", 28), ("6", "versions of each response", 28)]
y = 4.45
for big, small, fs in lines:
    bg.text(1.35, y, big, ha="center", va="center", fontsize=fs, fontweight="bold", color="#333333")
    bg.text(1.35, y - 0.4, small, ha="center", va="center", fontsize=12.5)
    y -= 0.98
bg.text(1.35, 0.68, "Feedback: none, full findings,\nor calibrated alerts;\nappended or used to revise", ha="center",
        va="center", fontsize=10.5, color="#333333", linespacing=1.3)

# Middle: omission and commission by version, one row per version
panel(2.7, 0.25, 5.85, 5.1, "#0072B2", "Omission and commission by version")


def dots(ax, block, key, lo_key, hi_key, labels):
    vals = []
    for i, (a, lab, col) in enumerate(ROWS):
        v = block["arms"].get(a)
        if not v or v.get(key) is None:
            continue
        m, lo, hi = 100 * v[key], 100 * v[lo_key][0], 100 * v[lo_key][1]
        vals.append(hi)
        ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="o", ms=9, color=col, elinewidth=2.0, capsize=3.5)
        ax.text(hi + 0.3, i, f"{m:.1f}", va="center", fontsize=12, color=col, fontweight="bold")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([lab for _, lab, _ in ROWS] if labels else [], fontsize=12)
    if not labels:
        ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_xlim(0, max(vals) * 1.4 if vals else 15)
    ax.tick_params(axis="x", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", linewidth=1)
    ax.axhline(3.5, color="#bbbbbb", linewidth=1)


ax_o = fig.add_axes([0.418, 0.215, 0.13, 0.585])
dots(ax_o, pool, "omit_mean", "omit_ci", None, labels=True)
ax_o.set_title("Required actions\nomitted (%)", fontsize=12.5, fontweight="bold", pad=8)
ax_c = fig.add_axes([0.593, 0.215, 0.13, 0.585])
dots(ax_c, c2, "p", "ci", None, labels=False)
ax_c.set_title("Probe-runs with\ncommission (%)", fontsize=12.5, fontweight="bold", pad=8)
bg.text(5.625, 0.55, f"Means with 95% CIs: {v2['n_test_cases']} held-out cases; {c2['n_probe_runs']} probe-runs.\n"
        "Feedback did not lower omission below self-revision;\nthe full findings lowered commission.",
        ha="center", va="center", fontsize=10.5, color="#444444", linespacing=1.2)

# Right: alert burden
panel(8.75, 0.25, 2.05, 5.1, "#E69F00", "Alert burden")
bg.text(9.775, 4.45, "Complete\nresponses flagged", ha="center", va="center", fontsize=12, fontweight="bold",
        linespacing=1.15)
bg.text(9.775, 3.7, f"{100 * pool['v1_false_alert_rate']['p']:.0f}%", ha="center", va="center", fontsize=26,
        fontweight="bold", color="#D55E00")
bg.text(9.775, 3.25, "full findings", ha="center", va="center", fontsize=11.5)
bg.text(9.775, 2.65, f"{100 * pool['false_alert_rate']['p']:.1f}%", ha="center", va="center", fontsize=26,
        fontweight="bold", color="#0072B2")
bg.text(9.775, 2.2, "calibrated alerts", ha="center", va="center", fontsize=11.5)
bg.text(9.775, 1.15, f"Words added:\n{pool['v1_words'][0]:,.0f} versus {pool['alert_words_per_alerted'][0]:,.0f}",
        ha="center", va="center", fontsize=11.5, linespacing=1.25, color="#333333")

fig.savefig(OUT / "graphical_abstract_r1.png", dpi=300)
print(f"wrote {OUT / 'graphical_abstract_r1.png'} from {src}")
