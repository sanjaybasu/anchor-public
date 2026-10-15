"""Figure 2: safety-critical omission in Cohort 1c, by configuration and model.

Reads CANONICAL_NUMBERS_R1.json (written by packaging/anchor/aih_r1_analysis.py); no value is
typed into this script. Panel A: mean share of High-grade checklist items that the delivered text
does not address, with case-clustered bootstrap 95% CIs. Panel B: paired differences for the four
primary contrasts and for configuration B vs. bare. Estimates are dodged by model, with one marker
shape per model.

Run: python figure2_r1.py [path/to/CANONICAL_NUMBERS_R1.json]
"""
import json
import sys
from pathlib import Path

from r1_style import ARM_LABEL, CANON, MODEL_STYLE, MODELS, apply_style, panel_letter

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else CANON
c1 = json.load(open(src))["c1c"]
series = [(mk, c1["per_model"][mk]) for mk in MODELS] + [("pooled", c1["pooled"])]
ARMS = ("bare", "guard", "guard_alt", "append", "rewrite")
CONTRASTS = (("guard_vs_bare", "Configuration A − bare"),
             ("append_vs_guard", "Appended − configuration A"),
             ("rewrite_vs_guard", "Rewrite − configuration A"),
             ("rewrite_vs_append", "Rewrite − appended"),
             ("guardalt_vs_bare", "Configuration B − bare"))
OFF = (-0.27, -0.09, 0.09, 0.27)


def pct(x):
    return 100.0 * x


fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.9, 6.6), gridspec_kw={"height_ratios": [1.0, 1.0],
                                                                         "hspace": 0.42})
for j, (mk, s) in enumerate(series):
    st = MODEL_STYLE[mk]
    xs, ys, lo, hi = [], [], [], []
    for i, a in enumerate(ARMS):
        v = s["arms"][a]
        m, (l, h) = pct(v["omit_mean"]), (pct(v["omit_ci"][0]), pct(v["omit_ci"][1]))
        xs.append(i + OFF[j]); ys.append(m); lo.append(m - l); hi.append(h - m)
    ax_a.errorbar(xs, ys, yerr=[lo, hi], fmt=st["marker"], ms=st["ms"], color=st["color"],
                  mfc=st["mfc"], mec=st["color"], mew=1.3, elinewidth=1.3, capsize=2.5,
                  label=st["label"], zorder=3, clip_on=False)
ax_a.set_xticks(range(len(ARMS)))
ax_a.set_xticklabels([ARM_LABEL[a] for a in ARMS])
ax_a.set_xlim(-0.55, len(ARMS) - 0.45)
ax_a.set_ylim(bottom=-1.0)
ax_a.set_ylabel("High-grade items not addressed (%)")
ax_a.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
for i in range(1, len(ARMS)):
    ax_a.axvline(i - 0.5, color="#d0d0d0", linewidth=0.8, zorder=0)
panel_letter(ax_a, "A")
ax_a.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), handletextpad=0.3,
            columnspacing=1.2)

for j, (mk, s) in enumerate(series):
    st = MODEL_STYLE[mk]
    for i, (ck, _) in enumerate(CONTRASTS):
        v = s["contrasts"][ck]
        m, (l, h) = pct(v["diff_mean"]), (pct(v["diff_ci"][0]), pct(v["diff_ci"][1]))
        ax_b.errorbar([m], [i + OFF[j]], xerr=[[m - l], [h - m]], fmt=st["marker"], ms=st["ms"],
                      color=st["color"], mfc=st["mfc"], mec=st["color"], mew=1.3, elinewidth=1.3,
                      capsize=2.5, zorder=3, clip_on=False)
ax_b.axvline(0, color="#777777", linewidth=1.0, linestyle="--", zorder=1)
ax_b.axhline(len(CONTRASTS) - 1.5, color="#bbbbbb", linewidth=0.8, zorder=0)
ax_b.set_yticks(range(len(CONTRASTS)))
ax_b.set_yticklabels([lab for _, lab in CONTRASTS])
ax_b.set_ylim(len(CONTRASTS) - 0.5, -0.5)
ax_b.set_xlabel("Paired difference in High-grade items not addressed (percentage points)")
ax_b.grid(axis="x", color="#e3e3e3", linewidth=0.8, zorder=0)
ax_b.text(1.0, len(CONTRASTS) - 1.0, "secondary", transform=ax_b.get_yaxis_transform(), ha="right",
          va="center", fontsize=9, color="#555555")
panel_letter(ax_b, "B", x=-0.33)

fig.savefig(OUT / "figure2_r1.png", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "figure2_r1.pdf", bbox_inches="tight", pad_inches=0.05)
print(f"wrote {OUT / 'figure2_r1.png'} from {src}")
