"""Figure 3: commission by delivered version, Cohort 2 (physician-reviewed probes).

Reads CANONICAL_NUMBERS_R1.json (written by packaging/anchor/aih_r1_analysis.py); no value is
typed into this script. Commission is a probe-run whose median judge score is 0; a provider refusal
counts as no commission. Panel A: share of probe-runs with commission, with Wilson 95% CIs for each
model and a probe-clustered bootstrap 95% CI for the three models together. Panel B: paired
differences with probe-clustered bootstrap 95% CIs. Estimates are dodged by model, with one marker
shape per model, as in Figure 2.

Run: python figure3_r1.py [path/to/CANONICAL_NUMBERS_R1.json]
"""
import json
import sys
from pathlib import Path

from r1_style import CANON, MODEL_STYLE, MODELS, apply_style, panel_letter

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else CANON
canon = json.load(open(src))
c2 = canon["c2"]
series = [(mk, c2["per_model"][mk]) for mk in MODELS] + [("pooled", c2)]
ARMS = ("guard", "regen", "append", "rewrite", "v2_append", "v2_rewrite")
LABEL = {"guard": "First\nresponse", "regen": "Self-\nrevision", "append": "Response and\nfindings",
         "rewrite": "Revision\nwith findings", "v2_append": "Response\nand alerts", "v2_rewrite": "Revision\nwith alerts"}
CONTRASTS = (("rewrite_vs_regen", "Revision with findings − self-revision"),
             ("v2_rewrite_vs_regen", "Revision with alerts − self-revision"),
             ("regen_vs_guard", "Self-revision − first response"),
             ("append_vs_guard", "Response and findings − first response"),
             ("v2_append_vs_guard", "Response and alerts − first response"),
             ("rewrite_vs_guard", "Revision with findings − first response"),
             ("v2_rewrite_vs_guard", "Revision with alerts − first response"))
OFF = (-0.27, -0.09, 0.09, 0.27)


def pct(x):
    return 100.0 * x


fig = plt.figure(figsize=(6.0, 7.4))
ax_a = fig.add_axes([0.11, 0.58, 0.87, 0.33])     # panel A spans the width
ax_b = fig.add_axes([0.53, 0.06, 0.42, 0.40])     # panel B leaves room for its contrast labels
for j, (mk, s) in enumerate(series):
    st = MODEL_STYLE[mk]
    xs, ys, lo, hi = [], [], [], []
    for i, a in enumerate(ARMS):
        v = s["arms"].get(a)
        if not v:
            continue
        m, (l, h) = pct(v["p"]), (pct(v["ci"][0]), pct(v["ci"][1]))
        xs.append(i + OFF[j]); ys.append(m); lo.append(m - l); hi.append(h - m)
    ax_a.errorbar(xs, ys, yerr=[lo, hi], fmt=st["marker"], ms=st["ms"], color=st["color"],
                  mfc=st["mfc"], mec=st["color"], mew=1.3, elinewidth=1.3, capsize=2.5,
                  label=st["label"], zorder=3, clip_on=False)
ax_a.set_xticks(range(len(ARMS)))
ax_a.set_xticklabels([LABEL[a] for a in ARMS])
ax_a.set_xlim(-0.55, len(ARMS) - 0.45)
ax_a.set_ylim(bottom=-1.0)
ax_a.set_ylabel("Probe-runs with commission (%)")
ax_a.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
for i in range(1, len(ARMS)):
    ax_a.axvline(i - 0.5, color="#d0d0d0", linewidth=0.8, zorder=0)
panel_letter(ax_a, "A")
ax_a.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), handletextpad=0.3,
            columnspacing=1.2)

for j, (mk, s) in enumerate(series):
    st = MODEL_STYLE[mk]
    for i, (ck, _) in enumerate(CONTRASTS):
        v = s["contrasts"].get(ck)
        if not v:
            continue
        m, (l, h) = pct(v["diff_mean"]), (pct(v["diff_ci"][0]), pct(v["diff_ci"][1]))
        ax_b.errorbar([m], [i + OFF[j]], xerr=[[m - l], [h - m]], fmt=st["marker"], ms=st["ms"],
                      color=st["color"], mfc=st["mfc"], mec=st["color"], mew=1.3, elinewidth=1.3,
                      capsize=2.5, zorder=3, clip_on=False)
ax_b.axvline(0, color="#777777", linewidth=1.0, linestyle="--", zorder=1)
ax_b.set_yticks(range(len(CONTRASTS)))
ax_b.set_yticklabels([lab for _, lab in CONTRASTS])
ax_b.set_ylim(len(CONTRASTS) - 0.5, -0.5)
ax_b.set_xlabel("Paired difference, percentage points")
ax_b.grid(axis="x", color="#e3e3e3", linewidth=0.8, zorder=0)
panel_letter(ax_b, "B", x=-1.18)

fig.savefig(OUT / "figure3_r1.png", bbox_inches="tight", pad_inches=0.05)
fig.savefig(OUT / "figure3_r1.pdf", bbox_inches="tight", pad_inches=0.05)
print(f"wrote {OUT / 'figure3_r1.png'} from {src}")
