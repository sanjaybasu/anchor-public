"""Shared labels, markers, and colors for the revision figures (Figures 2 and 3, graphical abstract).

One label per concept, matching the manuscript text. Okabe-Ito colorblind-safe palette; every
series also differs by marker shape so the figures read in grayscale.
"""
from pathlib import Path

import matplotlib as mpl

CANON = Path(__file__).resolve().parents[1] / "CANONICAL_NUMBERS_R1.json"

MODELS = ("gpt6", "gemini31", "claude55")
MODEL_STYLE = {
    "gpt6": dict(label="GPT-6 Astra", color="#0072B2", marker="o", ms=6.5, mfc="#0072B2"),
    "gemini31": dict(label="Gemini 3.1 Pro", color="#D55E00", marker="s", ms=6.0, mfc="#D55E00"),
    "claude55": dict(label="Claude Opus 5.5", color="#009E73", marker="^", ms=7.0, mfc="#009E73"),
    "pooled": dict(label="All three models", color="#111111", marker="D", ms=6.5, mfc="white"),
}

ARM_LABEL = {
    "bare": "Bare",
    "guard": "Configuration A",
    "guard_alt": "Configuration B",
    "append": "Appended",
    "rewrite": "Rewrite",
}
ARM_STYLE = {
    "bare": dict(color="#555555", marker="o", ms=6.5),
    "guard": dict(color="#CC79A7", marker="s", ms=6.0),
    "append": dict(color="#E69F00", marker="^", ms=7.0),
    "rewrite": dict(color="#56B4E9", marker="D", ms=6.0),
}

CLASS_LABEL = {
    "ddi": "Drug-drug\ninteraction",
    "lost_in_middle_lab": "Buried\nlaboratory value",
    "wrong_treatment": "Wrong first-line\ntreatment",
    "fake_citation": "Nonexistent\ncitation",
}
CLASS_ORDER = ("ddi", "lost_in_middle_lab", "wrong_treatment", "fake_citation")
CLASS_STYLE = {
    "ddi": dict(color="#0072B2", marker="o", ms=6.5),
    "lost_in_middle_lab": dict(color="#D55E00", marker="s", ms=6.0),
    "wrong_treatment": dict(color="#009E73", marker="^", ms=7.0),
    "fake_citation": dict(color="#CC79A7", marker="v", ms=7.0),
}

PD_ORDER = ("pd_full", "pd_no_safety", "pd_no_filter", "pd_no_grounding", "pd_no_hazard")
PD_LABEL = {
    "pd_full": "All four\ndirectives",
    "pd_no_safety": "Without\nclinical-\nsafety",
    "pd_no_filter": "Without\nprescribing-\nadvice",
    "pd_no_grounding": "Without\nevidence-\ngrounding",
    "pd_no_hazard": "Without\nhazard-\nscreen",
}


def apply_style():
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def panel_letter(ax, letter, x=-0.075, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
