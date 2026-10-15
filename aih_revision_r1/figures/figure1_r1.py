"""Figure 1: configurations, verification framework, and the two placements of its findings.

Conceptual schematic; reads no data. Arial, Okabe-Ito colorblind-safe palette. Box sizes
come from the rendered text, and the script stops if any text extends past its box.
Smallest text is 9 pt at the printed width of 6.9 in (17.5 cm). Writes figure1_r1.png
(300 dpi) and figure1_r1.pdf next to this script.

Run: python figure1_r1.py
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42})

BLUE, ORANGE, GREEN, PURPLE, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#555555", "#56B4E9"
TITLE, BODY, LS = 10, 9, 1.25
W, M = 6.9, 0.08                  # figure width and outer margin (inches; data units are inches)
PX, PY, GAP = 0.10, 0.09, 0.07    # inner padding and title-to-body gap
ROW_GAP, COL_GAP = 0.46, 0.30

fig = plt.figure(figsize=(W, 9.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(9.0, 0)               # y runs downward from the top edge
ax.axis("off")
RENDERER = fig.canvas.get_renderer()
CHECKS = []                       # (text, patch) pairs verified for containment before saving


def tint(c, m=0.87):
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(*(int(round(v + (255 - v) * m)) for v in (r, g, b)))


def extent(s, fs, weight="normal"):
    t = ax.text(0, 0, s, fontsize=fs, fontweight=weight, linespacing=LS)
    bb = t.get_window_extent(renderer=RENDERER)
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def need(title, body):
    tw, th = extent(title, TITLE, "bold")
    bw, bh = extent(body, BODY) if body else (0.0, 0.0)
    return max(tw, bw) + 2 * PX, PY + th + (GAP + bh if body else 0.0) + PY


def panel(x, y, w, h, color, fill=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", linewidth=1.2,
                       edgecolor=color, facecolor=tint(color) if fill else "white")
    ax.add_patch(p)
    return p


def put(p, x, y, s, fs, weight="normal", ha="center", color="#222222"):
    t = ax.text(x, y, s, fontsize=fs, fontweight=weight, ha=ha, va="top", color=color,
                linespacing=LS)
    CHECKS.append((t, p))
    return t


def box(x, y, w, h, title, body, color, fill=True, align="center"):
    nw, _ = need(title, body)
    if nw > w + 1e-6:
        raise SystemExit(f"text wider than box '{title}': needs {nw:.2f} in, has {w:.2f} in")
    p = panel(x, y, w, h, color, fill)
    put(p, x + w / 2, y + PY, title, TITLE, "bold", color="#111111")
    if body:
        th = extent(title, TITLE, "bold")[1]
        bx = x + w / 2 if align == "center" else x + PX
        put(p, bx, y + PY + th + GAP, body, BODY, ha=align)
    return p


def arrow(x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.4, color=color, shrinkA=0, shrinkB=0))


def label(x, y, s, color=GREY, ha="left"):
    ax.text(x, y, s, fontsize=BODY, color=color, ha=ha, va="center", linespacing=LS)


# Row 1: input, model under a configuration, response -----------------------------------
row1 = [
    (M, 1.80, "Case input",
     "Cohort 1c: case text and a\nrequest for the triage level,\ndifferential diagnosis,\n"
     "workup, and treatment plan\n\nCohorts 2 and 3:\na commission probe", BLUE, "center"),
    (None, 2.50, "Model and configuration",
     "GPT-6 Astra, Gemini 3.1 Pro,\nor Claude Opus 5.5\n\nBare: no system prompt\n"
     "Configuration A: structured\nsafety prompt; Gemini also\nuses search grounding\n"
     "Configuration B: safety\nprompt, no output format", PURPLE, "left"),
    (None, None, "Response",
     "Bare, configuration A,\nor configuration B\nresponse\n\nBare and configuration B\nresponses are scored\nas delivered", GREY, "center"),
]
x = M
for i, (bx, bw, *_rest) in enumerate(row1):
    if bw is None:
        bw = W - M - x
    row1[i] = (x, bw, *_rest)
    x += bw + COL_GAP
y1 = M
h1 = max(need(t, b)[1] for _, _, t, b, _, _ in row1)
for bx, bw, t, b, c, al in row1:
    box(bx, y1, bw, h1, t, b, c, align=al)
mid1 = y1 + h1 / 2
arrow(row1[0][0] + row1[0][1], mid1, row1[1][0], mid1)
arrow(row1[1][0] + row1[1][1], mid1, row1[2][0], mid1)

# Row 2: verification framework ------------------------------------------------------------
subs = [
    ("Rule certificate",
     "Facts extracted from the\nresponse are checked by a\nLogical Neural Network\n"
     "against 3,128 clinical rules\nand 34 fallback rules"),
    ("Specialist agents",
     "Six agents, each a call to the\nsame model: drug interactions,\nlaboratory values, guidelines,\n"
     "citations, safety net, and\ndifferential breadth"),
    ("Finding labels",
     "Each finding is labeled as\ntraceable to a named check\nor raised only by the\n"
     "safety-net agent, with a fixed\nstatement on untraceable\nreasoning"),
]
fw_title = "Verification framework: reviews the configuration A response"
y2 = y1 + h1 + ROW_GAP
fw_th = extent(fw_title, TITLE, "bold")[1]
sub_gap = 0.12
sub_w = (W - 2 * M - 2 * PX - 2 * sub_gap) / 3
sub_h = max(need(t, b)[1] for t, b in subs)
h2 = PY + fw_th + GAP + sub_h + PY
fw = panel(M, y2, W - 2 * M, h2, GREEN, fill=False)
put(fw, W / 2, y2 + PY, fw_title, TITLE, "bold", color="#111111")
for i, (t, b) in enumerate(subs):
    box(M + PX + i * (sub_w + sub_gap), y2 + PY + fw_th + GAP, sub_w, sub_h, t, b, GREEN)
resp_cx = row1[2][0] + row1[2][1] / 2
arrow(resp_cx, y1 + h1, resp_cx, y2)
label(resp_cx - 0.08, y1 + h1 + ROW_GAP / 2, "configuration A\nresponse", ha="right")

# Row 3: the two placements of the same findings -------------------------------------------
pl = [
    ("Placement 1: appended",
     "Delivered text: the unchanged\nconfiguration A response,\nfollowed by the findings", ORANGE),
    ("Placement 2: rewrite",
     "Delivered text: a rewritten response\nfrom the same model, given its\nconfiguration A response, the same\n"
     "findings, and an instruction to\naddress each finding that it\njudges clinically correct", SKY),
]
y3 = y2 + h2 + ROW_GAP
pw = (W - 2 * M - COL_GAP) / 2
h3 = max(need(t, b)[1] for t, b, _ in pl)
for i, (t, b, c) in enumerate(pl):
    px = M + i * (pw + COL_GAP)
    box(px, y3, pw, h3, t, b, c)
    arrow(px + pw / 2, y2 + h2, px + pw / 2, y3)
    label(px + pw / 2 + 0.08, y2 + h2 + ROW_GAP / 2, "same response and findings")

# Row 4: outcomes ---------------------------------------------------------------------------
oc_title = "Outcomes: three LLM judges, blinded to configuration, score the delivered text"
cols = [
    ("Safety-critical omission (Cohort 1c)",
     "Share of the High-grade checklist items\nthat the delivered text does not address"),
    ("Commission (Cohorts 2 and 3)",
     "The delivered text recommends the\nwrong action without a caveat"),
]
y4 = y3 + h3 + ROW_GAP
oc_th = extent(oc_title, TITLE, "bold")[1]
col_w = (W - 2 * M - 2 * PX) / 2
col_h = max(extent(h, BODY, "bold")[1] + 0.04 + extent(b, BODY)[1] for h, b in cols)
for h, b in cols:
    if max(extent(h, BODY, "bold")[0], extent(b, BODY)[0]) > col_w - 0.1:
        raise SystemExit(f"outcome column too narrow for '{h}'")
h4 = PY + oc_th + GAP + col_h + PY
oc = panel(M, y4, W - 2 * M, h4, BLUE)
put(oc, W / 2, y4 + PY, oc_title, TITLE, "bold", color="#111111")
for i, (h, b) in enumerate(cols):
    cx = M + PX + col_w * (i + 0.5)
    yy = y4 + PY + oc_th + GAP
    put(oc, cx, yy, h, BODY, "bold", color="#111111")
    put(oc, cx, yy + extent(h, BODY, "bold")[1] + 0.04, b, BODY)
for i in range(2):
    px = M + i * (pw + COL_GAP) + pw / 2
    arrow(px, y3 + h3, px, y4)

# Fit the canvas to the content, then confirm every text sits inside its box ---------------
H = y4 + h4 + M
fig.set_size_inches(W, H)
ax.set_ylim(H, 0)
fig.canvas.draw()
RENDERER = fig.canvas.get_renderer()
for t, p in CHECKS:
    tb = t.get_window_extent(renderer=RENDERER)
    pb = p.get_window_extent(renderer=RENDERER)
    if tb.x0 < pb.x0 or tb.x1 > pb.x1 or tb.y0 < pb.y0 or tb.y1 > pb.y1:
        raise SystemExit(f"text overflows its box: {t.get_text()[:40]!r}")

fig.savefig(OUT / "figure1_r1.png", dpi=300)
fig.savefig(OUT / "figure1_r1.pdf")
print(f"wrote {OUT / 'figure1_r1.png'} ({W:.2f} x {H:.2f} in; {len(CHECKS)} text blocks inside their boxes)")
