"""Figure 1: study design. Each first response is delivered in six versions: three kinds of feedback
crossed with two deliveries.

Conceptual schematic; reads no data. Arial, Okabe-Ito colorblind-safe palette. Box sizes
come from the rendered text, and the script stops if any text extends past its box.
Smallest text is 9.5 pt at the drawn width of 6.0 in, which is the width at which the figure appears. Writes figure1_r1.png
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
TITLE, BODY, LS = 10.5, 9.5, 1.22
W, M = 6.0, 0.06                  # figure width and outer margin (inches; data units are inches)
PX, PY, GAP = 0.10, 0.09, 0.07    # inner padding and title-to-body gap
ROW_GAP, COL_GAP = 0.42, 0.20

fig = plt.figure(figsize=(W, 11.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(11.0, 0)              # y runs downward from the top edge
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


# Row 1: case and first response --------------------------------------------------------
y1 = M
r1 = [("Case", "Cohort 1c: case text and a request for\nthe triage level, differential diagnosis,\nworkup, and treatment plan\nCohort 2: a commission probe", BLUE),
      ("First response", "The same model under\nconfiguration A, a structured\nclinical-safety prompt", PURPLE)]
w1 = (W - 2 * M - 0.45) / 2
h1 = max(need(t, b)[1] for t, b, _ in r1)
for k, (t, b, c) in enumerate(r1):
    box(M + k * (w1 + 0.45), y1, w1, h1, t, b, c)
arrow(M + w1, y1 + h1 / 2, M + w1 + 0.45, y1 + h1 / 2)

# Row 2: six delivered versions (feedback across, delivery down) --------------------------
LW = 0.95                                  # width of the delivery labels
cw = (W - 2 * M - LW - 2 * 0.10) / 3       # cell width
y2 = y1 + h1 + ROW_GAP
heads = [("No feedback", "No verification\nruns", GREY),
         ("Full findings", "Rule certificate on\nthe response and six\nagents run on the\nsame model", ORANGE),
         ("Calibrated alerts", "Rules grounded in the\ncase and blind drafts\nby two other models;\nat most two alerts\nabove a conformal\nthreshold", GREEN)]
hh = max(need(t, b)[1] for t, b, _ in heads)
xs = [M + LW + k * (cw + 0.10) for k in range(3)]
for x, (t, b, c) in zip(xs, heads):
    box(x, y2, cw, hh, t, b, c)
fr_cx = M + w1 + 0.45 + w1 / 2
ax.plot([fr_cx, fr_cx], [y1 + h1, y2 - 0.16], color=GREY, linewidth=1.4)
ax.plot([xs[0] + cw / 2, xs[2] + cw / 2], [y2 - 0.16] * 2, color=GREY, linewidth=1.4)
for x in xs:
    arrow(x + cw / 2, y2 - 0.16, x + cw / 2, y2)
cells = [[("First response", "Delivered\nunchanged"),
          ("Response and\nfindings", "The findings follow\nthe unchanged\nresponse"),
          ("Response and\nalerts", "The alerts follow\nthe unchanged\nresponse")],
         [("Self-revision", "The model revises\nits response with\nno feedback"),
          ("Revision with\nfindings", "The model revises\nits response with\nthe findings"),
          ("Revision with\nalerts", "The model revises\nits response with\nthe alerts")]]
rows = [("Appended\nfor the\nclinician", SKY), ("Returned\nto the model\nto revise", BLUE)]
ch = max(need(t, b)[1] for row in cells for t, b in row)
y = y2 + hh + 0.12
for (rl, rc), row in zip(rows, cells):
    ax.text(M + LW / 2 - 0.04, y + ch / 2, rl, fontsize=BODY, fontweight="bold", ha="center", va="center",
            color="#111111", linespacing=LS)
    for x, (t, b) in zip(xs, row):
        box(x, y, cw, ch, t, b, rc)
    y += ch + 0.10
label(M + LW, y + 0.02, "A first response with no alert is delivered unchanged in both alert versions.", ha="left")
y3 = y + 0.30

# Row 3: outcomes -----------------------------------------------------------------------
oc_title = "Outcomes: three LLM judges score each whole delivered\ntext, blinded to the version and the model"
cols = [("Omission (Cohort 1c)",
         "Share of the High-grade checklist\nitems not addressed. A second\njudging and a second first\nresponse measure noise."),
        ("Commission (Cohort 2)",
         "The delivered text recommends\nthe probe's wrong action\nwithout a caveat")]
oc_th = extent(oc_title, TITLE, "bold")[1]
col_w = (W - 2 * M - 2 * PX) / 2
col_h = max(extent(h, BODY, "bold")[1] + 0.04 + extent(b, BODY)[1] for h, b in cols)
for h, b in cols:
    if max(extent(h, BODY, "bold")[0], extent(b, BODY)[0]) > col_w - 0.1:
        raise SystemExit(f"outcome column too narrow for '{h}'")
h3 = PY + oc_th + GAP + col_h + PY
oc = panel(M, y3, W - 2 * M, h3, BLUE)
put(oc, W / 2, y3 + PY, oc_title, TITLE, "bold", color="#111111")
for k, (h, b) in enumerate(cols):
    cx = M + PX + col_w * (k + 0.5)
    yy = y3 + PY + oc_th + GAP
    put(oc, cx, yy, h, BODY, "bold", color="#111111")
    put(oc, cx, yy + extent(h, BODY, "bold")[1] + 0.04, b, BODY)
arrow(W / 2, y - 0.10 + 0.24, W / 2, y3)

H = y3 + h3 + M
for t, p_ in CHECKS:
    tb = t.get_window_extent(renderer=RENDERER).transformed(ax.transData.inverted())
    bb = p_.get_bbox()
    x0, x1 = sorted((tb.x0, tb.x1)); ya, yb = sorted((tb.y0, tb.y1))
    if x0 < bb.x0 - 1e-3 or x1 > bb.x1 + 1e-3 or ya < bb.y0 - 1e-3 or yb > bb.y1 + 1e-3:
        raise SystemExit(f"text outside its box: {t.get_text()[:40]!r}")
fig.set_size_inches(W, H)
ax.set_ylim(H, 0)
fig.savefig(OUT / "figure1_r1.png", dpi=300)
fig.savefig(OUT / "figure1_r1.pdf")
print(f"wrote {OUT / 'figure1_r1.png'} ({W:.2f} x {H:.2f} in; {len(CHECKS)} text blocks inside their boxes)")
