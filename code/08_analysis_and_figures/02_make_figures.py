#!/usr/bin/env python3
"""Generate the paper's chart figures as PDFs. Run from paper/ ."""
import math
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    # acmart/acmsmall body is Linux Libertine. DejaVu Serif is the closest metric-
    # compatible face available here. Figures are drawn at final print size, so
    # 7pt here renders as 7pt in the paper with no scaling.
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Libertine", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9a9a9a",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.color": "#6b6b6b", "ytick.color": "#6b6b6b",
    "axes.labelcolor": "#3a3a3a", "text.color": "#3a3a3a",
    "pdf.fonttype": 42, "figure.dpi": 300,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.5,
})
# pastel palette, one hue per role, deeper tints reserved for emphasis
BLUE   = "#7FA8D4";  BLUE_D  = "#3F6FA8"
CORAL  = "#E9A6A0";  CORAL_D = "#C2665C"
SAGE   = "#9CC5A1";  SAGE_D  = "#5E8F66"
SAND   = "#E3C79A";  SAND_D  = "#B8935A"
MAUVE  = "#BFA6CC";  MAUVE_D = "#8C6FA0"
INK    = "#4a4a4a";  GREY = "#9a9a9a";  LGREY = "#dcdcdc"
ACC, ACC2 = BLUE_D, CORAL_D

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*(c-h), 100*(c+h)


# ---------------------------------------------------------------- Fig 1
def fig_pipeline():
    import numpy as np
    # deduplicated by content, matching scan_ours_dedup.py so both scans count the same way.
    # counts from funnel12.json: all twelve mined repositories, criterion (2) as its own stage.
    STAGES = [(51300, "51,300", "functions scanned\n12 repositories"),
              ( 2883,  "2,883", "$\\geq$2 concern\ncategories"),
              ( 2646,  "2,646", "$\\geq$1 unfixable\nconcern"),
              ( 2017,  "2,017", "$\\leq$80 lines"),
              (  584,    "584", "covering test\nin container")]
    LOSSES = [(48417, "single or zero\nconcerns",      False, 0.000, "left"),
              (  237, "fully auto-fixable",            False, 0.055, "left"),
              (  629, "too long",                      False, 0.110, "left"),
              ( 1433, "no covering\ntest",              True, 0.055, "left")]
    counts = [c for c, _, _ in STAGES]
    for i, (lost, _, _, _, _) in enumerate(LOSSES):
        assert counts[i] - lost == counts[i+1], (counts[i], lost, counts[i+1])
    assert counts == [51300, 2883, 2646, 2017, 584]

    n = len(STAGES)
    fig, ax = plt.subplots(figsize=(3.33, 1.90))
    xs = [0.075, 0.315, 0.480, 0.635, 0.800]
    K = 0.150 / math.sqrt(counts[0])
    hw = [max(K * math.sqrt(c), 0.011) for c in counts]
    YC = 0.70

    def ease(t): return t * t * (3 - 2 * t)
    for i in range(n - 1):
        t = np.linspace(0, 1, 140)
        x = xs[i] + (xs[i+1] - xs[i]) * t
        h = hw[i] + (hw[i+1] - hw[i]) * ease(t)
        ax.fill_between(x, YC - h, YC + h, color=BLUE, alpha=0.58, lw=0, zorder=2)
    ax.fill_between([xs[-1], 0.900], [YC - hw[-1]]*2, [YC + hw[-1]]*2,
                    color=BLUE, alpha=0.58, lw=0, zorder=2)

    for i, (lost, lab, is_oracle, lift, ha) in enumerate(LOSSES):
        x = xs[i+1]
        col = CORAL_D if is_oracle else GREY
        drop = 0.17 - lift
        sgn = 1 if ha == "left" else -1
        ax.annotate("", xy=(x + 0.018*sgn, YC - hw[i+1] - drop),
                    xytext=(x + 0.003*sgn, YC - hw[i+1] - 0.012),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0 if is_oracle else 0.8,
                                    mutation_scale=7,
                                    connectionstyle=f"arc3,rad={-0.3*sgn}"), zorder=3)
        tx = x + 0.024*sgn
        ax.text(tx, YC - hw[i+1] - drop - 0.028, f"\u2212{lost:,}", fontsize=6.6,
                color=col, ha=ha, va="top", fontweight="bold")
        ax.text(tx, YC - hw[i+1] - drop - 0.118, lab, fontsize=5.5, color=col,
                ha=ha, va="top", linespacing=1.15)

    for k, ((c, num, lab), x, h) in enumerate(zip(STAGES, xs, hw)):
        ha = "left" if k == 0 else "center"
        tx = x - 0.072 if k == 0 else x
        ax.add_patch(Rectangle((x - 0.007, YC - h - 0.012), 0.014, 2*h + 0.024,
                               fc=BLUE_D, ec="none", zorder=4))
        ax.text(tx, YC + h + 0.045, num, fontsize=7.0, color=BLUE_D,
                ha=ha, va="bottom", fontweight="bold")
        ax.text(tx, YC + h + 0.155, lab, fontsize=5.5, color=INK,
                ha=ha, va="bottom", linespacing=1.15)

    ax.text(0.005, 0.265, "band width $\\propto\\sqrt{\\mathrm{count}}$",
            fontsize=5.6, color=GREY, style="italic", ha="left", va="bottom")
    ax.set_xlim(0.005, 0.995); ax.set_ylim(0.255, 1.13)
    ax.axis("off"); ax.grid(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig("figures/fig_pipeline.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig (arms)
def fig_arms():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    fig, ax = plt.subplots(figsize=(3.33, 2.42))
    XL, XP, WP, WG, GAP = 0.205, 0.225, 0.135, 0.085, 0.030
    ROWS = [
        ("select-one",     1, 1, "1.00", False, None),
        ("all concerns",   3, 1, "1.00", False, None),
        ("flat",           1, 3, "2.14", False, None),
        ("iterate",        1, 3, "1.52", False, "exit"),
        ("verifier-gated", 3, 2, "3.07", False, "revert"),
        ("localized",      3, 1, "1.00", True,  None),
    ]
    ys = [0.905 - i * 0.152 for i in range(6)]
    H = 0.082

    def box(x, y, w, fc, ec, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y - H/2), w, H,
                     boxstyle="round,pad=0.004,rounding_size=0.012",
                     fc=fc, ec=ec, lw=lw, zorder=3))

    def arrow(x0, x1, y, col=GREY, style="-|>", rad=0.0, lw=0.7, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle=style, color=col,
                     lw=lw, linestyle=ls, mutation_scale=6, zorder=2,
                     connectionstyle=f"arc3,rad={rad}"))

    for (name, chips, ngen, cost, loc, extra), y in zip(ROWS, ys):
        equal = cost == "1.00"
        ax.text(XL, y, name, fontsize=6.6, ha="right", va="center",
                color=CORAL_D if loc else INK, fontweight="bold" if loc else "normal")
        # prompt box with concern chips
        box(XP, y, WP, "#f4f7fb", BLUE_D if equal else GREY)
        for c in range(3):
            cx = XP + 0.022 + c * 0.020
            on = c < chips
            ax.add_patch(Circle((cx, y + 0.016), 0.0068,
                         fc=BLUE_D if on else "white", ec=BLUE_D if on else "#c9c9c9",
                         lw=0.5, zorder=4))
        ax.text(XP + WP/2, y - 0.019, "findings" if loc else "concerns",
                fontsize=5.2, ha="center", va="center",
                color=CORAL_D if loc else GREY)
        if loc:
            ax.add_patch(Rectangle((XP + 0.086, y + 0.007), 0.040, 0.018,
                         fc=CORAL, ec=CORAL_D, lw=0.5, zorder=4))
        x = XP + WP
        for g in range(ngen):
            arrow(x, x + GAP, y)
            box(x + GAP, y, WG, "#eef2f7", BLUE_D)
            ax.text(x + GAP + WG/2, y, "gen", fontsize=5.6, ha="center", va="center", color=BLUE_D)
            x = x + GAP + WG
        if extra == "exit":
            xg2 = XP + WP + 2*(GAP + WG) - WG/2          # centre of the 2nd gen box
            ax.add_patch(FancyArrowPatch((xg2, y + H/2 + 0.004),
                         (xg2 + 0.020, y + H/2 + 0.040), arrowstyle="-|>", color=SAGE_D,
                         lw=0.7, mutation_scale=5, zorder=5,
                         connectionstyle="arc3,rad=-0.2"))
            ax.text(xg2 + 0.026, y + H/2 + 0.044, "stop when clean", fontsize=5.2,
                    color=SAGE_D, ha="left", va="center")
        if extra == "revert":
            xa, xb = x - WG/2, XP + WP + GAP + WG/2
            ax.add_patch(FancyArrowPatch((xa, y - H/2 - 0.006), (xb, y - H/2 - 0.006),
                         arrowstyle="-|>", color=CORAL_D, lw=0.7, mutation_scale=5,
                         zorder=5, connectionstyle="arc3,rad=0.45"))
            ax.text((xa + xb)/2, y - H/2 - 0.052, "revert if not improved", fontsize=5.2,
                    color=CORAL_D, ha="center", va="center")
        arrow(x, x + 0.022, y)
        ax.text(0.985, y, cost, fontsize=6.6, ha="right", va="center",
                color=CORAL_D if equal else INK, fontweight="bold" if equal else "normal")

    ax.add_patch(Rectangle((0.955, ys[5] - 0.055), 0.001, 0.001, fc="none", ec="none"))
    ax.text(0.985, 0.985, "gen. / item", fontsize=5.8, ha="right", va="center", color=GREY)
    ax.text(XP, 0.985, "prompt contents", fontsize=5.8, ha="left", va="center", color=GREY)
    ax.annotate("", xy=(0.995, ys[0] + 0.048), xytext=(0.995, ys[5] - 0.048),
                arrowprops=dict(arrowstyle="-", color=LGREY, lw=0.6))
    ax.set_xlim(0, 1.0); ax.set_ylim(0.02, 1.03)
    ax.axis("off"); ax.grid(False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig("figures/fig_arms.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2
def fig_validation():
    rows = [("sphinx", 27, 33), ("pytest", 27, 32),
            ("scikit-learn", 41, 50), ("astropy", 79, 83)]
    fig, ax = plt.subplots(figsize=(3.33, 1.68))
    ys = range(len(rows))
    for y, (name, k, n) in zip(ys, rows):
        rate = 100*k/n
        lo, hi = wilson(k, n)
        ax.plot([lo, hi], [y, y], color=BLUE, lw=2.2, solid_capstyle="round",
                alpha=0.55, zorder=2)
        ax.scatter([rate], [y], s=18 + n*1.1, color=BLUE_D, zorder=3,
                   edgecolor="white", linewidth=0.7)
        ax.text(hi + 1.6, y, f"{rate:.1f}%", va="center", fontsize=6.5, color=INK)
        ax.text(hi + 8.2, y, f"n={n}", va="center", fontsize=6, color=GREY)
    ax.axvline(88.2, color=CORAL_D, lw=0.9, ls=(0, (3.5, 2)), zorder=1)
    ax.text(87.4, len(rows) - 0.38, "pooled\n88.2%", color=CORAL_D, fontsize=6.2,
            va="center", ha="right", linespacing=1.15)
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("behavioral pass rate of grader-correct predictions (%)")
    ax.set_xlim(60, 118); ax.set_ylim(-0.62, len(rows) - 0.18)
    ax.set_xticks([60, 70, 80, 90, 100]); ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    fig.tight_layout(pad=0.3); fig.savefig("figures/fig_validation.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3
def fig_ladder():
    x = [0, 1, 2]
    labels = ["DeepSeek 6.7B", "Qwen 14B", "Qwen 32B"]
    allc = [18.3, 39.6, 44.8]; flat = [20.0, 41.3, 42.2]; sel = [3.5, 11.3, 3.9]
    fig, ax = plt.subplots(figsize=(3.33, 2.10))
    ax.fill_between(x, sel, allc, color=BLUE, alpha=0.16, zorder=1, lw=0)
    ax.plot(x, flat, color=SAGE_D, lw=1.0, ls=(0, (3, 2)), marker="s", ms=3.2,
            mfc=SAGE, mec=SAGE_D, mew=0.7, zorder=3, label="flat, multi-pass")
    ax.plot(x, allc, color=BLUE_D, lw=1.5, marker="o", ms=4.0,
            mfc=BLUE, mec=BLUE_D, mew=0.8, zorder=4, label="all concerns, 1 gen")
    ax.plot(x, sel, color=CORAL_D, lw=1.5, marker="D", ms=3.8,
            mfc=CORAL, mec=CORAL_D, mew=0.8, zorder=4, label="select-one, 1 gen")
    for xi, a, sv in zip(x, allc, sel):
        ax.annotate(f"+{a-sv:.1f}", xy=(xi, (a + sv) / 2), fontsize=6.3, color=BLUE_D,
                    ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    ax.annotate("erratic, not flat", xy=(2, 3.9), xytext=(1.06, 15.5), fontsize=6.3,
                color=CORAL_D, ha="center",
                arrowprops=dict(arrowstyle="->", color=CORAL_D, lw=0.7,
                                connectionstyle="arc3,rad=-0.30"))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("items correct (%)"); ax.set_ylim(0, 50); ax.set_xlim(-0.22, 2.22)
    ax.set_axisbelow(True); ax.xaxis.grid(False)
    ax.legend(frameon=False, ncol=3, handlelength=1.5, columnspacing=1.0,
              handletextpad=0.4, borderpad=0.0, loc="lower center",
              bbox_to_anchor=(0.5, 1.00))
    fig.tight_layout(pad=0.25); fig.savefig("figures/fig_ladder.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4
def fig_imprecision():
    #           name            x      y    area   label offset   va
    pts = [("DeepSeek 6.7B",   55.9, 64.8,  3.5,  (0,  26), "bottom"),
           ("Qwen 14B",        41.9, 64.8, 11.3,  (0,  40), "bottom"),
           ("Qwen 32B",        34.6, 63.9,  3.9,  (0, -24), "top")]
    fig, ax = plt.subplots(figsize=(3.33, 2.25))
    ax.axhline(64.5, color=CORAL_D, lw=0.8, ls=(0, (3.5, 2)), zorder=1)
    ax.text(48.8, 65.05, "competence is flat", fontsize=6.2, color=CORAL_D,
            va="bottom", ha="center",
            bbox=dict(fc="white", ec="none", pad=0.9))
    ax.annotate("", xy=(39.5, 61.15), xytext=(57.0, 61.15),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9,
                                mutation_scale=8), zorder=1)
    ax.text(48.2, 61.38, "models sharpen", fontsize=6.2, color=GREY,
            style="italic", ha="center", va="bottom")
    for name, xi, yi, area, off, va in pts:
        ax.scatter([xi], [yi], s=area * 105, color=BLUE, alpha=0.42,
                   edgecolor=BLUE_D, linewidth=0.7, zorder=2)
        ax.scatter([xi], [yi], s=6, color=BLUE_D, zorder=3)
        ax.annotate(f"{name}\nfull-correct {area}%", xy=(xi, yi), xytext=off,
                    textcoords="offset points", ha="center", va=va,
                    fontsize=6.2, color=INK, linespacing=1.2)
    ax.set_xlabel("incidental resolution of untargeted concerns (%)")
    ax.set_ylabel("asked concern resolved (%)")
    ax.set_xlim(32, 61); ax.set_ylim(60.6, 69.4); ax.invert_xaxis()
    ax.set_yticks([61, 63, 65, 67]); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.25); fig.savefig("figures/fig_imprecision.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 6
def fig_composition():
    """All 21 concern combinations in the 584 items. Colour encodes how many categories the
    item carries, wedge size how often that exact combination occurs. Drawn at the paper's
    true 5.48in linewidth so labels render at their stated point size."""
    COMBOS = [  # (count, n_categories, label) -- grouped by n, descending within group
        (161, 2, "error-prone + outdated"),            (105, 2, "outdated + redundant"),
        ( 97, 2, "error-prone + redundant"),             ( 53, 2, "error-prone + collections"),
        ( 50, 2, "collections + outdated"),      ( 24, 2, "error-prone + dead code"),
        (  9, 2, "dead code + outdated"),            (  6, 2, "collections + redundant"),
        (  6, 2, "dead code + redundant"),             (  2, 2, "dead code + collections"),
        ( 25, 3, "error-prone + outdated + redundant"), ( 11, 3, "error-prone + collections + outdated"),
        ( 11, 3, "error-prone + collections + redundant"), (  6, 3, "error-prone + dead code + outdated"),
        (  3, 3, "collections + outdated + redundant"), (  3, 3, "error-prone + dead code + collections"),
        (  2, 3, "dead code + collections + outdated"), (  2, 3, "dead code + outdated + redundant"),
        (  2, 3, "error-prone + dead code + redundant"),
        (  5, 4, "error-prone + collections + outdated + redundant"),
        (  1, 4, "error-prone + dead code + outdated + redundant")]
    assert sum(c for c, _, _ in COMBOS) == 584
    assert len(COMBOS) == 21
    for n, want in ((2, 513), (3, 65), (4, 6)):
        assert sum(c for c, k, _ in COMBOS if k == n) == want

    # colour: hue = number of categories, lightness = rank within that group
    import matplotlib.colors as mc
    def ramp(base, k, n):
        r, g, b = mc.to_rgb(base)
        t = 0.0 if n == 1 else 0.62 * k / (n - 1)
        return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)
    groups = {2: BLUE_D, 3: CORAL_D, 4: SAGE_D}
    idx = {2: 0, 3: 0, 4: 0}; tot = {2: 10, 3: 9, 4: 2}
    colors = []
    for _, n, _ in COMBOS:
        colors.append(ramp(groups[n], idx[n], tot[n])); idx[n] += 1

    fig = plt.figure(figsize=(5.48, 1.98))
    ax  = fig.add_axes([0.004, 0.02, 0.325, 0.96]); ax.set_axis_off(); ax.grid(False)
    lg  = fig.add_axes([0.330, 0.01, 0.667, 0.97]); lg.set_axis_off(); lg.grid(False)

    ax.pie([c for c, _, _ in COMBOS], startangle=90, counterclock=False, colors=colors,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=0.7))
    ax.text(0, 0.13, "584", ha="center", va="center", fontsize=11, color=INK)
    ax.text(0, -0.14, "compound items", ha="center", va="center", fontsize=5.2, color=GREY)
    ax.text(0, -0.31, "21 combinations", ha="center", va="center", fontsize=5.2, color=GREY)

    # legend: short two-category labels left, longer three/four-category labels right
    lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    COLS = {2: (0.000, 0.365), 3: (0.395, 1.000), 4: (0.395, 1.000)}
    ROW, HDR = 0.0715, 0.080
    y = {2: 0.99, 3: 0.99, 4: None}
    for n, title in ((2, "two categories"), (3, "three categories"), (4, "four categories")):
        x0, xc = COLS[n]
        sub = sum(c for c, k, _ in COMBOS if k == n)
        if n == 4:
            y[4] = y[3] - 0.030
        yy = y[n]
        lg.text(x0, yy, f"{title}  \u00b7  {sub} ({100*sub/584:.1f}%)",
                fontsize=5.9, color=INK, va="top", ha="left", weight="bold")
        yy -= HDR
        for k, (c, kn, l) in enumerate(COMBOS):
            if kn != n: continue
            lg.add_patch(Rectangle((x0, yy - 0.040), 0.019, 0.036,
                                   fc=colors[k], ec="white", lw=0.4, clip_on=False))
            lg.text(x0 + 0.024, yy - 0.022, l, fontsize=5.2, color=INK, va="center", ha="left")
            lg.text(xc, yy - 0.022, f"{c}", fontsize=5.2, color=GREY, va="center", ha="right")
            yy -= ROW
        y[n] = yy
        if n == 3: y[3] = yy
    fig.savefig("figures/fig_composition.pdf"); plt.close(fig)

def fig_story():
    """Why the benchmark had to exist. Ways of prompting the SAME model are indistinguishable on
    ordinary one-problem code and 30.5 points apart on compound code. Table~5 (discrimination)."""
    # the two multi-pass variants (42.6 / 42.8) are merged -- they are one point apart and would
    # collide as labels, and the paper treats them as indistinguishable anyway
    ROWS = [("fix one problem at a time", 12.3, CORAL_D),
            ("name every problem",        39.2, SAGE_D),
            ("fix them one pass each",    42.7, BLUE_D)]
    SINGLE = 69.4
    assert round(max(r[1] for r in ROWS) - min(r[1] for r in ROWS), 1) == 30.4   # 12.3 -> 42.7

    fig = plt.figure(figsize=(5.48, 2.30))
    ax = fig.add_axes([0.075, 0.205, 0.62, 0.615]); ax.grid(False)
    for sp in ("top", "right", "bottom", "left"): ax.spines[sp].set_visible(False)
    XL, XR = 0.0, 1.0
    for name, b, col in ROWS:
        ax.plot([XL, XR], [SINGLE, b], color=col, lw=1.7, alpha=0.85, zorder=2)
        ax.scatter([XR], [b], s=26, color=col, zorder=4, edgecolor="white", linewidth=0.8)
        ax.text(XR + 0.10, b, f"{b:.1f}%   {name}", va="center", fontsize=6.6, color=col)
    ax.scatter([XL], [SINGLE], s=75, color=INK, zorder=5, edgecolor="white", linewidth=1.0)
    ax.text(XL - 0.07, SINGLE, "69.4%", ha="right", va="center", fontsize=7.4, color=INK, weight="bold")
    ax.text(XL - 0.07, SINGLE - 7.5, "every method\nlands here", ha="right", va="top",
            fontsize=6.3, color=GREY, linespacing=1.3)

    ax.annotate("", xy=(XR + 0.035, 12.3), xytext=(XR + 0.035, 42.7),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.text(XR - 0.04, 27.5, "30.4 points\napart", ha="right", va="center", fontsize=6.5,
            color=INK, linespacing=1.2)

    ax.set_xlim(-0.72, 2.95); ax.set_ylim(4, 80); ax.set_yticks([])
    ax.set_xticks([XL, XR]); ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xticklabels(["ordinary code", "our benchmark"], fontsize=6.8)
    fig.text(0.075, 0.945, "The same model, asked three different ways",
             fontsize=7.8, color=INK, weight="bold")
    fig.text(0.075, 0.870, "left, functions carrying one problem, which are 94% of real code.   "
                           "right, functions carrying two or more.", fontsize=6.4, color=GREY)
    fig.savefig("figures/fig_story.pdf"); plt.close(fig)

def fig_motivation():
    """Item 1's motivating example. Real astropy function (_notify_all, rules RET504 + RET505 + UP032),
    real Qwen2.5-14B outputs from bare_outputs_qwen14b.json and sweep_outputs_p15_loc.json. Code lines
    are the real ones, with the long error message abbreviated by an ellipsis."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    RED, GRN, AMB = "#C2665C", "#4E8C62", "#B8863A"
    fig = plt.figure(figsize=(5.48, 2.92))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-10.5, 100)
    ax.set_axis_off(); ax.grid(False)

    def chip(x, y, w, h):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                                    fc="#F7F9FB", ec=LGREY, lw=0.8, zorder=2))
    def mark(x, y, ok):
        if ok:
            ax.plot([x-1.0, x-0.25, x+1.2], [y+0.1, y-0.8, y+1.2], color=GRN, lw=1.6,
                    solid_capstyle="round", solid_joinstyle="round", zorder=5)
        else:
            for dx, dy in ((1, 1), (1, -1)):
                ax.plot([x-1.0*dx, x+1.0*dx], [y-1.0*dy, y+1.0*dy], color=RED, lw=1.6,
                        solid_capstyle="round", zorder=5)

    ax.text(1, 96, "One astropy function, three separate problems", fontsize=8.4,
            color=INK, weight="bold")
    ax.text(1, 91.2, "Same 14B model in both rewrites. Prompts quoted verbatim.",
            fontsize=6.5, color=GREY)

    COLS = [(48, "a variable used\nonly to return it"), (64, "an else that can\nnever be needed"),
            (79.5, "text built the\nold way")]
    for x, lab in COLS:
        ax.text(x, 86.5, lab, fontsize=6.2, color=AMB, weight="bold", ha="center",
                va="top", linespacing=1.25)
    ax.text(93, 84.6, "verdict", fontsize=6.4, color=GREY, weight="bold", ha="center")

    ROWS = [
        (58, "the real code", GREY, [
            'ids = self._notify_all_(key, msg)',
            'return ids',
            'else:',
            '    raise Error("... {} ...".format(key))'], (False, False, False), None),
        (31, "PROMPT: \u201cImprove the code quality of this function\u201d", RED, [
            'ids = self._notify_all_(key, msg)',
            'return ids',
            '',
            'raise Error(f"... {key} ...")'], (False, True, True), "rejected"),
        (4, "PROMPT: + the three findings, with line numbers", GRN, [
            'return self._notify_all_(key, msg)',
            '',
            '',
            'raise Error(f"... {key} ...")'], (True, True, True), "accepted"),
    ]
    for y, label, col, code, oks, verdict in ROWS:
        ax.text(1, y + 17.5, label, fontsize=6.4, color=col, weight="bold")
        chip(1, y, 42, 16.2)
        for i2, ln in enumerate(code):
            if ln:
                ax.text(2.4, y + 13.2 - i2*3.7, ln, fontsize=5.5, family="monospace",
                        color=INK, va="center")
        for (x, _), ok in zip(COLS, oks):
            mark(x, y + 8.0, ok)
        if verdict:
            ax.text(93, y + 8.0, verdict, fontsize=6.8, ha="center", va="center",
                    color=GRN if all(oks) else RED, weight="bold")
        else:
            ax.text(93, y + 8.0, "starting point", fontsize=6.2, ha="center",
                    va="center", color=GREY)

    ax.annotate("", xy=(48, 26.0), xytext=(48, 18.5),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9, mutation_scale=8))
    ax.text(50.2, 22.2, "the only difference", fontsize=5.9, color=GREY, va="center")

    ax.add_patch(Rectangle((1, -9.5), 98, 8.4, fc="#FBF3EE", ec=CORAL, lw=0.7, zorder=1))
    ax.text(3.2, -3.4, "It removed the needless else and modernised the text, then stopped.",
            fontsize=6.5, color=INK, va="center", weight="bold")
    ax.text(3.2, -6.9, "On 43.2% of our 584 items the model does this, fixing some concerns and leaving others.",
            fontsize=6.5, color=INK, va="center")
    fig.savefig("figures/fig_motivation.pdf"); plt.close(fig)

def fig_bands():
    # SE of the paired difference = sqrt(b+c)/n  (McNemar), computed from the P16 run
    rows = [("targeting", 26.9, 20.2, 2.02),
            ("localization", 20.7, 24.0, 2.27)]
    fig, ax = plt.subplots(figsize=(3.33, 1.62))
    for y, (name, ours, theirs, se) in zip([1, 0], rows):
        ax.plot([min(ours, theirs), max(ours, theirs)], [y, y], color=LGREY, lw=3.0,
                solid_capstyle="round", zorder=1)
        ax.plot([theirs - 1.96*se, theirs + 1.96*se], [y, y], color=BLUE_D, lw=1.0, zorder=3)
        ax.scatter([ours], [y], marker="D", s=30, facecolor="white",
                   edgecolor=CORAL_D, linewidth=1.2, zorder=4)
        ax.scatter([theirs], [y], marker="o", s=30, color=BLUE_D, zorder=5,
                   edgecolor="white", linewidth=0.6)
        ax.annotate(f"{theirs:+.1f}", xy=(theirs, y), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=6.3, color=BLUE_D)
        ax.annotate(f"{ours:+.1f}", xy=(ours, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=6.3, color=CORAL_D)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["localization", "targeting"])
    ax.set_xlabel("gain over the arm below it (percentage points)")
    ax.set_xlim(14, 32); ax.set_ylim(-0.52, 1.42)
    ax.set_axisbelow(True); ax.yaxis.grid(False)
    h = [plt.Line2D([], [], marker="D", ls="", mfc="white", mec=CORAL_D, mew=1.2, ms=4.6),
         plt.Line2D([], [], marker="o", ls="", color=BLUE_D, ms=4.6)]
    ax.legend(h, ["ours", "independent corpus"], frameon=False, fontsize=6.2,
              ncol=2, loc="lower center", handletextpad=0.4, borderpad=0.0,
              columnspacing=1.4, bbox_to_anchor=(0.5, 1.00))
    fig.tight_layout(pad=0.25); fig.savefig("figures/fig_bands.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 7
def fig_costbenefit():
    pts = [("select-one",     1.00, 12.3, (8, 0),    "left"),
           ("all concerns",   1.00, 39.2, (9, -7),   "left"),
           ("localized",      1.00, 59.9, (8, 0),    "left"),
           ("iterate",        1.52, 42.8, (0, 9),    "center"),
           ("flat",           2.14, 42.6, (0, -13),  "center"),
           ("verifier-gated", 3.07, 45.9, (0, 9),    "center")]
    fig, ax = plt.subplots(figsize=(3.33, 2.15))
    ax.axvspan(0.93, 1.07, color=BLUE, alpha=0.13, zorder=0, lw=0)
    ax.plot([1.42, 3.35], [59.9, 59.9], color=SAGE_D, lw=0.8, ls=(0, (3.5, 2)), zorder=1)
    ax.text(3.35, 61.2, "Pareto frontier", fontsize=6.0, color=SAGE_D, ha="right")
    for name, x, y, off, ha in pts:
        best = name == "localized"
        ax.scatter([x], [y], s=52 if best else 30,
                   color=CORAL if best else BLUE,
                   edgecolor=CORAL_D if best else BLUE_D,
                   linewidth=0.9, zorder=3)
        ax.annotate(name, xy=(x, y), xytext=off, textcoords="offset points",
                    fontsize=6.4, ha=ha, va="center",
                    color=CORAL_D if best else INK,
                    fontweight="bold" if best else "normal")
    ax.annotate("", xy=(1.02, 12.3), xytext=(1.02, 59.9),
                arrowprops=dict(arrowstyle="<->", color=BLUE_D, lw=0.7,
                                shrinkA=4, shrinkB=4), zorder=2)
    ax.text(1.09, 24.5, "47 points\nat one generation", fontsize=6.3, color=BLUE_D,
            ha="left", va="center", linespacing=1.2)
    ax.set_xlabel("generations per item"); ax.set_ylabel("items correct (%)")
    ax.set_xlim(0.84, 3.62); ax.set_ylim(4, 70)
    ax.set_xticks([1, 1.5, 2, 2.5, 3]); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.25); fig.savefig("figures/fig_costbenefit.pdf"); plt.close(fig)



# ---------------------------------------------------------------- Skeleton
def fig_skeleton():
    """LAYOUT SKELETON for the motivating figure. Not final art. Every number here is measured;
    the boxes and arrows are placeholders showing WHERE each beat of the argument sits so the
    panel can be redrawn properly. Four beats: the artifact, what models do with it, why you
    cannot measure it on ordinary code, what this paper builds."""
    fig = plt.figure(figsize=(5.48, 5.15))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.16,
                          left=0.015, right=0.985, top=0.925, bottom=0.015)
    def panel(g, tag, title):
        ax = fig.add_subplot(g); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(LGREY); sp.set_linewidth(0.6)
        ax.add_patch(Rectangle((0.15, 8.75), 0.62, 0.85, fc=ACC, ec="none", zorder=3))
        ax.text(0.46, 9.17, tag, fontsize=7, color="white", ha="center", va="center",
                fontweight="bold", zorder=4)
        ax.text(1.0, 9.17, title, fontsize=7.4, color=INK, ha="left", va="center", fontweight="bold")
        return ax
    def box(ax, x, y, w, h, fc, label, sub=None, fs=6.2):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=GREY, lw=0.45, zorder=2))
        ax.text(x + w/2, y + h*(0.66 if sub else 0.5), label, fontsize=fs, color=INK,
                ha="center", va="center", zorder=3, linespacing=1.15)
        if sub:
            ax.text(x + w/2, y + h*0.18, sub, fontsize=5.6, color="#6b6b6b",
                    ha="center", va="center", zorder=3, linespacing=1.15)

    # ---- A. the artifact -------------------------------------------------
    ax = panel(gs[0, 0], "A", "One real function")
    box(ax, 0.6, 4.3, 8.8, 3.6, "#f7f7f7", "", None)
    ax.text(1.0, 7.45, "astropy  _notify_all()", fontsize=6.0, color="#6b6b6b", ha="left", va="center")
    for i, (c, lab) in enumerate([(CORAL, "dead code"), (SAND, "dead branch"),
                                  (MAUVE, "old syntax")]):
        y = 6.55 - i*0.85
        ax.add_patch(Rectangle((1.0, y - 0.2), 4.6, 0.42, fc=c, ec="none", alpha=0.55, zorder=3))
        ax.text(5.85, y, lab, fontsize=6.0, color=INK, ha="left", va="center", zorder=3)
    ax.text(5.0, 3.45, "3 concerns  ·  1 function  ·  its own test is green",
            fontsize=6.3, color=INK, ha="center", va="center")
    ax.text(5.0, 2.35, "REPLACE WITH: clean code panel,\nthree problems circled in three colors",
            fontsize=5.6, color=ACC2, ha="center", va="center", style="italic", linespacing=1.3)
    ax.text(5.0, 0.85, "Nothing is broken. The test passes before and after.",
            fontsize=6.0, color=INK, ha="center", va="center")

    # ---- B. what models do ----------------------------------------------
    ax = panel(gs[0, 1], "B", "What models do with it")
    ax.text(0.5, 8.15, "ask plainly", fontsize=6.4, color=INK, ha="left", va="center", fontweight="bold")
    for x, w, c, lab in [(0.5, 2.05, SAGE, "21.9%"), (2.55, 4.05, SAND, "43.2%"), (6.6, 2.9, CORAL, "32.7%")]:
        ax.add_patch(Rectangle((x, 6.95), w, 0.82, fc=c, ec="none", zorder=3))
        ax.text(x + w/2, 7.36, lab, fontsize=6.0, color=INK, ha="center", va="center", zorder=4)
    ax.text(0.5, 6.35, "all fixed          some left behind          none",
            fontsize=5.5, color="#6b6b6b", ha="left", va="center")
    ax.text(0.5, 5.35, "ask one concern at a time", fontsize=6.4, color=INK,
            ha="left", va="center", fontweight="bold")
    ax.text(0.5, 4.70, "(the design shipped by SE mixture-of-experts systems)",
            fontsize=5.5, color="#6b6b6b", ha="left", va="center")
    box(ax, 0.5, 3.15, 4.3, 1.25, "#f2f2f2", "obeys exactly\n36.6%", None, fs=6.0)
    box(ax, 5.2, 3.15, 4.3, 1.25, "#f2f2f2", "scores\n12.3%", None, fs=6.0)
    ax.annotate("", xy=(5.15, 3.78), xytext=(4.85, 3.78),
                arrowprops=dict(arrowstyle="->", color=CORAL_D, lw=0.8))
    ax.text(5.0, 2.55, "obeying the instruction IS failing",
            fontsize=6.4, color=CORAL_D, ha="center", va="center", fontweight="bold")
    ax.text(5.0, 1.75, "every item has $\\geq$2 concerns, so a model that\n"
                       "does exactly what it was told scores zero",
            fontsize=5.7, color=INK, ha="center", va="center", linespacing=1.3)
    ax.text(5.0, 0.55, "REPLACE WITH: the two prompts side by side,\nwith their real outputs",
            fontsize=5.5, color=ACC2, ha="center", va="center", style="italic", linespacing=1.3)

    # ---- C. why it cannot be measured on ordinary code -------------------
    ax = panel(gs[1, 0], "C", "Why ordinary code cannot show this")
    ax.add_patch(plt.Circle((2.5, 6.0), 1.55, fc=LGREY, ec=GREY, lw=0.5, zorder=2))
    ax.add_patch(mpl.patches.Wedge((2.5, 6.0), 1.55, 90, 90 + 0.06*360, fc=ACC, ec="none", zorder=3))
    ax.text(2.5, 6.0, "94%", fontsize=7.2, color=INK, ha="center", va="center",
            fontweight="bold", zorder=4)
    ax.text(2.5, 3.95, "of real flagged functions\ncarry only ONE concern",
            fontsize=5.9, color=INK, ha="center", va="center", linespacing=1.25)
    box(ax, 5.3, 6.55, 4.2, 1.75, "#f2f2f2", "4 methods on\n1-concern code", "spread 0.0 pt", fs=6.0)
    box(ax, 5.3, 4.35, 4.2, 1.75, CORAL, "same 4 on\ncompound code", "spread 30.5 pt", fs=6.0)
    ax.text(5.0, 2.75, "A benchmark you could HARVEST is 94% inert.",
            fontsize=6.3, color=INK, ha="center", va="center", fontweight="bold")
    ax.text(5.0, 1.85, "It cannot tell these methods apart at all.",
            fontsize=6.0, color=INK, ha="center", va="center")
    ax.text(5.0, 0.75, "REPLACE WITH: two flat lines vs a fan of lines",
            fontsize=5.5, color=ACC2, ha="center", va="center", style="italic")

    # ---- D. what the paper builds ---------------------------------------
    ax = panel(gs[1, 1], "D", "So the artifact had to be built")
    for i, (lab, sub) in enumerate([("584 compound functions", "9 repositories, every item $\\geq$2 concerns"),
                                    ("a test per item", "green at that commit, in an era-matched container"),
                                    ("a measured grader", "13.0% false-accept rate, not assumed sound")]):
        y = 7.25 - i*1.45
        box(ax, 0.5, y, 9.0, 1.15, "#f4f7fa", lab, sub, fs=6.2)
    ax.annotate("", xy=(5.0, 2.75), xytext=(5.0, 3.35),
                arrowprops=dict(arrowstyle="->", color=ACC, lw=1.0))
    box(ax, 0.5, 1.35, 9.0, 1.3, SAGE, "the gaps become visible", "12.3% $\\rightarrow$ 39.2% $\\rightarrow$ 59.9%  at one generation", fs=6.6)
    ax.text(5.0, 0.55, "REPLACE WITH: the three bars growing, left to right",
            fontsize=5.5, color=ACC2, ha="center", va="center", style="italic")

    fig.text(0.5, 0.975, "SKELETON  ·  layout draft for the motivating figure  ·  numbers are measured",
             fontsize=6.4, color=ACC2, ha="center", va="center", fontweight="bold")
    fig.savefig("figures/fig_skeleton.pdf")
    fig.savefig("/tmp/skel.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Motivating example
def fig_example():
    """The motivating example, from real data. astropy get_ref carries three concerns in three
    categories and the zero-shot 14B model fixes exactly one. Code is verbatim except that the
    four-line docstring is collapsed to one, which touches no flagged line. Rule identities and
    columns come from ruff; line numbers are as displayed here."""
    FS, CW, X0, NCH, LH = 7.0, 0.9227, 6.0, 84, 3.30   # CW measured for monospace at FS
    BEFORE = [
        "def get_ref(self):",
        '    """Lookup the Field instance that this FieldRef references."""',
        "    for field in self._table._votable.iter_fields_and_params():",
        "        if isinstance(field, Field) and field.ID == self.ref:",
        "            return field",
        "    vo_raise(\"No field named '{}'\".format(self.ref),",
        "             self._config, self._pos, KeyError)",
    ]
    AFTER = list(BEFORE); AFTER[5] = "    vo_raise(f\"No field named '{self.ref}'\","
    C1, C2, C3 = CORAL, SAND, MAUVE
    S2, S3 = (3, 40, 60), (5, 13, 51)
    fig = plt.figure(figsize=(5.48, 4.86))
    ax = fig.add_axes([0.012, 0.008, 0.976, 0.935])
    ax.set_xlim(0, NCH); ax.set_ylim(0, 100); ax.axis("off")
    mono = dict(family="monospace", fontsize=FS, va="center")

    def badge(x, y, n, col):
        ax.add_patch(plt.Circle((x, y), 1.32, fc=col, ec="none", zorder=5))
        ax.text(x, y, n, fontsize=5.5, color="white", ha="center", va="center",
                fontweight="bold", zorder=6)

    def block(ytop, lines, spans, fixed_row=None):
        ax.add_patch(Rectangle((X0-1.7, ytop-(len(lines)-1)*LH-LH*0.5), 0.5,
                               len(lines)*LH, fc=C1, ec="none", zorder=3))
        if fixed_row is not None:
            ax.add_patch(Rectangle((X0-0.4, ytop-fixed_row*LH-LH*0.42), NCH-X0-2.5,
                                   LH*0.84, fc=SAGE, ec="none", alpha=0.45, zorder=1))
        for row, c0, c1, col in spans:
            ax.add_patch(Rectangle((X0+c0*CW, ytop-row*LH-LH*0.42), (c1-c0)*CW,
                                   LH*0.84, fc=col, ec="none", alpha=0.6, zorder=2))
        for i, ln in enumerate(lines):
            ax.text(X0-2.7, ytop-i*LH, str(i+1), fontsize=5.5, color="#a5a5a5",
                    ha="right", va="center")
            ax.text(X0, ytop-i*LH, ln, color=INK, zorder=4, **mono)
        return ytop-(len(lines)-1)*LH

    ax.text(NCH/2, 97.3, "One function. Three problems. The model fixes one.",
            fontsize=8.8, color=INK, ha="center", va="center", fontweight="bold")
    ax.text(NCH/2, 93.4, "astropy/io/votable/tree.py line 1779. The output is the 14B model's, unedited.",
            fontsize=6.0, color="#6b6b6b", ha="center", va="center")

    ax.text(0.4, 88.8, "BEFORE", fontsize=6.8, color=INK, fontweight="bold", va="center")
    t1 = 84.8
    y = block(t1, BEFORE, [(S2[0], S2[1], S2[2], C2), (S3[0], S3[1], S3[2], C3)])
    for n, row, col in [("1", 0, C1), ("2", S2[0], C2), ("3", S3[0], C3)]:
        badge(1.2, t1-row*LH, n, col)

    rows = [("1", C1, "RET503", "error-prone",
             "line 5 hands back a field, but when nothing matches",
             "the function just ends and returns None instead."),
            ("2", C2, "SIM300", "redundant logic",
             "the comparison is back to front. It should read",
             "self.ref == field.ID, i.e. the subject first."),
            ("3", C3, "UP032", "outdated syntax",
             '"...{}".format(x) is the old way to build a string.',
             'Modern Python writes f"...{x}".')]
    yy = y - 5.0
    for n, col, rule, cat, l1, l2 in rows:
        badge(1.2, yy, n, col)
        ax.text(3.3, yy, rule, fontsize=6.4, color=INK, va="center", fontweight="bold",
                family="monospace")
        ax.text(13.8, yy, cat, fontsize=6.3, color="#6b6b6b", va="center", style="italic")
        ax.text(29.5, yy, l1, fontsize=6.4, color=INK, va="center")
        ax.text(29.5, yy-2.5, l2, fontsize=6.4, color=INK, va="center")
        yy -= 5.9

    yy -= 1.1
    ax.text(0.4, yy, "AFTER", fontsize=6.8, color=INK, fontweight="bold", va="center")
    ax.text(9.2, yy, "handed only the function, asked to improve its quality",
            fontsize=6.0, color="#6b6b6b", va="center")
    t2 = yy - 4.2
    y2 = block(t2, AFTER, [(S2[0], S2[1], S2[2], C2)], fixed_row=5)
    ax.text(NCH-1.0, t2-5*LH, "fixed", fontsize=5.8, color=SAGE_D, ha="right",
            va="center", fontweight="bold")
    for n, row, col in [("1", 0, C1), ("2", S2[0], C2)]:
        badge(1.2, t2-row*LH, n, col)
    ax.text(0.4, y2-4.6, "Two of the three are still there. The linter is still red.",
            fontsize=7.4, color=CORAL_D, va="center", fontweight="bold")
    fig.savefig("figures/fig_example.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("/tmp/example.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); fig_arms(); fig_validation(); fig_ladder(); fig_imprecision(); fig_bands()
    fig_costbenefit(); fig_composition(); fig_story(); fig_motivation(); fig_skeleton(); fig_example()
    print("wrote figures/*.pdf")
