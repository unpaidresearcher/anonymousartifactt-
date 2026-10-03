import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# ── Colors ──
TEXT   = "#201f1d"   # ink
N300   = "#d7d3d3"   # gridlines
N500   = "#9b9797"   # baseline dots, axis
N600   = "#7d7979"   # n labels, axis arrow
N700   = "#605d5d"   # tick labels
FRAME  = "#bab6b6"   # dotted rounded outer border
ARMS = {  # name: (dot color, value-label color)
    "Select-one":   ("#c26e12", "#904700"),
    "One-pass-all": ("#868604", "#5a5a00"),
    "Flat":         ("#008f77", "#00614d"),
    "Iterate":      ("#007daa", "#005681"),
}
BASE = ("Ruff Native Repair", N500, N700)

# ── Data (repair accuracy, %) ──
compound = {"base": 0.0, "Select-one": 12.3, "One-pass-all": 39.2, "Flat": 42.6, "Iterate": 42.8}
control  = {"base": 22.6, "Select-one": 69.4, "One-pass-all": 69.4, "Flat": 69.4, "Iterate": 69.4}

fig, ax = plt.subplots(figsize=(8.6, 1.9), dpi=300)
Y_C, Y_S = 1.0, 0.0            # row positions
ax.set_xlim(-24, 92); ax.set_ylim(-0.55, 1.55)
ax.axis("off")

# gridlines + axis
for x in (0, 25, 50, 75):
    ax.plot([x, x], [-0.35, 1.45], color=N300, lw=0.8, ls=(0, (1, 2.5)), zorder=0)
    ax.plot([x, x], [-0.42, -0.46], color=N500, lw=0.8)
    ax.text(x, -0.55, f"{x}%", ha="center", va="top", color=N700, fontsize=9.5)
ax.plot([0, 75], [-0.42, -0.42], color=N500, lw=0.8)
ax.text(-24, -0.55, "Repair Accuracy", va="top", color=N600, fontsize=9.5)
ax.annotate("", xy=(-2, -0.62), xytext=(-9.5, -0.62),
            arrowprops=dict(arrowstyle="-|>", color=N600, lw=1, mutation_scale=8))

# row labels
ax.text(-24, Y_C + 0.02, "Compound Items", fontweight="bold", fontsize=11.5, color=TEXT)
ax.text(-24, Y_C - 0.14, "n = 584", fontsize=9.5, color=N600)
ax.text(-24, Y_S + 0.20, "Single-Concern", fontweight="bold", fontsize=11.5, color=TEXT)
ax.text(-24, Y_S - 0.02, "Control", fontweight="bold", fontsize=11.5, color=TEXT)
ax.text(-24, Y_S - 0.22, "n = 186", fontsize=9.5, color=N600)

dot = dict(s=46, edgecolors="white", linewidths=1.1, zorder=3)

# compound row (Flat/Iterate nudged ±0.07 vertically so they don't overlap)
ax.scatter(compound["base"], Y_C, c=BASE[1], **dot)
ax.text(compound["base"], Y_C + 0.13, "0.0", ha="center", color=BASE[2], fontsize=9.5)
off = {"Select-one": 0, "One-pass-all": 0, "Flat": +0.11, "Iterate": -0.11}
for k, (c, tc) in ARMS.items():
    x = compound[k]; y = Y_C + off[k]
    ax.scatter(x, y, c=c, **dot)
    if k in ("Flat", "Iterate"):
        ax.text(x + 2.3, y, f"{x}", va="center", color=tc, fontsize=9.5)
    else:
        ax.text(x, Y_C + 0.2, f"{x}", ha="center", color=tc, fontsize=9.5)

# 30.5 pt spread: dotted double-headed arrow, exact endpoints
lo, hi = compound["Select-one"], compound["Iterate"]
ax.add_patch(FancyArrowPatch((lo, Y_C - 0.28), (hi, Y_C - 0.28), arrowstyle="<|-|>",
             mutation_scale=9, color=TEXT, lw=1.1, ls=(0, (1.5, 2.5))))
ax.text((lo + hi) / 2, Y_C - 0.36, f"{hi - lo:.1f} pt Spread", ha="center", va="top",
        fontweight="bold", fontsize=10.5, color=TEXT)

# control row
ax.scatter(control["base"], Y_S, c=BASE[1], **dot)
ax.text(control["base"] + 1.5, Y_S - 0.05, "22.6", va="top", color=BASE[2], fontsize=9.5)
for i, (k, (c, _)) in enumerate(ARMS.items()):      # stacked column at 69.4
    ax.scatter(69.4, Y_S + 0.24 - i * 0.16, c=c, s=40, edgecolors="white", linewidths=1, zorder=3)
ax.text(72, Y_S + 0.12, "69.4", fontweight="bold", fontsize=10.5, color=TEXT)
ax.text(72, Y_S - 0.12, "0.0 pt Spread", fontsize=9.5, color=N700)

# legend on top
handles = [plt.Line2D([], [], ls="", marker="o", ms=7, mfc=BASE[1], mec="none", label=BASE[0])] + \
          [plt.Line2D([], [], ls="", marker="o", ms=7, mfc=c, mec="none", label=k) for k, (c, _) in ARMS.items()]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=10,
           handletextpad=0.4, columnspacing=1.6, bbox_to_anchor=(0.5, 1.02))

# dotted rounded outer border
from matplotlib.patches import FancyBboxPatch
fig.patches.append(FancyBboxPatch((0.005, 0.01), 0.99, 0.98, transform=fig.transFigure,
    boxstyle="round,pad=0,rounding_size=0.03", fill=False, ec=FRAME, lw=1.2, ls=(0, (1, 2))))

plt.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.12)
plt.savefig("figures/spread.pdf", bbox_inches="tight")
