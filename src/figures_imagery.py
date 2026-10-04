"""Figures for the imagery-only analyses (window 2-3.5 s: after the cue, before any feedback)."""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
INK, INK2, MUTED, GRID, AXIS, BG = "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#55554f", "#000000"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 13, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "legend.frameon": False,
                     "legend.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 14, "axes.titlelocation": "left"})
D = json.load(open("results/deeper.json"))

# 4 ── imagery-only decoding, session 1 -> session 25
f, ax = plt.subplots(figsize=(4.3, 4.4))
for p, col, dy in (("P1", INK2, 0), ("P2", INK2, 0), ("P3", INK2, 0)):
    a, b = D[f"{p}_pre"]["imagery_tt"], D[f"{p}_post"]["imagery_tt"]
    ax.plot([0, 1], [a, b], color=col, lw=2.4, marker="o", ms=9, mec=BG, mew=2, solid_capstyle="round")
    ax.text(-0.08, a + (1.6 if p == "P1" else -1.6 if p == "P3" else 0), f"{p}  {a:.0f}", ha="right", va="center", fontsize=12, color=INK)
    ax.text(1.08, b + (-1.2 if p == "P3" else 1.2 if p == "P2" else 0), f"{b:.0f}", ha="left", va="center", fontsize=12, color=INK)
ax.axhline(50, color=AXIS, lw=1); ax.text(0.5, 47.5, "chance", ha="center", fontsize=10, color=MUTED)
ax.set_xlim(-0.45, 1.3); ax.set_ylim(44, 100); ax.set_xticks([0, 1]); ax.set_xticklabels(["session 1", "session 25"]); ax.grid(axis="x", visible=False)
ax.set_ylabel("accuracy from imagery alone (%)")
f.savefig("figures/4_imagery_only.png", dpi=220, bbox_inches="tight"); plt.close(f)

# 5 ── which hemisphere responds? beta ERD, imagery-only window
f, axs = plt.subplots(1, 3, figsize=(9.4, 4.4), sharey=True); w = 0.36
for ax, p in zip(axs, ("P1", "P2", "P3")):
    groups = [(f"{p}_pre", "left"), (f"{p}_pre", "right"), (f"{p}_post", "left"), (f"{p}_post", "right")]
    xs = np.array([0, 1, 2.5, 3.5])
    for k, (ch, col) in enumerate((("C3", BLUE), ("C4", ORANGE))):
        v = np.array([D[s][f"erd|imagery|{h}"][ch] for s, h in groups])
        ax.bar(xs + (k - 0.5) * w, v[:, 0], w - 0.04, color=col, label=f"{ch}: {'left' if ch == 'C3' else 'right'} motor cortex")
        ax.errorbar(xs + (k - 0.5) * w, v[:, 0], [v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=INK2, lw=1.2)
    ax.axhline(0, color=AXIS, lw=1); ax.set_xticks(xs); ax.set_xticklabels(["left\nhand", "right\nhand"] * 2); ax.grid(axis="x", visible=False); ax.set_title(p)
    ax.text(0.5, 1.15, "session 1", ha="center", fontsize=11, color=INK2); ax.text(3.0, 1.15, "session 25", ha="center", fontsize=11, color=INK2)
    ax.axvline(1.75, color=AXIS, lw=1)
axs[0].set_ylim(-4.2, 1.5); axs[0].set_ylabel("beta power change (dB)\nmore negative = more engaged")
axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, fontsize=11, handlelength=1.2)
f.savefig("figures/5_hemisphere.png", dpi=220, bbox_inches="tight"); plt.close(f)
