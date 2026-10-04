"""Decoding figures (accuracy, zero calibration, time-resolved) -> figures/*.png, dark theme to match the slide template."""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
sys.path.insert(0, "src"); from mi import *

BLUE, ORANGE, AQUA, RED = "#3987e5", "#d95926", "#199e70", "#e66767"
INK, INK2, MUTED, GRID, AXIS, BG = "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#55554f", "#000000"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 13, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 1, "axes.axisbelow": True, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "legend.labelcolor": INK2,
                     "legend.frameon": False, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 14, "axes.titlelocation": "left"})
LAB = [s.replace("_", "\n") for s in SESSIONS] + ["mean"]
save = lambda f, n: (f.savefig(f"figures/{n}.png", dpi=220, bbox_inches="tight"), plt.close(f))

def bars(ax, series, ylim=(40, 100), label_last=True):
    n, w = len(series), 0.8 / len(series)
    for i, (name, vals, col) in enumerate(series):
        x = np.arange(len(vals)) + (i - (n - 1) / 2) * w
        ax.bar(x, np.array(vals) - ylim[0], w - 0.04, bottom=ylim[0], color=col, label=name)
        if label_last or i == n - 1:
            for xi, v in zip(x, vals):
                if i == n - 1 or xi == x[-1]:
                    ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(LAB))); ax.set_xticklabels(LAB); ax.set_ylim(*ylim); ax.grid(axis="x", visible=False)
    ax.axvline(len(LAB) - 1.5, color=AXIS, lw=1); ax.set_ylabel("test-run accuracy (%)")

wm = lambda v: list(v) + [float(np.mean(v))]
A = json.load(open("results/adapt.json"))

# 1 ── accuracy vs reference
f, ax = plt.subplots(figsize=(9.2, 4.3))
bars(ax, [("g.tec reference: CSP + LDA", wm([REF[s][0] for s in SESSIONS]), MUTED),
          ("g.tec reference: PCA + TVLDA", wm([REF[s][1] for s in SESSIONS]), ORANGE),
          ("ours: filter-bank Riemannian + online recentering", wm([A[s]["within | FB | online recentering"] for s in SESSIONS]), BLUE)], label_last=False)
ax.axhline(50, color=AXIS, lw=1); 
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=11, handlelength=1.2); ax.set_ylim(40, 104)
save(f, "1_accuracy")

# 2 ── zero calibration
f, ax = plt.subplots(figsize=(9.2, 4.3))
bars(ax, [("calibrated today: CSP + LDA reference (80 training trials)", wm([REF[s][0] for s in SESSIONS]), MUTED),
          ("zero calibration: model from the patient's other session", wm([A[s]["zero-calib | FB | same patient, other session | online"] for s in SESSIONS]), AQUA)], label_last=False)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=11, handlelength=1.2); ax.set_ylim(40, 104)
save(f, "2_zero_calibration")

# 3 ── time-resolved decoding per band (CSP+LDA, 1 s windows, train run -> test run), mean over 6 sessions
bands = {"beta 13–30 Hz": ((13, 30), BLUE), "mu 8–13 Hz": ((8, 13), ORANGE), "55–95 Hz (muscle / stimulation artefact range)": ((55, 95), MUTED)}
starts = np.arange(-1.0, 7.01, 0.25); acc = {k: [] for k in bands}
for s in SESSIONS:
    Xtr, ytr, fs = load_bank(s + "_training", [b for b, _ in bands.values()]); Xte, yte, _ = load_bank(s + "_test", [b for b, _ in bands.values()])
    for bi, k in enumerate(bands):
        row = []
        for st in starts:
            w = sl(fs, st, st + 1); cov = lambda X: Covariances("oas").transform(X[:, bi, :, w])
            row.append(make_pipeline(CSP(4, log=True), LinearDiscriminantAnalysis()).fit(cov(Xtr), ytr).score(cov(Xte), yte) * 100)
        acc[k].append(row)
json.dump({"window_end": (starts + 1).tolist(), **{k: np.array(v).tolist() for k, v in acc.items()}}, open("results/time_resolved.json", "w"))
f, ax = plt.subplots(figsize=(9.2, 4.3)); t = starts + 1                       # plot at window END: causal reading
ax.axvspan(3.5, 8, color=GRID, alpha=0.9, lw=0); ax.axvline(2, color=INK2, lw=1)
ax.text(2.06, 97, "cue", fontsize=11, color=INK2, va="top"); ax.text(3.58, 97, "FES + visual feedback on", fontsize=11, color=INK2, va="top")
for k, (_, col) in bands.items():
    m = np.mean(acc[k], 0); ax.plot(t, m, color=col, lw=2.2, label=k, solid_capstyle="round")
ax.axhline(50, color=AXIS, lw=1); ax.set_xlim(0, 8); ax.set_ylim(40, 100); ax.grid(axis="x", visible=False)
ax.set_xlabel("time in trial (s) — end of the 1 s analysis window"); ax.set_ylabel("test-run accuracy (%)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=11)
save(f, "3_time_resolved")

print("ok")
