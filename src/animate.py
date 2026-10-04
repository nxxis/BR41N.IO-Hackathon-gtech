"""Animation for the video: P1's beta desynchronisation map through the trial, session 1 vs session 25."""
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, mne
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert
sys.path.insert(0, "src"); from mi import *
mne.set_log_level("ERROR")
BLUE, RED, INK, INK2, MUTED, GRID, BG, LIME = "#3987e5", "#e66767", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#000000", "#d9f24a"
DIV = LinearSegmentedColormap.from_list("div", ["#b7d3f6", "#6da7ec", BLUE, "#383835", RED, "#f3a9a8", "#fbdcdc"])
plt.rcParams.update({"font.family": "Helvetica Neue", "figure.facecolor": BG, "savefig.facecolor": BG, "text.color": INK})
P = sys.argv[1] if len(sys.argv) > 1 else "P1"; fs = 256
info = mne.create_info(CH, fs, "eeg"); info.set_montage("standard_1005")

def course(sess, hand):
    """median beta power over trials (CAR), 0.5 s smoothing, in dB vs the 0.5-2 s pre-cue rest -> (ch, samples)"""
    out = []
    for r in ("training", "test"):
        y, on, lab, _ = load_raw(f"{sess}_{r}")
        out.append((np.abs(hilbert(epoch(y, on, fs, (13, 30)), axis=-1)) ** 2)[lab == hand])
    p = np.median(np.concatenate(out), 0); k = fs // 2
    p = np.stack([np.convolve(c, np.ones(k) / k, "same") for c in p])
    d = 10 * np.log10(p / p[:, sl(fs, 0.5, 2)].mean(-1, keepdims=True))
    return d - d.mean(0, keepdims=True)      # relative to the average sensor: shows WHICH side engages more

maps = {(st, h): course(f"{P}_{st}", h) for st in ("pre", "post") for h in (1, -1)}
times = np.arange(1.8, 7.71, 0.1)
fig = plt.figure(figsize=(7.2, 6.4))
axs = np.array([[fig.add_axes([0.20 + j * 0.39, 0.50 - i * 0.37, 0.34, 0.34]) for j in range(2)] for i in range(2)])
fig.text(0.37, 0.865, "imagine LEFT hand", ha="center", fontsize=14, color=INK2); fig.text(0.76, 0.865, "imagine RIGHT hand", ha="center", fontsize=14, color=INK2)
fig.text(0.02, 0.67, "session 1", fontsize=15, fontweight="bold", va="center"); fig.text(0.02, 0.30, "session 25", fontsize=15, fontweight="bold", va="center")
fig.text(0.02, 0.955, f"Patient {P}: which side engages more (blue)", fontsize=15, fontweight="bold")
fig.text(0.02, 0.915, "beta rhythm, relative to the average sensor: blue = more engaged, red = less", fontsize=10.5, color=MUTED)
phase = fig.text(0.98, 0.955, "", ha="right", fontsize=14, fontweight="bold", color=LIME)
tl = fig.add_axes([0.20, 0.055, 0.73, 0.03]); tl.set_xlim(0, 8); tl.set_ylim(0, 1); tl.axis("off")
tl.barh(0.5, 8, 0.35, color=GRID); bar = tl.barh(0.5, 0, 0.35, color=LIME)[0]
for x, t in ((0, "beep"), (2, "cue"), (3.5, "stimulation on"), (8, "8 s")): tl.text(x, -1.3, t, ha="left" if x < 8 else "right", fontsize=10, color=MUTED); tl.plot([x, x], [0.1, 0.9], color=MUTED, lw=1)

def draw(t):
    n = sl(fs, t, t + 0.01).start
    for i, st in enumerate(("pre", "post")):
        for j, h in enumerate((1, -1)):
            ax = axs[i, j]; ax.clear()
            mne.viz.plot_topomap(maps[(st, h)][:, n], info, axes=ax, show=False, cmap=DIV, vlim=(-2, 2), contours=0, sensors=False, extrapolate="local")
            [l.set_color(MUTED) for l in ax.lines]
            ax.text(-0.118, 0.01, "L", fontsize=10, color=MUTED, ha="center"); ax.text(0.118, 0.01, "R", fontsize=10, color=MUTED, ha="center")
    phase.set_text("rest" if t < 2 else "imagery only" if t < 3.5 else "imagery + stimulation"); bar.set_width(t)

if "--frame" in sys.argv:
    draw(3.2); fig.savefig(f"figures/6_{P}_flip_frame.png", dpi=150)
else:
    FuncAnimation(fig, draw, frames=times).save(f"figures/6_{P}_flip.gif", writer=PillowWriter(fps=10), dpi=110)
