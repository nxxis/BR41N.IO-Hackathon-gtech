"""Robustness of the main finding. Lateralisation index = contralateral minus ipsilateral ERD (dB),
negative = normal (opposite hemisphere more engaged). Varies band, spatial reference and window.
Plus: binomial tests for imagery-only decoding, and whether lateralisation predicts classifier success."""
import sys, json, numpy as np
from scipy.signal import hilbert
from scipy.stats import binomtest, mannwhitneyu
sys.path.insert(0, "src"); from mi import *
rng = np.random.default_rng(1); ix = CH.index
NB = {"C3": ["FC3", "C5", "C1", "CP3"], "C4": ["FC4", "C6", "C2", "CP4"]}

def db(sess, band, win, ref):
    P, L = [], []
    for r in ("training", "test"):
        y, on, lab, fs = load_raw(f"{sess}_{r}")
        if ref == "laplacian": yl = np.stack([y[:, ix(c)] - y[:, [ix(n) for n in NB[c]]].mean(1) for c in ("C3", "C4")], 1)
        elif ref == "car":     yl = (y - y.mean(1, keepdims=True))[:, [ix("C3"), ix("C4")]]
        else:                  yl = y[:, [ix("C3"), ix("C4")]]
        p = np.abs(hilbert(epoch(yl, on, fs, band), axis=-1)) ** 2
        P.append(10 * np.log10(p[:, :, sl(fs, *win)].mean(-1) / p[:, :, sl(fs, 0.5, 2)].mean(-1))); L.append(lab)
    return np.concatenate(P), np.concatenate(L)

def li(d, lab):                      # per trial: contra - ipsi
    return np.where(lab == 1, d[:, 1] - d[:, 0], d[:, 0] - d[:, 1])

print("Lateralisation index, median over trials (dB). negative = contralateral hemisphere more engaged")
print("band  ref        window   | " + "  ".join(s.rjust(8) for s in SESSIONS) + " | pre->post p: P1     P2     P3")
for bn, band in (("beta", (13, 30)), ("mu", (8, 13)), ("8-30", (8, 30))):
    for ref in ("laplacian", "car", "raw"):
        for win in ((2.0, 3.5), (2.5, 3.5), (3.5, 8.0)):
            L = {s: li(*db(s, band, win, ref)) for s in SESSIONS}
            ps = [mannwhitneyu(L[f"{p}_pre"], L[f"{p}_post"]).pvalue for p in ("P1", "P2", "P3")]
            print(f"{bn:5s} {ref:10s} {win[0]}-{win[1]} | " + "  ".join(f"{np.median(L[s]):+8.2f}" for s in SESSIONS) + " |           " + "  ".join(f"{p:.3f}" for p in ps))

print("\nPer hand, beta, Laplacian, 2-3.5 s: median contra-ipsi (dB)")
for s in SESSIONS:
    d, lab = db(s, (13, 30), (2.0, 3.5), "laplacian"); l = li(d, lab)
    print(f"   {s:8s} left hand {np.median(l[lab==1]):+5.2f}   right hand {np.median(l[lab==-1]):+5.2f}")

print("\nImagery-only decoding vs chance (binomial, one-sided)")
D = json.load(open("results/deeper.json"))
for s in SESSIONS:
    n = 79 if s == "P1_post" else 80; k = round(D[s]["imagery_tt"] * n / 100)
    print(f"   {s:8s} {k}/{n} = {k/n*100:.1f}%  p = {binomtest(k, n, 0.5, alternative='greater').pvalue:.4f}")
