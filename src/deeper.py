"""Deeper questions:
A  imagery-only decoding (2-3.5 s: after cue, before any FES / visual feedback), pre vs post
B  like-for-like with the reference metric: time-resolved accuracy in 0.5 s steps, max in the feedback phase
C  Wilson 95% CIs for the headline accuracies
D  hemisphere asymmetry of beta ERD per hand, imagery-only vs feedback window, bootstrap CIs"""
import sys, json, numpy as np
from scipy.signal import hilbert
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from statsmodels.stats.proportion import proportion_confint
sys.path.insert(0, "src"); from mi import *
rng = np.random.default_rng(0); out = {}

def fb_riemann(fs, win, C=0.01):
    return make_pipeline(BankFeatures(fs, (win,), "ts"), StandardScaler(), LogisticRegression(C=C, max_iter=3000))

print("A  imagery-only window 2.0-3.5 s (no feedback yet) | train->test, and 10-fold CV on all 160 trials")
data = {s: {r: load_bank(f"{s}_{r}") for r in ("training", "test")} for s in SESSIONS}
for s in SESSIONS:
    (Xtr, ytr, fs), (Xte, yte, _) = data[s]["training"], data[s]["test"]
    tt = clone(fb_riemann(fs, (2.0, 3.5))).fit(Xtr, ytr).predict(Xte) == yte
    Xa, ya = np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte])
    cv = cross_val_predict(fb_riemann(fs, (2.0, 3.5)), Xa, ya, cv=StratifiedKFold(10, shuffle=True, random_state=0)) == ya
    lo, hi = proportion_confint(cv.sum(), len(cv), method="wilson")
    out[s] = {"imagery_tt": tt.mean() * 100, "imagery_cv": cv.mean() * 100, "imagery_cv_ci": [lo * 100, hi * 100]}
    print(f"   {s:8s} train->test {tt.mean()*100:5.1f}   CV {cv.mean()*100:5.1f}  [{lo*100:.0f}-{hi*100:.0f}]")
for st in ("pre", "post"):
    print(f"   mean {st}: train->test {np.mean([out[s]['imagery_tt'] for s in SESSIONS if st in s]):.1f}  CV {np.mean([out[s]['imagery_cv'] for s in SESSIONS if st in s]):.1f}")

print("\nB  reference-style metric: 1.5 s sliding window, 0.5 s steps, train->test; max over feedback phase vs fixed-window single decision")
ends = np.arange(4.0, 8.01, 0.5)
for s in SESSIONS:
    (Xtr, ytr, fs), (Xte, yte, _) = data[s]["training"], data[s]["test"]
    acc = [clone(fb_riemann(fs, (e - 1.5, e))).fit(Xtr, ytr).score(Xte, yte) * 100 for e in ends]
    out[s]["timeres"] = acc; out[s]["max_metric"] = max(acc)
    print(f"   {s:8s} " + " ".join(f"{a:4.0f}" for a in acc) + f" | max {max(acc):5.1f} | ref CSP+LDA {REF[s][0]:5.1f}  TVLDA {REF[s][1]:5.1f}")
print(f"   mean of max: {np.mean([out[s]['max_metric'] for s in SESSIONS]):.1f}  (ref {np.mean([REF[s][0] for s in SESSIONS]):.1f} / {np.mean([REF[s][1] for s in SESSIONS]):.1f})")

print("\nC  Wilson 95% CI, headline single-decision accuracy (FB Riemann + online recentering)")
A = json.load(open("results/adapt.json"))
for s in SESSIONS:
    a = A[s]["within | FB | online recentering"]; n = len(data[s]["test"][1]); lo, hi = proportion_confint(round(a * n / 100), n, method="wilson")
    out[s]["ci"] = [lo * 100, hi * 100]; print(f"   {s:8s} {a:5.1f}  [{lo*100:.0f}-{hi*100:.0f}]  n={n}")
k = sum(round(A[s]["within | FB | online recentering"] * len(data[s]["test"][1]) / 100) for s in SESSIONS); n = sum(len(data[s]["test"][1]) for s in SESSIONS)
print("   pooled", f"{k}/{n} = {k/n*100:.1f}", [round(x * 100, 1) for x in proportion_confint(k, n, method="wilson")])

print("\nD  beta ERD (dB vs 0.5-2 s rest), Laplacian C3 / C4; median over trials [bootstrap 95% CI]")
LAP = {"C3": ("C3", ["FC3", "C5", "C1", "CP3"]), "C4": ("C4", ["FC4", "C6", "C2", "CP4"])}; ix = CH.index
def trial_db(sess, band, win):
    P, L = [], []
    for r in ("training", "test"):
        y, on, lab, fs = load_raw(f"{sess}_{r}")
        yl = np.stack([y[:, ix(c)] - y[:, [ix(n) for n in nb]].mean(1) for c, nb in LAP.values()], 1)
        p = np.abs(hilbert(epoch(yl, on, fs, band), axis=-1)) ** 2
        P.append(10 * np.log10(p[:, :, sl(fs, *win)].mean(-1) / p[:, :, sl(fs, 0.5, 2)].mean(-1))); L.append(lab)
    return np.concatenate(P), np.concatenate(L)
def med_ci(x):
    b = np.median(rng.choice(x, (2000, len(x))), 1); return np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5)
for wn, win in (("imagery-only 2-3.5 s", (2.0, 3.5)), ("feedback 3.5-8 s", (3.5, 8.0))):
    print("  ", wn); print("      sess      hand   C3 (left hemi)          C4 (right hemi)")
    for s in SESSIONS:
        db, lab = trial_db(s, (13, 30), win)
        for hand, c in (("left", 1), ("right", -1)):
            m3, m4 = med_ci(db[lab == c, 0]), med_ci(db[lab == c, 1])
            out[s][f"erd|{wn[:7]}|{hand}"] = {"C3": m3, "C4": m4}
            print(f"      {s:8s}  {hand:5s}  {m3[0]:+5.2f} [{m3[1]:+5.2f},{m3[2]:+5.2f}]   {m4[0]:+5.2f} [{m4[1]:+5.2f},{m4[2]:+5.2f}]")
    for p in ("P1", "P2", "P3"):
        for hand, c in (("left", 1), ("right", -1)):
            (a, la), (b, lb) = trial_db(f"{p}_pre", (13, 30), win), trial_db(f"{p}_post", (13, 30), win)
            print(f"      {p} {hand:5s} pre->post MWU p:  C3 {mannwhitneyu(a[la==c,0], b[lb==c,0]).pvalue:.4f}   C4 {mannwhitneyu(a[la==c,1], b[lb==c,1]).pvalue:.4f}")
json.dump(out, open("results/deeper.json", "w"), indent=1, default=float)
