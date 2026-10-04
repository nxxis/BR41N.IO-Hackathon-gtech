"""Riemannian recentering experiments.
(1) within-session train->test, with/without causal online recentering of the test run
(2) zero-calibration: train on other sessions/patients, test on the target test run."""
import sys, json, numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.geodesic import geodesic_riemann
sys.path.insert(0, "src"); from mi import *

WIN = (2.5, 8.0)
BANDS = {"8-30": [(8, 30)], "FB": FBANK}

def isqrt(M): return np.real(fractional_matrix_power(M, -0.5))
def whiten(C, W): return W @ C @ W.T

def covs(name, bands):
    X, y, fs = load_bank(name, bands)
    return np.stack([Covariances("oas").transform(X[:, b, :, sl(fs, *WIN)]) for b in range(len(bands))], 1), y  # trials, bands, ch, ch

def rc_batch(C):
    """recenter every band of a whole run to identity (uses all its trials, no labels)"""
    out = np.empty_like(C)
    for b in range(C.shape[1]):
        W = isqrt(mean_riemann(C[:, b])); out[:, b] = [whiten(c, W) for c in C[:, b]]
    return out

def rc_online(C, M0=None, n0=8):
    """causal recentering: trial i is whitened by the running mean of trials <= i (prior M0 worth n0 trials)"""
    out = np.empty_like(C)
    for b in range(C.shape[1]):
        M, n = (C[0, b], 0) if M0 is None else (M0[b], n0)
        for i in range(len(C)):
            n += 1; M = geodesic_riemann(M, C[i, b], 1.0 / n)
            out[i, b] = whiten(C[i, b], isqrt(M))
    return out

def feats(C): return np.hstack([TangentSpace(metric="riemann").fit(np.eye(C.shape[-1])[None]).transform(C[:, b]) for b in range(C.shape[1])])
def clf(nb): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1 if nb == 1 else 0.01, max_iter=3000))

res = {}
for bn, bands in BANDS.items():
    data = {s: {r: covs(f"{s}_{r}", bands) for r in ("training", "test")} for s in SESSIONS}
    for s in SESSIONS:
        (Ctr, ytr), (Cte, yte) = data[s]["training"], data[s]["test"]
        Mtr = [mean_riemann(Ctr[:, b]) for b in range(Ctr.shape[1])]
        Wtr = [isqrt(M) for M in Mtr]
        fix = lambda C: np.stack([[whiten(c, Wtr[b]) for c in C[:, b]] for b in range(C.shape[1])], 1)
        m = clf(len(bands)).fit(feats(rc_batch(Ctr)), ytr)
        r = res.setdefault(s, {})
        r[f"within | {bn} | no adaptation"] = m.score(feats(fix(Cte)), yte) * 100
        r[f"within | {bn} | online recentering"] = m.score(feats(rc_online(Cte, Mtr)), yte) * 100
        r[f"within | {bn} | offline recentering"] = m.score(feats(rc_batch(Cte)), yte) * 100
        # zero calibration: no data from the target session's training run
        p, st = s.split("_"); other = "post" if st == "pre" else "pre"
        src = {"same patient, other session": [f"{p}_{other}"],
               "other patients": [q for q in SESSIONS if not q.startswith(p)],
               "all other sessions": [q for q in SESSIONS if q != s]}
        for sn, ss in src.items():
            Xs = np.vstack([feats(rc_batch(data[q][run][0])) for q in ss for run in ("training", "test")])
            ys = np.concatenate([data[q][run][1] for q in ss for run in ("training", "test")])
            m0 = clf(len(bands)).fit(Xs, ys)
            r[f"zero-calib | {bn} | {sn} | online"] = m0.score(feats(rc_online(Cte)), yte) * 100
            r[f"zero-calib | {bn} | {sn} | offline"] = m0.score(feats(rc_batch(Cte)), yte) * 100
json.dump(res, open("results/adapt.json", "w"), indent=1)
print("condition".ljust(58) + "".join(s.rjust(9) for s in SESSIONS) + "   mean")
for n in res[SESSIONS[0]]:
    print(n.ljust(58) + "".join(f"{res[s][n]:9.1f}" for s in SESSIONS) + f"  {np.mean([res[s][n] for s in SESSIONS]):5.1f}")
