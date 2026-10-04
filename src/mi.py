"""Shared loading / feature code for the stroke-rehab motor-imagery analysis."""
import numpy as np, scipy.io as sio
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import CSP

D = "data/stroke-rehab/stroke-rehab/"
CH = "FC3 FCz FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 Pz".split()
SESSIONS = [f"{p}_{s}" for p in ("P1", "P2", "P3") for s in ("pre", "post")]
REF = {"P1_pre": (77.1, 92.9), "P1_post": (93.9, 97.0), "P2_pre": (68.4, 72.4),
       "P2_post": (96.1, 97.4), "P3_pre": (74.4, 93.6), "P3_post": (79.7, 100.0)}  # CSP+LDA, PCA+TVLDA
FBANK = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
T0, T1 = -2.0, 8.0   # epoch limits relative to trial onset (cue at 2 s, feedback from 3.5 s)


def load_raw(name):
    m = sio.loadmat(D + name + ".mat")
    fs = int(m["fs"][0, 0]); trig = m["trig"].ravel().astype(int)
    on = np.flatnonzero((trig[1:] != 0) & (trig[:-1] == 0)) + 1
    on = on[(on + T0 * fs >= 0) & (on + T1 * fs <= len(trig))]
    return m["y"], on, trig[on], fs


def epoch(y, on, fs, band, car=False):
    """Band-pass the continuous recording, then cut trials -> (trials, channels, samples)."""
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    yf = sosfiltfilt(sos, y, axis=0)
    if car:
        yf = yf - yf.mean(1, keepdims=True)
    return np.stack([yf[i + int(T0 * fs):i + int(T1 * fs)].T for i in on])


def load_bank(name, bands=FBANK, car=False):
    """-> X (trials, bands, channels, samples), labels (+1 left, -1 right), fs"""
    y, on, lab, fs = load_raw(name)
    return np.stack([epoch(y, on, fs, b, car) for b in bands], 1), lab, fs


def sl(fs, a, b):
    return slice(int((a - T0) * fs), int((b - T0) * fs))


class BankFeatures(BaseEstimator, TransformerMixin):
    """Per (band, window) block: covariance -> CSP log-variance or Riemannian tangent space."""
    def __init__(self, fs=256, windows=((2.5, 8.0),), kind="csp", nfilter=4):
        self.fs, self.windows, self.kind, self.nfilter = fs, windows, kind, nfilter

    def _covs(self, X):
        for b in range(X.shape[1]):
            for a, e in self.windows:
                yield Covariances("oas").transform(X[:, b, :, sl(self.fs, a, e)])

    def fit(self, X, y):
        self.models_ = []
        for C in self._covs(X):
            m = CSP(nfilter=self.nfilter, log=True) if self.kind == "csp" else TangentSpace()
            self.models_.append(m.fit(C, y))
        return self

    def transform(self, X):
        return np.hstack([m.transform(C) for m, C in zip(self.models_, self._covs(X))])


def lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


WIN_ONE = ((2.5, 8.0),)
WIN_MULTI = ((2.5, 4.5), (3.5, 5.5), (4.5, 6.5), (5.5, 7.5))

def pipelines(fs):
    """name -> (band indices into FBANK+[(8,30)], pipeline)"""
    return {
        "CSP+LDA (8-30)":        ([7], make_pipeline(BankFeatures(fs, WIN_ONE, "csp"), lda())),
        "FBCSP+sLDA":            (list(range(7)), make_pipeline(BankFeatures(fs, WIN_ONE, "csp"), lda())),
        "TV-FBCSP+sLDA":         (list(range(7)), make_pipeline(BankFeatures(fs, WIN_MULTI, "csp"), lda())),
        "Riemann TS+LR (8-30)":  ([7], make_pipeline(BankFeatures(fs, WIN_ONE, "ts"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))),
        "FB Riemann TS+LR":      (list(range(7)), make_pipeline(BankFeatures(fs, WIN_ONE, "ts"), StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))),
        "TV-FB Riemann TS+LR":   (list(range(7)), make_pipeline(BankFeatures(fs, WIN_MULTI, "ts"), StandardScaler(), LogisticRegression(C=0.005, max_iter=3000))),
    }
