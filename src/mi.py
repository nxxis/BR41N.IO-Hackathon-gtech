"""Shared loading / feature code for the stroke-rehab motor-imagery analysis."""
import os
from pathlib import Path
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

ROOT = Path(__file__).resolve().parents[1]
D = Path(os.environ.get("STROKE_REHAB_DATA", ROOT / "data" / "stroke-rehab"))
CH = "FC3 FCz FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 Pz".split()
SESSIONS = [f"{p}_{s}" for p in ("P1", "P2", "P3") for s in ("pre", "post")]
REF = {"P1_pre": (77.1, 92.9), "P1_post": (93.9, 97.0), "P2_pre": (68.4, 72.4),
       "P2_post": (96.1, 97.4), "P3_pre": (74.4, 93.6), "P3_post": (79.7, 100.0)}  # CSP+LDA, PCA+TVLDA
FBANK = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
T0, T1 = -2.0, 8.0   # Cue at 2 s; diagram feedback boundary ~3.5 s, actual timestamps absent.


def load_raw(name, data_dir=None, return_metadata=False):
    """Validate held trial blocks; keep the original full-epoch boundary rule.

    Accept flat and nested organiser extraction layouts. Metadata makes any
    boundary exclusion explicit without changing the established epoch limits.
    """
    allowed = {f"{s}_{run}" for s in SESSIONS for run in ("training", "test")}
    if name not in allowed:
        raise ValueError("Expected one of the twelve named organiser recordings")
    folder = Path(data_dir) if data_dir is not None else Path(D)
    candidates = (folder / f"{name}.mat", folder / "stroke-rehab" / f"{name}.mat")
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("Recording not found; set STROKE_REHAB_DATA to the extracted MAT folder")
    m = sio.loadmat(path, variable_names=["fs", "y", "trig"])
    if not {"fs", "y", "trig"} <= m.keys():
        raise ValueError("Recording requires fs, y and trig")
    rate = np.asarray(m["fs"])
    if rate.size != 1 or not np.isfinite(rate.item()) or float(rate.item()) != 256:
        raise ValueError("Expected a scalar 256 Hz sample rate")
    fs = int(rate.item())
    y, raw_trig = np.asarray(m["y"]), np.asarray(m["trig"])
    trig = raw_trig.ravel()
    if y.ndim != 2 or y.shape != (len(trig), len(CH)):
        raise ValueError("Expected samples by 16 channels and a matching trigger")
    if not np.isfinite(y).all() or not np.isfinite(trig).all():
        raise ValueError("Recording contains non-finite samples")
    if not set(np.unique(trig)) <= {-1, 0, 1}:
        raise ValueError("Unexpected trigger code")
    active = trig != 0
    on = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    end = np.flatnonzero(active & np.r_[~active[1:], True]) + 1
    if len(on) != 80 or len(end) != 80 or not np.all(end - on == 8 * fs):
        raise ValueError("Expected 80 held eight-second trial blocks")
    if any(not np.all(trig[a:b] == trig[a]) for a, b in zip(on, end)):
        raise ValueError("Label changes within a held trial block")
    labels = trig[on].astype(int)
    if np.sum(labels == 1) != 40 or np.sum(labels == -1) != 40:
        raise ValueError("Expected 40 trials per hand")
    keep = (on + T0 * fs >= 0) & (on + T1 * fs <= len(trig))
    metadata = {"recording": name, "total_trials": len(on),
                "retained_trials": int(keep.sum()), "boundary_excluded_trials": int((~keep).sum()),
                "epoch_seconds": [T0, T1]}
    result = (y, on[keep], labels[keep], fs)
    return (*result, metadata) if return_metadata else result


def epoch(y, on, fs, band, car=False):
    """Offline, noncausal band-pass then cut trials; later samples can affect early estimates."""
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
