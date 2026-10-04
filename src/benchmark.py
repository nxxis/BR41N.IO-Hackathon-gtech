"""Train run -> test run accuracy for each pipeline, with 5x5 CV on the training run
(the only thing used for model selection) and the g.tec reference numbers."""
import sys, json, numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.base import clone
sys.path.insert(0, "src"); from mi import *

res = {}
for s in SESSIONS:
    Xtr, ytr, fs = load_bank(s + "_training", FBANK + [(8, 30)]); Xte, yte, _ = load_bank(s + "_test", FBANK + [(8, 30)])
    res[s] = {}
    for name, (bi, pipe) in pipelines(fs).items():
        cv = cross_val_score(clone(pipe), Xtr[:, bi], ytr, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)).mean()
        te = clone(pipe).fit(Xtr[:, bi], ytr).score(Xte[:, bi], yte)
        res[s][name] = {"cv": cv * 100, "test": te * 100}
    print(s, "done", flush=True)
json.dump(res, open("results/benchmark.json", "w"), indent=1)
names = list(next(iter(res.values())))
print("\nTEST accuracy % (train-run CV in brackets)")
print("pipeline".ljust(24) + "".join(s.rjust(15) for s in SESSIONS) + "   mean")
for n in names:
    print(n.ljust(24) + "".join(f"{res[s][n]['test']:6.1f} ({res[s][n]['cv']:4.0f})".rjust(15) for s in SESSIONS) + f"  {np.mean([res[s][n]['test'] for s in SESSIONS]):5.1f}")
print("ref CSP+LDA".ljust(24) + "".join(f"{REF[s][0]:6.1f}       ".rjust(15) for s in SESSIONS) + f"  {np.mean([REF[s][0] for s in SESSIONS]):5.1f}")
print("ref PCA+TVLDA".ljust(24) + "".join(f"{REF[s][1]:6.1f}       ".rjust(15) for s in SESSIONS) + f"  {np.mean([REF[s][1] for s in SESSIONS]):5.1f}")
