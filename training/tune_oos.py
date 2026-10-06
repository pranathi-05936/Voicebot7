"""Tune the out-of-scope rule (p1 < t and p1/p2 < r) on the VALIDATION split using
the deployed NumPy inference path. Rule: maximise OOS recall while losing at most
2 points of in-scope accuracy relative to plain argmax. Writes the result to
artifacts/meta.json (then re-run export_numpy.py)."""
import json, sys
import numpy as np
sys.path.insert(0, "../api/_lib")
from intent_np import IntentModel

m = IntentModel("../api/_model")
raw = json.load(open("data/data_full.json"))
split = raw["val"] + raw["oos_val"]
L = m.labels; OOS = L.index("oos")
y = np.array([L.index(l) for _, l in split])
P = np.stack([m.probs(t)[0] for t, _ in split])
s = -np.sort(-P, 1); arg = P.argmax(1); ins = y != OOS
base = (arg[ins] == y[ins]).mean()
best = None
for t in np.arange(0, 0.95, 0.01):
    for r in [2, 3, 4, 5, 6, 8, 10, 1e9]:
        pred = arg.copy(); pred[(s[:, 0] < t) & (s[:, 0] / s[:, 1] < r)] = OOS
        a, o = (pred[ins] == y[ins]).mean(), (pred[~ins] == OOS).mean()
        if a >= base - 0.02 and (best is None or o > best[0] or (o == best[0] and a > best[3])):
            best = (o, round(float(t), 2), r, a)
print("val argmax in-scope", base, "-> best (oos_recall, t, r, in_scope):", best)
meta = json.load(open("artifacts/meta.json"))
meta.update(oos_threshold=best[1], oos_ratio=float(best[2]))
json.dump(meta, open("artifacts/meta.json", "w"))
