"""Evaluate the trained model on the CLINC150 test split with the final
out-of-scope decision rule used by the app, and store the numbers in
artifacts/metrics.json under "final_rule"."""
import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from model import IntentPredictor, pad_batch

META_PATH, MET_PATH = "artifacts/meta.json", "artifacts/metrics.json"
meta = json.load(open(META_PATH))
meta.update(oos_threshold=0.34, oos_ratio=5.0)   # tuned on the validation split
json.dump(meta, open(META_PATH, "w"))

p = IntentPredictor("artifacts")
raw = json.load(open("data/data_full.json"))
L = p.labels; OOS = L.index("oos")


def probs(split):
    X = [p.vocab.encode(t) for t, _ in split]
    y = np.array([L.index(l) for _, l in split])
    with torch.no_grad():
        P = torch.cat([torch.softmax(p.model(pad_batch(X[i:i + 512])), -1)
                       for i in range(0, len(X), 512)]).numpy()
    return P, y


def apply_rule(P):
    pred = P.argmax(1)
    s = -np.sort(-P, 1)
    for i in range(len(P)):
        if p.decide(s[i]):
            pred[i] = OOS
    return pred


def metrics(y, pred):
    ins = y != OOS
    return {"overall_acc": float((pred == y).mean()),
            "in_scope_acc": float((pred[ins] == y[ins]).mean()),
            "oos_recall": float((pred[~ins] == OOS).mean()),
            "oos_precision": float((y[pred == OOS] == OOS).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro"))}


res = {}
for name in ("val", "test"):
    P, y = probs(raw[name] + raw["oos_" + name])
    res[name] = metrics(y, apply_rule(P))
    if name == "test":
        pred = apply_rule(P)
        per = {l: float((pred[y == i] == i).mean()) for i, l in enumerate(L)}
        res["hardest_intents"] = sorted(per.items(), key=lambda kv: kv[1])[:8]
        res["perfect_intents"] = sum(v == 1.0 for v in per.values())
print(json.dumps(res, indent=2))
m = json.load(open(MET_PATH)); m["final_rule"] = res
m["oos_rule"] = {"threshold": 0.34, "ratio": 5.0}
json.dump(m, open(MET_PATH, "w"), indent=2)
