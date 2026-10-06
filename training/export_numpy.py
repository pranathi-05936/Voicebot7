"""Export the trained PyTorch model to NumPy for the Vercel API, verify that the
NumPy forward pass matches PyTorch, and evaluate the *deployed* inference path
(one sentence at a time, exactly as the API runs) on the CLINC150 test split.

Usage (from training/):  python export_numpy.py
"""
import json
import os
import shutil
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

from model import IntentPredictor, pad_batch

OUT = "../api/_model"
os.makedirs(OUT, exist_ok=True)

p = IntentPredictor("artifacts")
state = {k: v.float().numpy() for k, v in p.model.state_dict().items()}
np.savez_compressed(f"{OUT}/weights.npz", **{k: v.astype(np.float16) for k, v in state.items()})
shutil.copy("artifacts/meta.json", f"{OUT}/meta.json")
print("exported", len(state), "tensors ->", f"{OUT}/weights.npz",
      f"({os.path.getsize(f'{OUT}/weights.npz') / 1e6:.1f} MB)")

sys.path.insert(0, "../api/_lib")
from intent_np import IntentModel  # noqa: E402

npm = IntentModel(OUT)

# 1) numerical equivalence with PyTorch on sample sentences
raw = json.load(open("data/data_full.json"))
samples = [t for t, _ in raw["test"][::300]] + ["hi", "what time is it"]
max_diff = 0.0
for s in samples:
    with torch.no_grad():
        pt = torch.softmax(p.model(pad_batch([p.vocab.encode(s)])), -1)[0].numpy()
    pn, _ = npm.probs(s)
    max_diff = max(max_diff, float(np.abs(pt - pn).max()))
print(f"max |p_torch - p_numpy| over {len(samples)} sentences: {max_diff:.2e}")

# 2) deployed-path evaluation (NumPy, per sentence, final OOS rule)
L = npm.labels
OOS = L.index("oos")
split = raw["test"] + raw["oos_test"]
y = np.array([L.index(l) for _, l in split])
pred = np.array([L.index(npm.predict(t)["intent"]) for t, _ in split])
arg = np.array([int(np.argmax(npm.probs(t)[0])) for t, _ in split])
ins = y != OOS


def summary(pr):
    return {"overall_acc": float((pr == y).mean()),
            "in_scope_acc": float((pr[ins] == y[ins]).mean()),
            "oos_recall": float((pr[~ins] == OOS).mean()),
            "oos_precision": float((y[pr == OOS] == OOS).mean()) if (pr == OOS).any() else 0.0,
            "macro_f1": float(f1_score(y, pr, average="macro"))}


res = {"argmax": summary(arg), "final_rule": summary(pred), "max_prob_diff_vs_torch": max_diff}
per = {l: float((pred[y == i] == i).mean()) for i, l in enumerate(L)}
res["hardest_intents"] = sorted(per.items(), key=lambda kv: kv[1])[:8]
res["perfect_intents"] = int(sum(v == 1.0 for v in per.values()))
print(json.dumps(res, indent=2))
m = json.load(open("artifacts/metrics.json"))
m["deployed_numpy"] = res
json.dump(m, open("artifacts/metrics.json", "w"), indent=2)

# summary for the web app's "About" panel
json.dump({"dataset": m["dataset"], "trainable_params": m["model_params"],
           "frozen_params": m.get("frozen_params"),
           "baseline_tfidf_logreg": m["baseline_tfidf_logreg"],
           "test": res["final_rule"], "test_argmax": res["argmax"]},
          open(f"{OUT}/summary.json", "w"), indent=2)
