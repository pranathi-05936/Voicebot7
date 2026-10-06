"""Pure-NumPy inference for the trained BiLSTM-attention intent classifier.

The network is trained in PyTorch (training/train.py); training/export_numpy.py
exports its weights to api/_model/weights.npz. Re-implementing the forward pass
in NumPy keeps the serverless function tiny (no PyTorch), so it fits Vercel's
function size limit while producing the same outputs as the PyTorch model.
"""
import json
import os
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
MAX_LEN = 32


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_direction(x, W_ih, W_hh, b):
    """Single-direction LSTM over x (T, D). PyTorch gate order: i, f, g, o."""
    H = W_hh.shape[1]
    h = np.zeros(H, dtype=np.float32)
    c = np.zeros(H, dtype=np.float32)
    xw = x @ W_ih.T + b                  # precompute input projections (T, 4H)
    out = np.empty((x.shape[0], H), dtype=np.float32)
    for t in range(x.shape[0]):
        z = xw[t] + W_hh @ h
        i, f, g, o = z[:H], z[H:2 * H], z[2 * H:3 * H], z[3 * H:]
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        out[t] = h
    return out


class IntentModel:
    def __init__(self, model_dir):
        meta = json.load(open(os.path.join(model_dir, "meta.json")))
        self.itos = meta["itos"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.labels = meta["labels"]
        self.layers = meta["hparams"]["layers"]
        self.threshold = meta.get("oos_threshold", 0.34)
        self.ratio = meta.get("oos_ratio", 5.0)
        w = np.load(os.path.join(model_dir, "weights.npz"))
        self.w = {k: w[k].astype(np.float32) for k in w.files}

    def encode(self, text):
        ids = [self.stoi.get(t, 1) for t in TOKEN_RE.findall(text.lower())][:MAX_LEN]
        return ids or [1]

    def probs(self, text):
        w = self.w
        ids = np.array(self.encode(text))
        x = np.concatenate([w["embedding.weight"][ids],
                            w["task_embedding.weight"][ids]], axis=1)
        for l in range(self.layers):
            fwd = _lstm_direction(x, w[f"lstm.weight_ih_l{l}"], w[f"lstm.weight_hh_l{l}"],
                                  w[f"lstm.bias_ih_l{l}"] + w[f"lstm.bias_hh_l{l}"])
            bwd = _lstm_direction(x[::-1], w[f"lstm.weight_ih_l{l}_reverse"],
                                  w[f"lstm.weight_hh_l{l}_reverse"],
                                  w[f"lstm.bias_ih_l{l}_reverse"]
                                  + w[f"lstm.bias_hh_l{l}_reverse"])[::-1]
            x = np.concatenate([fwd, bwd], axis=1)          # (T, 2H)
        scores = x @ w["attn.weight"][0] + w["attn.bias"][0]
        alpha = np.exp(scores - scores.max()); alpha /= alpha.sum()
        pooled = alpha @ x
        logits = w["fc.weight"] @ pooled + w["fc.bias"]
        p = np.exp(logits - logits.max())
        return p / p.sum(), alpha

    def predict(self, text, top_k=3):
        p, alpha = self.probs(text)
        idx = np.argsort(-p)[:top_k]
        top = [(self.labels[i], float(p[i])) for i in idx]
        p1, p2 = top[0][1], top[1][1]
        # out-of-scope: low confidence AND no clear margin over the runner-up
        oos = p1 < self.threshold and p1 / max(p2, 1e-9) < self.ratio
        intent = "oos" if oos else top[0][0]
        tokens = TOKEN_RE.findall(text.lower())[:MAX_LEN]
        attention = [(t, float(a)) for t, a in zip(tokens, alpha)] if tokens else []
        return {"intent": intent, "confidence": p1, "top": top,
                "out_of_scope": oos, "attention": attention}
