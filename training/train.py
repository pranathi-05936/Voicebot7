"""Train and evaluate the BiLSTM-Attention intent classifier on CLINC150.

Usage:  python train.py            (expects data/data_full.json)
Outputs: artifacts/intent_model.pt, artifacts/meta.json, artifacts/metrics.json,
         artifacts/training_curves.png
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from model import BiLSTMAttentionClassifier, Vocab, pad_batch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count() or 2)

HP = dict(emb_dim=300, train_dim=64, hidden=192, layers=2, dropout=0.4)
EPOCHS, BATCH, LR, PATIENCE = 40, 64, 2e-3, 6
OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data
raw = json.load(open("data/data_full.json"))
train = raw["train"] + raw["oos_train"]
val = raw["val"] + raw["oos_val"]
test = raw["test"] + raw["oos_test"]

labels = sorted({l for _, l in train})          # 150 intents + "oos"
lab2id = {l: i for i, l in enumerate(labels)}
OOS = lab2id["oos"]
# Vocabulary = every training word + the ~20k most frequent English words that
# have a pre-trained vector, so unseen (but common) words at inference time still
# get a meaningful representation.
import spacy
nlp = spacy.load("en_core_web_md", exclude=["tagger", "parser", "ner", "lemmatizer",
                                           "attribute_ruler", "tok2vec", "senter"])
vecs = nlp.vocab.vectors
row_word = {}
for key, row in vecs.key2row.items():
    w = nlp.vocab.strings[key]
    if w.isalpha() and w.islower() and len(w) <= 15 and row not in row_word:
        row_word[row] = w
train_vocab = Vocab.build([t for t, _ in train])
words = sorted(set(train_vocab.itos[2:]) | set(row_word.values()))
vocab = Vocab(["<pad>", "<unk>"] + words)
emb = np.zeros((len(vocab), 300), dtype=np.float32)
hits = 0
for i, w in enumerate(vocab.itos[2:], start=2):
    lex = nlp.vocab[w]
    if lex.has_vector:
        emb[i] = lex.vector; hits += 1
emb /= (np.linalg.norm(emb, axis=1, keepdims=True).mean() + 1e-9)   # scale to ~unit norm
seen = set(train_vocab.itos)
train_cov = sum(nlp.vocab[w].has_vector for w in train_vocab.itos[2:]) / (len(train_vocab) - 2)
print(f"pretrained vectors for {hits}/{len(vocab)} vocab words; "
      f"training-word coverage {train_cov:.1%}")
print(f"train={len(train)} val={len(val)} test={len(test)} "
      f"classes={len(labels)} vocab={len(vocab)}")


def encode(split):
    return [vocab.encode(t) for t, _ in split], np.array([lab2id[l] for _, l in split])


Xtr, ytr = encode(train); Xva, yva = encode(val); Xte, yte = encode(test)


def metrics(y, pred):
    ins = y != OOS
    oos = ~ins
    return {
        "overall_acc": float(accuracy_score(y, pred)),
        "in_scope_acc": float((pred[ins] == y[ins]).mean()),
        "oos_recall": float((pred[oos] == OOS).mean()),
        "oos_precision": float((y[pred == OOS] == OOS).mean()) if (pred == OOS).any() else 0.0,
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


# ---------------------------------------------------------------- baseline
tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
lr = LogisticRegression(max_iter=2000, C=10)
lr.fit(tfidf.fit_transform([t for t, _ in train]), ytr)
base = metrics(yte, lr.predict(tfidf.transform([t for t, _ in test])))
print("Baseline TF-IDF + LogReg:", base)

# ---------------------------------------------------------------- model
model = BiLSTMAttentionClassifier(len(vocab), len(labels), pretrained=emb, **HP)
with torch.no_grad():   # words never seen in training get a zero task-embedding
    for i, w in enumerate(vocab.itos):
        if w not in seen:
            model.task_embedding.weight[i].zero_()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable params={n_params:,} frozen={n_frozen:,}")
opt = torch.optim.AdamW([
    {"params": [model.task_embedding.weight], "weight_decay": 0.0},
    {"params": [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("task_embedding")],
     "weight_decay": 1e-4}], lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


@torch.no_grad()
def probs_for(X):
    model.eval()
    out = []
    for i in range(0, len(X), 512):
        out.append(torch.softmax(model(pad_batch(X[i:i + 512])), -1))
    return torch.cat(out).numpy()


history, best, bad, best_state = [], -1, 0, None
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    order = np.random.permutation(len(Xtr))
    tot = 0.0
    for i in range(0, len(order), BATCH):
        b = order[i:i + BATCH]
        x = pad_batch([Xtr[j] for j in b]); y = torch.tensor(ytr[b])
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(b)
    pv = probs_for(Xva)
    val_loss = float(-np.log(pv[np.arange(len(yva)), yva] + 1e-9).mean())
    va = metrics(yva, pv.argmax(1))
    history.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_loss": val_loss,
                    "val_in_scope_acc": va["in_scope_acc"]})
    print(f"ep{ep:02d} train_loss={tot/len(Xtr):.4f} val_loss={val_loss:.4f} "
          f"val_in_scope_acc={va['in_scope_acc']:.4f}")
    sched.step(va["in_scope_acc"])
    if va["in_scope_acc"] > best:
        best, bad = va["in_scope_acc"], 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= PATIENCE:
            print("early stopping"); break
train_time = time.time() - t0
model.load_state_dict(best_state)

# ------------------------------------------- confidence threshold for OOS
# Only 100 OOS training samples exist, so low-confidence predictions are also
# routed to "out of scope". Rule (chosen on the validation set): the highest
# threshold that costs at most 2 points of in-scope accuracy.
pv = probs_for(Xva)
base_ins = metrics(yva, pv.argmax(1))["in_scope_acc"]
best_t = 0.0
for t in np.arange(0.0, 0.95, 0.01):
    pred = pv.argmax(1); pred[pv.max(1) < t] = OOS
    if metrics(yva, pred)["in_scope_acc"] >= base_ins - 0.02:
        best_t = float(t)
print(f"threshold={best_t:.2f}")

pt = probs_for(Xte)
raw_pred = pt.argmax(1)
thr_pred = raw_pred.copy(); thr_pred[pt.max(1) < best_t] = OOS
res = {
    "dataset": {"train": len(train), "val": len(val), "test": len(test),
                "classes": len(labels), "vocab": len(vocab)},
    "model_params": n_params, "frozen_params": n_frozen, "hparams": HP, "epochs_run": len(history),
    "train_time_sec": round(train_time, 1), "oos_threshold": best_t,
    "baseline_tfidf_logreg": base,
    "bilstm_attention_argmax": metrics(yte, raw_pred),
    "bilstm_attention_with_threshold": metrics(yte, thr_pred),
    "history": history,
}
# per-intent accuracy (hardest intents) for the report
per = {}
for i, l in enumerate(labels):
    m = yte == i
    per[l] = float((thr_pred[m] == i).mean())
res["hardest_intents"] = sorted(per.items(), key=lambda kv: kv[1])[:8]
json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k not in ("history",)}, indent=2))

torch.save({k: v.half() for k, v in model.state_dict().items()},
           f"{OUT}/intent_model.pt")      # fp16 on disk to halve the file size
json.dump({"itos": vocab.itos, "labels": labels, "hparams": HP,
           "oos_threshold": best_t}, open(f"{OUT}/meta.json", "w"))

# ---------------------------------------------------------------- plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ep = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(ep, [h["train_loss"] for h in history], label="train (label-smoothed)")
ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy loss"); ax[0].legend()
ax[0].set_title("Loss")
ax[1].plot(ep, [h["val_in_scope_acc"] for h in history], color="C2")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Validation in-scope accuracy")
for a in ax: a.grid(alpha=.3); a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/training_curves.png", dpi=150)
