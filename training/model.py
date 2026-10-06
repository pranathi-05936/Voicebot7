"""Intent classification model shared by training and the web app.

Architecture: word embeddings -> 2-layer bidirectional LSTM -> additive
self-attention pooling -> dropout -> fully connected softmax over intents.
"""
import json
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq=1):
        counts = {}
        for t in texts:
            for w in tokenize(t):
                counts[w] = counts.get(w, 0) + 1
        words = sorted(w for w, c in counts.items() if c >= min_freq)
        return cls([PAD, UNK] + words)

    def encode(self, text, max_len=32):
        ids = [self.stoi.get(w, 1) for w in tokenize(text)][:max_len]
        return ids or [1]

    def __len__(self):
        return len(self.itos)


class BiLSTMAttentionClassifier(nn.Module):
    """Two embedding channels are concatenated per token:
    * a frozen 300-d pre-trained word-vector table (spaCy en_core_web_md / GloVe-style),
      which lets the model handle words never seen in the training set, and
    * a small trainable task-specific embedding learned from scratch.
    """

    def __init__(self, vocab_size, num_classes, emb_dim=300, train_dim=64,
                 hidden=192, layers=2, dropout=0.4, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(pretrained))
        self.embedding.weight.requires_grad = False
        self.task_embedding = nn.Embedding(vocab_size, train_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(0.25)
        self.lstm = nn.LSTM(emb_dim + train_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x, return_attention=False):
        mask = x != 0                                    # (B, T)
        e = torch.cat([self.embedding(x), self.task_embedding(x)], dim=-1)
        h, _ = self.lstm(self.emb_drop(e))               # (B, T, 2H)
        scores = self.attn(h).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=1)                 # attention weights
        pooled = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        logits = self.fc(self.drop(pooled))
        return (logits, alpha) if return_attention else logits


def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [0] * (L - len(s)) for s in seqs], dtype=torch.long)


class IntentPredictor:
    """Loads the trained artifacts and predicts intents for raw text."""

    def __init__(self, model_dir="artifacts"):
        meta = json.load(open(f"{model_dir}/meta.json"))
        self.vocab = Vocab(meta["itos"])
        self.labels = meta["labels"]
        self.model = BiLSTMAttentionClassifier(len(self.vocab), len(self.labels),
                                               **meta["hparams"])
        state = torch.load(f"{model_dir}/intent_model.pt", map_location="cpu")
        self.model.load_state_dict({k: v.float() for k, v in state.items()})
        self.model.eval()
        self.threshold = meta.get("oos_threshold", 0.34)
        self.ratio = meta.get("oos_ratio", 5.0)

    def decide(self, probs_sorted):
        """Out-of-scope rule: reject only if the top probability is low AND it does
        not clearly dominate the runner-up (p1 < threshold and p1/p2 < ratio).
        The ratio test keeps short but unambiguous inputs such as "hi" or "thanks"."""
        p1, p2 = probs_sorted[0], probs_sorted[1]
        return p1 < self.threshold and p1 / max(p2, 1e-9) < self.ratio

    @torch.no_grad()
    def predict(self, text, top_k=3):
        x = pad_batch([self.vocab.encode(text)])
        probs = F.softmax(self.model(x), dim=-1)[0]
        p, idx = probs.topk(top_k)
        return [(self.labels[i], float(v)) for v, i in zip(p.tolist(), idx.tolist())]
