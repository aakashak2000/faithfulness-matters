"""Shared pieces: data loading, models, fusion rules, training loop, metrics.

Architecture (same as the legacy MELD build):
  text head   : Linear(384 -> 7) on frozen MiniLM embeddings
  audio head  : Linear(1024 -> 7) on frozen WavLM-large embeddings (masked mean pooled)
  fusion      : late fusion of the two probability vectors, three ways
                  average      p = 0.5 * p_t + 0.5 * p_a
                  confidence   w_t = c_t / (c_t + c_a), c = max class prob
                  learned gate w_t = MLP([e_t; e_a]) in (0, 1)
                  p = w_t * p_t + (1 - w_t) * p_a
Heads train on the clf split; the gate trains only on the held-out gate split,
using the heads' out-of-sample probabilities (no stacking leakage).
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"
FIGURES = ROOT / "figures"
DEVICE = "cpu"  # heads are tiny; CPU is fast and fully deterministic


# ----------------------------------------------------------------------------- data

def load_dataset(path=DATA / "dataset.pt"):
    D = torch.load(path, map_location="cpu", weights_only=False)
    D["meta"] = {s: pd.DataFrame(m) for s, m in D["meta"].items()}
    _add_duplicate_flags(D)
    return D


def _add_duplicate_flags(D):
    """Mark dev/test rows whose audio is byte-identical to another row.
    MELD sometimes reuses one clip for neighbouring utterances."""
    S = D["splits"]
    owners = defaultdict(list)
    for split in ["clf", "gate", "dev", "test"]:
        for i, row in enumerate(S[split]["audio"]):
            owners[row.numpy().tobytes()].append(split)
    for split in ["dev", "test"]:
        shared, with_train = [], []
        for row in S[split]["audio"]:
            o = owners[row.numpy().tobytes()]
            shared.append(len(o) > 1)
            with_train.append(any(s in ("clf", "gate") for s in o))
        D["meta"][split]["audio_shared"] = shared
        D["meta"][split]["audio_shared_with_train"] = with_train


def class_weights(y, n_classes=7):
    """'balanced' weights: n / (k * count_c)."""
    counts = torch.bincount(y, minlength=n_classes).float()
    return (len(y) / (n_classes * counts.clamp_min(1))).to(torch.float32)


# ----------------------------------------------------------------------------- models

class Head(nn.Module):
    def __init__(self, dim, n_classes=7):
        super().__init__()
        self.lin = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.lin(x)  # logits


class LearnedGate(nn.Module):
    def __init__(self, text_dim=384, audio_dim=1024, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(text_dim + audio_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, e_t, e_a):
        return self.net(torch.cat([e_t, e_a], dim=1))  # (n, 1) text weight


# ----------------------------------------------------------------------------- fusion rules

def fuse_average(p_t, p_a):
    w_t = torch.full((p_t.shape[0], 1), 0.5)
    return w_t * p_t + (1 - w_t) * p_a, w_t


def fuse_confidence(p_t, p_a):
    c_t = p_t.max(dim=1, keepdim=True).values
    c_a = p_a.max(dim=1, keepdim=True).values
    w_t = c_t / (c_t + c_a)
    return w_t * p_t + (1 - w_t) * p_a, w_t


def fuse_gate(gate, e_t, e_a, p_t, p_a):
    w_t = gate(e_t, e_a)
    return w_t * p_t + (1 - w_t) * p_a, w_t


# ----------------------------------------------------------------------------- training

def train_full_batch(model, loss_fn, dev_loss_fn, lr=1e-3, max_steps=5000,
                     eval_every=50, patience=20):
    """Full-batch Adam with early stopping on dev loss. Returns best dev loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                d = float(dev_loss_fn())
            if d < best - 1e-5:
                best, bad = d, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    return best


def train_head(X, y, X_dev, y_dev, weights, seed):
    torch.manual_seed(seed)
    head = Head(X.shape[1])
    train_full_batch(head,
                     lambda: F.cross_entropy(head(X), y, weight=weights),
                     lambda: F.cross_entropy(head(X_dev), y_dev, weight=weights))
    return head


def nll_on_probs(p, y, weights):
    return F.nll_loss(torch.log(p.clamp_min(1e-8)), y, weight=weights)


def train_gate(e_t, e_a, p_t, p_a, y, dev, weights, seed):
    """dev = (e_t, e_a, p_t, p_a, y) for early stopping."""
    torch.manual_seed(seed)
    gate = LearnedGate(e_t.shape[1], e_a.shape[1])
    train_full_batch(
        gate,
        lambda: nll_on_probs(fuse_gate(gate, e_t, e_a, p_t, p_a)[0], y, weights),
        lambda: nll_on_probs(fuse_gate(gate, *dev[:4])[0], dev[4], weights))
    return gate


# ----------------------------------------------------------------------------- metrics

def metrics(y_true, p):
    y_pred = p.argmax(dim=1).numpy()
    y_true = y_true.numpy()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def per_class_f1(y_true, p, emotions):
    scores = f1_score(y_true.numpy(), p.argmax(dim=1).numpy(), average=None,
                      labels=list(range(len(emotions))), zero_division=0)
    return dict(zip(emotions, scores))


def set_determinism(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
