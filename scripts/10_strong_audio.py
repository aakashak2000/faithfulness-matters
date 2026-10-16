"""Step 10: layer-weighted WavLM audio features (fixes C and D) and a stronger audio head.

Pre-registered before running (see docs/plans/step-03-strong-audio-head.md):
  - Features come from 09_extract_layers.py: per-layer mean (and std) pooled WavLM-large.
  - The audio embedding becomes a learned softmax-weighted sum over the 25 layers.
      C   : mean pooling only            -> 1024-d
      C+D : mean and std, each with its own layer weights, concatenated -> 2048-d
    Each layer is optionally standardised (per-dim mean/std from the clf split) first.
  - The head stays Linear(dim -> 7). Same loss (class-weighted CE) and same early stopping
    (dev weighted CE, full-batch Adam lr 1e-3) as the baseline in common.train_head, so the
    only thing that changes is the features.
  - Layer weights are fit on clf with seed 0 and selected on DEV weighted-F1. The chosen
    weights are then frozen; every split gets one fixed embedding, as in the baseline.
  - Test is scored once per variant for reporting and never used to choose a variant.
  - The headline model is unchanged; this is a robustness variant.

Stage 1 (this script):
  1. map train WAVs to the clf/gate rows of the cached bundle (layer-24 mean vs cached audio)
  2. fit the four variants {C, C+D} x {raw, standardised}, pick the best on dev
  3. write data/dataset_strong.pt (same as dataset.pt with the new audio embedding)

Usage: python scripts/10_strong_audio.py
Outputs: results/strong_audio_row_map.csv, results/strong_audio_variants.csv,
         results/strong_audio_layer_weights.csv, data/dataset_strong.pt
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import (DATA, RESULTS, class_weights, load_dataset, set_determinism,
                    train_full_batch)

SEED = 0


class LayerWeighted(nn.Module):
    """softmax-weighted sum over layers of (optionally standardised) pooled features,
    then a linear head. use_std adds the std-pooled features with their own layer weights."""

    def __init__(self, mu, sd, use_std, standardise):
        super().__init__()
        self.use_std, self.standardise = use_std, standardise
        self.register_buffer("mu", mu)  # (2, 25, 1024) mean-feats then std-feats
        self.register_buffer("sd", sd)
        self.lw = nn.Parameter(torch.zeros(2 if use_std else 1, 25))
        self.lin = nn.Linear(1024 * (2 if use_std else 1), 7)

    def embed(self, X):
        """X: (N, 2, 25, 1024). Returns the fixed-weight embedding (N, dim)."""
        if self.standardise:
            X = (X - self.mu) / self.sd
        w = torch.softmax(self.lw, dim=1)  # (k, 25)
        parts = [(X[:, k] * w[k][None, :, None]).sum(1) for k in range(self.lw.shape[0])]
        return torch.cat(parts, dim=1)

    def forward(self, X):
        return self.lin(self.embed(X))


def load_feats(split):
    L = torch.load(DATA / f"wavlm_layers_{split}.pt", map_location="cpu", weights_only=False)
    X = torch.stack([L["mean"].float(), L["std"].float()], dim=1)  # (N, 2, 25, 1024)
    return L, X


def map_train_rows(D, L):
    """Match each clf/gate row of the cached bundle to a train WAV by layer-24 mean."""
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    emo_idx = {e: i for i, e in enumerate(D["emotions"])}
    lab = {(a, b): emo_idx[e] for a, b, e in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Emotion)}
    file_y = torch.tensor([lab[tuple(i)] for i in L["ids"]])
    F_last = F.normalize(L["last_mean"], dim=1)
    rows, used = [], set()
    for split in ["clf", "gate"]:
        A = F.normalize(D["splits"][split]["audio"], dim=1)
        y = D["splits"][split]["y"]
        sim = A @ F_last.T  # (n, 9988)
        sim[y[:, None] != file_y[None, :]] = -1  # label must agree
        order = torch.argsort(sim, dim=1, descending=True)[:, :20]
        for r in range(len(A)):
            pick = next((int(j) for j in order[r] if int(j) not in used), None)
            assert pick is not None, f"no free file for {split} row {r}"
            used.add(pick)
            rows.append({"split": split, "row": r, "file_index": pick,
                         "dialogue_id": L["ids"][pick][0], "utterance_id": L["ids"][pick][1],
                         "cosine": float(sim[r, pick])})
    mp = pd.DataFrame(rows)
    print(f"row map: {len(mp)} rows, cosine min {mp.cosine.min():.4f} "
          f"p1 {mp.cosine.quantile(.01):.4f} median {mp.cosine.median():.4f}; "
          f"files used {len(used)} of {len(L['ids'])}")
    assert len(used) == len(mp) == len(L["ids"])
    assert mp.cosine.min() > 0.98, "some rows match poorly; inspect results/strong_audio_row_map.csv"
    mp.to_csv(RESULTS / "strong_audio_row_map.csv", index=False)
    return mp


def wf1(model, X, y):
    with torch.no_grad():
        return f1_score(y.numpy(), model(X).argmax(1).numpy(), average="weighted")


def main():
    D = load_dataset()
    S = D["splits"]
    Ltr, Xtr = load_feats("train")
    Ldv, Xdv = load_feats("dev")
    Lte, Xte = load_feats("test")
    mp = map_train_rows(D, Ltr)

    def split_X(name):
        m = mp[mp.split == name].sort_values("row")
        return Xtr[torch.tensor(m.file_index.values)]
    X = {"clf": split_X("clf"), "gate": split_X("gate"), "dev": Xdv, "test": Xte}
    y = {k: S[k]["y"] for k in X}
    w = class_weights(y["clf"])
    mu, sd = X["clf"].mean(0), X["clf"].std(0) + 1e-6

    rows, models = [], {}
    for use_std in (False, True):
        for standardise in (False, True):
            name = f"{'C+D' if use_std else 'C'}_{'std' if standardise else 'raw'}"
            set_determinism(SEED)
            m = LayerWeighted(mu, sd, use_std, standardise)
            train_full_batch(m, lambda: F.cross_entropy(m(X["clf"]), y["clf"], weight=w),
                             lambda: F.cross_entropy(m(X["dev"]), y["dev"], weight=w))
            models[name] = m
            r = {"variant": name, **{f"{s}_wF1": wf1(m, X[s], y[s]) for s in ["clf", "dev", "test"]}}
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    best = res.loc[res.dev_wF1.idxmax(), "variant"]
    res["selected_on_dev"] = res.variant == best
    res.to_csv(RESULTS / "strong_audio_variants.csv", index=False)
    print(f"\nselected on dev weighted-F1: {best}")

    m = models[best]
    lw = torch.softmax(m.lw, 1).detach().numpy()
    pd.DataFrame(lw.T, columns=["mean"] + (["std"] if m.use_std else [])).rename_axis("layer") \
        .to_csv(RESULTS / "strong_audio_layer_weights.csv")
    print("top layers by weight (mean pooling):", np.argsort(-lw[0])[:5].tolist())

    # new dataset: same everything, audio embedding replaced by the frozen layer-weighted one
    D2 = torch.load(DATA / "dataset.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        for s in ["clf", "gate", "dev", "test"]:
            D2["splits"][s]["audio"] = m.embed(X[s]).contiguous()
    D2["strong_audio_variant"] = best
    torch.save(D2, DATA / "dataset_strong.pt")
    print(f"saved {DATA / 'dataset_strong.pt'} (audio dim {D2['splits']['clf']['audio'].shape[1]})")


if __name__ == "__main__":
    main()
