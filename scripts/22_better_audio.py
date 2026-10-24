"""Step 22: a better audio head from the features we already extracted (variant, not the headline).

Motivation: step 10's head used Adam with early stopping on dev loss and reached dev 0.383 / test
0.395 (C_raw). A per-layer probe on dev only (grouped CV, standardised features, strong L2) scored
up to 0.437, so the head was probably under-trained, not the features. This script trains a
standardised, L2-regularised linear head to convergence (LBFGS) on the clf split and picks the
configuration on DEV.

Pre-registered before running:
  - Features: data/wavlm_layers_{train,dev,test}.pt (09_extract_layers.py); clf rows via
    results/strong_audio_row_map.csv. Standardise with clf mean and std (per dimension).
  - Grid (all fixed now):
      single layer L in {8, 12, 16, 18, 20, 22, 24}  x  pooling in {mean, mean+std}
      concatenation of layers {12, 16, 20, 24}        x  pooling in {mean, mean+std}
      L2 coefficient wd in {1e-3, 1e-2, 1e-1}  (loss = class-weighted CE + 0.5 * wd * ||W||^2)
  - Loss: class-weighted cross-entropy, weights = common.class_weights(clf labels) (as in 02).
    Optimiser: LBFGS, strong-Wolfe line search, at most 200 iterations. Deterministic (zero init).
  - Selection: highest DEV weighted-F1. Test weighted-F1 and macro-F1 are reported for every
    configuration but never used to choose. The unregularised step 10 head (dev 0.3827,
    test 0.3955) is the reference.
  - Keep rule: the selected head is a candidate variant only if its dev weighted-F1 beats 0.3827
    by at least 0.02; otherwise we say the features, not the training, are the limit.
  - The headline model is unchanged.

Usage: python scripts/22_better_audio.py
Output: results/better_audio_head.csv, results/better_audio_head_selected.json
"""
import json
import time

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import DATA, RESULTS, class_weights, load_dataset

LAYERS = [8, 12, 16, 18, 20, 22, 24]
CONCAT = [12, 16, 20, 24]
WDS = [1e-3, 1e-2, 1e-1]
REF_DEV, REF_TEST = 0.3827, 0.3955


def feats(split, rows=None):
    L = torch.load(DATA / f"wavlm_layers_{split}.pt", map_location="cpu", weights_only=False)
    mean, std = L["mean"], L["std"]
    if rows is not None:
        mean, std = mean[rows], std[rows]
    return mean, std


def build(mean, std, layers, pool):
    parts = []
    for l in layers:
        parts.append(mean[:, l].float())
        if pool == "mean+std":
            parts.append(std[:, l].float())
    return torch.cat(parts, dim=1)


def fit(X, y, w, wd):
    W = torch.zeros(X.shape[1], 7, requires_grad=True)
    b = torch.zeros(7, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y, weight=w) + 0.5 * wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def score(X, y, W, b):
    pred = (X @ W + b).argmax(1).numpy()
    return (f1_score(y.numpy(), pred, average="weighted", zero_division=0),
            f1_score(y.numpy(), pred, average="macro", zero_division=0))


def main():
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    mtr, str_ = feats("train")
    rows = torch.tensor(mp[mp.split == "clf"].sort_values("row").file_index.values)
    mclf, sclf = mtr[rows], str_[rows]
    del mtr, str_
    mdv, sdv = feats("dev")
    mte, ste = feats("test")
    y = {"clf": S["clf"]["y"], "dev": S["dev"]["y"], "test": S["test"]["y"]}
    w = class_weights(y["clf"])

    configs = [(f"layer{l}", [l]) for l in LAYERS] + [("concat12-16-20-24", CONCAT)]
    rows_out = []
    t0 = time.time()
    for name, layers in configs:
        for pool in ("mean", "mean+std"):
            Xc = build(mclf, sclf, layers, pool)
            mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
            Xc = (Xc - mu) / sd
            Xd = (build(mdv, sdv, layers, pool) - mu) / sd
            Xt = (build(mte, ste, layers, pool) - mu) / sd
            for wd in WDS:
                W, b = fit(Xc, y["clf"], w, wd)
                dw, dm = score(Xd, y["dev"], W, b)
                tw, tm = score(Xt, y["test"], W, b)
                rows_out.append({"features": name, "pool": pool, "wd": wd, "dim": Xc.shape[1],
                                 "dev_wF1": dw, "dev_macroF1": dm, "test_wF1": tw, "test_macroF1": tm})
            print(f"{name:20s} {pool:9s} best dev so far {max(r['dev_wF1'] for r in rows_out):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows_out)
    best = res.loc[res.dev_wF1.idxmax()]
    res["selected_on_dev"] = (res.index == res.dev_wF1.idxmax())
    res.to_csv(RESULTS / "better_audio_head.csv", index=False)
    keep = bool(best.dev_wF1 >= REF_DEV + 0.02)
    out = {"selected": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
           "reference_step10": {"dev_wF1": REF_DEV, "test_wF1": REF_TEST},
           "candidate_variant": keep}
    (RESULTS / "better_audio_head_selected.json").write_text(json.dumps(out, indent=2, default=str))
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)
    print("\nTop 8 by dev weighted-F1:")
    print(res.sort_values("dev_wF1", ascending=False).head(8).to_string(index=False))
    print(f"\nselected on dev: {best.features} {best.pool} wd={best.wd}: dev {best.dev_wF1:.4f} "
          f"test {best.test_wF1:.4f} (macro {best.test_macroF1:.4f}); step 10 reference dev {REF_DEV} test {REF_TEST}")
    print("candidate variant (dev gain >= 0.02):", keep)


if __name__ == "__main__":
    main()
