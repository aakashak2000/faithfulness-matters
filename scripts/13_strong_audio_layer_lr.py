"""Step 13: let the layer weights actually learn (plan item 3.3d, "option A").

Finding that motivates it: in 10_strong_audio.py the learned layer weights stayed at their
starting value (all 25 layers at 0.040, results/strong_audio_layer_weights.csv), so the
"layer-weighted" head was an equal average of the layers. Early stopping ended training before
the weights could move at lr 1e-3.

Pre-registered before running:
  - Feature set: the variant selected on dev in step 10 (C_raw: mean pooling, no standardising).
  - Change: the 25 layer-weight parameters get their own learning rate; the head stays at 1e-3.
    Grid for the layer-weight lr: {1e-3 (= step 10), 3e-2, 1e-1, 3e-1}.
  - Everything else as in step 10 / common.train_head: class-weighted CE, full-batch Adam,
    early stopping on dev weighted CE (patience 20 x 50 steps), seed 0.
  - Choose the lr by DEV weighted-F1. Test is reported for every row but never used to choose.
  - Keep the new head only if its dev weighted-F1 beats the 1e-3 row by at least 0.005
    (within noise for 1,108 clips); otherwise the step 10 head stays.
  - Also report how far the weights moved: max layer weight and the five largest layers.
  - If kept, data/dataset_strong.pt is rewritten with the new frozen embedding.

Usage: python scripts/13_strong_audio_layer_lr.py
Outputs: results/strong_audio_layer_lr.csv, results/strong_audio_layer_weights_lr.csv
"""
import importlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import DATA, RESULTS, class_weights, load_dataset, set_determinism

m10 = importlib.import_module("10_strong_audio")
SEED = 0
GRID = [1e-3, 3e-2, 1e-1, 3e-1]
HEAD_LR = 1e-3
MIN_GAIN = 0.005


def train(model, X, y, w, layer_lr, max_steps=5000, eval_every=50, patience=20):
    opt = torch.optim.Adam([{"params": [model.lw], "lr": layer_lr},
                            {"params": model.lin.parameters(), "lr": HEAD_LR}])
    best, state, bad, steps = float("inf"), None, 0, 0
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(X["clf"]), y["clf"], weight=w).backward()
        opt.step()
        steps = step
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                d = float(F.cross_entropy(model(X["dev"]), y["dev"], weight=w))
            if d < best - 1e-5:
                best, bad = d, 0
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(state)
    model.eval()
    return steps


def main():
    var = pd.read_csv(RESULTS / "strong_audio_variants.csv")
    name = var.loc[var.selected_on_dev, "variant"].iloc[0]
    assert name == "C_raw", f"script assumes the step 10 winner is C_raw, got {name}"

    D = load_dataset()
    S = D["splits"]
    Ltr, Xtr = m10.load_feats("train")
    _, Xdv = m10.load_feats("dev")
    _, Xte = m10.load_feats("test")
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")

    def split_X(n):
        m = mp[mp.split == n].sort_values("row")
        return Xtr[torch.tensor(m.file_index.values)]
    X = {"clf": split_X("clf"), "gate": split_X("gate"), "dev": Xdv, "test": Xte}
    y = {k: S[k]["y"] for k in X}
    w = class_weights(y["clf"])
    mu, sd = X["clf"].mean(0), X["clf"].std(0) + 1e-6

    rows, models = [], {}
    for lr in GRID:
        set_determinism(SEED)
        m = m10.LayerWeighted(mu, sd, False, False)
        steps = train(m, X, y, w, lr)
        models[lr] = m
        lw = torch.softmax(m.lw, 1).detach().numpy()[0]
        r = {"variant": name, "layer_lr": lr, "steps": steps,
             **{f"{s}_wF1": m10.wf1(m, X[s], y[s]) for s in ["clf", "dev", "test"]},
             "max_layer_weight": float(lw.max()), "top5_layers": " ".join(map(str, np.argsort(-lw)[:5]))}
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    base_dev = float(res.loc[res.layer_lr == 1e-3, "dev_wF1"].iloc[0])
    best_lr = float(res.loc[res.dev_wF1.idxmax(), "layer_lr"])
    gain = float(res.dev_wF1.max()) - base_dev
    keep = best_lr != 1e-3 and gain >= MIN_GAIN
    res["chosen"] = res.layer_lr == (best_lr if keep else 1e-3)
    res.to_csv(RESULTS / "strong_audio_layer_lr.csv", index=False)
    print(f"\nbest layer lr on dev {best_lr} (dev gain over 1e-3: {gain:+.4f}); "
          f"{'keeping the new head' if keep else 'keeping the step 10 head'}")

    if keep:
        m = models[best_lr]
        lw = torch.softmax(m.lw, 1).detach().numpy()[0]
        pd.DataFrame({"layer": range(25), "mean": lw}).to_csv(
            RESULTS / "strong_audio_layer_weights_lr.csv", index=False)
        D2 = torch.load(DATA / "dataset_strong.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for s in ["clf", "gate", "dev", "test"]:
                D2["splits"][s]["audio"] = m.embed(X[s]).contiguous()
        D2["strong_audio_layer_lr"] = best_lr
        torch.save(D2, DATA / "dataset_strong.pt")
        print("rewrote data/dataset_strong.pt with the learned layer weights")


if __name__ == "__main__":
    main()
