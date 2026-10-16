"""Step 11: regularisation arm for the strong audio head (plan item 3.3b).

Pre-registered before running:
  - Feature set = the variant selected on dev by 10_strong_audio.py (results/strong_audio_variants.csv).
  - Grid: Adam weight decay in {0, 1e-5, 1e-4, 1e-3, 1e-2}, applied to the whole model
    (head and layer weights). Everything else as in 10 (class-weighted CE, early stopping on
    dev weighted CE, full-batch Adam lr 1e-3, seed 0).
  - Choose the weight decay by DEV weighted-F1. Test is reported for the chosen value and
    for the unregularised head, never used to choose.
  - If the chosen value beats weight decay 0 by less than 0.005 dev weighted-F1, the
    unregularised head stays (a gain that small is within noise for 1,108 clips).
  - If a regularised head is kept, data/dataset_strong.pt is rewritten with its frozen
    embedding; otherwise it is left alone.

Usage: python scripts/11_strong_audio_reg.py
Outputs: results/strong_audio_regularisation.csv (and data/dataset_strong.pt if updated)
"""
import importlib

import pandas as pd
import torch
import torch.nn.functional as F

from common import DATA, RESULTS, class_weights, load_dataset, set_determinism

m10 = importlib.import_module("10_strong_audio")
SEED = 0
GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
MIN_GAIN = 0.005


def train_wd(model, X, y, w, wd, lr=1e-3, max_steps=5000, eval_every=50, patience=20):
    """Same loop as common.train_full_batch, with weight decay."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = float("inf"), None, 0
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(X["clf"]), y["clf"], weight=w).backward()
        opt.step()
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


def main():
    var = pd.read_csv(RESULTS / "strong_audio_variants.csv")
    name = var.loc[var.selected_on_dev, "variant"].iloc[0]
    use_std, standardise = name.startswith("C+D"), name.endswith("_std")
    print(f"feature set selected in step 10: {name}")

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
    for wd in GRID:
        set_determinism(SEED)
        m = m10.LayerWeighted(mu, sd, use_std, standardise)
        train_wd(m, X, y, w, wd)
        models[wd] = m
        r = {"variant": name, "weight_decay": wd,
             **{f"{s}_wF1": m10.wf1(m, X[s], y[s]) for s in ["clf", "dev", "test"]}}
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    best_wd = float(res.loc[res.dev_wF1.idxmax(), "weight_decay"])
    base_dev = float(res.loc[res.weight_decay == 0.0, "dev_wF1"].iloc[0])
    gain = float(res.dev_wF1.max()) - base_dev
    keep = best_wd != 0.0 and gain >= MIN_GAIN
    res["chosen"] = res.weight_decay == (best_wd if keep else 0.0)
    res.to_csv(RESULTS / "strong_audio_regularisation.csv", index=False)
    print(f"\nbest weight decay on dev {best_wd} (dev gain over 0: {gain:+.4f}); "
          f"{'keeping regularised head' if keep else 'keeping the unregularised head'}")

    if keep:
        m = models[best_wd]
        D2 = torch.load(DATA / "dataset_strong.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for s in ["clf", "gate", "dev", "test"]:
                D2["splits"][s]["audio"] = m.embed(X[s]).contiguous()
        D2["strong_audio_weight_decay"] = best_wd
        torch.save(D2, DATA / "dataset_strong.pt")
        print("rewrote data/dataset_strong.pt with the regularised layer weights")


if __name__ == "__main__":
    main()
