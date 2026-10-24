"""Step 28: a better text head from multi-layer RoBERTa-large features (item 1 of the F1 push; a VARIANT).

Same recipe that took the audio head from 36.8 to 46.7 weighted-F1 (step 22 and 25): standardised multi-layer
features, a linear head trained to convergence with an L2 penalty, configuration chosen on DEV.

Pre-registered before running:
  - Features: data/roberta_layers_{train,dev,test}.pt (27_extract_text_layers.py), mean pooling over tokens;
    clf rows via results/strong_audio_row_map.csv. Standardise with clf mean and std (per dimension).
  - Grid: single layers {8, 12, 16, 20, 24}; concat of {12, 16, 20, 24}; concat of {8, 10, ..., 24};
    all 25 layers.  L2 coefficient wd in {0.1, 0.3, 1, 3, 10} (loss = class-weighted CE + 0.5 * wd * ||W||^2).
    LBFGS, strong-Wolfe, at most 200 iterations, zero init (as in step 22).
  - Selection: highest DEV weighted-F1. Test weighted-F1 and macro-F1 are reported for every configuration, never
    used to choose. Reference: the headline MiniLM text head (3-seed test weighted-F1 0.4914, macro 0.3530);
    its dev score is computed here from checkpoints/seed0.pt.
  - Keep rule: the new head is a candidate variant if its dev weighted-F1 beats the MiniLM head's by at least 0.02.
  - The headline model is unchanged.

Usage: python scripts/28_text_head_grid.py
Output: results/text_head_grid.csv, results/text_head_selected.json
"""
import json
import time

import pandas as pd
import torch
from sklearn.metrics import f1_score

from common import CKPT, DATA, RESULTS, Head, class_weights, load_dataset
import importlib

b22 = importlib.import_module("22_better_audio")
NINE = [8, 10, 12, 14, 16, 18, 20, 22, 24]
CONFIGS = [(f"layer{l}", [l]) for l in (8, 12, 16, 20, 24)] + [("concat4", [12, 16, 20, 24]), ("concat9", NINE),
                                                                ("all25", list(range(25)))]
WDS = [0.1, 0.3, 1, 3, 10]


def load_text(split):
    L = torch.load(DATA / f"roberta_layers_{split}.pt", map_location="cpu", weights_only=False)
    return L["mean"]


def pick(m, layers):
    return torch.cat([m[:, l].float() for l in layers], dim=1)


def main():
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    rows = torch.tensor(mp[mp.split == "clf"].sort_values("row").file_index.values)
    mtr, mdv, mte = load_text("train")[rows], load_text("dev"), load_text("test")
    y = {k: S[k]["y"] for k in ("clf", "dev", "test")}
    w = class_weights(y["clf"])

    # reference: the headline MiniLM text head on dev and test (seed 0)
    ck = torch.load(CKPT / "seed0.pt", map_location="cpu", weights_only=False)
    h = Head(384)
    h.load_state_dict(ck["text_head"])
    with torch.no_grad():
        ref_dev = f1_score(y["dev"].numpy(), h(S["dev"]["text"]).argmax(1).numpy(), average="weighted")
        ref_test = f1_score(y["test"].numpy(), h(S["test"]["text"]).argmax(1).numpy(), average="weighted")
    print(f"reference MiniLM head (seed 0): dev {ref_dev:.4f} test {ref_test:.4f}", flush=True)

    out, t0 = [], time.time()
    for name, layers in CONFIGS:
        Xc = pick(mtr, layers)
        mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
        Xc = (Xc - mu) / sd
        Xd, Xt = (pick(mdv, layers) - mu) / sd, (pick(mte, layers) - mu) / sd
        for wd in WDS:
            W, b = b22.fit(Xc, y["clf"], w, wd)
            dw, dm = b22.score(Xd, y["dev"], W, b)
            tw, tm = b22.score(Xt, y["test"], W, b)
            out.append({"features": name, "wd": wd, "dim": Xc.shape[1], "dev_wF1": dw, "dev_macroF1": dm,
                        "test_wF1": tw, "test_macroF1": tm})
            print(f"{name:9s} wd={wd:<4} dev {dw:.4f} test {tw:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(out)
    best = res.loc[res.dev_wF1.idxmax()]
    res["selected_on_dev"] = res.index == res.dev_wF1.idxmax()
    res.to_csv(RESULTS / "text_head_grid.csv", index=False)
    keep = bool(best.dev_wF1 >= ref_dev + 0.02)
    (RESULTS / "text_head_selected.json").write_text(json.dumps(
        {"selected": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
         "reference_minilm": {"dev_wF1": ref_dev, "test_wF1": ref_test}, "candidate_variant": keep}, indent=2, default=str))
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)
    print("\nTop 8 by dev weighted-F1:")
    print(res.sort_values("dev_wF1", ascending=False).head(8).to_string(index=False))
    print(f"\nselected on dev: {best.features} wd={best.wd}: dev {best.dev_wF1:.4f} test {best.test_wF1:.4f} "
          f"(macro {best.test_macroF1:.4f}); MiniLM reference dev {ref_dev:.4f} test {ref_test:.4f}; candidate variant: {keep}")


if __name__ == "__main__":
    main()
