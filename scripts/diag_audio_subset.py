"""Diagnostic (item 5): does the faithfulness result change where the audio head is reliable?

Motivation: the audio head is weak (weighted-F1 0.368), so removing audio may change little
and "always text" is a strong baseline. Here we recompute the primary pairs on subsets
where audio is more likely to matter.

DEFINITIONS (written before running; no protocol definition in 04/05 is changed)
  audio_correct               audio head's own argmax == true label (per seed, test row)
  audio_confident             audio head's top-class probability c_a >= T_seed, where T_seed is
                              the MEDIAN of c_a over the dev split for that seed's audio head
                              (fit on dev, frozen, applied to test)
  audio_correct_and_confident both of the above
  all                         full test set (reference; equals results/faithfulness_summary.csv)
  Subsets use test labels for "correct". This is descriptive subgroup reporting, not
  tuning: nothing is selected or thresholded on test.
Pairs (primary only): fusion_learned_gate <- gate_w_t, fusion_confidence <- conf_w_t;
  methods mean and blank. Metrics as in 05: agreement, always-text baseline, gain,
  Spearman; exact ties excluded. Gain CI: 1000-draw dialogue bootstrap (rng 0), 3 seeds
  averaged inside each draw, as in 06. Spearman has no CI here.

Output: results/audio_subset_faithfulness.csv
Usage:  python scripts/diag_audio_subset.py   (needs checkpoints/, results/interventions_log.csv)
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_dataset  # noqa: E402

load_models = importlib.import_module("04_interventions").load_models
SEEDS = [0, 1, 2]
PAIRS = [("fusion_learned_gate", "gate_w_t"), ("fusion_confidence", "conf_w_t")]
N_BOOT = 1000


def main():
    D = load_dataset()
    test, dev = D["splits"]["test"], D["splits"]["dev"]
    log = pd.read_csv(RESULTS / "interventions_log.csv")

    masks, thresholds = {}, {}
    for seed in SEEDS:
        _, audio_head, _ = load_models(seed)
        with torch.no_grad():
            ca_dev = torch.softmax(audio_head(dev["audio"]), 1).max(1).values
            p_test = torch.softmax(audio_head(test["audio"]), 1)
        t = float(ca_dev.median())
        thresholds[seed] = t
        ca = p_test.max(1).values.numpy()
        correct = (p_test.argmax(1) == test["y"]).numpy()
        confident = ca >= t
        masks[seed] = {"_pred": p_test.argmax(1).numpy(), "all": np.ones_like(correct), "audio_correct": correct,
                       "audio_confident": confident, "audio_correct_and_confident": correct & confident}
    print("dev-median audio confidence thresholds:", {k: round(v, 4) for k, v in thresholds.items()})

    first = log[(log.seed == 0) & (log.model == "fusion_learned_gate") & (log.method == "mean")].sort_values("row")
    dlg = pd.factorize(first.dialogue_id)[0]
    n_dlg = dlg.max() + 1
    rng = np.random.default_rng(0)
    draws = [np.bincount(rng.integers(0, n_dlg, n_dlg), minlength=n_dlg)[dlg] for _ in range(N_BOOT)]

    out = []
    for model, signal in PAIRS:
        for method in ["mean", "blank"]:
            S, E = [], []
            for seed in SEEDS:
                d = log[(log.seed == seed) & (log.model == model) & (log.method == method)].sort_values("row")
                assert (d.audio_pred.values == masks[seed]["_pred"]).all()  # log matches recomputed head
                S.append(2 * d[signal].values - 1)
                E.append(d.dp_text.abs().values - d.dp_audio.abs().values)
            for subset in [k for k in masks[0] if k != "_pred"]:
                def point(w):
                    ag, bs, rho = [], [], []
                    for k, seed in enumerate(SEEDS):
                        v = (S[k] != 0) & (E[k] != 0) & masks[seed][subset]
                        ww = w[v]
                        ag.append((ww * (np.sign(S[k][v]) == np.sign(E[k][v]))).sum() / ww.sum())
                        bs.append((ww * (E[k][v] > 0)).sum() / ww.sum())
                    return np.mean(ag), np.mean(bs)
                a, b = point(np.ones(len(dlg)))
                gains = [np.subtract(*point(w)) for w in draws]
                lo, hi = np.percentile(gains, [2.5, 97.5])
                rho = np.mean([spearmanr(S[k][masks[seed][subset]], E[k][masks[seed][subset]])[0]
                               for k, seed in enumerate(SEEDS)])
                n = np.mean([masks[seed][subset].sum() for seed in SEEDS])
                out.append({"model": model, "signal": signal, "method": method, "subset": subset,
                            "n_per_seed": round(n), "agreement": a, "always_text_baseline": b,
                            "gain": a - b, "gain_ci_low": lo, "gain_ci_high": hi, "spearman": rho})
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "audio_subset_faithfulness.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    print(res.drop(columns=["signal"]).to_string(index=False))


if __name__ == "__main__":
    main()
