"""Step 16: does weighting the signal by how much the two heads disagree track effect size
better? (action item 17, optional signal)

Idea: if the text and audio heads output almost the same distribution, removing either one
changes little whatever the weight says. The raw weight ignores this. A signal that scales the
weight difference by the heads' disagreement can rank effect sizes better.

Definitions, fixed before running:
  s        = 2 * w_t - 1                         (the existing signal, per pair of 06)
  D1       = TVD(p_text, p_audio) = 0.5 * sum |p_text - p_audio|        (primary variant)
  D2       = |p_text[c] - p_audio[c]|, c = predicted class of the fused model   (second variant)
  s_new    = s * D                                (sign(s_new) = sign(s) because D >= 0, so
                                                   agreement and gain are UNCHANGED by construction;
                                                   only the ranking, i.e. Spearman, can change)
  Evaluated: Spearman of s_new vs e for e in {dp, tvd} (effect differences as in 14), all four
  pairs of 06, resample_full only, original-input head outputs (no intervention needed).
  Report the Spearman of s, of s * D1, of s * D2, and the paired bootstrap difference
  (s * D - s), dialogue clusters, 1000 draws, rng 0, 3 seeds averaged.
  Reading rule: an improvement counts only if the paired interval excludes 0 for the primary
  variant D1 under the primary effect dp. D2 and tvd are sensitivity checks, reported either way.
  Nothing is tuned; there are no free parameters.

Usage: python scripts/16_weight_disagreement.py
Output: results/weight_disagreement.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_dataset  # noqa: E402

iv04 = importlib.import_module("04_interventions")
an06 = importlib.import_module("06_analysis")
SEEDS = [0, 1, 2]


def spearman_rows(s, e):
    """Spearman per seed, averaged. s, e: (seeds, n)."""
    return np.mean([np.corrcoef(rankdata(a), rankdata(b))[0, 1] for a, b in zip(s, e)])


@torch.no_grad()
def disagreement(model, signal_log):
    D = load_dataset()
    test = D["splits"]["test"]
    d1, d2 = [], []
    for seed in SEEDS:
        th, ah, gate = iv04.load_models(seed)
        p_t, p_a = torch.softmax(th(test["text"]), 1), torch.softmax(ah(test["audio"]), 1)
        d1.append((0.5 * (p_t - p_a).abs().sum(1)).numpy())
        c = signal_log[(signal_log.seed == seed) & (signal_log.model == model)].sort_values("row").pred.values
        idx = np.arange(len(c))
        d2.append(np.abs(p_t.numpy()[idx, c] - p_a.numpy()[idx, c]))
    return np.stack(d1), np.stack(d2)


def main():
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    log = pd.read_csv(RESULTS / "resampling_full_log.csv")

    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]

    rows = []
    for model, signal, label in an06.PAIRS:
        d = log[log.model == model].sort_values(["seed", "row"])
        s = np.stack([2 * d[d.seed == k][signal].values - 1 for k in SEEDS])
        D1, D2 = disagreement(model, log)
        eff = {"dp": np.stack([d[d.seed == k].dp_text.abs().values - d[d.seed == k].dp_audio.abs().values for k in SEEDS]),
               "tvd": np.stack([d[d.seed == k].tvd_text.values - d[d.seed == k].tvd_audio.values for k in SEEDS])}
        for ename, e in eff.items():
            base_pt = spearman_rows(s, e)
            base_b = np.array([spearman_rows(s[:, i], e[:, i]) for i in draws])
            for vname, Dv in (("D1_tvd_heads", D1), ("D2_class_gap", D2)):
                s2 = s * Dv
                pt = spearman_rows(s2, e)
                b = np.array([spearman_rows(s2[:, i], e[:, i]) for i in draws])
                lo, hi = np.percentile(b - base_b, [2.5, 97.5])
                rows.append({"pair": label, "effect": ename, "variant": vname,
                             "spearman_signal": base_pt, "spearman_weighted": pt,
                             "diff": pt - base_pt, "ci_low": lo, "ci_high": hi,
                             "excludes_zero": bool(lo > 0 or hi < 0)})
        print("done", label, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "weight_disagreement.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
