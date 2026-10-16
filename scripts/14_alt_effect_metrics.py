"""Step 14: do the headline findings survive other ways of measuring "effect"? (items 11 and 12)

The primary effect (fixed in 04_interventions.py) is |dp|, the change in the fused probability
of the originally predicted class. A reviewer can ask whether the conclusion depends on that
choice. This script recomputes every pair x method with three other effect measures, plus a
threshold-free agreement measure (AUC). Nothing in the protocol changes: these are extra
columns of the same log.

Effect difference e = effect(text removed) - effect(audio removed), per utterance and seed:
  dp          |dp_text| - |dp_audio|             (primary, reproduces earlier tables)
  tvd         tvd_text - tvd_audio               (change of the whole distribution)
  flip        flip_text - flip_audio             (did the predicted class change; ties are common,
                                                  so only utterances where exactly one removal
                                                  flips the prediction are scored)
  signed_drop dp_audio - dp_text                 (signed: how much more the predicted-class
                                                  probability FELL when text was removed)
Signal s = 2 * w_t - 1 (unchanged). "Text mattered more" T = e > 0.

Metrics per pair x method x effect: agreement, always-text baseline, gain, Spearman (as in 06),
plus AUC = P(s_i > s_j) for a random T=1 utterance i and T=0 utterance j (ties count half).
AUC needs no threshold and no baseline: 0.5 is chance, and an always-text signal scores 0.5.

Bootstrap: identical to 06 (dialogue clusters, 1000 draws, rng 0, 3 seeds averaged in each draw).
Methods: resample_full (primary), mean, blank (sensitivity).

Pre-registered reading rule, written before running:
  - "Gate is at most a small edge over always-text" is robust to the effect measure if, for
    resample_full under tvd, flip and signed_drop, the gate's gain 95% interval includes 0 or its
    point estimate stays below +0.05. If any alternative gives a clearly larger gain, we say the
    headline depends on the effect measure and report which one.
  - "Confidence rule far above the baseline" is robust if its gain interval's lower bound is
    above +0.10 under all three alternatives.
  - The dp rows must reproduce the earlier tables to 1e-9 (checked below).

Usage: python scripts/14_alt_effect_metrics.py
Output: results/alt_effect_metrics.csv
"""
import importlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import RESULTS

an06 = importlib.import_module("06_analysis")

METHODS = ["resample_full", "mean", "blank"]
EFFECTS = {
    "dp": lambda d: d.dp_text.abs() - d.dp_audio.abs(),
    "tvd": lambda d: d.tvd_text - d.tvd_audio,
    "flip": lambda d: d.flip_text.astype(float) - d.flip_audio.astype(float),
    "signed_drop": lambda d: d.dp_audio - d.dp_text,
}
NAMES = ["agreement", "always_text_baseline", "gain", "spearman", "auc"]


def arrays(log, model, signal, method, effect):
    d = log[(log.model == model) & (log.method == method)].sort_values(["seed", "row"])
    seeds = sorted(d.seed.unique())
    s = np.stack([2 * d[d.seed == k][signal].values - 1 for k in seeds])
    e = np.stack([EFFECTS[effect](d[d.seed == k]).values for k in seeds])
    return s, e


def stats(s, e):
    """agreement, baseline, gain, spearman, auc; averaged over seeds. s, e: (seeds, n)."""
    agree, base, rho, auc = [], [], [], []
    for sk, ek in zip(s, e):
        v = (sk != 0) & (ek != 0)
        agree.append((np.sign(sk[v]) == np.sign(ek[v])).mean())
        base.append((ek[v] > 0).mean())
        rho.append(np.corrcoef(rankdata(sk), rankdata(ek))[0, 1])
        t = ek[v] > 0
        n1, n0 = t.sum(), (~t).sum()
        r = rankdata(sk[v])
        auc.append((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    a, b = np.mean(agree), np.mean(base)
    return a, b, a - b, np.mean(rho), np.mean(auc)


def main():
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    log = pd.concat([ref, pd.read_csv(RESULTS / "resampling_full_log.csv")], ignore_index=True)

    order = ref[(ref.seed == ref.seed.min()) & (ref.model == an06.PAIRS[0][0])
                & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]

    rows = []
    for model, signal, label in an06.PAIRS:
        for method in METHODS:
            for effect in EFFECTS:
                s, e = arrays(log, model, signal, method, effect)
                point = stats(s, e)
                boots = np.array([stats(s[:, i], e[:, i]) for i in draws])
                lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
                for k, name in enumerate(NAMES):
                    rows.append({"pair": label, "method": method, "effect": effect, "metric": name,
                                 "estimate": point[k], "ci_low": lo[k], "ci_high": hi[k]})
            print(f"done {label} / {method}", flush=True)
    res = pd.DataFrame(rows)

    # the dp rows must reproduce the tables already in results/
    old = pd.read_csv(RESULTS / "resampling_full_faithfulness.csv")
    chk = res[(res.effect == "dp") & (res.metric != "auc")].merge(
        old, on=["pair", "method", "metric"], suffixes=("", "_old"))
    assert len(chk) == 4 * 3 * 4, len(chk)
    for c in ["estimate", "ci_low", "ci_high"]:
        assert np.abs(chk[c] - chk[c + "_old"]).max() < 1e-9, c
    print("dp rows reproduce results/resampling_full_faithfulness.csv")

    res.to_csv(RESULTS / "alt_effect_metrics.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    for metric in ["gain", "auc", "spearman"]:
        t = res[(res.method == "resample_full") & (res.metric == metric)].copy()
        t["value"] = t.apply(lambda r: f"{r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
                             if metric == "gain" else f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
        print(f"\n== {metric}, resample_full ==")
        print(t.pivot(index="pair", columns="effect", values="value")[list(EFFECTS)].to_string())


if __name__ == "__main__":
    main()
