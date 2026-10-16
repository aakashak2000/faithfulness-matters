"""Step 12: paired comparison, gate weight vs confidence weight, on the learned-gate model.

Question (action item 7): both signals are scored against the SAME interventions on the
SAME model (the learned-gate model), so their gains can be compared directly. Their separate
intervals overlap (gate +0.019 vs confidence +0.025 under resample_full), which does not
settle whether they differ. This script bootstraps the DIFFERENCE.

Definitions, fixed before running:
  - difference = metric(confidence weight) - metric(gate weight), on model fusion_learned_gate,
    for metric in {gain over always-text, agreement, Spearman}. The always-text baseline is
    identical for both signals (same effects), so the gain difference equals the agreement
    difference; both are reported for readability.
  - Primary method: resample_full (the pre-registered primary intervention). mean, blank,
    resample and resample_alt are sensitivity checks.
  - Same bootstrap as 06: dialogue-level clusters, 1000 draws, rng 0, same draws for both
    signals (that is what makes it paired), 3 seeds averaged inside each draw. Intervals are
    conditional on the 3 trained models.
  - Reading rule: if the 95% interval of the primary difference excludes 0, the two signals
    are distinguishable; otherwise we say they cannot be told apart.
  - Caveat that stays attached to any confidence result: the confidence weight is built from
    the same probabilities it mixes, so a higher agreement is partly structural and is not
    evidence of a better explanation.
  - Checks: the per-signal point estimates and intervals must reproduce
    results/resampling_full_faithfulness.csv.

Usage: python scripts/12_paired_gate_vs_confidence.py
Output: results/paired_gate_vs_confidence.csv
"""
import importlib

import numpy as np
import pandas as pd

from common import RESULTS

an06 = importlib.import_module("06_analysis")

MODEL = "fusion_learned_gate"
SIGNALS = {"gate": "gate_w_t", "confidence": "conf_w_t"}
PRIMARY = "resample_full"
METHODS = ["resample_full", "mean", "blank", "resample", "resample_alt"]
METRICS = {"gain": 2, "agreement": 0, "spearman": 3}  # index into an06.stats output


def main():
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    log = pd.concat([ref, pd.read_csv(RESULTS / "resampling_log.csv"),
                     pd.read_csv(RESULTS / "resampling_full_log.csv")], ignore_index=True)

    # same draws as 06 / 07 / 08 (rng 0, same dialogue order)
    order = ref[(ref.seed == ref.seed.min()) & (ref.model == an06.PAIRS[0][0])
                & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]

    rows = []
    for method in METHODS:
        s_g, e = an06.arrays(log, MODEL, SIGNALS["gate"], method)
        s_c, e2 = an06.arrays(log, MODEL, SIGNALS["confidence"], method)
        assert np.array_equal(e, e2), "effects must be identical for a paired comparison"
        pt = {k: np.array(an06.stats(s, e)) for k, s in [("gate", s_g), ("confidence", s_c)]}
        bt = {k: np.array([an06.stats(s[:, i], e[:, i]) for i in draws])
              for k, s in [("gate", s_g), ("confidence", s_c)]}
        for name, j in METRICS.items():
            d = bt["confidence"][:, j] - bt["gate"][:, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"method": method, "metric": name,
                         "gate": pt["gate"][j], "confidence": pt["confidence"][j],
                         "diff_conf_minus_gate": pt["confidence"][j] - pt["gate"][j],
                         "ci_low": lo, "ci_high": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
        print(f"done {method}", flush=True)
    res = pd.DataFrame(rows)

    # reproduce the per-signal numbers already in results/
    old = pd.read_csv(RESULTS / "resampling_full_faithfulness.csv")
    for label, sig in [("Learned gate ← gate weight", "gate"), ("Learned gate ← confidence", "confidence")]:
        o = old[(old.pair == label) & (old.method == PRIMARY) & (old.metric == "gain")].estimate.item()
        n = res[(res.method == PRIMARY) & (res.metric == "gain")][sig].item()
        assert abs(o - n) < 1e-9, f"{sig}: {n} vs {o}"
    print("per-signal point estimates reproduce results/resampling_full_faithfulness.csv")

    res.to_csv(RESULTS / "paired_gate_vs_confidence.csv", index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 4)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
