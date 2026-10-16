"""Item 15: back two wording claims in MIDTERM_STATUS.md with a results file.

  1. "Dropping the 42 shared-audio rows changes every metric by <= 0.002"
     -> compute agreement, always-text baseline, gain and Spearman with and without the
        rows flagged audio_shared, for every pair x method, and report the largest change
        per metric.
  2. "Agreement varies by <= 0.01 across seeds"
     -> compute the per-seed agreement for every pair x method and report std and range.

Methods: mean, blank (interventions_log.csv) and resample_full (resampling_full_log.csv).
Point estimates only (no bootstrap), averaged over seeds like 06_analysis.stats.

Usage: python scripts/diag_wording_checks.py
Output: results/wording_checks.json
"""
import importlib
import json

import numpy as np
import pandas as pd

from common import RESULTS

an06 = importlib.import_module("06_analysis")
METHODS = ["mean", "blank", "resample_full"]


def main():
    log = pd.concat([pd.read_csv(RESULTS / "interventions_log.csv"),
                     pd.read_csv(RESULTS / "resampling_full_log.csv")], ignore_index=True)
    shared = (log[(log.model == "fusion_learned_gate") & (log.method == "mean")
                  & (log.seed == 0)].sort_values("row").audio_shared.values.astype(bool))
    print(f"shared rows: {shared.sum()}")

    names = ["agreement", "always_text_baseline", "gain", "spearman"]
    rows, seed_rows = [], []
    for model, signal, label in an06.PAIRS:
        for method in METHODS:
            s, e = an06.arrays(log, model, signal, method)
            full = np.array(an06.stats(s, e))
            drop = np.array(an06.stats(s[:, ~shared], e[:, ~shared]))
            rows.append({"pair": label, "method": method,
                         **{f"{n}_change": float(d) for n, d in zip(names, drop - full)}})
            per_seed = [an06.stats(s[k:k + 1], e[k:k + 1])[0] for k in range(s.shape[0])]
            seed_rows.append({"pair": label, "method": method,
                              "agreement_per_seed": [round(float(x), 4) for x in per_seed],
                              "std": float(np.std(per_seed, ddof=1)),
                              "range": float(np.ptp(per_seed))})
    ch = pd.DataFrame(rows)
    out = {
        "n_shared_rows": int(shared.sum()),
        "shared_audio_max_abs_change": {n: round(float(ch[f"{n}_change"].abs().max()), 4) for n in names},
        "shared_audio_by_pair": rows,
        "seed_agreement_max_std": round(max(r["std"] for r in seed_rows), 4),
        "seed_agreement_max_range": round(max(r["range"] for r in seed_rows), 4),
        "seed_agreement_by_pair": seed_rows,
    }
    (RESULTS / "wording_checks.json").write_text(json.dumps(out, indent=2))
    print("shared-audio rows, largest absolute change per metric:", out["shared_audio_max_abs_change"])
    print(f"agreement across seeds: largest std {out['seed_agreement_max_std']}, "
          f"largest range {out['seed_agreement_max_range']}")
    worst = max(seed_rows, key=lambda r: r["range"])
    print("widest range:", worst["pair"], worst["method"], worst["agreement_per_seed"])


if __name__ == "__main__":
    main()
