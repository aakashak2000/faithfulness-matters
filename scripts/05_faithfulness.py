"""Step 5: faithfulness metrics from the intervention log.

For each (seed, model, signal, method, subset):
  signal difference  s = 2 * w_t - 1            (> 0: the signal says text mattered more)
  effect difference  e = |dp_text| - |dp_audio| (> 0: removing text changed more)

  agreement            share of utterances where sign(s) == sign(e)
  always_text_baseline share where sign(e) > 0, i.e. what a signal that always says
                       "text" would score. Agreement only means something above this.
  agreement_nonzero    agreement after dropping |e| < NEAR_ZERO (threshold fixed in advance)
  spearman             rank correlation of s and e across utterances
  also: mean |dp| and flip rate for each modality

Signal/model pairs:
  primary   learned gate model  <- gate_w_t      (the model's own weight)
  primary   confidence model    <- conf_w_t      (the model's own weight)
  secondary learned gate model  <- conf_w_t      (an external plausible signal)
  secondary average model       <- conf_w_t

Method robustness: does the effect ranking survive switching mean <-> blank?

Outputs
  results/faithfulness_per_seed.csv
  results/faithfulness_summary.csv    mean and std over seeds
  results/method_robustness.csv
Usage: python scripts/05_faithfulness.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS

NEAR_ZERO = 0.02
PAIRS = [
    ("fusion_learned_gate", "gate_w_t", "primary"),
    ("fusion_confidence", "conf_w_t", "primary"),
    ("fusion_learned_gate", "conf_w_t", "secondary"),
    ("fusion_average", "conf_w_t", "secondary"),
]
SUBSETS = {"full": lambda d: d, "no_shared_audio": lambda d: d[~d.audio_shared]}


def score(d, signal):
    s = 2 * d[signal].values - 1
    e = d.dp_text.abs().values - d.dp_audio.abs().values
    valid = (s != 0) & (e != 0)
    agree = np.sign(s[valid]) == np.sign(e[valid])
    keep = valid & (np.abs(e) >= NEAR_ZERO)
    rho, p = spearmanr(s, e)
    return {
        "n": len(d),
        "agreement": agree.mean(),
        "always_text_baseline": (e[valid] > 0).mean(),
        "signal_says_text": (s[valid] > 0).mean(),
        "agreement_nonzero": (np.sign(s[keep]) == np.sign(e[keep])).mean(),
        "n_nonzero": int(keep.sum()),
        "spearman": rho,
        "spearman_p": p,
        "mean_abs_dp_text": d.dp_text.abs().mean(),
        "mean_abs_dp_audio": d.dp_audio.abs().mean(),
        "flip_rate_text": d.flip_text.mean(),
        "flip_rate_audio": d.flip_audio.mean(),
    }


def main():
    log = pd.read_csv(RESULTS / "interventions_log.csv")
    rows = []
    for (seed, method), g in log.groupby(["seed", "method"]):
        for model, signal, role in PAIRS:
            d_model = g[g.model == model]
            for subset, f in SUBSETS.items():
                rows.append({"seed": seed, "method": method, "model": model, "signal": signal,
                             "role": role, "subset": subset, **score(f(d_model), signal)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS / "faithfulness_per_seed.csv", index=False)

    keys = ["role", "model", "signal", "method", "subset"]
    cols = ["agreement", "always_text_baseline", "signal_says_text", "agreement_nonzero",
            "spearman", "mean_abs_dp_text", "mean_abs_dp_audio", "flip_rate_text", "flip_rate_audio"]
    summary = per_seed.groupby(keys)[cols].agg(["mean", "std"])
    summary.to_csv(RESULTS / "faithfulness_summary.csv")

    # method robustness: same model, effect difference under mean vs blank
    rob = []
    for (seed, model), g in log.groupby(["seed", "model"]):
        a = g[g.method == "mean"].set_index("row")
        b = g[g.method == "blank"].set_index("row")
        ea = a.dp_text.abs() - a.dp_audio.abs()
        eb = (b.dp_text.abs() - b.dp_audio.abs()).loc[ea.index]
        rho, _ = spearmanr(ea, eb)
        rob.append({"seed": seed, "model": model,
                    "same_winner_share": (np.sign(ea) == np.sign(eb)).mean(),
                    "spearman_mean_vs_blank": rho})
    rob = pd.DataFrame(rob)
    rob.to_csv(RESULTS / "method_robustness.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    main_view = (per_seed[per_seed.subset == "full"]
                 .groupby(["role", "model", "signal", "method"])
                 [["agreement", "always_text_baseline", "agreement_nonzero", "spearman",
                   "mean_abs_dp_text", "mean_abs_dp_audio"]].mean())
    print("=== Faithfulness, full test set, mean over seeds ===")
    print(main_view)
    shared = (per_seed.pivot_table(index=["model", "signal", "method"], columns="subset",
                                   values=["agreement", "spearman"], aggfunc="mean"))
    print("\n=== Sensitivity to the 42 shared-audio rows ===")
    print(shared)
    print("\n=== Do mean and blank interventions agree? (mean over seeds) ===")
    print(rob.groupby("model")[["same_winner_share", "spearman_mean_vs_blank"]].mean())
    print("\n=== Per-seed spread of the primary metrics ===")
    prim = per_seed[(per_seed.role == "primary") & (per_seed.subset == "full")]
    print(prim.pivot_table(index=["model", "method"], columns="seed",
                           values=["agreement", "spearman"]))
    print(f"\nSaved tables to {RESULTS}")


if __name__ == "__main__":
    main()
