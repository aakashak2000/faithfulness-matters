"""Step 44: gate-adaptation analysis on existing row-level arrays (no training, no new swaps). Written 2026-10-23.

Questions, fixed before running (audit follow-up P1):
  Q1  For M1 and M5, under the raw rule and the corrected rule: winner frequency, weight agreement and gain over the constant
      baseline for the FROZEN-gate swap and for the ADAPTIVE (recomputed) gate swap.
  Q2  Paired utterance-level winner disagreement between frozen and adaptive: how often adaptation reverses the winner, in which
      direction, and which winner the weight agrees with on the reversed rows.
  Q3  Signed decomposition per modality: effect_adaptive = effect_frozen + gate term, with gate term = real minus frozen signed mean
      change (for the raw rule this equals mean_d (w_d - w0)(P_d[c] - p_other[c])). Report mean absolute sizes, the signed contribution of the
      gate term to the text-minus-audio effect gap (|fro+g| - |fro| for each modality, then text minus audio), and how often the gate term
      dampens (opposite sign to the frozen change) or amplifies it.
  Q4  Dialogue-cluster bootstrap (1,000 draws, rng 0, gates averaged inside a draw) for paired differences adaptive minus frozen.
  Q5  Cross-tabulation (corrected rule vs raw rule, adaptive gate): predicted class changed (yes/no) x swap winner changed (yes/no) x
      weight agreement (lost / gained / unchanged), averaged over gates, to check whether prior correction changes agreement only where the
      predicted class changes.
Inputs: data/frozen_gate_arrays_{M1,M5}.npz (steps 38/39, raw rule) and data/corrected_rule_arrays_{M1,M5}.npz (step 43, both rules).
Output: results/gate_adaptation_{M1,M5}.csv, results/gate_adaptation_crosstab.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, load_dataset  # noqa: E402

N_BOOT = 1000


def per_gate_metrics(s, er, ef, fr_t, fr_a, rl_t, rl_a):
    """s: (G,n) signed weight signal; er, ef: (G,n) text-minus-audio effect difference (adaptive, frozen); fr/rl: signed mean changes."""
    G = s.shape[0]
    out = {}
    wr, wf = s > 0, s > 0
    for tag, e in (("adaptive", er), ("frozen", ef)):
        pt = (e > 0).mean(1)
        ag = ((s > 0) == (e > 0)).mean(1)
        out[f"p_text_{tag}"] = pt
        out[f"agreement_{tag}"] = ag
        out[f"majority_{tag}"] = np.maximum(pt, 1 - pt)
        out[f"gain_{tag}"] = ag - np.maximum(pt, 1 - pt)
    out["diff_agreement"] = out["agreement_adaptive"] - out["agreement_frozen"]
    out["diff_gain"] = out["gain_adaptive"] - out["gain_frozen"]
    rev = (er > 0) != (ef > 0)
    out["reversal_share"] = rev.mean(1)
    out["reversal_frozen_text_to_audio"] = (rev & (ef > 0)).mean(1)
    out["reversal_frozen_audio_to_text"] = (rev & (ef <= 0)).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["reversed_weight_agrees_adaptive"] = np.array([((s[g] > 0) == (er[g] > 0))[rev[g]].mean() if rev[g].any() else np.nan for g in range(G)])
        out["reversed_weight_agrees_frozen"] = np.array([((s[g] > 0) == (ef[g] > 0))[rev[g]].mean() if rev[g].any() else np.nan for g in range(G)])
    gt, ga = rl_t - fr_t, rl_a - fr_a
    out["mean_abs_frozen_text"], out["mean_abs_frozen_audio"] = np.abs(fr_t).mean(1), np.abs(fr_a).mean(1)
    out["mean_abs_gate_term_text"], out["mean_abs_gate_term_audio"] = np.abs(gt).mean(1), np.abs(ga).mean(1)
    dt, da = np.abs(rl_t) - np.abs(fr_t), np.abs(rl_a) - np.abs(fr_a)       # change in effect size caused by the gate term
    out["mean_signed_size_change_text"], out["mean_signed_size_change_audio"] = dt.mean(1), da.mean(1)
    out["mean_gap_change_text_minus_audio"] = (dt - da).mean(1)
    out["share_gate_term_dampens_text"] = (gt * fr_t < 0).mean(1)
    out["share_gate_term_dampens_audio"] = (ga * fr_a < 0).mean(1)
    out["share_gate_term_exceeds_frozen_text"] = (np.abs(gt) > np.abs(fr_t)).mean(1)
    out["share_gate_term_exceeds_frozen_audio"] = (np.abs(ga) > np.abs(fr_a)).mean(1)
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def main():
    D = load_dataset()
    dlg = D["meta"]["test"]["dialogue_id"].values
    groups = [np.flatnonzero(dlg == g) for g in np.unique(dlg)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
    cross_rows = []
    for tag in ("M1", "M5"):
        R = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        s = R["s_w"]
        rows = []
        for rule in ("raw", "cor"):
            args = (R[f"e_{rule}_real"], R[f"e_{rule}_fro"], R[f"{rule}_fro_text_mean"], R[f"{rule}_fro_audio_mean"],
                    R[f"{rule}_real_text_mean"], R[f"{rule}_real_audio_mean"])
            point = per_gate_metrics(s, *args)
            boots = [per_gate_metrics(s[:, i], *(a[:, i] for a in args)) for i in draws]
            for k, v in point.items():
                b = np.array([bb[k] for bb in boots])
                rows.append({"model": tag, "rule": rule, "metric": k, "value": v, "lo": np.nanpercentile(b, 2.5), "hi": np.nanpercentile(b, 97.5)})
        pd.DataFrame(rows).to_csv(RESULTS / f"gate_adaptation_{tag}.csv", index=False)
        d = pd.DataFrame(rows)
        for rule in ("raw", "cor"):
            x = d[d.rule == rule].set_index("metric")
            print(f"\n{tag} {rule}")
            for k in ("p_text_frozen", "p_text_adaptive", "agreement_frozen", "agreement_adaptive", "gain_frozen", "gain_adaptive", "diff_agreement", "diff_gain",
                      "reversal_share", "reversal_frozen_text_to_audio", "reversal_frozen_audio_to_text", "reversed_weight_agrees_adaptive", "reversed_weight_agrees_frozen",
                      "mean_abs_frozen_text", "mean_abs_gate_term_text", "mean_abs_frozen_audio", "mean_abs_gate_term_audio", "mean_gap_change_text_minus_audio",
                      "share_gate_term_dampens_text", "share_gate_term_dampens_audio"):
                print(f"  {k:38s} {x.loc[k,'value']:.4f} [{x.loc[k,'lo']:.4f}, {x.loc[k,'hi']:.4f}]")
        # Q5 cross-tab: corrected vs raw rule, adaptive gate
        chg = R["c_raw"] != R["c_cor"]
        wchg = (R["e_raw_real"] > 0) != (R["e_cor_real"] > 0)
        ag_r, ag_c = (s > 0) == (R["e_raw_real"] > 0), (s > 0) == (R["e_cor_real"] > 0)
        change = np.where(ag_c & ~ag_r, "gained", np.where(~ag_c & ag_r, "lost", "unchanged"))
        G = s.shape[0]
        for cl in (True, False):
            for wc in (True, False):
                for ch in ("gained", "lost", "unchanged"):
                    m = (chg == cl) & (wchg == wc) & (change == ch)
                    cross_rows.append({"model": tag, "class_changed": cl, "winner_changed": wc, "agreement_change": ch, "mean_rows_per_gate": m.sum() / G, "share_of_all": m.mean()})
    ct = pd.DataFrame(cross_rows)
    ct.to_csv(RESULTS / "gate_adaptation_crosstab.csv", index=False)
    print("\nCross-tab (share of all rows, averaged over gates):")
    print(ct.pivot_table(index=["model", "class_changed", "winner_changed"], columns="agreement_change", values="share_of_all").round(4).to_string())


if __name__ == "__main__":
    main()
