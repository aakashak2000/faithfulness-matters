"""Step 15: gate diagnostic, "gate held fixed" (action item 10).

Question: when we remove a modality, the learned gate RECOMPUTES its weight from the changed
input. So the measured effect mixes two things: (a) the head output changing at the OLD weight,
and (b) the gate moving its weight. Does that re-weighting explain why the gate's original weight
fails to match the measured effect?

Definitions, fixed before running (this is my definition of item 10; the item was only named
before, see docs/plans/step-03... follow-up notes):
  Full-pool resampling donors and exclusion rule exactly as 08_resampling_full.py
  (all 7,990 clf rows, byte-identical embeddings excluded).
  For the learned-gate model, with w0 = gate weight on the original input, p_t, p_a the original
  head outputs, c the originally predicted class, P_d the donor head output, w_d the gate weight
  recomputed for donor d:
    fixed-weight effect   dp_text_fixed  = w0 * (mean_d P_t,d[c] - p_t[c])
                          dp_audio_fixed = (1 - w0) * (mean_d P_a,d[c] - p_a[c])
    re-weighting term     dp_text_reweight = dp_text(log) - dp_text_fixed   (same for audio)
  and the identity dp = fixed + re-weighting holds exactly because the fused output is linear
  in the weight.
  Outputs:
    1. gain / agreement / Spearman of the gate weight vs the FIXED-weight effects (dp and tvd),
       next to the same metrics for the recomputed effects (already in results/). Bootstrap as
       in 06 (dialogue clusters, 1000 draws, rng 0, 3 seeds averaged).
    2. share of utterances where the sign of the effect difference differs between fixed and
       recomputed effects (how often re-weighting changes "which modality mattered more").
    3. median |head term| and |re-weighting term| per modality.
  Caveat attached to any result: with the weight held fixed, the effect of each modality is
  proportional to its weight by construction, so agreement with the fixed-weight effect is partly
  structural (the same caveat as the confidence rule). It measures how much of the gate's
  mismatch is due to its own re-weighting, not whether the gate is a good explanation.
  Reading rule: if the gate's gain under fixed-weight effects is clearly higher than under
  recomputed effects (interval of the difference excludes 0), re-weighting accounts for part of
  the gap; if not, the mismatch comes from the heads.
  Check: for fusion_average (weight constant 0.5) the same code must reproduce the dp_text and
  dp_audio of results/resampling_full_log.csv to 1e-5 (validates donor masks and averaging).

Usage: python scripts/15_gate_fixed.py
Output: results/gate_fixed_diagnostic.csv, results/gate_fixed_summary.json
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_dataset  # noqa: E402

iv04 = importlib.import_module("04_interventions")
rs07 = importlib.import_module("07_resampling")
an06 = importlib.import_module("06_analysis")
SEEDS = [0, 1, 2]


@torch.no_grad()
def fixed_effects(seed, e_t, e_a, pool_t, pool_a, ok_t, ok_a):
    th, ah, gate = iv04.load_models(seed)
    orig, p_t, p_a = iv04.run_all_models(e_t, e_a, th, ah, gate)
    n = e_t.shape[0]
    idx = torch.arange(n)
    Pt_bar = (ok_t.float() @ torch.softmax(th(pool_t), 1)) / ok_t.sum(1, keepdim=True)
    Pa_bar = (ok_a.float() @ torch.softmax(ah(pool_a), 1)) / ok_a.sum(1, keepdim=True)
    out = {}
    for m, w0 in (("fusion_learned_gate", orig["fusion_learned_gate"][1]),
                  ("fusion_average", torch.full((n, 1), 0.5))):
        c = orig[m][0].argmax(1)
        dpt = (w0 * (Pt_bar - p_t))[idx, c]
        dpa = ((1 - w0) * (Pa_bar - p_a))[idx, c]
        tvt = 0.5 * (w0 * (Pt_bar - p_t)).abs().sum(1)
        tva = 0.5 * ((1 - w0) * (Pa_bar - p_a)).abs().sum(1)
        out[m] = (dpt.numpy(), dpa.numpy(), tvt.numpy(), tva.numpy())
    return out


def main():
    D = load_dataset()
    test, clf = D["splits"]["test"], D["splits"]["clf"]
    e_t, e_a = test["text"], test["audio"]
    log = pd.read_csv(RESULTS / "resampling_full_log.csv")
    ok_t, ok_a = rs07.valid_masks(e_t, clf["text"]), rs07.valid_masks(e_a, clf["audio"])

    rows, S_fix, E_fix, E_fix_tvd, E_rec, E_rec_tvd, S_sig = [], [], [], [], [], [], []
    summary = {}
    for seed in SEEDS:
        fx = fixed_effects(seed, e_t, e_a, clf["text"], clf["audio"], ok_t, ok_a)
        # check on the constant-weight model
        a = log[(log.seed == seed) & (log.model == "fusion_average")].sort_values("row")
        assert np.allclose(fx["fusion_average"][0], a.dp_text.values, atol=1e-5), "average-model check (text)"
        assert np.allclose(fx["fusion_average"][1], a.dp_audio.values, atol=1e-5), "average-model check (audio)"
        g = log[(log.seed == seed) & (log.model == "fusion_learned_gate")].sort_values("row")
        dpt, dpa, tvt, tva = fx["fusion_learned_gate"]
        rt, ra = g.dp_text.values - dpt, g.dp_audio.values - dpa
        E_fix.append(np.abs(dpt) - np.abs(dpa))
        E_fix_tvd.append(tvt - tva)
        E_rec.append(g.dp_text.abs().values - g.dp_audio.abs().values)
        E_rec_tvd.append(g.tvd_text.values - g.tvd_audio.values)
        S_sig.append(2 * g.gate_w_t.values - 1)
        summary[f"seed{seed}"] = {
            "median_abs_head_term_text": float(np.median(np.abs(dpt))),
            "median_abs_reweight_term_text": float(np.median(np.abs(rt))),
            "median_abs_head_term_audio": float(np.median(np.abs(dpa))),
            "median_abs_reweight_term_audio": float(np.median(np.abs(ra)))}
    print("check passed: fusion_average fixed-weight code reproduces results/resampling_full_log.csv")
    s, ef, eft, er, ert = (np.stack(x) for x in (S_sig, E_fix, E_fix_tvd, E_rec, E_rec_tvd))

    # bootstrap (same draws as 06)
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == gid) for gid in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]
    names = ["agreement", "always_text_baseline", "gain", "spearman"]
    boots = {}
    for label, e in (("recomputed_dp", er), ("fixed_dp", ef), ("recomputed_tvd", ert), ("fixed_tvd", eft)):
        point = np.array(an06.stats(s, e))
        b = np.array([an06.stats(s[:, i], e[:, i]) for i in draws])
        boots[label] = b
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        for k, nme in enumerate(names):
            rows.append({"effects": label, "metric": nme, "estimate": point[k], "ci_low": lo[k], "ci_high": hi[k]})
    for kind in ("dp", "tvd"):
        d = boots[f"fixed_{kind}"][:, 2] - boots[f"recomputed_{kind}"][:, 2]
        est = [r for r in rows if r["effects"] == f"fixed_{kind}" and r["metric"] == "gain"][0]["estimate"] \
            - [r for r in rows if r["effects"] == f"recomputed_{kind}" and r["metric"] == "gain"][0]["estimate"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"effects": f"fixed_minus_recomputed_{kind}", "metric": "gain", "estimate": est,
                     "ci_low": lo, "ci_high": hi})
    flip_dp = float(np.mean([np.sign(a) != np.sign(b) for a, b in zip(ef.ravel(), er.ravel()) if a != 0 and b != 0]))
    summary["share_sign_of_effect_difference_changes_dp"] = flip_dp
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "gate_fixed_diagnostic.csv", index=False)
    (RESULTS / "gate_fixed_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)
    print(res.to_string(index=False))
    print(f"\nshare of utterances where the sign of the dp effect difference changes: {flip_dp:.3f}")
    for k, v in summary.items():
        if k.startswith("seed"):
            print(k, {a: round(b, 4) for a, b in v.items()})


if __name__ == "__main__":
    main()
