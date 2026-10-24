"""Step 41: does the ORDER of mean and absolute value in the primary swap effect matter?
Primary (06/33/37, the main results): effect_k = | mean over donors d of fused_d[c] - fused_0[c] |   (average the donor
fused vectors first, then take the change in the predicted class c, then the absolute value).
Variant:                                effect_k = mean over donors d of | fused_d[c] - fused_0[c] |   (absolute value per donor).
Both are computed from the same loops (scripts 38 and 39, which store e_real = primary and e_mabs = variant for the
recomputed-gate swap, and e_fro / e_mabs_fro for the frozen gate). Compared per model: P(text mattered more), the weight's
agreement, gain over the majority modality (dialogue bootstrap, 1000 draws, rng 0, gates averaged in each draw, paired),
Spearman with the weight, the share of utterances whose winner differs, and the corrected signal's agreement with each.
Output: results/mean_of_abs_check.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS  # noqa: E402

an06 = importlib.import_module("06_analysis")
a40 = importlib.import_module("40_audit_bootstrap")


def main():
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(an06.N_BOOT)]
    out = []
    for tag in ("M0b", "M1", "M2", "M3", "M5"):
        R = np.load(DATA / f"frozen_gate_arrays_{tag}.npz")
        w, c = R["s_w"], R["e_c34"]
        row = {"model": tag}
        for lab, e in (("primary", R["e_real"]), ("mean_of_abs", R["e_mabs"])):
            pt = a40.stat(w, e)
            pc = a40.stat(c, e)
            b = np.array([a40.stat(w[:, i], e[:, i]) for i in draws])
            lo, hi = np.percentile(b[:, 1], [2.5, 97.5])
            v = e != 0
            row.update({f"{lab}_p_text": float((e[v] > 0).mean()), f"{lab}_weight_agreement": pt[0], f"{lab}_weight_gain": pt[1],
                        f"{lab}_weight_gain_lo": lo, f"{lab}_weight_gain_hi": hi, f"{lab}_weight_spearman": pt[2],
                        f"{lab}_corrected_agreement": pc[0], f"{lab}_corrected_spearman": pc[2]})
        both = (R["e_real"] != 0) & (R["e_mabs"] != 0)
        row["share_winner_differs"] = float((np.sign(R["e_real"][both]) != np.sign(R["e_mabs"][both])).mean())
        row["spearman_primary_vs_variant"] = float(np.mean([np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
                                                           for a, b in zip(R["e_real"], R["e_mabs"])]))
        row["frozen_share_winner_differs"] = float((np.sign(R["e_fro"][both]) != np.sign(R["e_mabs_fro"][both])).mean())
        out.append(row)
        print(tag, {k: round(float(v), 3) for k, v in row.items() if k != "model"}, flush=True)
    pd.DataFrame(out).to_csv(RESULTS / "mean_of_abs_check.csv", index=False)


if __name__ == "__main__":
    main()
