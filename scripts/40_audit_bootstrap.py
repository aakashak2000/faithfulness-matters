"""Step 40: paired dialogue-bootstrap uncertainty for the audit comparisons (weight vs corrected signal vs real swap).
Unit = dialogue (test dialogues resampled with replacement), 1000 draws, rng 0 (same draws as 06/33/37), gates averaged
inside each draw. Every difference is PAIRED: weight and corrected signal are scored on the same resample against the same
real-swap effects. Reported per model: agreement, Spearman and gain over the majority modality for the weight and the
corrected signal, their paired differences with 95% intervals, and the share of utterances whose winner the gate's
re-weighting reverses. Input: data/frozen_gate_arrays_*.npz from 38 and 39. Output: results/audit_bootstrap.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS  # noqa: E402

an06 = importlib.import_module("06_analysis")


def stat(s, e):
    ag, pt, rho = [], [], []
    for sk, ek in zip(s, e):
        v = (sk != 0) & (ek != 0)
        ag.append((np.sign(sk[v]) == np.sign(ek[v])).mean())
        pt.append((ek[v] > 0).mean())
        rho.append(np.corrcoef(rankdata(sk), rankdata(ek))[0, 1])
    a, p = np.mean(ag), np.mean(pt)
    return np.array([a, a - max(p, 1 - p), np.mean(rho)])


def main():
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(an06.N_BOOT)]
    out = []
    for tag in ("M0b", "M1", "M2", "M3", "M5"):
        R = np.load(DATA / f"frozen_gate_arrays_{tag}.npz")
        e, w, c = R["e_real"], R["s_w"], R["e_c34"]
        pt = stat(w, e), stat(c, e)
        bw = np.array([stat(w[:, i], e[:, i]) for i in draws])
        bc = np.array([stat(c[:, i], e[:, i]) for i in draws])
        rev = np.array([((e[:, i] * R["e_fro"][:, i]) < 0).mean() for i in draws])
        row = {"model": tag}
        for j, nm in enumerate(("agreement", "gain_over_majority", "spearman")):
            for lab, p_, b in (("weight", pt[0], bw), ("corrected", pt[1], bc)):
                lo, hi = np.percentile(b[:, j], [2.5, 97.5])
                row.update({f"{lab}_{nm}": p_[j], f"{lab}_{nm}_lo": lo, f"{lab}_{nm}_hi": hi})
            d = bc[:, j] - bw[:, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            row.update({f"diff_{nm}": pt[1][j] - pt[0][j], f"diff_{nm}_lo": lo, f"diff_{nm}_hi": hi})
        lo, hi = np.percentile(rev, [2.5, 97.5])
        row.update({"share_reversed_by_gate": float(((e * R["e_fro"]) < 0).mean()), "share_reversed_lo": lo, "share_reversed_hi": hi})
        out.append(row)
        print(tag, {k: round(float(v), 3) for k, v in row.items() if k != "model" and ("diff" in k or "reversed" in k)}, flush=True)
    pd.DataFrame(out).to_csv(RESULTS / "audit_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
