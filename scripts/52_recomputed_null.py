"""Step 52: recomputed-prediction permutation null, version 2 (Amendment 2 of docs/decomposition_protocol.md, written 2026-11-17
before this script was run). Raw rule, frozen gate. Under a shuffled weight the fused class, distinctiveness and swap winner are
all recomputed, so the null is a consistent random-assignment gate. Exploratory. Steps 50 and 51 outputs are not modified.

Usage:  python scripts/52_recomputed_null.py
Output: results/decomposition_null_v2.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS  # noqa: E402

r49 = importlib.import_module("49_rescoring")
r51 = importlib.import_module("51_gate_free_tau")
auroc, N_BOOT = r49.auroc, r49.N_BOOT
N_PERM_POINT, N_PERM_DRAW = 200, 20


def winner(w, pt, pa, mt, ma):
    """Fused class, label (text won, raw rule, frozen gate) and correctness inputs for weight vector w."""
    ii = np.arange(len(w))
    c = (w[:, None] * pt + (1 - w[:, None]) * pa).argmax(1)
    dt, da = np.abs(pt[ii, c] - mt[ii, c]), np.abs(pa[ii, c] - ma[ii, c])
    return c, w * dt > (1 - w) * da, w * dt - (1 - w) * da


def stats(W, pt, pa, mt, ma, y, idx, rng, n_perm):
    res = {}
    P, Q, MT, MA, Y = pt[idx], pa[idx], mt[idx], ma[idx], y[idx]
    for g in range(W.shape[0]):
        w = W[g, idx]
        c, lab, e = winner(w, P, Q, MT, MA)
        v = (w != 0.5) & (e != 0)
        r = {"agreement_actual": ((w > 0.5) == lab)[v].mean(), "auroc_actual": auroc(w[v], lab[v]), "accuracy_actual": (c == Y).mean()}
        nl = []
        for _ in range(n_perm):
            wp = rng.permutation(w)
            cp, lp_, ep = winner(wp, P, Q, MT, MA)
            vp = (wp != 0.5) & (ep != 0)
            nl.append((((wp > 0.5) == lp_)[vp].mean(), auroc(wp[vp], lp_[vp]), (cp == Y).mean()))
        na, nu, nc = np.mean(nl, 0)
        r.update({"agreement_null_v2": na, "auroc_null_v2": nu, "accuracy_null_v2": nc,
                  "agreement_excess_v2": r["agreement_actual"] - na, "auroc_excess_v2": r["auroc_actual"] - nu,
                  "agreement_null_share_v2": na / r["agreement_actual"]})
        for k, x in r.items():
            res.setdefault(k, []).append(x)
    return {k: float(np.mean(x)) for k, x in res.items()}


def main():
    rows = []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        A = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        W = (A["s_w"] + 1) / 2
        pt, pa, mt, ma, y, meta = r51.pool_means(tag)
        # validation
        bad_c = bad_l = 0
        for g in range(len(W)):
            c, lab, _ = winner(W[g], pt, pa, mt, ma)
            nz = A["e_raw_fro"][g] != 0
            bad_c += int((c != A["c_raw"][g]).sum())
            bad_l += int((lab != (A["e_raw_fro"][g] > 0))[nz].sum())
        print(f"VALIDATION recomputed class mismatches {bad_c}, label mismatches {bad_l}", flush=True)
        assert bad_c == 0 and bad_l == 0
        dlg = meta["dialogue_id"].values
        groups = [np.flatnonzero(dlg == g) for g in np.unique(dlg)]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
        prng = np.random.default_rng(1)
        p = stats(W, pt, pa, mt, ma, y, np.arange(len(y)), prng, N_PERM_POINT)
        bs = [stats(W, pt, pa, mt, ma, y, d, prng, N_PERM_DRAW) for d in draws]
        for k, v in p.items():
            lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
            rows.append({"model": tag, "null": "recomputed_prediction_v2", "metric": k, "value": v, "lo": lo, "hi": hi})
        print(f"{tag} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "decomposition_null_v2.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
