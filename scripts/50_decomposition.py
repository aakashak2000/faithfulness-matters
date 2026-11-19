"""Step 50: decomposition of the weight's agreement with the swap into structural and learned parts. Protocol: docs/decomposition_protocol.md
(written 2026-11-17 before this script was run). Exploratory. No swap is rerun and no model is trained.

Version 3 (2026-11-19, docs/decomposition_protocol.md): also run without the 42 audio_shared rows
(results/decomposition_no_audio_shared.csv); the all-row output is unchanged. Helpers now come from common.py and audit_common.py.

Usage:  python scripts/50_decomposition.py
Needs: data/corrected_rule_arrays_{M1,M5}.npz, data/audit_cache/gates_{M1,M5}.pt
Output: results/decomposition.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import load_rescoring_inputs  # noqa: E402
from common import DATA, RESULTS, auroc, dialogue_draws, ece, spearman  # noqa: E402

N_BOOT = importlib.import_module("06_analysis").N_BOOT
N_PERM_POINT, N_PERM_DRAW = 200, 20
OUT = {"all": "decomposition.csv", "without_audio_shared": "decomposition_no_audio_shared.csv"}


def stats(W, TAU, LAB, R, ONLY, RC, idx, rng, n_perm):
    """Per gate, then averaged. W, TAU, LAB: (gates, n). R, ONLY, RC: (n,)."""
    res = {}
    for g in range(W.shape[0]):
        w, tau, lab = W[g, idx], TAU[g, idx], LAB[g, idx]
        v = np.isfinite(tau) & (w != 0.5)
        w, tau, lab = w[v], tau[v], lab[v]
        r, only, rc = R[idx][v], ONLY[idx][v], RC[idx][v]
        a_auc, a_ag = auroc(w, lab), ((w > 0.5) == lab).mean()
        nulls = []
        for _ in range(n_perm):
            wp = rng.permutation(w)
            lp_ = wp > tau
            nulls.append((auroc(wp, lp_), ((wp > 0.5) == lp_).mean()))
        n_auc, n_ag = np.mean(nulls, 0)
        row = {"auroc_actual": a_auc, "auroc_null": n_auc, "auroc_excess": a_auc - n_auc,
               "agreement_actual": a_ag, "agreement_null": n_ag, "agreement_excess": a_ag - n_ag,
               "w_R_auroc": auroc(w[only], r[only]), "w_tau_spearman": spearman(w, tau),
               "textshare_R_auroc": auroc(1 - tau[only], r[only]), "w_r_spearman": spearman(w, rc)}
        for k, x in row.items():
            res.setdefault(k, []).append(x)
    return {k: float(np.nanmean(x)) for k, x in res.items()}


def main():
    rows_by = {"all": [], "without_audio_shared": []}
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        A = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        Xm = load_rescoring_inputs(tag)
        X = Xm["test"]
        W = (A["s_w"] + 1) / 2
        assert np.abs(W - X["w"]).max() < 1e-5
        dt = np.abs(A["raw_fro_text_mean"]) / W
        da = np.abs(A["raw_fro_audio_mean"]) / (1 - W)
        TAU = da / (dt + da)
        E = A["e_raw_fro"]
        LAB = E > 0

        # ---- validation
        nz = E != 0
        mism = ((W > TAU) != LAB)[nz].sum()
        cf = max(np.abs(A["closed_form_err_text"]).max(), np.abs(A["closed_form_err_audio"]).max())
        ref = pd.read_csv(RESULTS / f"corrected_rule_swap_{tag}.csv")
        ref = ref[(ref.rule == "raw") & (ref.effect == "frozen_gate")].iloc[0].agreement
        ag = np.mean([((A["s_w"][g] > 0) == (E[g] > 0))[(A["s_w"][g] != 0) & nz[g]].mean() for g in range(len(W))])
        print(f"VALIDATION identity mismatches {mism}, closed-form err {cf:.1e}, agreement {ag:.10f} vs step 43 {ref:.10f}", flush=True)
        assert mism == 0 and cf < 1e-4 and abs(ag - ref) < 1e-9
        y = X["y"]
        t_ok, a_ok = X["pt"].argmax(1) == y, X["pa"].argmax(1) == y
        ONLY = t_ok != a_ok
        R = t_ok
        RC = np.log(np.clip(X["pt"][np.arange(len(y)), y], 1e-12, None)) - np.log(np.clip(X["pa"][np.arange(len(y)), y], 1e-12, None))
        draws = dialogue_draws(Xm["meta"]["dialogue_id"].values, N_BOOT)
        n = W.shape[1]
        shared = Xm["meta"]["audio_shared"].values.astype(bool)
        for rl, base in (("all", np.arange(n)), ("without_audio_shared", np.flatnonzero(~shared))):
            rows = rows_by[rl]
            mask = np.zeros(n, bool)
            mask[base] = True
            prim = np.mean([((W[g][base] > TAU[g][base]) == (A["e_cor_real"][g][base] > 0)).mean() for g in range(len(W))])
            print(f"[{rl}] share of rows where 1[w > tau] matches the primary effect (corrected, recomputed): {prim:.4f}", flush=True)
            prng = np.random.default_rng(1)
            pt_ = stats(W, TAU, LAB, R, ONLY, RC, base, prng, N_PERM_POINT)
            bs = [stats(W, TAU, LAB, R, ONLY, RC, d[mask[d]], prng, N_PERM_DRAW) for d in draws]
            for k, v in pt_.items():
                arr = np.array([b[k] for b in bs])
                lo, hi = np.nanpercentile(arr, [2.5, 97.5])
                rows.append({"model": tag, "part": "decomposition", "metric": k, "value": v, "lo": lo, "hi": hi})
            rows.append({"model": tag, "part": "decomposition", "metric": "share_tau_rule_matches_primary_effect", "value": prim})

            # ---- heads (gate-free)
            yb = y[base]
            for head, P in (("text", X["pt"] if rl == "all" else X["pt"][base]), ("audio", X["pa"] if rl == "all" else X["pa"][base])):
                ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
                for k, v in (("ece", ece(P, yb, bins=15)), ("mean_entropy", ent.mean()), ("mean_max_prob", P.max(1).mean()), ("accuracy", (P.argmax(1) == yb).mean())):
                    rows.append({"model": tag, "part": f"head_{head}", "metric": k, "value": float(v)})
            # all rows: use the full arrays so float32 summation order (and the committed output) is unchanged
            dtb, dab, tb = (dt, da, TAU) if rl == "all" else (dt[:, base], da[:, base], TAU[:, base])
            for k, v in (("mean_d_text", dtb.mean()), ("median_d_text", np.median(dtb, 1).mean()), ("mean_d_audio", dab.mean()),
                         ("median_d_audio", np.median(dab, 1).mean()), ("median_tau", np.median(tb, 1).mean()), ("share_tau_above_05", (tb > 0.5).mean())):
                rows.append({"model": tag, "part": "distinctiveness", "metric": k, "value": float(v)})
        print(f"{tag} done", flush=True)
    for rl, fn in OUT.items():
        df = pd.DataFrame(rows_by[rl])
        df.to_csv(RESULTS / fn, index=False)
        print(f"wrote {RESULTS / fn} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
