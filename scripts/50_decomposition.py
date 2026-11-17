"""Step 50: decomposition of the weight's agreement with the swap into structural and learned parts. Protocol: docs/decomposition_protocol.md
(written 2026-11-17 before this script was run). Exploratory. No swap is rerun and no model is trained.

Usage:  python scripts/50_decomposition.py
Needs: data/corrected_rule_arrays_{M1,M5}.npz, data/audit_cache/gates_{M1,M5}.pt
Output: results/decomposition.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS  # noqa: E402

r49 = importlib.import_module("49_rescoring")
auroc, load_model, N_BOOT = r49.auroc, r49.load_model, r49.N_BOOT
N_PERM_POINT, N_PERM_DRAW = 200, 20


def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return out


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
    rows = []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        A = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        Xm = load_model(tag)
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
        prim = np.mean([((W[g] > TAU[g]) == (A["e_cor_real"][g] > 0)).mean() for g in range(len(W))])
        print(f"share of rows where 1[w > tau] matches the primary effect (corrected, recomputed): {prim:.4f}", flush=True)

        y = X["y"]
        t_ok, a_ok = X["pt"].argmax(1) == y, X["pa"].argmax(1) == y
        ONLY = t_ok != a_ok
        R = t_ok
        RC = np.log(np.clip(X["pt"][np.arange(len(y)), y], 1e-12, None)) - np.log(np.clip(X["pa"][np.arange(len(y)), y], 1e-12, None))

        dlg = Xm["meta"]["dialogue_id"].values
        groups = [np.flatnonzero(dlg == g) for g in np.unique(dlg)]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
        prng = np.random.default_rng(1)
        n = W.shape[1]
        pt_ = stats(W, TAU, LAB, R, ONLY, RC, np.arange(n), prng, N_PERM_POINT)
        bs = [stats(W, TAU, LAB, R, ONLY, RC, d, prng, N_PERM_DRAW) for d in draws]
        for k, v in pt_.items():
            arr = np.array([b[k] for b in bs])
            lo, hi = np.nanpercentile(arr, [2.5, 97.5])
            rows.append({"model": tag, "part": "decomposition", "metric": k, "value": v, "lo": lo, "hi": hi})
        rows.append({"model": tag, "part": "decomposition", "metric": "share_tau_rule_matches_primary_effect", "value": prim})

        # ---- heads (gate-free)
        for head, P in (("text", X["pt"]), ("audio", X["pa"])):
            ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
            for k, v in (("ece", ece(P, y)), ("mean_entropy", ent.mean()), ("mean_max_prob", P.max(1).mean()), ("accuracy", (P.argmax(1) == y).mean())):
                rows.append({"model": tag, "part": f"head_{head}", "metric": k, "value": float(v)})
        for k, v in (("mean_d_text", dt.mean()), ("median_d_text", np.median(dt, 1).mean()), ("mean_d_audio", da.mean()),
                     ("median_d_audio", np.median(da, 1).mean()), ("median_tau", np.median(TAU, 1).mean()), ("share_tau_above_05", (TAU > 0.5).mean())):
            rows.append({"model": tag, "part": "distinctiveness", "metric": k, "value": float(v)})
        print(f"{tag} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "decomposition.csv", index=False)
    print(f"wrote {RESULTS / 'decomposition.csv'} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
