"""Step 54: reference sensitivity (part A) and swap-free selective attribution (part B). Protocol: docs/reference_pools_protocol.md
(written 2026-11-19 before this script was run). Exploratory; no training of models (part B fits one logistic regression on dev).

Usage:  python scripts/54_selective_attribution.py
Needs:  data/reference_pools_{M1,M5}.npz (step 53), data/corrected_rule_arrays_{M}.npz (step 43), data/audit_dev_arrays_{M}.npz (step 42)
Output: results/reference_sensitivity.csv, results/selective_attribution.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import donor_pool_means  # noqa: E402
from common import DATA, RESULTS, dialogue_draws, js_div  # noqa: E402

N_BOOT = importlib.import_module("06_analysis").N_BOOT
COVER = np.arange(1, 21) / 20


def features(W, pt, pa, mt, ma):
    """Swap-free features per (gate, row). W: (gates, n)."""
    n = pt.shape[0]
    ii = np.arange(n)
    ent = lambda p: -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    c0 = (0.5 * pt + 0.5 * pa).argmax(1)
    d0t, d0a = np.abs(pt[ii, c0] - mt[ii, c0]), np.abs(pa[ii, c0] - ma[ii, c0])
    shared = [pt.max(1), pa.max(1), ent(pt), ent(pa), (pt.argmax(1) == pa.argmax(1)).astype(float), js_div(pt, pa), d0t, d0a]
    F, R2 = [], []
    for w in W:
        c = (w[:, None] * pt + (1 - w[:, None]) * pa).argmax(1)
        dt, da = np.abs(pt[ii, c] - mt[ii, c]), np.abs(pa[ii, c] - ma[ii, c])
        tau = da / np.clip(dt + da, 1e-12, None)
        r2 = w * dt / np.clip(w * dt + (1 - w) * da, 1e-12, None)
        F.append(np.column_stack([w, np.abs(w - 0.5), r2, np.abs(r2 - 0.5), tau, np.abs(w - tau), dt, da] + shared))
        R2.append(r2)
    return np.stack(F), np.stack(R2)          # (gates, n, f), (gates, n)


def risk_curve(score, wrong):
    order = np.argsort(-score, kind="stable")
    cw = np.cumsum(wrong[order])
    k = np.maximum(1, np.ceil(COVER * len(score)).astype(int))
    return cw[k - 1] / k


def main():
    rows_a, rows_b = [], []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        P = np.load(DATA / f"reference_pools_{tag}.npz")
        A = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        Dv = np.load(DATA / f"audit_dev_arrays_{tag}.npz")
        Wt, Wd = (A["s_w"] + 1) / 2, (Dv["s_w"] + 1) / 2
        assert np.abs(P["test_global50_s_w"] - A["s_w"]).max() < 1e-6 and np.abs(P["dev_global50_s_w"] - Dv["s_w"]).max() < 1e-6
        G = len(Wt)

        # ---------------- part A: winners under each pool (test)
        inc = P["test_speaker_included"]
        meta_t = None
        win = {}
        for dname, key, fullkey in (("abs_of_mean", "e_cor_real", "e_cor_real"), ("mean_of_abs", "e_cor_real_mabs", "e_cor_real_mabs")):
            win[dname] = {"full": A[fullkey] > 0, **{p: P[f"test_{p}_{key}"] > 0 for p in ("global50", "speaker50", "nn50")}}
        pt, pa, mt, ma, y, meta_t = donor_pool_means(tag, "test")
        draws = dialogue_draws(meta_t["dialogue_id"].values, N_BOOT)
        base = np.flatnonzero(inc)
        mask = np.zeros(len(inc), bool)
        mask[base] = True

        def part_a(idx, wd):
            r = {}
            pairs = [("full", "global50"), ("full", "speaker50"), ("full", "nn50"), ("global50", "speaker50"), ("global50", "nn50"), ("speaker50", "nn50")]
            for a, b in pairs:
                r[f"disagree_{a}_{b}"] = np.mean([(wd[a][g, idx] != wd[b][g, idx]).mean() for g in range(G)])
            allp = np.stack([wd[p][:, idx] for p in ("full", "global50", "speaker50", "nn50")])
            r["share_not_unanimous"] = np.mean((allp.min(0) != allp.max(0)).mean(1))
            r["share_speaker_or_nn_disagree_full"] = np.mean([((wd["speaker50"][g, idx] != wd["full"][g, idx]) | (wd["nn50"][g, idx] != wd["full"][g, idx])).mean() for g in range(G)])
            r["excess_either_over_noise_floor"] = r["share_speaker_or_nn_disagree_full"] - r["disagree_full_global50"]
            r["excess_speaker_over_noise_floor"] = r["disagree_full_speaker50"] - r["disagree_full_global50"]
            r["excess_nn_over_noise_floor"] = r["disagree_full_nn50"] - r["disagree_full_global50"]
            for p in ("full", "global50", "speaker50", "nn50"):
                r[f"gate_agreement_{p}"] = np.mean([((Wt[g, idx] > 0.5) == wd[p][g, idx]).mean() for g in range(G)])
            return r

        for dname, wd in win.items():
            pt_ = part_a(base, wd)
            bs = [part_a(d[mask[d]], wd) for d in draws]
            for k, v in pt_.items():
                lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
                rows_a.append({"model": tag, "definition": dname, "metric": k, "value": v, "lo": lo, "hi": hi, "n_rows": len(base)})
        print(f"{tag} part A done: speaker50 rows {len(base)}", flush=True)

        # ---------------- part B: selective attribution
        ptd, pad, mtd, mad, yd, _ = donor_pool_means(tag, "dev")
        Fd, R2d = features(Wd, ptd, pad, mtd, mad)
        Ft, R2t = features(Wt, pt, pa, mt, ma)
        claims = {"gate": (Wd > 0.5, Wt > 0.5, lambda W, R2: np.abs(W - 0.5)), "R2": (R2d > 0.5, R2t > 0.5, lambda W, R2: np.abs(R2 - 0.5))}
        dev_win = {p: P[f"dev_{p}_e_cor_real"] > 0 for p in ("global50", "nn50")}
        rng = np.random.default_rng(7)
        rand = rng.random(Wt.shape)
        t_ok, a_ok = pt.argmax(1) == y, pa.argmax(1) == y
        only = t_ok != a_ok
        targets = {"speaker50": win["abs_of_mean"]["speaker50"], "global50": win["abs_of_mean"]["global50"],
                   "nn50": win["abs_of_mean"]["nn50"], "full": win["abs_of_mean"]["full"]}
        for cname, (cd, ct, marg) in claims.items():
            yfit = ((cd == dev_win["global50"]) & (cd == dev_win["nn50"])).ravel()
            Xd = Fd.reshape(-1, Fd.shape[2])
            mu, sd = Xd.mean(0), Xd.std(0) + 1e-9
            lr = LogisticRegression().fit((Xd - mu) / sd, yfit)
            learned = lr.predict_proba((Ft.reshape(-1, Ft.shape[2]) - mu) / sd)[:, 1].reshape(Wt.shape)
            scores = {"learned": learned, "margin": marg(Wt, R2t), "random": rand}
            print(f"{tag} {cname}: dev fit, share of claims holding under both dev pools {yfit.mean():.3f}", flush=True)
            for tname, tw in targets.items():
                wrong = ct != tw

                def part_b(idx):
                    r = {}
                    for sname, S in scores.items():
                        curves = np.stack([risk_curve(S[g, idx], wrong[g, idx]) for g in range(G)])
                        r[f"aurc_{sname}"] = curves.mean(1).mean()
                        for cv in (0.5, 0.75, 0.9):
                            r[f"risk_at_{int(cv * 100)}_{sname}"] = curves[:, int(round(cv * 20)) - 1].mean()
                        if tname == "speaker50":
                            ok1 = only[idx]
                            hits = []
                            for g in range(G):
                                s_, c_, t_ = S[g, idx][ok1], ct[g, idx][ok1], t_ok[idx][ok1]
                                keep = np.argsort(-s_, kind="stable")[: int(np.ceil(0.5 * len(s_)))]
                                hits.append((c_[keep] == t_[keep]).mean())
                            r[f"correct_head_named_at_50_{sname}"] = np.mean(hits)
                    r["aurc_learned_minus_margin"] = r["aurc_learned"] - r["aurc_margin"]
                    r["risk_full_coverage"] = np.mean([wrong[g, idx].mean() for g in range(G)])
                    return r
                pt_ = part_b(base)
                bs = [part_b(d[mask[d]]) for d in draws]
                for k, v in pt_.items():
                    lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
                    rows_b.append({"model": tag, "claim": cname, "target_pool": tname, "metric": k, "value": v, "lo": lo, "hi": hi})
            print(f"{tag} {cname} part B done", flush=True)
    pd.DataFrame(rows_a).to_csv(RESULTS / "reference_sensitivity.csv", index=False)
    pd.DataFrame(rows_b).to_csv(RESULTS / "selective_attribution.csv", index=False)
    print("wrote results/reference_sensitivity.csv and results/selective_attribution.csv", flush=True)


if __name__ == "__main__":
    main()
