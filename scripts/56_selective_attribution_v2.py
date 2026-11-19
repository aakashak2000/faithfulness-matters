"""Step 56: phase 1 version 2 (Amendment 1 of docs/reference_pools_protocol.md, written 2026-11-19 before this script was run).
Part B refitted on dev WITHOUT nn50 (global50 + speaker50), evaluated on test against the held-out nn50 pool on all rows, with the
identity score |w - tau| as the comparator the fusion equation gives for free. Part A additions: direction of full-to-nn50 flips,
nn50 comparisons on all rows, and the noise floor over four random draws. Version 1 outputs are not modified. Exploratory.

Usage:  python scripts/56_selective_attribution_v2.py
Output: results/reference_sensitivity_v2.csv, results/selective_attribution_v2.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS, dialogue_draws  # noqa: E402

N_BOOT = importlib.import_module("06_analysis").N_BOOT
COVER = np.arange(1, 21) / 20


def boot_rows(fn, base, draws, n):
    mask = np.zeros(n, bool)
    mask[base] = True
    pt_ = fn(base)
    bs = [fn(d[mask[d]]) for d in draws]
    return [(k, v, *np.percentile([b[k] for b in bs], [2.5, 97.5])) for k, v in pt_.items()]


def main():
    rows_a, rows_b = [], []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        P1, P2 = np.load(DATA / f"reference_pools_{tag}.npz"), np.load(DATA / f"reference_pools_v2_{tag}.npz")
        A, Dv = np.load(DATA / f"corrected_rule_arrays_{tag}.npz"), np.load(DATA / f"audit_dev_arrays_{tag}.npz")
        Wt, Wd = (A["s_w"] + 1) / 2, (Dv["s_w"] + 1) / 2
        G = len(Wt)
        pt, pa, mt, ma, y, meta = ac.donor_pool_means(tag, "test")
        n = len(y)
        allr = np.arange(n)
        spk_rows = np.flatnonzero(P1["test_speaker_included"])
        draws = dialogue_draws(meta["dialogue_id"].values, N_BOOT)
        win = {"full": A["e_cor_real"] > 0, "nn50": P1["test_nn50_e_cor_real"] > 0, "speaker50": P1["test_speaker50_e_cor_real"] > 0,
               "global50": P1["test_global50_e_cor_real"] > 0}
        for s in (545, 546, 547):
            win[f"global50_s{s}"] = P2[f"test_global50_s{s}_e_cor_real"] > 0

        # ---------------- part A additions
        don_spk = ac.donor_speakers()
        meta_r, emb, okd, _, _ = ac.pool_runner(tag, 0.0)
        nnm, _ = ac.build_pool_masks(okd["test"], emb["test"], emb["clf"], meta_r["test"]["speaker"].values, don_spk, ("nn50",))
        _, _, mt_nn, ma_nn, _, _ = ac.donor_pool_means(tag, "test", ok=nnm["nn50"])
        ii = np.arange(n)
        dshift = {}
        for g in range(G):
            c = (Wt[g][:, None] * pt + (1 - Wt[g][:, None]) * pa).argmax(1)
            for nm, P_, M_, Mn in (("d_text", pt, mt, mt_nn), ("d_audio", pa, ma, ma_nn)):
                dshift.setdefault(f"{nm}_full", []).append(np.abs(P_[ii, c] - M_[ii, c]))
                dshift.setdefault(f"{nm}_nn50", []).append(np.abs(P_[ii, c] - Mn[ii, c]))
        dshift = {k: np.stack(v) for k, v in dshift.items()}

        def part_a(idx):
            r = {}
            f, nn = win["full"][:, idx], win["nn50"][:, idx]
            r["disagree_full_nn50"] = (f != nn).mean()
            r["disagree_global50_nn50"] = (win["global50"][:, idx] != nn).mean()
            r["flip_text_to_audio"] = (f & ~nn).mean()
            r["flip_audio_to_text"] = (~f & nn).mean()
            for k in ("d_text", "d_audio"):
                r[f"mean_{k}_full"] = dshift[f"{k}_full"][:, idx].mean()
                r[f"mean_{k}_nn50"] = dshift[f"{k}_nn50"][:, idx].mean()
            floors = [(win["full"][:, idx] != win[p][:, idx]).mean() for p in ("global50", "global50_s545", "global50_s546", "global50_s547")]
            r["noise_floor_mean4"], r["noise_floor_min4"], r["noise_floor_max4"] = np.mean(floors), np.min(floors), np.max(floors)
            r["excess_nn50_over_noise_mean4"] = r["disagree_full_nn50"] - r["noise_floor_mean4"]
            return r

        for lbl, base in (("all_rows", allr), ("speaker_rows", spk_rows)):
            for k, v, lo, hi in boot_rows(part_a, base, draws, n):
                rows_a.append({"model": tag, "rows": lbl, "n_rows": len(base), "metric": k, "value": v, "lo": lo, "hi": hi})
        print(f"{tag} part A additions done", flush=True)

        # ---------------- part B version 2
        ptd, pad, mtd, mad, yd, _ = ac.donor_pool_means(tag, "dev")
        Fd, R2d, TAUd = ac.swapfree_features(Wd, ptd, pad, mtd, mad)
        Ft, R2t, TAUt = ac.swapfree_features(Wt, pt, pa, mt, ma)
        dev_inc = P2["dev_speaker_included"]
        dwin = {"global50": P1["dev_global50_e_cor_real"] > 0, "speaker50": P2["dev_speaker50_e_cor_real"] > 0}
        rand = np.random.default_rng(7).random(Wt.shape)
        t_ok = pt.argmax(1) == y
        only = t_ok != (pa.argmax(1) == y)
        dist = np.abs(Wt - TAUt)
        claims = {
            "R2": (R2d > 0.5, R2t > 0.5, np.abs(R2t - 0.5), dist),
            "gate": (Wd > 0.5, Wt > 0.5, np.abs(Wt - 0.5), np.where((Wt > 0.5) == (Wt > TAUt), dist, -dist)),
        }
        for cname, (cd, ct, margin, ident) in claims.items():
            yfit = ((cd == dwin["global50"]) & (cd == dwin["speaker50"]))[:, dev_inc].ravel()
            Xd = Fd[:, dev_inc].reshape(-1, Fd.shape[2])
            mu, sd = Xd.mean(0), Xd.std(0) + 1e-9
            lr = LogisticRegression().fit((Xd - mu) / sd, yfit)
            learned = lr.predict_proba((Ft.reshape(-1, Ft.shape[2]) - mu) / sd)[:, 1].reshape(Wt.shape)
            scores = {"learned": learned, "identity": ident, "margin": margin, "random": rand}
            print(f"{tag} {cname}: refit on dev global50+speaker50 ({dev_inc.sum()} rows per gate), share holding {yfit.mean():.3f}", flush=True)
            for tname, rows in (("nn50", allr), ("speaker50", spk_rows), ("global50", allr), ("full", allr)):
                wrong = ct != win[tname]

                def part_b(idx):
                    r = {}
                    for sname, S in scores.items():
                        curves = np.stack([ac.risk_curve(S[g, idx], wrong[g, idx], COVER) for g in range(G)])
                        r[f"aurc_{sname}"] = curves.mean()
                        for cv in (0.5, 0.75, 0.9):
                            r[f"risk_at_{int(cv * 100)}_{sname}"] = curves[:, int(round(cv * 20)) - 1].mean()
                        if tname == "nn50":
                            ok1 = only[idx]
                            hits = []
                            for g in range(G):
                                s_, c_, t_ = S[g, idx][ok1], ct[g, idx][ok1], t_ok[idx][ok1]
                                keep = np.argsort(-s_, kind="stable")[: int(np.ceil(0.5 * len(s_)))]
                                hits.append((c_[keep] == t_[keep]).mean())
                            r[f"correct_head_named_at_50_{sname}"] = np.mean(hits)
                    r["aurc_learned_minus_identity"] = r["aurc_learned"] - r["aurc_identity"]
                    r["aurc_learned_minus_margin"] = r["aurc_learned"] - r["aurc_margin"]
                    r["aurc_identity_minus_margin"] = r["aurc_identity"] - r["aurc_margin"]
                    r["risk_full_coverage"] = wrong[:, idx].mean()
                    return r
                for k, v, lo, hi in boot_rows(part_b, rows, draws, n):
                    rows_b.append({"model": tag, "claim": cname, "target_pool": tname, "n_rows": len(rows), "metric": k, "value": v, "lo": lo, "hi": hi})
            print(f"{tag} {cname} part B done", flush=True)
    pd.DataFrame(rows_a).to_csv(RESULTS / "reference_sensitivity_v2.csv", index=False)
    pd.DataFrame(rows_b).to_csv(RESULTS / "selective_attribution_v2.csv", index=False)
    print("wrote results/reference_sensitivity_v2.csv and results/selective_attribution_v2.csv", flush=True)


if __name__ == "__main__":
    main()
