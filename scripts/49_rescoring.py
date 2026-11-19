"""Step 49: re-scoring of the existing corrected-rule swap results (item 0 of the revised plan). Protocol: docs/rescoring_protocol.md
(written 2026-11-17 before this script was run). Exploratory. No swap is rerun and no model is trained; the arrays of steps 42/43 are
re-summarised and the head probabilities are rebuilt with audit_common exactly as in step 43.

Version 2 (2026-11-19, docs/rescoring_protocol.md "Version 2"): blocks B to E also without the 42 audio_shared rows; class-dependent
metrics set to missing for a gate with fewer than 5 rows of either class; n_valid_draws and underpowered reported.
results/rescoring.csv (version 1) is kept unchanged; this script now writes results/rescoring_v2.csv.

Usage:  python scripts/49_rescoring.py
Needs: data/corrected_rule_arrays_{M1,M5}.npz, data/audit_dev_arrays_{M1,M5}.npz, data/audit_cache/gates_{M1,M5}.pt
Output: results/rescoring_v2.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import load_rescoring_inputs  # noqa: E402
from common import DATA, RESULTS, auroc, bal_acc, dialogue_draws, js_div, kappa  # noqa: E402

an06 = importlib.import_module("06_analysis")
N_BOOT = an06.N_BOOT
ALPHA = {"M1": 0.25, "M5": 0.5}
MIN_CLASS = 5          # version 2: fewer rows than this in either class makes a gate's class-dependent metrics missing
MIN_VALID_DRAWS = 950  # version 2: below this a row is flagged underpowered
NAN = float("nan")


# ---------------------------------------------------------------- per-gate metric blocks on a row subset
def block_metrics(S, E, idx, extra):
    """S, E: (gates, n). idx: row indices (with repeats under the bootstrap). extra: per-gate dict of fixed things."""
    res = {}
    for g in range(S.shape[0]):
        s, e = S[g, idx], E[g, idx]
        v = (s != 0) & (e != 0)
        if "keep" in extra:
            v &= extra["keep"][g][idx]
        s, e = s[v], e[v]
        lab, pred = e > 0, s > 0
        ok = min(lab.sum(), (~lab).sum()) >= MIN_CLASS
        r = {"agreement": (pred == lab).mean(), "auroc": auroc(s, lab) if ok else NAN, "bal_acc_05": bal_acc(pred, lab) if ok else NAN,
             "kappa_05": kappa(pred, lab) if ok else NAN, "audio_recall_05": (~pred[~lab]).mean() if ok else NAN,
             "share_w_below_05": (~pred).mean(), "share_text_won": lab.mean(), "n_audio_won": float((~lab).sum()), "n_rows": float(v.sum())}
        if "t" in extra:
            pt_ = (s + 1) / 2 > extra["t"][g]
            r.update({"bal_acc_devt": bal_acc(pt_, lab) if ok else NAN, "kappa_devt": kappa(pt_, lab) if ok else NAN,
                      "audio_recall_devt": (~pt_[~lab]).mean() if ok else NAN})
        if "rel" in extra:
            rel = extra["rel"][idx][v]
            ok1 = extra["only"][idx][v]
            ok_r = min(rel[ok1].sum(), (~rel[ok1]).sum()) >= MIN_CLASS
            ok_i = min(lab[ok1].sum(), (~lab[ok1]).sum()) >= MIN_CLASS
            ra, ia = (auroc(s[ok1], rel[ok1]) if ok_r else NAN), (auroc(s[ok1], lab[ok1]) if ok_i else NAN)
            r.update({"rel_auroc": ra, "inf_auroc": ia, "rel_minus_inf": ra - ia,
                      "correct_head_won": (rel[ok1] == lab[ok1]).mean(), "n_only_one": float(ok1.sum())})
        if "mean_abs_e" in extra:
            r["mean_abs_e"] = np.abs(e).mean()
        for k, x in r.items():
            res.setdefault(k, []).append(x)
    return {k: (float(np.nanmean(x)) if np.isfinite(x).any() else NAN) for k, x in res.items()}


def boot(S, E, rows, extra, draws):
    """rows: the subset of test rows; draws are dialogue-cluster resamples of all rows, intersected with the subset."""
    mask = np.zeros(S.shape[1], bool)
    mask[rows] = True
    pt = block_metrics(S, E, np.flatnonzero(mask), extra)
    bs = [block_metrics(S, E, d[mask[d]], extra) for d in draws]
    out = []
    for k, v in pt.items():
        arr = np.array([b[k] for b in bs])
        nv = int(np.isfinite(arr).sum())
        lo, hi = np.nanpercentile(arr, [2.5, 97.5]) if nv else (NAN, NAN)
        out.append((k, v, lo, hi, nv, bool(np.isnan(v) or nv < MIN_VALID_DRAWS)))
    return out


def main():
    rows_out = []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        R = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        dv = np.load(DATA / f"audit_dev_arrays_{tag}.npz")
        assert abs(float(R["alpha"]) - ALPHA[tag]) < 1e-9 and abs(float(dv["alpha"]) - ALPHA[tag]) < 1e-9
        X = load_rescoring_inputs(tag)
        S, E = R["s_w"], R["e_cor_real"]

        # ---- validation (protocol): row order, step 43 agreement, step 45 AUC
        d_test = np.abs(2 * X["test"]["w"] - 1 - S).max()
        d_dev = np.abs(2 * X["dev"]["w"] - 1 - dv["s_w"]).max()
        print(f"VALIDATION gate weight rebuild: max |diff| test {d_test:.1e}, dev {d_dev:.1e}", flush=True)
        assert d_test < 1e-5 and d_dev < 1e-5
        ref = pd.read_csv(RESULTS / f"corrected_rule_swap_{tag}.csv")
        ref = ref[(ref.rule == "cor") & (ref.effect == "primary_abs_of_mean")].iloc[0]
        ag = block_metrics(S, E, np.arange(S.shape[1]), {})["agreement"]
        print(f"VALIDATION agreement {ag:.10f} vs step 43 {ref.agreement:.10f}", flush=True)
        assert abs(ag - ref.agreement) < 1e-9
        y = X["test"]["y"]
        t_ok, a_ok = X["test"]["pt"].argmax(1) == y, X["test"]["pa"].argmax(1) == y
        only = t_ok != a_ok
        auc45 = np.mean([auroc(X["test"]["w"][g][only], t_ok[only]) for g in range(S.shape[0])])
        if tag == "M1":
            ref45 = pd.read_csv(RESULTS / "gate_input_diagnostic.csv").set_index("model").loc["M1", "auc_text_only_vs_audio_only"]
            print(f"VALIDATION reliability AUROC {auc45:.5f} vs step 45 {ref45:.5f}", flush=True)
            assert abs(auc45 - ref45) < 1e-4
        else:
            print(f"reliability AUROC on all only-one-correct rows (no step 45 reference for M5): {auc45:.5f}", flush=True)

        # ---- bootstrap draws, exactly as step 43
        draws = dialogue_draws(X["meta"]["dialogue_id"].values, N_BOOT)
        n = S.shape[1]
        shared = X["meta"]["audio_shared"].values.astype(bool)
        subsets = (("all", np.arange(n)), ("without_audio_shared", np.flatnonzero(~shared)))

        def emit(block, subset, rows_lbl, stats, effect="cor_real"):
            for k, v, lo, hi, nv, under in stats:
                rows_out.append({"model": tag, "block": block, "effect": effect, "subset": subset, "rows": rows_lbl, "metric": k,
                                 "value": v, "lo": lo, "hi": hi, "n_valid_draws": nv, "underpowered": under})

        # ---- block A (+ sensitivity effects on all rows)
        for rl, base in subsets:
            emit("A", "all", rl, boot(S, E, base, {}, draws))
        emit("A", "all", "all", boot(S, R["e_cor_fro"], subsets[0][1], {}, draws), effect="cor_frozen")
        emit("A", "all", "all", boot(S, R["e_raw_real"], subsets[0][1], {}, draws), effect="raw_real")

        # ---- block B: dev-fitted threshold on w
        ts = []
        for g in range(S.shape[0]):
            sd, ed = dv["s_w"][g], dv["e_cor_real"][g]
            v = (sd != 0) & (ed != 0)
            wd, ld = (sd[v] + 1) / 2, ed[v] > 0
            cands = np.quantile(wd, np.linspace(0, 1, 201))
            ts.append(cands[int(np.argmax([bal_acc(wd > c, ld) for c in cands]))])
        print(f"dev thresholds on w: {np.round(ts, 3)}", flush=True)
        for rl, base in subsets:
            emit("B", "all", rl, [x for x in boot(S, E, base, {"t": np.array(ts)}, draws) if x[0].endswith("devt")])

        # ---- block C: meaningful effects, tau from dev per gate (gate-specific row masks inside one bootstrap)
        for q, lbl in ((50, "abs_e_above_dev_median"), (75, "abs_e_above_dev_p75")):
            tau = np.percentile(np.abs(dv["e_cor_real"]), q, axis=1)
            for rl, base in subsets:
                emit("C", lbl, rl, boot(S, E, base, {"keep": np.abs(E) > tau[:, None]}, draws))

        # ---- block D: reliability versus influence
        for rl, base in subsets:
            emit("D", "only_one_head_correct", rl, [x for x in boot(S, E, base, {"rel": t_ok, "only": only}, draws)
                                                    if x[0] in ("rel_auroc", "inf_auroc", "rel_minus_inf", "correct_head_won", "n_only_one")])

        # ---- block E: disagreement quartiles, cuts from dev
        jd = js_div(X["dev"]["pt"], X["dev"]["pa"])
        jt = js_div(X["test"]["pt"], X["test"]["pa"])
        cuts = np.quantile(jd, [0.25, 0.5, 0.75])
        print(f"dev JS quartile cuts: {np.round(cuts, 4)}", flush=True)
        qt = np.digitize(jt, cuts)
        for k in range(4):
            for rl, base in subsets:
                rows_k = np.intersect1d(base, np.flatnonzero(qt == k))
                emit("E", f"js_quartile_{k + 1}", rl, [x for x in boot(S, E, rows_k, {"mean_abs_e": True}, draws)
                                                       if x[0] in ("agreement", "auroc", "bal_acc_05", "share_text_won", "mean_abs_e", "n_rows")])
        print(f"{tag} done", flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(RESULTS / "rescoring_v2.csv", index=False)
    print(f"wrote {RESULTS / 'rescoring_v2.csv'} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
