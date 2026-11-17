"""Step 49: re-scoring of the existing corrected-rule swap results (item 0 of the revised plan). Protocol: docs/rescoring_protocol.md
(written 2026-11-17 before this script was run). Exploratory. No swap is rerun and no model is trained; the arrays of steps 42/43 are
re-summarised and the head probabilities are rebuilt with audit_common exactly as in step 43.

Usage:  python scripts/49_rescoring.py
Needs: data/corrected_rule_arrays_{M1,M5}.npz, data/audit_dev_arrays_{M1,M5}.npz, data/audit_cache/gates_{M1,M5}.pt
Output: results/rescoring.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS, LearnedGate  # noqa: E402

an06 = importlib.import_module("06_analysis")
N_BOOT = an06.N_BOOT
ALPHA = {"M1": 0.25, "M5": 0.5}


# ---------------------------------------------------------------- small metrics (numpy, fast enough for the bootstrap)
def auroc(score, label):
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bal_acc(pred, label):
    return 0.5 * ((pred[label]).mean() + (~pred[~label]).mean())


def kappa(pred, label):
    po = (pred == label).mean()
    pe = pred.mean() * label.mean() + (1 - pred.mean()) * (1 - label.mean())
    return (po - pe) / (1 - pe)


def js_div(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log2(np.clip(a, 1e-12, None)) - np.log2(np.clip(b, 1e-12, None)))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------------- rebuild head probabilities and gate weights
def load_model(tag):
    cache = torch.load(ac.CACHE / f"gates_{tag}.pt")
    out = {}
    if tag == "M1":
        B = ac.build_harness()
        th, ah = B["specs"]["M1 stronger audio"]
        ET, EA = B["ET"], B["EA"]
        with torch.no_grad():
            for sp in ("dev", "test"):
                pt, pa = torch.softmax(th(ET[sp]), 1).numpy(), torch.softmax(ah(EA[sp]), 1).numpy()
                ws = []
                for _, sd in cache.items():
                    g = ac.LearnedGate(B["dT"] + 384, B["dA"] + 1024)
                    g.load_state_dict(sd)
                    g.eval()
                    ws.append(g(ET[sp], EA[sp]).squeeze(1).numpy())
                out[sp] = dict(pt=pt, pa=pa, w=np.stack(ws), y=B["y"][sp].numpy())
        out["meta"] = B["D"]["meta"]["test"]
    else:
        M = ac.build_m5()
        with torch.no_grad():
            for sp in ("dev", "test"):
                pt, pa = torch.softmax(M["Z2"]["text"][sp], 1).numpy(), torch.softmax(M["Z2"]["audio"][sp], 1).numpy()
                ws = []
                for _, sd in cache.items():
                    g = LearnedGate(384, 1024)
                    g.load_state_dict(sd)
                    g.eval()
                    ws.append(g(*M["G"][sp]).squeeze(1).numpy())
                out[sp] = dict(pt=pt, pa=pa, w=np.stack(ws), y=M["y"][sp].numpy())
        out["meta"] = M["D"]["meta"]["test"]
    return out


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
        r = {"agreement": (pred == lab).mean(), "auroc": auroc(s, lab), "bal_acc_05": bal_acc(pred, lab), "kappa_05": kappa(pred, lab),
             "audio_recall_05": (~pred[~lab]).mean(), "share_w_below_05": (~pred).mean(), "share_text_won": lab.mean(),
             "n_audio_won": float((~lab).sum()), "n_rows": float(v.sum())}
        if "t" in extra:
            pt_ = (s + 1) / 2 > extra["t"][g]
            r.update({"bal_acc_devt": bal_acc(pt_, lab), "kappa_devt": kappa(pt_, lab), "audio_recall_devt": (~pt_[~lab]).mean()})
        if "rel" in extra:
            rel = extra["rel"][idx][v]
            ok1 = extra["only"][idx][v]
            r.update({"rel_auroc": auroc(s[ok1], rel[ok1]), "inf_auroc": auroc(s[ok1], lab[ok1]),
                      "rel_minus_inf": auroc(s[ok1], rel[ok1]) - auroc(s[ok1], lab[ok1]),
                      "correct_head_won": (rel[ok1] == lab[ok1]).mean(), "n_only_one": float(ok1.sum())})
        if "mean_abs_e" in extra:
            r["mean_abs_e"] = np.abs(e).mean()
        for k, x in r.items():
            res.setdefault(k, []).append(x)
    return {k: float(np.nanmean(x)) for k, x in res.items()}


def boot(S, E, rows, extra, draws):
    """rows: the subset of test rows; draws are dialogue-cluster resamples of all rows, intersected with the subset."""
    mask = np.zeros(S.shape[1], bool)
    mask[rows] = True
    pt = block_metrics(S, E, np.flatnonzero(mask), extra)
    bs = [block_metrics(S, E, d[mask[d]], extra) for d in draws]
    out = []
    for k, v in pt.items():
        arr = np.array([b[k] for b in bs])
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out.append((k, v, lo, hi))
    return out


def main():
    rows_out = []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        R = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        dv = np.load(DATA / f"audit_dev_arrays_{tag}.npz")
        assert abs(float(R["alpha"]) - ALPHA[tag]) < 1e-9 and abs(float(dv["alpha"]) - ALPHA[tag]) < 1e-9
        X = load_model(tag)
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
        dlg = X["meta"]["dialogue_id"].values
        groups = [np.flatnonzero(dlg == g) for g in np.unique(dlg)]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
        n = S.shape[1]
        allrows = np.arange(n)

        def emit(block, subset, stats, effect="cor_real"):
            for k, v, lo, hi in stats:
                rows_out.append({"model": tag, "block": block, "effect": effect, "subset": subset, "metric": k, "value": v, "lo": lo, "hi": hi})

        # ---- block A (+ sensitivity effects, + without audio_shared)
        emit("A", "all", boot(S, E, allrows, {}, draws))
        shared = X["meta"]["audio_shared"].values.astype(bool)
        emit("A", "without_audio_shared", boot(S, E, np.flatnonzero(~shared), {}, draws))
        emit("A", "all", boot(S, R["e_cor_fro"], allrows, {}, draws), effect="cor_frozen")
        emit("A", "all", boot(S, R["e_raw_real"], allrows, {}, draws), effect="raw_real")

        # ---- block B: dev-fitted threshold on w
        ts = []
        for g in range(S.shape[0]):
            sd, ed = dv["s_w"][g], dv["e_cor_real"][g]
            v = (sd != 0) & (ed != 0)
            wd, ld = (sd[v] + 1) / 2, ed[v] > 0
            cands = np.quantile(wd, np.linspace(0, 1, 201))
            ts.append(cands[int(np.argmax([bal_acc(wd > c, ld) for c in cands]))])
        print(f"dev thresholds on w: {np.round(ts, 3)}", flush=True)
        emit("B", "all", [x for x in boot(S, E, allrows, {"t": np.array(ts)}, draws) if x[0].endswith("devt")])

        # ---- block C: meaningful effects, tau from dev per gate (gate-specific row masks inside one bootstrap)
        for q, lbl in ((50, "abs_e_above_dev_median"), (75, "abs_e_above_dev_p75")):
            tau = np.percentile(np.abs(dv["e_cor_real"]), q, axis=1)
            emit("C", lbl, boot(S, E, allrows, {"keep": np.abs(E) > tau[:, None]}, draws))

        # ---- block D: reliability versus influence
        emit("D", "only_one_head_correct", [x for x in boot(S, E, allrows, {"rel": t_ok, "only": only}, draws)
                                            if x[0] in ("rel_auroc", "inf_auroc", "rel_minus_inf", "correct_head_won", "n_only_one")])

        # ---- block E: disagreement quartiles, cuts from dev
        jd = js_div(X["dev"]["pt"], X["dev"]["pa"])
        jt = js_div(X["test"]["pt"], X["test"]["pa"])
        cuts = np.quantile(jd, [0.25, 0.5, 0.75])
        print(f"dev JS quartile cuts: {np.round(cuts, 4)}", flush=True)
        qt = np.digitize(jt, cuts)
        for k in range(4):
            emit("E", f"js_quartile_{k + 1}", [x for x in boot(S, E, np.flatnonzero(qt == k), {"mean_abs_e": True}, draws)
                                               if x[0] in ("agreement", "auroc", "bal_acc_05", "share_text_won", "mean_abs_e", "n_rows")])
        print(f"{tag} done", flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(RESULTS / "rescoring.csv", index=False)
    print(f"wrote {RESULTS / 'rescoring.csv'} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
