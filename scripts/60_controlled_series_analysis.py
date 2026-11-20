"""Step 60: statistics for the phase 2 controlled series (docs/controlled_series_protocol.md, revision 2). Reads the arrays of step 59.
Exploratory. Applies the protocol's reading rules: supported (interval on the predicted side of 0), contradicted (entirely on the
opposite side), otherwise inconclusive.

Usage:  python scripts/60_controlled_series_analysis.py
Output: results/controlled_series.csv (and prints the hypothesis readouts)
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, auroc, dialogue_draws, load_dataset, spearman  # noqa: E402

N_BOOT = importlib.import_module("06_analysis").N_BOOT
N_PERM = 20
GROUPS = {"C0": ["C0"], "C1": ["C1_seed0", "C1_seed1", "C1_seed2"], "C2": ["C2_seed0", "C2_seed1", "C2_seed2"],
          "S0": ["S0"], "S1": ["S1"], "S2": ["S2"], "P": ["P"]}
REF = {"C1": "C0", "C2": "C0", "P": "C0", "S1": "S0", "S2": "S0"}
BASELINES = {"C1": ["C1_seed0_C0gates", "C1_seed1_C0gates", "C1_seed2_C0gates"], "C2": ["C2_seed0_C0gates", "C2_seed1_C0gates", "C2_seed2_C0gates"],
             "S1": ["S1_C0gates"], "S2": ["S2_C0gates"], "P": ["P_C0gates"]}


def logit(w):
    w = np.clip(w, 1e-6, 1 - 1e-6)
    return np.log(w / (1 - w))


def gates_of(Z, runs):
    """One record per gate: its weight and the arm's per-head arrays."""
    out = []
    for r in runs:
        for g in range(Z[f"{r}__w"].shape[0]):
            out.append({k: Z[f"{r}__{k}"][g] if Z[f"{r}__{k}"].ndim == 2 and k not in ("pt", "pa", "mt", "ma") else Z[f"{r}__{k}"]
                        for k in ("w", "e_cor_real", "e_cor_fro", "cor_real_text_mean", "cor_fro_text_mean", "cor_real_audio_mean",
                                  "cor_fro_audio_mean", "c_cor", "pt", "pa", "mt", "ma")} | {"run": r})
    return out


def winner(w, pt, pa, mt, ma):
    ii = np.arange(len(w))
    c = (w[:, None] * pt + (1 - w[:, None]) * pa).argmax(1)
    dt, da = np.abs(pt[ii, c] - mt[ii, c]), np.abs(pa[ii, c] - ma[ii, c])
    return w * dt > (1 - w) * da, da / np.clip(dt + da, 1e-12, None)


def gate_stats(G, idx, y, rng, n_perm, full=True):
    res = {}
    for g in G:
        w, pt, pa, mt, ma = g["w"][idx], g["pt"][idx], g["pa"][idx], g["mt"][idx], g["ma"][idx]
        yy = y[idx]
        t_ok, a_ok = pt.argmax(1) == yy, pa.argmax(1) == yy
        only = t_ok != a_ok
        ii = np.arange(len(idx))
        c0 = (0.5 * pt + 0.5 * pa).argmax(1)
        d0t, d0a = np.abs(pt[ii, c0] - mt[ii, c0]), np.abs(pa[ii, c0] - ma[ii, c0])
        tau0 = d0a / np.clip(d0t + d0a, 1e-12, None)
        lab, tau = winner(w, pt, pa, mt, ma)
        e = g["e_cor_real"][idx]
        r = {"agreement_primary": ((w > 0.5) == (e > 0)).mean(),
             "share_gate_claim_vs_identity_disagree": ((w > 0.5) != (w > tau)).mean(),
             "fused_weighted_f1": f1_score(yy, g["c_cor"][idx], average="weighted", zero_division=0)}
        for mod in ("text", "audio"):
            head = g[f"cor_fro_{mod}_mean"][idx]
            r[f"mean_abs_head_path_{mod}"] = np.abs(head).mean()
            r[f"mean_abs_gate_path_{mod}"] = np.abs(g[f"cor_real_{mod}_mean"][idx] - head).mean()
        if full:
            r["mean_logit_w"] = logit(w).mean()
            r["auroc_w_reliability"] = auroc(w[only], t_ok[only])
            r["spearman_w_tau0"] = spearman(w, tau0)
            a_ag = ((w > 0.5) == lab).mean()
            nl = []
            for _ in range(n_perm):
                wp = rng.permutation(w)
                lp_, _ = winner(wp, pt, pa, mt, ma)
                nl.append(((wp > 0.5) == lp_).mean())
            r["excess_over_random_assignment"] = a_ag - np.mean(nl)
        for k, v in r.items():
            res.setdefault(k, []).append(v)
    return {k: float(np.nanmean(v)) for k, v in res.items()}


def main():
    Z = np.load(DATA / "controlled_series_arrays.npz")
    bad = [k for k in Z.files if Z[k].dtype.kind == "f" and not np.isfinite(Z[k]).all()]
    assert not bad, f"non-finite arrays: {bad}"      # added after review: donor means and all other arrays must be finite
    D = load_dataset()
    y = D["splits"]["test"]["y"].numpy()
    n = len(y)
    draws = dialogue_draws(D["meta"]["test"]["dialogue_id"].values, N_BOOT)
    settings = pd.read_csv(RESULTS / "controlled_series_arm_settings.csv", index_col=0)
    G = {a: gates_of(Z, runs) for a, runs in GROUPS.items()}
    Gb = {a: gates_of(Z, runs) for a, runs in BASELINES.items()}
    rows = []

    def emit(arm, kind, stats_pt, stats_bs):
        for k, v in stats_pt.items():
            arr = np.array([b[k] for b in stats_bs])
            lo, hi = np.nanpercentile(arr, [2.5, 97.5])
            rows.append({"arm": arm, "analysis": kind, "metric": k, "value": v, "lo": lo, "hi": hi})

    allr = np.arange(n)
    pt_ = {a: gate_stats(G[a], allr, y, np.random.default_rng(1), 200) for a in G}
    bs_ = {a: [] for a in G}
    rng = np.random.default_rng(1)
    for d in draws:
        for a in G:
            bs_[a].append(gate_stats(G[a], d, y, rng, N_PERM))
    for a in G:
        emit(a, "retrained_gate", pt_[a], bs_[a])
        if a in REF:
            sh = [b["mean_logit_w"] - r["mean_logit_w"] for b, r in zip(bs_[a], bs_[REF[a]])]
            v = pt_[a]["mean_logit_w"] - pt_[REF[a]]["mean_logit_w"]
            lo, hi = np.percentile(sh, [2.5, 97.5])
            rows.append({"arm": a, "analysis": "retrained_gate", "metric": f"shift_mean_logit_w_vs_{REF[a]}", "value": v, "lo": lo, "hi": hi})
    for a in ("C1", "C2"):
        for run in GROUPS[a]:
            v = gate_stats(gates_of(Z, [run]), allr, y, np.random.default_rng(1), 0, full=False) | {
                "mean_logit_w": float(np.mean([logit(Z[f'{run}__w'][g]).mean() for g in range(Z[f'{run}__w'].shape[0])]))}
            rows.append({"arm": run, "analysis": "per_seed", "metric": "shift_mean_logit_w_vs_C0", "value": v["mean_logit_w"] - pt_["C0"]["mean_logit_w"]})
    for a, gb in Gb.items():
        p = gate_stats(gb, allr, y, None, 0, full=False)
        b = [gate_stats(gb, d, y, None, 0, full=False) for d in draws]
        emit(a, "C0_gates_mechanistic_baseline", p, b)

    # planted arm: provenance, audio head wrong, audio caused the fused error
    planted = Z["P__planted"].astype(bool)
    gP = G["P"]
    pa, pt = gP[0]["pa"], gP[0]["pt"]
    t_ok, a_ok = pt.argmax(1) == y, pa.argmax(1) == y
    conf = pt.max(1) / (pt.max(1) + pa.max(1))

    def planted_stats(idx):
        r = {}
        targets = {"provenance": planted[idx], "audio_head_wrong": ~a_ok[idx]}
        for nm, lab in targets.items():
            r[f"auroc_w_{nm}"] = np.mean([auroc(g["w"][idx], lab) for g in gP])
            r[f"auroc_confidence_rule_{nm}"] = auroc(conf[idx], lab)
        caused = [((g["c_cor"][idx] != y[idx]) & t_ok[idx] & ~a_ok[idx]) for g in gP]
        r["auroc_w_audio_caused_fused_error"] = np.mean([auroc(g["w"][idx], c) for g, c in zip(gP, caused)])
        r["auroc_confidence_rule_audio_caused_fused_error"] = np.mean([auroc(conf[idx], c) for c in caused])
        r["mean_abs_audio_effect_planted"] = np.mean([np.abs(g["cor_real_audio_mean"][idx][planted[idx]]).mean() for g in gP])
        r["mean_abs_audio_effect_clean"] = np.mean([np.abs(g["cor_real_audio_mean"][idx][~planted[idx]]).mean() for g in gP])
        r["share_planted"] = planted[idx].mean()
        return r
    emit("P", "planted_outcomes", planted_stats(allr), [planted_stats(d) for d in draws])

    # secondary: gate shift against the change in audio-head dev NLL
    dnll = {a: np.mean([settings.loc[r, "dev_audio_nll"] for r in GROUPS[a]]) for a in G}
    pts = [(pt_[a]["mean_logit_w"] - pt_[REF[a]]["mean_logit_w"], dnll[a] - dnll[REF[a]]) for a in ("C1", "C2", "S1", "S2", "P")]
    for a, (s, dn) in zip(("C1", "C2", "S1", "S2", "P"), pts):
        rows.append({"arm": a, "analysis": "nll_vs_shift", "metric": "dev_audio_nll_change", "value": dn})
    rows.append({"arm": "all", "analysis": "nll_vs_shift", "metric": "spearman_shift_vs_dev_nll_change", "value": spearman(*np.array(pts).T)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "controlled_series.csv", index=False)

    # ---- readouts
    def side(arm, metric):
        r = df[(df.arm == arm) & (df.metric == metric) & (df.analysis == "retrained_gate")].iloc[0]
        return r.value, r.lo, r.hi
    v, lo, hi = side("C2", "shift_mean_logit_w_vs_C0")
    seeds = df[(df.analysis == "per_seed") & df.arm.str.startswith("C2")].value.values
    ha = "supported" if lo > 0 and (seeds > 0).all() else ("contradicted" if hi < 0 else "inconclusive")
    print(f"H-A (C2 shift toward text): {v:.3f} [{lo:.3f}, {hi:.3f}], seeds {np.round(seeds, 3)} -> {ha}")
    s1, s2 = side("S1", "shift_mean_logit_w_vs_S0"), side("S2", "shift_mean_logit_w_vs_S0")
    pat = ("likelihood pattern (audio weight falls in both)" if s1[1] > 0 and s2[1] > 0 else
           "sharpness pattern (audio weight rises in S1, falls in S2)" if s1[2] < 0 and s2[1] > 0 else "inconclusive")
    print(f"H-B: S1 {s1[0]:.3f} [{s1[1]:.3f}, {s1[2]:.3f}], S2 {s2[0]:.3f} [{s2[1]:.3f}, {s2[2]:.3f}] -> {pat}")
    r = df[(df.arm == "P") & (df.metric == "auroc_w_provenance")].iloc[0]
    print(f"H-D (gate detects planted rows): AUROC {r.value:.3f} [{r.lo:.3f}, {r.hi:.3f}] -> {'supported' if r.lo > 0.5 else ('contradicted' if r.hi < 0.5 else 'inconclusive')}")
    print("wrote results/controlled_series.csv")


if __name__ == "__main__":
    main()
