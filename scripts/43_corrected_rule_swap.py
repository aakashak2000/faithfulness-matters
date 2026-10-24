"""Step 43: the corrected-rule swap test (follow-up prompted by the 2026-10-23 audit). Protocol: docs/corrected_rule_swap_protocol.md
(written and committed BEFORE this script was run on test data). The original experiment (steps 33 to 41) is unchanged and its
outputs are not touched; this step only adds new files.

Usage:  python scripts/43_corrected_rule_swap.py --selftest
        python scripts/43_corrected_rule_swap.py --model M1
        python scripts/43_corrected_rule_swap.py --model M5          (only after M1 has been validated)
Needs: data/audit_cache/gates_{M1,M5}.pt and data/audit_dev_arrays_{M1,M5}.npz from step 42, data/frozen_gate_arrays_{M1,M5}.npz from step 38/39.
Outputs: results/corrected_rule_swap_{M1,M5}.csv, data/corrected_rule_arrays_{M1,M5}.npz (local)
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS  # noqa: E402

an06 = importlib.import_module("06_analysis")
N_BOOT = an06.N_BOOT
ALPHA = {"M1": 0.25, "M5": 0.5}          # frozen in the protocol from dev (results/harness_models.csv, data/v2_pipeline_v2.pt)


# ---------------------------------------------------------------- sanity tests (protocol section 8)
def selftest():
    torch.manual_seed(0)
    n, k = 6, 40
    lp = torch.log(torch.tensor([.3, .2, .1, .1, .1, .1, .1]))
    pt = torch.softmax(torch.randn(n, 7), 1)
    pa = torch.softmax(torch.randn(n, 7), 1)
    pool = torch.softmax(torch.randn(k, 7) * 2, 1)
    pre = torch.randn(n, k, 8)
    w2 = torch.randn(8)
    ok = torch.ones(n, k, dtype=torch.bool)
    pool_fn = lambda kind, sl: pool[sl][None]
    pre_fn = lambda kind, sl: pre[:, sl]
    w0 = torch.sigmoid(torch.randn(n))
    o = ac.core2(pt, pa, w0, pool_fn, pre_fn, w2, torch.tensor(0.1), ok, ok, lp, 0.5)
    # 1. alpha = 0: both rules identical
    o0 = ac.core2(pt, pa, w0, pool_fn, pre_fn, w2, torch.tensor(0.1), ok, ok, lp, 0.0)
    assert np.abs(o0["e_raw_real"] - o0["e_cor_real"]).max() < 1e-6 and (o0["c_raw"] == o0["c_cor"]).all()
    # 2. manual numpy reference for the corrected rule, real gate
    P0 = w0[:, None] * pt + (1 - w0[:, None]) * pa
    Q0 = torch.softmax(P0.log() + 0.5 * lp, 1)
    c = Q0.argmax(1)
    wd = torch.sigmoid(torch.relu(pre) @ w2 + 0.1)
    F = wd[:, :, None] * pool[None] + (1 - wd[:, :, None]) * pa[:, None, :]
    Q = torch.softmax(F.log() + 0.5 * lp, 2)
    ref = Q[torch.arange(n), :, c].mean(1) - Q0[torch.arange(n), c]
    assert np.abs(ref.numpy() - o["cor_real_text_mean"]).max() < 1e-5
    # 3. a donor equal to the explained utterance's own modality (constant gate at w0) gives zero effect
    wc = 0.37
    same_pool = lambda kind, sl: (pt if kind == "text" else pa)[:, None, :].expand(n, sl.stop - sl.start if sl.stop else 1, 7) if False else None
    oz = ac.core2(pt, pa, torch.full((n,), wc), lambda kind, sl: (pt[:, None, :] if kind == "text" else pa[:, None, :]).expand(n, min(sl.stop, 5) - sl.start, 7),
                  lambda kind, sl: torch.zeros(n, min(sl.stop, 5) - sl.start, 8), torch.zeros(8), torch.logit(torch.tensor(wc)),
                  torch.ones(n, 5, dtype=torch.bool), torch.ones(n, 5, dtype=torch.bool), lp, 0.5)
    assert max(np.abs(oz[f"{v}_real_{m}_mean"]).max() for v in ("raw", "cor") for m in ("text", "audio")) < 1e-6
    # 4. gate constant: real == frozen
    oc = ac.core2(pt, pa, torch.full((n,), wc), pool_fn, lambda kind, sl: torch.zeros(n, sl.stop - sl.start if sl.stop <= k else k - sl.start, 8),
                  torch.zeros(8), torch.logit(torch.tensor(wc)), ok, ok, lp, 0.5)
    assert np.abs(oc["cor_real_text_mean"] - oc["cor_fro_text_mean"]).max() < 1e-6
    # 5. donor permutation invariance
    perm = torch.randperm(k)
    op = ac.core2(pt, pa, w0, lambda kind, sl: pool[perm][sl][None], lambda kind, sl: pre[:, perm][:, sl], w2, torch.tensor(0.1), ok, ok, lp, 0.5)
    assert np.abs(op["e_cor_real"] - o["e_cor_real"]).max() < 1e-5
    # 6. raw-rule frozen closed form
    assert max(o["closed_form_err_text"].max(), o["closed_form_err_audio"].max()) < 1e-5
    # 7. donor masking: masked donors do not contribute
    ok2 = ok.clone()
    ok2[:, 10:] = False
    om = ac.core2(pt, pa, w0, pool_fn, pre_fn, w2, torch.tensor(0.1), ok2, ok2, lp, 0.5)
    oc10 = ac.core2(pt, pa, w0, lambda kind, sl: pool[:10][sl][None], lambda kind, sl: pre[:, :10][:, sl], w2, torch.tensor(0.1),
                    torch.ones(n, 10, dtype=torch.bool), torch.ones(n, 10, dtype=torch.bool), lp, 0.5)
    assert np.abs(om["e_cor_real"] - oc10["e_cor_real"]).max() < 1e-5
    print("selftest passed (7 checks)", flush=True)


# ---------------------------------------------------------------- statistics
def boot_stats(S, E, const_text, draws, with_spearman=False):
    """S, E: (gates, n) weight signal (2w-1) and effect difference (text minus audio). const_text: (gates,) bool or None."""
    def one(idx):
        s, e = S[:, idx], E[:, idx]
        v = (s != 0) & (e != 0)
        ag = np.array([((s[g][v[g]] > 0) == (e[g][v[g]] > 0)).mean() for g in range(len(s))])
        pt = np.array([(e[g][v[g]] > 0).mean() for g in range(len(s))])
        r = {"agreement": ag.mean(), "p_text": pt.mean(), "gain_test_majority": (ag - np.maximum(pt, 1 - pt)).mean(), "gain_always_text": (ag - pt).mean()}
        if const_text is not None:
            ca = np.where(const_text, pt, 1 - pt)
            r["dev_constant_agreement"] = ca.mean()
            r["gain_dev_constant"] = (ag - ca).mean()
        if with_spearman:
            r["spearman"] = np.mean([np.corrcoef(rankdata(s[g]), rankdata(e[g]))[0, 1] for g in range(len(s))])
        return r
    pt_ = one(np.arange(S.shape[1]))
    bs = [one(i) for i in draws]
    out = {}
    for k_, v_ in pt_.items():
        arr = np.array([b[k_] for b in bs])
        out[k_] = v_
        out[k_ + "_lo"], out[k_ + "_hi"] = np.percentile(arr, [2.5, 97.5])
    return out, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", choices=["M1", "M5"])
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    selftest()
    t0 = time.time()
    tag, alpha = args.model, ALPHA[args.model]
    cache = torch.load(ac.CACHE / f"gates_{tag}.pt")
    outs = []
    if tag == "M1":
        B = ac.build_harness()
        name = "M1 stronger audio"
        th, ah = B["specs"][name]
        y, lp = B["y"], B["lp"]
        D = B["D"]
        for seed, sd in cache.items():
            G_ = ac.LearnedGate(B["dT"] + 384, B["dA"] + 1024)
            G_.load_state_dict(sd)
            G_.eval()
            pt, pa, w0, out = ac.harness_split(B, th, ah, G_, "test", alpha)
            outs.append((pt, pa, w0, out))
            print(f"gate {seed} done ({time.time() - t0:.0f}s)", flush=True)
    else:
        M = ac.build_m5()
        y, lp, D = M["y"], M["lp"], M["D"]
        assert abs(M["alpha_saved"] - alpha) < 1e-9
        for seed, sd in cache.items():
            g = ac.m5_gate(M, seed)
            g.load_state_dict(sd)
            g.eval()
            outs.append(ac.m5_split(M, g, "test", alpha))
            print(f"gate {seed} done ({time.time() - t0:.0f}s)", flush=True)
    R = {k: np.stack([o[3][k] for o in outs]) for k in outs[0][3] if isinstance(outs[0][3][k], np.ndarray)}
    np.savez_compressed(DATA / f"corrected_rule_arrays_{tag}.npz", alpha=alpha, **R)

    # ---- sanity checks against earlier outputs
    z = np.load(DATA / f"frozen_gate_arrays_{tag}.npz")
    d_raw = float(np.abs(R["e_raw_real"] - z["e_real"]).max())
    d_fro = float(np.abs(R["e_raw_fro"] - z["e_fro"]).max())
    print(f"VALIDATION raw-rule effects vs step 38/39 arrays: max |diff| real {d_raw:.2e}, frozen {d_fro:.2e}", flush=True)
    assert d_raw < 1e-4 and d_fro < 1e-4, "raw-rule reproduction failed; do not interpret the corrected-rule rows"
    f1c = np.mean([f1_score(y["test"].numpy(), o[3]["c_cor"], average="weighted", zero_division=0) for o in outs])
    f1r = np.mean([f1_score(y["test"].numpy(), o[3]["c_raw"], average="weighted", zero_division=0) for o in outs])
    ref = pd.read_csv(RESULTS / "audit_f1_recompute.csv")
    ref = ref[(ref.model == tag) & (ref.rule == "gate")].set_index("correction")
    print(f"VALIDATION classes tracked reproduce step 42 F1: raw {f1r:.4f} vs {ref.loc['alpha0','weighted_f1']:.4f}; corrected {f1c:.4f} vs {ref.loc['dev_alpha','weighted_f1']:.4f}", flush=True)
    assert abs(f1r - ref.loc["alpha0", "weighted_f1"]) < 1e-6 and abs(f1c - ref.loc["dev_alpha", "weighted_f1"]) < 1e-6

    # ---- statistics
    meta = D["meta"]["test"]
    dlg = meta["dialogue_id"].values
    groups = [np.flatnonzero(dlg == g) for g in np.unique(dlg)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
    dv = np.load(DATA / f"audit_dev_arrays_{tag}.npz")
    assert abs(float(dv["alpha"]) - alpha) < 1e-9, "dev constant was computed with a different alpha than the frozen one"
    S = R["s_w"]
    rows = []
    specs = [("raw", "primary_abs_of_mean", "e_raw_real", "e_raw_real", True), ("cor", "primary_abs_of_mean", "e_cor_real", "e_cor_real", True),
             ("raw", "mean_of_abs", "e_raw_real_mabs", "e_raw_real_mabs", False), ("cor", "mean_of_abs", "e_cor_real_mabs", "e_cor_real_mabs", False),
             ("raw", "flip_share", "e_raw_real_flip", "e_raw_real_flip", False), ("cor", "flip_share", "e_cor_real_flip", "e_cor_real_flip", False),
             ("raw", "frozen_gate", "e_raw_fro", None, False), ("cor", "frozen_gate", "e_cor_fro", None, False)]
    stash = {}
    for rule, eff, tkey, dkey, sp in specs:
        E = R[tkey]
        const = (dv[dkey] > 0).mean(1) > 0.5 if dkey else None
        st, bs = boot_stats(S, E, const, draws, with_spearman=sp)
        stash[(rule, eff)] = bs
        row = {"model": tag, "rule": rule, "effect": eff, "alpha": alpha if rule == "cor" else 0.0, "n_gates": len(S), **st}
        if dkey:
            row["dev_share_text_won"] = float((dv[dkey] > 0).mean())
        rows.append(row)
    # corrected minus raw, paired over the same bootstrap draws
    for eff in ("primary_abs_of_mean", "mean_of_abs", "flip_share", "frozen_gate"):
        for m in ("agreement", "gain_dev_constant", "gain_test_majority"):
            if m not in stash[("raw", eff)][0]:
                continue
            a = np.array([b[m] for b in stash[("cor", eff)]]) - np.array([b[m] for b in stash[("raw", eff)]])
            p = [r for r in rows if r["rule"] == "cor" and r["effect"] == eff][0][m] - [r for r in rows if r["rule"] == "raw" and r["effect"] == eff][0][m]
            rows.append({"model": tag, "rule": "cor_minus_raw", "effect": eff, "metric": m, "diff": p, "diff_lo": np.percentile(a, 2.5), "diff_hi": np.percentile(a, 97.5)})
    # predicted-class changes between the rules
    chg = R["orig_class_changes"]
    for lbl, mask in (("class_changed", chg.any(0)), ("class_same", ~chg.any(0))):
        if mask.sum() == 0:
            continue
        for rule, key in (("raw", "e_raw_real"), ("cor", "e_cor_real")):
            ag = ((S[:, mask] > 0) == (R[key][:, mask] > 0)).mean()
            rows.append({"model": tag, "rule": rule, "effect": f"primary_on_{lbl}", "n_rows": int(mask.sum()), "share_rows": float(mask.mean()),
                         "agreement": float(ag), "p_text": float((R[key][:, mask] > 0).mean())})
    pd.DataFrame(rows).to_csv(RESULTS / f"corrected_rule_swap_{tag}.csv", index=False)
    show = pd.DataFrame([r for r in rows if r["rule"] in ("raw", "cor") and "agreement_lo" in r])
    print(show[["rule", "effect", "p_text", "agreement", "agreement_lo", "agreement_hi", "gain_dev_constant", "gain_dev_constant_lo", "gain_dev_constant_hi",
                "gain_test_majority"]].round(4).to_string(), flush=True)
    print(pd.DataFrame([r for r in rows if r["rule"] == "cor_minus_raw"]).round(4).to_string(), flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
