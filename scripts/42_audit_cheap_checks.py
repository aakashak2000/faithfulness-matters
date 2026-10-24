"""Step 42: cheap audit checks (2026-10-23). Nothing here changes an earlier result; outputs are new files.

Checks, fixed before running:
  C1  Recompute weighted-F1 and macro-F1 from labels and argmax predictions (own sklearn calls) for M0b, M1, M2, M3 (10 gates)
      and M5 (3 gates), with and without the dev-fitted class-prior correction q = softmax(log p + alpha log pi).
      alpha is re-selected here on dev (gate rule, mean over gates, grid {0, .25, .5, .75, 1}) and compared with the value stored in
      results/harness_models.csv (M0b..M3) or data/v2_pipeline_v2.pt (M5). Also checks that q sums to 1 and that argmax q is what is scored.
  C2  Dev-set swap effects (original rule and corrected rule, gate recomputed per donor) for every model and gate; the DEV winners
      select a constant modality (text if text won on more than half of the dev utterances, per gate). Test agreement of that fixed
      constant is reported next to the weight's agreement and the descriptive test-majority baseline. No test outcome selects anything.
  C3  Duplicate-audio sensitivity from the saved step 38/39 arrays: full test set, minus the 39 within-test duplicate rows, minus all
      42 flagged rows; primary effect, mean-of-absolute-changes and frozen-gate effect; dialogue bootstrap (1000 draws, rng 0).
Usage: python scripts/42_audit_cheap_checks.py [--skip-m5]
Outputs: results/audit_f1_recompute.csv, results/audit_dev_constant.csv, results/audit_duplicate_sensitivity.csv,
         data/audit_dev_arrays_{tag}.npz (local), data/audit_cache/gates_{tag}.pt (local)
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS  # noqa: E402

an06 = importlib.import_module("06_analysis")
N_BOOT, SEEDS10 = an06.N_BOOT, list(range(10))


def f1s(y, pred):
    return (f1_score(y, pred, average="weighted", zero_division=0), f1_score(y, pred, average="macro", zero_division=0))


def fused_rules(pt, pa, w):
    return {"text_only": pt, "audio_only": pa, "average": 0.5 * pt + 0.5 * pa, "gate": w[:, None] * pt + (1 - w[:, None]) * pa}


def f1_rows(tag, y, lp, probs_by_split, csv_alpha=None):
    """probs_by_split[split][seed] = dict rule -> (n,7). Returns rows and my dev alpha."""
    seeds = sorted(probs_by_split["dev"])
    dev = {a: np.mean([f1s(y["dev"].numpy(), (probs_by_split["dev"][s]["gate"].log() + a * lp).argmax(1).numpy())[0] for s in seeds]) for a in ac.ALPHAS}
    a_best = max(dev, key=dev.get)
    rows = []
    for rule in ("text_only", "audio_only", "average", "gate"):
        for atag, a in (("alpha0", 0.0), ("dev_alpha", a_best)):
            res, nrm = [], []
            for s in seeds:
                p = probs_by_split["test"][s][rule]
                q = ac.qrule(p, a, lp)
                nrm.append((q.sum(1) - 1).abs().max().item())
                assert (q.argmax(1) == (p.clamp_min(1e-12).log() + a * lp).argmax(1)).all()
                res.append(f1s(y["test"].numpy(), q.argmax(1).numpy()))
            rows.append({"model": tag, "rule": rule, "correction": atag, "alpha": a, "n_gates": len(seeds),
                         "weighted_f1": float(np.mean([r[0] for r in res])), "macro_f1": float(np.mean([r[1] for r in res])),
                         "max_abs_row_sum_error": max(nrm), "my_dev_alpha": a_best, "csv_alpha": csv_alpha})
    return rows, a_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-m5", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ac.CACHE.mkdir(exist_ok=True)
    hm = pd.read_csv(RESULTS / "harness_models.csv").set_index("model")
    f1_all, const_rows = [], []
    dev_store = {}

    # ---------------- harness models
    B = ac.build_harness()
    y, lp = B["y"], B["lp"]
    print(f"harness built ({time.time() - t0:.0f}s)", flush=True)
    for name, (th, ah) in B["specs"].items():
        tag = name.split()[0]
        gates = {}
        probs = {"dev": {}, "test": {}}
        dev_eff = []
        for seed in SEEDS10:
            Gp = ac.harness_gate(B, th, ah, seed)
            gates[seed] = Gp.state_dict()
            for sp in ("dev", "test"):
                with torch.no_grad():
                    pt, pa = torch.softmax(th(B["ET"][sp]), 1), torch.softmax(ah(B["EA"][sp]), 1)
                    w = Gp(B["ET"][sp], B["EA"][sp]).squeeze(1)
                probs[sp][seed] = fused_rules(pt, pa, w)
        torch.save(gates, ac.CACHE / f"gates_{tag}.pt")
        rows, a_mine = f1_rows(tag, y, lp, probs, float(hm.loc[name, "alpha_dev"]))
        f1_all += rows
        print(f"{tag}: my dev alpha {a_mine} (csv {hm.loc[name, 'alpha_dev']}); gate wF1 a0 "
              f"{[r['weighted_f1'] for r in rows if r['rule']=='gate'][0]:.4f} (csv {hm.loc[name, 'gate_weighted_f1_a0']:.4f}) / dev alpha "
              f"{[r['weighted_f1'] for r in rows if r['rule']=='gate'][1]:.4f} (csv {hm.loc[name, 'gate_weighted_f1_adev']:.4f}) ({time.time() - t0:.0f}s)", flush=True)
        # dev swaps (both rules) with my dev alpha
        out_list = []
        for seed in SEEDS10:
            G_ = ac.LearnedGate(B["dT"] + 384, B["dA"] + 1024)
            G_.load_state_dict(gates[seed])
            G_.eval()
            _, _, _, out = ac.harness_split(B, th, ah, G_, "dev", a_mine)
            assert max(out["closed_form_err_text"].max(), out["closed_form_err_audio"].max()) < 1e-5
            out_list.append(out)
        keys = ["e_raw_real", "e_cor_real", "e_raw_real_mabs", "e_cor_real_mabs", "e_raw_real_flip", "e_cor_real_flip", "s_w"]
        dev_store[tag] = {k: np.stack([o[k] for o in out_list]) for k in keys}
        np.savez_compressed(DATA / f"audit_dev_arrays_{tag}.npz", alpha=a_mine, **dev_store[tag])
        print(f"{tag}: dev swaps done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- M5
    if not args.skip_m5:
        M = ac.build_m5()
        probs = {"dev": {}, "test": {}}
        gates = {}
        for seed in range(3):
            g = ac.m5_gate(M, seed)
            gates[seed] = g.state_dict()
            for sp in ("dev", "test"):
                pt, pa = torch.softmax(M["Z2"]["text"][sp], 1), torch.softmax(M["Z2"]["audio"][sp], 1)
                with torch.no_grad():
                    w = g(*M["G"][sp]).squeeze(1)
                probs[sp][seed] = fused_rules(pt, pa, w)
        torch.save(gates, ac.CACHE / "gates_M5.pt")
        rows, a_mine = f1_rows("M5", M["y"], M["lp"], probs, float(M["alpha_saved"]))
        f1_all += rows
        print(f"M5: my dev alpha {a_mine} (saved {M['alpha_saved']}); gate wF1 "
              f"{[r['weighted_f1'] for r in rows if r['rule']=='gate'][0]:.4f} / {[r['weighted_f1'] for r in rows if r['rule']=='gate'][1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        out_list = []
        for seed in range(3):
            g = ac.m5_gate(M, seed)
            g.load_state_dict(gates[seed])
            g.eval()
            _, _, _, out = ac.m5_split(M, g, "dev", float(M["alpha_saved"]))
            assert max(out["closed_form_err_text"].max(), out["closed_form_err_audio"].max()) < 1e-5
            out_list.append(out)
        keys = ["e_raw_real", "e_cor_real", "e_raw_real_mabs", "e_cor_real_mabs", "e_raw_real_flip", "e_cor_real_flip", "s_w"]
        dev_store["M5"] = {k: np.stack([o[k] for o in out_list]) for k in keys}
        np.savez_compressed(DATA / "audit_dev_arrays_M5.npz", alpha=float(M["alpha_saved"]), **dev_store["M5"])
        print(f"M5: dev swaps done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(f1_all).to_csv(RESULTS / "audit_f1_recompute.csv", index=False)

    # ---------------- C2 dev-selected constant (original rule on the saved test arrays; corrected rule is evaluated in step 43)
    D = ac.load_dataset()
    meta = D["meta"]["test"]
    dlg = meta["dialogue_id"].values
    groups = [np.flatnonzero(dlg == g) for g in np.unique(dlg)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]

    def agree(s, e, idx=None):
        s_, e_ = (s, e) if idx is None else (s[:, idx], e[:, idx])
        return ((s_ > 0) == (e_ > 0)).mean(1)

    for tag in dev_store:
        z = np.load(DATA / f"frozen_gate_arrays_{tag}.npz")
        s, e = z["s_w"], z["e_real"]
        dev_text = (dev_store[tag]["e_raw_real"] > 0).mean(1)         # per gate: share of dev utterances where text won
        const_text = dev_text > 0.5
        const_ag = np.where(const_text, (e > 0).mean(1), (e <= 0).mean(1))
        test_text = (e > 0).mean(1)
        w_ag = agree(s, e)
        boot = []
        for idx in draws:
            ca = np.where(const_text, (e[:, idx] > 0).mean(1), (e[:, idx] <= 0).mean(1)).mean()
            tm = (e[:, idx] > 0).mean(1)
            boot.append([agree(s, e, idx).mean(), ca, agree(s, e, idx).mean() - ca, agree(s, e, idx).mean() - np.maximum(tm, 1 - tm).mean()])
        b = np.array(boot)
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        const_rows.append({"model": tag, "n_gates": len(s), "dev_share_text_won": float(dev_text.mean()),
                           "dev_constant": "text" if const_text.mean() > 0.5 else "audio", "all_gates_same_constant": bool(const_text.all() or (~const_text).all()),
                           "test_share_text_won": float(test_text.mean()), "weight_agreement": float(w_ag.mean()),
                           "dev_constant_agreement": float(const_ag.mean()), "gain_over_dev_constant": float(w_ag.mean() - const_ag.mean()),
                           "gain_dev_const_lo": lo[2], "gain_dev_const_hi": hi[2],
                           "test_majority_agreement_descriptive": float(np.maximum(test_text, 1 - test_text).mean()),
                           "gain_over_test_majority": float(w_ag.mean() - np.maximum(test_text, 1 - test_text).mean()),
                           "gain_test_maj_lo": lo[3], "gain_test_maj_hi": hi[3]})
    pd.DataFrame(const_rows).to_csv(RESULTS / "audit_dev_constant.csv", index=False)
    print(pd.DataFrame(const_rows).round(4).to_string(), flush=True)

    # ---------------- C3 duplicate-audio sensitivity
    shared, shared_tr = meta["audio_shared"].values.astype(bool), meta["audio_shared_with_train"].values.astype(bool)
    print("flag counts: audio_shared", shared.sum(), "audio_shared_with_train", shared_tr.sum(), "both", (shared & shared_tr).sum(), flush=True)
    within = shared & ~shared_tr
    subsets = {"full": np.ones(len(shared), bool), "minus_within_test_dups": ~within, "minus_all_flagged": ~shared}
    dup_rows = []
    for tag in dev_store:
        z = np.load(DATA / f"frozen_gate_arrays_{tag}.npz")
        for eff_name, key in (("primary_abs_of_mean", "e_real"), ("mean_of_abs", "e_mabs"), ("frozen_gate", "e_fro")):
            for sub, keep in subsets.items():
                idxk = np.flatnonzero(keep)
                s, e = z["s_w"][:, idxk], z[key][:, idxk]
                ag = ((s > 0) == (e > 0)).mean(1)
                pt = (e > 0).mean(1)
                gain = ag - np.maximum(pt, 1 - pt)
                sp = np.mean([np.corrcoef(rankdata(a), rankdata(b_))[0, 1] for a, b_ in zip(s, e)])
                bs = []
                for idx in draws:
                    idx2 = idx[keep[idx]]
                    s2, e2 = z["s_w"][:, idx2], z[key][:, idx2]
                    a2 = ((s2 > 0) == (e2 > 0)).mean(1)
                    p2 = (e2 > 0).mean(1)
                    bs.append((a2 - np.maximum(p2, 1 - p2)).mean())
                lo, hi = np.percentile(bs, [2.5, 97.5])
                dup_rows.append({"model": tag, "effect": eff_name, "subset": sub, "n": int(keep.sum()), "p_text": float(pt.mean()),
                                 "weight_agreement": float(ag.mean()), "gain_over_test_majority": float(gain.mean()), "gain_lo": lo, "gain_hi": hi, "spearman": float(sp)})
    pd.DataFrame(dup_rows).to_csv(RESULTS / "audit_duplicate_sensitivity.csv", index=False)
    d = pd.DataFrame(dup_rows)
    print(d[d.effect == "primary_abs_of_mean"].round(4).to_string(), flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
