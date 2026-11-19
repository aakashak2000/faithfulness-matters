"""Step 58: does the nn50 headline survive with the gate held fixed during the swap? (Amendment 3 of docs/reference_pools_protocol.md,
written 2026-11-19 before this script was run). Reruns the global50 (seed 544) and nn50 test pools of step 53, keeping the head-path-only
effects (gate weight held at its original value during the swap) and the signed donor-averaged changes, so that the gate path
(recomputed minus head-path-only) can be measured before absolute values. Exploratory; no training.

Usage:  python scripts/58_nn50_gate_path.py
Output: results/reference_sensitivity_v4.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS, dialogue_draws  # noqa: E402

N_BOOT = importlib.import_module("06_analysis").N_BOOT
ALPHA = {"M1": 0.25, "M5": 0.5}
KEYS = ("e_cor_real", "e_cor_fro", "cor_real_text_mean", "cor_fro_text_mean", "cor_real_audio_mean", "cor_fro_audio_mean")


def main():
    rows = []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        meta, emb, okd, gates, split_fn = ac.pool_runner(tag, ALPHA[tag])
        masks, _ = ac.build_pool_masks(okd["test"], emb["test"], emb["clf"], meta["test"]["speaker"].values, ac.donor_speakers(), ("global50", "nn50"))
        full_ok, R = okd["test"], {}
        for pool, m in masks.items():
            okd["test"] = m
            res = [split_fn(g, "test") for g in gates]
            R[pool] = {k: np.stack([r[k] for r in res]) for k in KEYS}
        okd["test"] = full_ok
        A, P1 = np.load(DATA / f"corrected_rule_arrays_{tag}.npz"), np.load(DATA / f"reference_pools_{tag}.npz")
        R["full"] = {k: A[k] for k in KEYS}
        for pool in ("global50", "nn50"):
            d = np.abs(R[pool]["e_cor_real"] - P1[f"test_{pool}_e_cor_real"]).max()
            print(f"VALIDATION {tag} {pool} recomputed-gate effects vs step 53: max |diff| {d:.1e}", flush=True)
            assert d < 1e-6
        win = {(p, mode): R[p][f"e_cor_{mode}"] > 0 for p in R for mode in ("real", "fro")}
        n = A["s_w"].shape[1]

        def stat(idx):
            r = {}
            for mode, lbl in (("real", "recomputed_gate"), ("fro", "head_path_only")):
                f = win[("full", mode)][:, idx]
                r[f"flip_full_nn50_{lbl}"] = (f != win[("nn50", mode)][:, idx]).mean()
                r[f"noise_full_global50_{lbl}"] = (f != win[("global50", mode)][:, idx]).mean()
                r[f"excess_nn50_{lbl}"] = r[f"flip_full_nn50_{lbl}"] - r[f"noise_full_global50_{lbl}"]
            r["ratio_head_path_excess_to_recomputed_excess"] = r["excess_nn50_head_path_only"] / r["excess_nn50_recomputed_gate"]
            for p in ("full", "nn50"):
                for mod in ("text", "audio"):
                    head = R[p][f"cor_fro_{mod}_mean"][:, idx]
                    gate = R[p][f"cor_real_{mod}_mean"][:, idx] - head
                    r[f"mean_abs_head_path_{mod}_{p}"] = np.abs(head).mean()
                    r[f"mean_abs_gate_path_{mod}_{p}"] = np.abs(gate).mean()
                    r[f"share_gate_path_opposes_head_path_{mod}_{p}"] = (np.sign(gate) == -np.sign(head)).mean()
            return r

        draws = dialogue_draws(meta["test"]["dialogue_id"].values, N_BOOT)
        p_ = stat(np.arange(n))
        bs = [stat(dr) for dr in draws]
        for k, v in p_.items():
            lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
            rows.append({"model": tag, "metric": k, "value": v, "lo": lo, "hi": hi})
        print(f"{tag} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "reference_sensitivity_v4.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
