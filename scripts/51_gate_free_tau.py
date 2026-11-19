"""Step 51: gate-free distinctiveness check (Amendment 1 of docs/decomposition_protocol.md, written 2026-11-17 before this script was run).
Distinctiveness is recomputed at c0 = argmax(0.5 p_text + 0.5 p_audio), a class that does not depend on the gate, with the same donor pools
and valid-donor masks as step 43. Exploratory.
Version 3 (2026-11-19): also run without the 42 audio_shared rows (results/decomposition_gate_free_tau_no_audio_shared.csv); prints the
DIAGNOSTIC behind the validation change in the protocol; helpers moved to common.py and audit_common.py. All-row output unchanged.

Usage:  python scripts/51_gate_free_tau.py
Output: results/decomposition_gate_free_tau.csv, results/decomposition_gate_free_tau_no_audio_shared.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import donor_pool_means  # noqa: E402
from common import DATA, RESULTS, auroc, dialogue_draws, spearman  # noqa: E402

N_BOOT = importlib.import_module("06_analysis").N_BOOT
OUT = {"all": "decomposition_gate_free_tau.csv", "without_audio_shared": "decomposition_gate_free_tau_no_audio_shared.csv"}


def main():
    rows_by = {"all": [], "without_audio_shared": []}
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        A = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        W = (A["s_w"] + 1) / 2
        dt = np.abs(A["raw_fro_text_mean"]) / W
        da = np.abs(A["raw_fro_audio_mean"]) / (1 - W)
        pt, pa, mt, ma, y, meta = donor_pool_means(tag)
        n = len(y)
        c0 = (0.5 * pt + 0.5 * pa).argmax(1)
        ii = np.arange(n)
        d0t, d0a = np.abs(pt[ii, c0] - mt[ii, c0]), np.abs(pa[ii, c0] - ma[ii, c0])
        tau0 = d0a / (d0t + d0a)
        C = A["c_raw"]
        same = C == c0[None]
        gt, ga = np.where(same, np.abs(d0t[None] - dt), 0), np.where(same, np.abs(d0a[None] - da), 0)
        for name, gap, wk in (("text", gt, W), ("audio", ga, 1 - W)):
            j = np.unravel_index(np.argmax(gap), gap.shape)
            print(f"DIAGNOSTIC {name}: max |d0 - d| on rows with c0 == c {gap[j]:.1e}, at a {name} weight of {wk[j]:.2e}; "
                  f"max on the effect scale (gap x weight) {(gap * wk).max():.1e}", flush=True)
        err = max((gt * W).max(), (ga * (1 - W)).max())   # effect scale, see protocol
        print(f"VALIDATION on rows with c0 == c: max |d0 - d| x weight {err:.1e}; share c0 == c {same.mean():.4f}", flush=True)
        assert err < 1e-5

        t_ok, a_ok = pt.argmax(1) == y, pa.argmax(1) == y
        only = t_ok != a_ok
        draws = dialogue_draws(meta["dialogue_id"].values, N_BOOT)
        shared = meta["audio_shared"].values.astype(bool)

        def st(idx):
            v = np.isfinite(tau0[idx])
            o = only[idx] & v
            return {"w_tau0_spearman": np.mean([spearman(W[g, idx][v], tau0[idx][v]) for g in range(len(W))]),
                    "textshare0_R_auroc": auroc(1 - tau0[idx][o], t_ok[idx][o]),
                    "median_tau0": np.median(tau0[idx][v])}
        for rl, base in (("all", ii), ("without_audio_shared", np.flatnonzero(~shared))):
            mask = np.zeros(n, bool)
            mask[base] = True
            p_ = st(base)
            bs = [st(d[mask[d]]) for d in draws]
            for k, v in p_.items():
                lo, hi = np.nanpercentile([b[k] for b in bs], [2.5, 97.5])
                rows_by[rl].append({"model": tag, "metric": k, "value": v, "lo": lo, "hi": hi})
            rows_by[rl].append({"model": tag, "metric": "share_c0_equals_c", "value": float(same[:, base].mean())})
        print(f"{tag} done", flush=True)
    for rl, fn in OUT.items():
        df = pd.DataFrame(rows_by[rl])
        df.to_csv(RESULTS / fn, index=False)
        print(f"wrote {fn}\n" + df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
