"""Step 51: gate-free distinctiveness check (Amendment 1 of docs/decomposition_protocol.md, written 2026-11-17 before this script was run).
Distinctiveness is recomputed at c0 = argmax(0.5 p_text + 0.5 p_audio), a class that does not depend on the gate, with the same donor pools
and valid-donor masks as step 43. Exploratory.

Usage:  python scripts/51_gate_free_tau.py
Output: results/decomposition_gate_free_tau.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS  # noqa: E402

r49 = importlib.import_module("49_rescoring")
r50 = importlib.import_module("50_decomposition")
auroc, spearman, N_BOOT = r49.auroc, r50.spearman, r49.N_BOOT


def pool_means(tag):
    """Per test row: mean donor probability vector for text and audio, over valid donors only. Returns pt, pa, mt, ma, y, meta."""
    B = ac.build_harness() if tag == "M1" else None
    M = ac.build_m5() if tag == "M5" else None
    with torch.no_grad():
        if tag == "M1":
            th, ah = B["specs"]["M1 stronger audio"]
            pt, pa = torch.softmax(th(B["ET"]["test"]), 1), torch.softmax(ah(B["EA"]["test"]), 1)
            Pt, Pa = torch.softmax(th(B["ET"]["clf"]), 1), torch.softmax(ah(B["EA"]["clf"]), 1)
            ok_t, ok_a = (m.float() for m in B["ok"]["test"])
            mt = (ok_t @ Pt) / ok_t.sum(1, keepdim=True)
            ma = (ok_a @ Pa) / ok_a.sum(1, keepdim=True)
            return pt.numpy(), pa.numpy(), mt.numpy(), ma.numpy(), B["y"]["test"].numpy(), B["D"]["meta"]["test"]
        pt, pa = torch.softmax(M["Z2"]["text"]["test"], 1), torch.softmax(M["Z2"]["audio"]["test"], 1)
        out = {}
        for kind in ("text", "audio"):
            ok = M["ok"]["test"][0 if kind == "text" else 1].float()
            ms = []
            for s in range(0, ok.shape[0], 100):
                P = torch.softmax(M["donor"][kind][None] + M["ctxpart"][kind]["test"][s:s + 100, None], 2)   # (rows, donors, 7)
                o = ok[s:s + 100]
                ms.append((o[:, :, None] * P).sum(1) / o.sum(1, keepdim=True))
            out[kind] = torch.cat(ms)
        return pt.numpy(), pa.numpy(), out["text"].numpy(), out["audio"].numpy(), M["y"]["test"].numpy(), M["D"]["meta"]["test"]


def main():
    rows = []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        A = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        W = (A["s_w"] + 1) / 2
        dt = np.abs(A["raw_fro_text_mean"]) / W
        da = np.abs(A["raw_fro_audio_mean"]) / (1 - W)
        pt, pa, mt, ma, y, meta = pool_means(tag)
        n = len(y)
        c0 = (0.5 * pt + 0.5 * pa).argmax(1)
        ii = np.arange(n)
        d0t, d0a = np.abs(pt[ii, c0] - mt[ii, c0]), np.abs(pa[ii, c0] - ma[ii, c0])
        tau0 = d0a / (d0t + d0a)
        C = A["c_raw"]
        same = C == c0[None]
        err = max((np.abs(d0t[None] - dt) * W)[same].max(), (np.abs(d0a[None] - da) * (1 - W))[same].max())   # effect scale, see protocol
        print(f"VALIDATION on rows with c0 == c: max |d0 - d| x weight {err:.1e}; share c0 == c {same.mean():.4f}", flush=True)
        assert err < 1e-5

        t_ok, a_ok = pt.argmax(1) == y, pa.argmax(1) == y
        only = t_ok != a_ok
        dlg = meta["dialogue_id"].values
        groups = [np.flatnonzero(dlg == g) for g in np.unique(dlg)]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]

        def st(idx):
            v = np.isfinite(tau0[idx])
            o = only[idx] & v
            return {"w_tau0_spearman": np.mean([spearman(W[g, idx][v], tau0[idx][v]) for g in range(len(W))]),
                    "textshare0_R_auroc": auroc(1 - tau0[idx][o], t_ok[idx][o]),
                    "median_tau0": np.median(tau0[idx][v])}
        p = st(ii)
        bs = [st(d) for d in draws]
        for k, v in p.items():
            lo, hi = np.nanpercentile([b[k] for b in bs], [2.5, 97.5])
            rows.append({"model": tag, "metric": k, "value": v, "lo": lo, "hi": hi})
        rows.append({"model": tag, "metric": "share_c0_equals_c", "value": float(same.mean())})
        print(f"{tag} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "decomposition_gate_free_tau.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
