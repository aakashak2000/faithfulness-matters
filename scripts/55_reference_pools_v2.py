"""Step 55: extra donor pools for phase 1 version 2 (Amendment 1 of docs/reference_pools_protocol.md, written 2026-11-19 before
this script was run). Dev: speaker50 (for refitting part B without nn50). Test: global50 with seeds 545, 546, 547 (noise floor).
Same masks and swap routine as step 53. Validation: dev global50 rebuilt with seed 544 reproduces step 53.

Usage:  python scripts/55_reference_pools_v2.py
Output: data/reference_pools_v2_{M1,M5}.npz
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA  # noqa: E402

ALPHA = {"M1": 0.25, "M5": 0.5}
KEEP = ("e_cor_real", "e_cor_real_mabs", "e_raw_real", "s_w", "c_cor", "c_raw")


def run(tag):
    t0 = time.time()
    don_spk = ac.donor_speakers()
    meta, emb, okd, gates, split_fn = ac.pool_runner(tag, ALPHA[tag])
    v1 = np.load(DATA / f"reference_pools_{tag}.npz")
    out = {}

    def swap(split, m, name):
        full = okd[split]
        okd[split] = m
        res = [split_fn(g, split) for g in gates]
        okd[split] = full
        for k in KEEP:
            out[f"{split}_{name}_{k}"] = np.stack([r[k] for r in res])
        print(f"{tag} {split} {name}: {len(gates)} gates ({time.time() - t0:.0f}s)", flush=True)

    masks, inc = ac.build_pool_masks(okd["dev"], emb["dev"], emb["clf"], meta["dev"]["speaker"].values, don_spk, ("global50", "speaker50"), seed=544)
    swap("dev", masks["global50"], "global50_check")
    d = np.abs(out["dev_global50_check_e_cor_real"] - v1["dev_global50_e_cor_real"]).max()
    print(f"VALIDATION {tag} dev global50 rebuilt vs step 53: max |diff| {d:.1e}", flush=True)
    assert d < 1e-6
    out["dev_speaker_included"] = inc
    print(f"{tag} dev speaker50 included {inc.mean():.4f}", flush=True)
    swap("dev", masks["speaker50"], "speaker50")
    for seed in (545, 546, 547):
        m, _ = ac.build_pool_masks(okd["test"], emb["test"], emb["clf"], meta["test"]["speaker"].values, don_spk, ("global50",), seed=seed)
        swap("test", m["global50"], f"global50_s{seed}")
    np.savez_compressed(DATA / f"reference_pools_v2_{tag}.npz", **out)
    print(f"wrote data/reference_pools_v2_{tag}.npz", flush=True)


if __name__ == "__main__":
    for tag in ("M5", "M1"):
        run(tag)
