"""Step 53: swap test under alternative donor pools (phase 1, part A inputs). Protocol: docs/reference_pools_protocol.md
(written 2026-11-19 before this script was run). Exploratory; no training. The step 43 swap routine (audit_common.core2) is reused
unchanged: a pool is a per-utterance donor mask with exactly 50 valid donors.

Pools: global50 (random), speaker50 (random, same speaker; rows without 50 such donors fall back to global50 and are flagged),
nn50 (nearest by cosine in the swapped modality's gate embedding). Dev: global50, nn50. Test: global50, speaker50, nn50.

Usage:  python scripts/53_reference_pools.py [--model M1|M5]      (default: both)
Output: data/reference_pools_{M1,M5}.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA  # noqa: E402

ALPHA = {"M1": 0.25, "M5": 0.5}
POOLS = {"dev": ("global50", "nn50"), "test": ("global50", "speaker50", "nn50")}
KEEP = ("e_cor_real", "e_cor_real_mabs", "e_raw_real", "s_w", "c_cor", "c_raw")


def run(tag):
    t0 = time.time()
    don_spk = ac.donor_speakers()
    out = {}
    meta, emb_all, okd, gates, split_fn = ac.pool_runner(tag, ALPHA[tag])

    # validation: full pool, gate 0, test reproduces step 43
    ref = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")["e_cor_real"][0]
    d = np.abs(split_fn(gates[0], "test")["e_cor_real"] - ref).max()
    print(f"VALIDATION {tag} full pool gate 0 vs step 43: max |diff| {d:.1e} ({time.time() - t0:.0f}s)", flush=True)
    assert d < 1e-6

    for split in ("dev", "test"):
        full_ok = okd[split]
        masks, included = ac.build_pool_masks(full_ok, emb_all[split], emb_all["clf"], meta[split]["speaker"].values, don_spk, POOLS[split])
        out[f"{split}_speaker_included"] = included
        print(f"{tag} {split}: masks built; speaker50 included {included.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        for pool, m in masks.items():
            okd[split] = m
            res = [split_fn(g, split) for g in gates]
            for k in KEEP:
                out[f"{split}_{pool}_{k}"] = np.stack([r[k] for r in res])
            print(f"{tag} {split} {pool}: {len(gates)} gates done ({time.time() - t0:.0f}s)", flush=True)
        okd[split] = full_ok
    np.savez_compressed(DATA / f"reference_pools_{tag}.npz", **out)
    print(f"wrote data/reference_pools_{tag}.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["M1", "M5"])
    args = ap.parse_args()
    for tag in ([args.model] if args.model else ["M1", "M5"]):
        run(tag)


if __name__ == "__main__":
    main()
