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
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, LearnedGate  # noqa: E402

ALPHA = {"M1": 0.25, "M5": 0.5}
K = 50
POOLS = {"dev": ("global50", "nn50"), "test": ("global50", "speaker50", "nn50")}
KEEP = ("e_cor_real", "e_cor_real_mabs", "e_raw_real", "s_w", "c_cor", "c_raw")


def donor_speakers():
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    spk = {(a, b): s for a, b, s in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Speaker)}
    tr = P["ids"]["train"]
    return np.array([spk[tuple(int(x) for x in tr[i])] for i in np.asarray(P["rows"]["clf"])])


def build_masks(ok, emb, emb_pool, row_spk, don_spk, split):
    """ok: (valid_text, valid_audio) bool tensors (n, K_all). emb: (text, audio) row embeddings; emb_pool: donor embeddings."""
    n = ok[0].shape[0]
    rng = np.random.default_rng(544)
    masks, included = {}, np.ones(n, bool)
    for pool in POOLS[split]:
        mt, ma = torch.zeros_like(ok[0]), torch.zeros_like(ok[1])
        if pool == "nn50":
            for m, valid, e, ep in ((mt, ok[0], emb[0], emb_pool[0]), (ma, ok[1], emb[1], emb_pool[1])):
                sim = torch.nn.functional.normalize(e.float(), dim=1) @ torch.nn.functional.normalize(ep.float(), dim=1).T
                sim[~valid] = -float("inf")
                m.scatter_(1, sim.topk(K, dim=1).indices, True)
        else:
            for i in range(n):
                for m, valid in ((mt, ok[0]), (ma, ok[1])):
                    cand = np.flatnonzero(valid[i].numpy())
                    if pool == "speaker50":
                        same = cand[don_spk[cand] == row_spk[i]]
                        if len(same) < K:
                            included[i] = False
                            m[i] = masks["global50"][0 if m is mt else 1][i]
                            continue
                        cand = same
                    m[i, rng.choice(cand, K, replace=False)] = True
        for m, valid in ((mt, ok[0]), (ma, ok[1])):
            assert (m.sum(1) == K).all() and not (m & ~valid).any(), f"{pool}: mask check failed"
        masks[pool] = (mt, ma)
    return masks, included


def run(tag):
    t0 = time.time()
    cache = torch.load(ac.CACHE / f"gates_{tag}.pt")
    don_spk = donor_speakers()
    out = {}
    if tag == "M1":
        B = ac.build_harness()
        th, ah = B["specs"]["M1 stronger audio"]
        meta, emb_all, okd = B["D"]["meta"], B["H"], B["ok"]
        gates = []
        for _, sd in cache.items():
            g = ac.LearnedGate(B["dT"] + 384, B["dA"] + 1024)
            g.load_state_dict(sd)
            g.eval()
            gates.append(g)
        split_fn = lambda g, split: ac.harness_split(B, th, ah, g, split, ALPHA[tag])[3]
    else:
        M = ac.build_m5()
        meta, emb_all, okd = M["D"]["meta"], M["G"], M["ok"]
        gates = []
        for _, sd in cache.items():
            g = LearnedGate(384, 1024)
            g.load_state_dict(sd)
            g.eval()
            gates.append(g)
        split_fn = lambda g, split: ac.m5_split(M, g, split, ALPHA[tag])[3]

    # validation: full pool, gate 0, test reproduces step 43
    ref = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")["e_cor_real"][0]
    d = np.abs(split_fn(gates[0], "test")["e_cor_real"] - ref).max()
    print(f"VALIDATION {tag} full pool gate 0 vs step 43: max |diff| {d:.1e} ({time.time() - t0:.0f}s)", flush=True)
    assert d < 1e-6

    for split in ("dev", "test"):
        full_ok = okd[split]
        masks, included = build_masks(full_ok, emb_all[split], emb_all["clf"], meta[split]["speaker"].values, don_spk, split)
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
