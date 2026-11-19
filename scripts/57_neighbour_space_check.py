"""Step 57: neighbour-space check (Amendment 2 of docs/reference_pools_protocol.md, written 2026-11-19 before this script was run).
New pool nnp50: the 50 valid donors nearest in each head's OUTPUT space (Euclidean distance between probability vectors), so that
"similar" means the same thing for the text and the audio head. Compares flip direction with the embedding-space nn50 of step 53.
Also reports how often the gate claim and the identity disagree, (w > 0.5) != (w > tau). Exploratory; no training.
Post hoc (added after the first run, labelled so in the protocol): the same flip statistics with the gate held fixed (e_cor_fro),
to test whether the nnp50 direction comes from the gate's own reaction rather than the heads.

Usage:  python scripts/57_neighbour_space_check.py
Output: data/reference_pools_v3_{M1,M5}.npz, results/reference_sensitivity_v3.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS, dialogue_draws  # noqa: E402

N_BOOT = importlib.import_module("06_analysis").N_BOOT
ALPHA = {"M1": 0.25, "M5": 0.5}
K = ac.POOL_K


def prob_space_masks(tag, model, ok):
    """nnp50 masks for the test split: nearest valid donors by Euclidean distance between head probability vectors."""
    masks = []
    with torch.no_grad():
        if tag == "M1":
            th, ah = model["specs"]["M1 stronger audio"]
            heads = ((th, model["ET"]), (ah, model["EA"]))
            for (h, E), valid in zip(heads, ok):
                pr, pd_ = torch.softmax(h(E["test"]), 1), torch.softmax(h(E["clf"]), 1)
                dist = torch.cdist(pr, pd_)
                dist[~valid] = float("inf")
                m = torch.zeros_like(valid)
                m.scatter_(1, dist.topk(K, dim=1, largest=False).indices, True)
                masks.append(m)
        else:
            for kind, valid in zip(("text", "audio"), ok):
                pr = torch.softmax(model["Z2"][kind]["test"], 1)
                m = torch.zeros_like(valid)
                for s in range(0, pr.shape[0], 100):
                    P = torch.softmax(model["donor"][kind][None] + model["ctxpart"][kind]["test"][s:s + 100, None], 2)
                    dist = ((P - pr[s:s + 100, None]) ** 2).sum(-1).sqrt()
                    dist[~valid[s:s + 100]] = float("inf")
                    m[s:s + 100].scatter_(1, dist.topk(K, dim=1, largest=False).indices, True)
                masks.append(m)
    for m, valid in zip(masks, ok):
        assert (m.sum(1) == K).all() and not (m & ~valid).any()
    return tuple(masks)


def main():
    rows = []
    for tag in ("M1", "M5"):
        print(f"== {tag}", flush=True)
        meta, emb, okd, gates, split_fn, model = ac.pool_runner(tag, ALPHA[tag], return_model=True)
        full_ok = okd["test"]
        m = prob_space_masks(tag, model, full_ok)
        okd["test"] = m
        res = [split_fn(g, "test") for g in gates]
        okd["test"] = full_ok
        e_nnp = np.stack([r["e_cor_real"] for r in res])
        e_nnp_fro = np.stack([r["e_cor_fro"] for r in res])
        np.savez_compressed(DATA / f"reference_pools_v3_{tag}.npz", test_nnp50_e_cor_real=e_nnp, test_nnp50_e_cor_fro=e_nnp_fro,
                            test_nnp50_s_w=np.stack([r["s_w"] for r in res]))

        A = np.load(DATA / f"corrected_rule_arrays_{tag}.npz")
        P1, P2 = np.load(DATA / f"reference_pools_{tag}.npz"), np.load(DATA / f"reference_pools_v2_{tag}.npz")
        assert np.abs(np.stack([r["s_w"] for r in res]) - A["s_w"]).max() < 1e-6
        W = (A["s_w"] + 1) / 2
        G, n = W.shape
        pt, pa, mt, ma, y, meta_t = ac.donor_pool_means(tag, "test")
        _, _, mt_p, ma_p, _, _ = ac.donor_pool_means(tag, "test", ok=m)
        _, _, TAU = ac.swapfree_features(W, pt, pa, mt, ma)
        ii = np.arange(n)
        d = {}
        for g in range(G):
            c = (W[g][:, None] * pt + (1 - W[g][:, None]) * pa).argmax(1)
            for nm, P_, M_, Mp in (("d_text", pt, mt, mt_p), ("d_audio", pa, ma, ma_p)):
                d.setdefault(f"{nm}_full", []).append(np.abs(P_[ii, c] - M_[ii, c]))
                d.setdefault(f"{nm}_nnp50", []).append(np.abs(P_[ii, c] - Mp[ii, c]))
        d = {k: np.stack(v) for k, v in d.items()}
        win_full, win_nnp, win_nn = A["e_cor_real"] > 0, e_nnp > 0, P1["test_nn50_e_cor_real"] > 0
        win_nnp_fro = e_nnp_fro > 0
        globs = [P1["test_global50_e_cor_real"] > 0] + [P2[f"test_global50_s{s}_e_cor_real"] > 0 for s in (545, 546, 547)]

        def stat(idx):
            f, p, nn = win_full[:, idx], win_nnp[:, idx], win_nn[:, idx]
            floor = np.mean([(f != gw[:, idx]).mean() for gw in globs])
            r = {"disagree_full_nnp50": (f != p).mean(), "excess_nnp50_over_noise_mean4": (f != p).mean() - floor,
                 "flip_text_to_audio_nnp50": (f & ~p).mean(), "flip_audio_to_text_nnp50": (~f & p).mean(),
                 "flip_difference_nnp50": (f & ~p).mean() - (~f & p).mean(),
                 "flip_difference_nn50": (f & ~nn).mean() - (~f & nn).mean(),
                 "disagree_nn50_nnp50": (nn != p).mean(),
                 "posthoc_flip_text_to_audio_nnp50_frozen_gate": (f & ~win_nnp_fro[:, idx]).mean(),
                 "posthoc_flip_audio_to_text_nnp50_frozen_gate": (~f & win_nnp_fro[:, idx]).mean(),
                 "posthoc_mean_abs_effect_diff_nnp50_recomputed": np.abs(e_nnp[:, idx]).mean(),
                 "posthoc_mean_abs_effect_diff_nnp50_frozen": np.abs(e_nnp_fro[:, idx]).mean(),
                 "share_gate_claim_vs_identity_disagree": ((W[:, idx] > 0.5) != (W[:, idx] > TAU[:, idx])).mean()}
            for k in ("d_text", "d_audio"):
                r[f"mean_{k}_full"], r[f"mean_{k}_nnp50"] = d[f"{k}_full"][:, idx].mean(), d[f"{k}_nnp50"][:, idx].mean()
            return r

        draws = dialogue_draws(meta_t["dialogue_id"].values, N_BOOT)
        p_ = stat(np.arange(n))
        bs = [stat(dr) for dr in draws]
        for k, v in p_.items():
            lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
            rows.append({"model": tag, "metric": k, "value": v, "lo": lo, "hi": hi})
        print(f"{tag} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "reference_sensitivity_v3.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
