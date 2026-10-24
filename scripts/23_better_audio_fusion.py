"""Step 23: fuse the better audio head (step 22) with the text head and rerun the primary tests.

This is a VARIANT ("strong audio") reported next to the headline model; the headline is unchanged.

Pre-registered before running:
  - Audio features: layers {12, 16, 20, 24}, mean + std pooling, concatenated (8,192 numbers),
    standardised with clf statistics. Audio head: linear, class-weighted CE + 0.5 * wd * ||W||^2 with
    wd = 0.1, LBFGS, 200 iterations, zero init (the configuration selected on DEV in step 22,
    results/better_audio_head_selected.json; it sits at the edge of the wd grid, a stated limit).
  - Text head: the headline seed{n} text head (checkpoints/seed{n}.pt), unchanged.
  - Gate: same architecture and training as 02_train.py (MLP on [text embedding; audio features],
    trained on the gate split with the heads' out-of-sample probabilities, early stopping on dev loss),
    seeds 0 to 2. Fusion rules: average, confidence, learned gate.
  - Reported: test accuracy, macro-F1 and weighted-F1 for text, audio, average, confidence, gate
    (mean +- sd over seeds), next to the headline numbers.
  - Faithfulness, PRIMARY only: full-pool resampling exactly as 08_resampling_full.py (donors = the 7,990
    clf rows, byte-identical excluded), effect = |dp| difference, signals and 4 pairs as in 06,
    dialogue-cluster bootstrap (1000 draws, rng 0, 3 seeds averaged). Report agreement, always-text
    baseline, gain and Spearman for each pair, next to the headline model.
  - Reading rule: no claim is made in advance about the direction of the change. The point is to see how
    the faithfulness numbers move when the audio model is stronger (a second point on the model-quality axis).
  - Check: with the headline checkpoints and the same code path, the headline numbers must be reproduced
    (08 / results/resampling_full_faithfulness.csv).

Usage: python scripts/23_better_audio_fusion.py
Outputs: results/better_audio_baselines.csv, results/better_audio_faithfulness.csv,
         checkpoints/audio2/seed{N}.pt (local only)
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, DATA, RESULTS, Head, LearnedGate, class_weights, fuse_average,  # noqa: E402
                    fuse_confidence, fuse_gate, load_dataset, metrics, set_determinism, train_gate)

iv04 = importlib.import_module("04_interventions")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
an06 = importlib.import_module("06_analysis")
b22 = importlib.import_module("22_better_audio")
LAYERS, POOL, WD = [12, 16, 20, 24], "mean+std", 0.1
SEEDS = [0, 1, 2]
OUT = CKPT / "audio2"
MODELS = rs07.MODELS


def build_audio(S, mp):
    mtr, str_ = b22.feats("train")
    feat = {}
    for sp in ("clf", "gate"):
        rows = torch.tensor(mp[mp.split == sp].sort_values("row").file_index.values)
        feat[sp] = b22.build(mtr[rows], str_[rows], LAYERS, POOL)
    del mtr, str_
    for sp in ("dev", "test"):
        m, s = b22.feats(sp)
        feat[sp] = b22.build(m, s, LAYERS, POOL)
    mu, sd = feat["clf"].mean(0), feat["clf"].std(0) + 1e-6
    return {k: ((v - mu) / sd).contiguous() for k, v in feat.items()}


def load_models2(seed):
    ck = torch.load(OUT / f"seed{seed}.pt", map_location="cpu", weights_only=False)
    th, ah, gate = Head(384), Head(ck["audio_dim"]), LearnedGate(384, ck["audio_dim"])
    th.load_state_dict(ck["text_head"]); ah.load_state_dict(ck["audio_head"]); gate.load_state_dict(ck["gate"])
    for m in (th, ah, gate):
        m.eval()
    return th, ah, gate


def frames(seed, tag, orig, cls, pb):
    n = len(cls["fusion_learned_gate"])
    out = []
    for m in MODELS:
        p0, _ = orig[m]
        dpt, _, tvt = iv04.effect(pb[m][0], p0, cls[m])
        dpa, _, tva = iv04.effect(pb[m][1], p0, cls[m])
        out.append(pd.DataFrame({
            "seed": seed, "row": range(n), "model": m, "method": tag,
            "gate_w_t": orig["fusion_learned_gate"][1].squeeze(1).numpy(),
            "conf_w_t": orig["fusion_confidence"][1].squeeze(1).numpy(),
            "dp_text": dpt.numpy(), "dp_audio": dpa.numpy()}))
    return pd.concat(out, ignore_index=True)


def faithfulness(D, e_a_by_split, load_fn, tag):
    S = D["splits"]
    test, clf = S["test"], S["clf"]
    e_t, e_a = test["text"], e_a_by_split["test"]
    ok_t, ok_a = rs07.valid_masks(e_t, clf["text"]), rs07.valid_masks(e_a, e_a_by_split["clf"])
    old = iv04.load_models
    iv04.load_models = load_fn
    try:
        fr = []
        for seed in SEEDS:
            orig, _, _, cls, pb, _, _ = rs08.exact_average(seed, e_t, e_a, clf["text"], e_a_by_split["clf"], ok_t, ok_a)
            fr.append(frames(seed, tag, orig, cls, pb))
            print(f"  {tag} seed {seed} swaps done", flush=True)
    finally:
        iv04.load_models = old
    return pd.concat(fr, ignore_index=True)


def main():
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    A = build_audio(S, mp)
    y = {k: S[k]["y"] for k in ("clf", "gate", "dev", "test")}
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. audio head (deterministic) and per-seed gates
    W, b = b22.fit(A["clf"], y["clf"], w_clf, WD)
    ah = Head(A["clf"].shape[1])
    ah.lin.weight.data, ah.lin.bias.data = W.T.contiguous(), b.clone()
    ah.eval()
    base_rows = []
    for seed in SEEDS:
        ck0 = torch.load(CKPT / f"seed{seed}.pt", map_location="cpu", weights_only=False)
        th = Head(384)
        th.load_state_dict(ck0["text_head"]); th.eval()
        set_determinism(seed)
        with torch.no_grad():
            pr = {s: (torch.softmax(th(S[s]["text"]), 1), torch.softmax(ah(A[s]), 1)) for s in ["gate", "dev", "test"]}
        gate = train_gate(S["gate"]["text"], A["gate"], *pr["gate"], y["gate"],
                          (S["dev"]["text"], A["dev"], *pr["dev"], y["dev"]), w_gate, seed)
        torch.save({"text_head": th.state_dict(), "audio_head": ah.state_dict(), "gate": gate.state_dict(),
                    "audio_dim": A["clf"].shape[1], "seed": seed}, OUT / f"seed{seed}.pt")
        with torch.no_grad():
            p_t, p_a = pr["test"]
            outs = {"text_only": p_t, "audio_only": p_a, "fusion_average": fuse_average(p_t, p_a)[0],
                    "fusion_confidence": fuse_confidence(p_t, p_a)[0],
                    "fusion_learned_gate": fuse_gate(gate, S["test"]["text"], A["test"], p_t, p_a)[0]}
        for k, p in outs.items():
            base_rows.append({"model": k, "seed": seed, **metrics(y["test"], p)})
        print(f"seed {seed}: gate weighted-F1 {base_rows[-1]['weighted_f1']:.4f}", flush=True)
    base = pd.DataFrame(base_rows)
    base.to_csv(RESULTS / "better_audio_baselines.csv", index=False)
    head = pd.read_csv(RESULTS / "baselines_per_seed.csv")
    order = ["text_only", "audio_only", "fusion_average", "fusion_confidence", "fusion_learned_gate"]
    t = pd.concat([base.groupby("model")[["accuracy", "macro_f1", "weighted_f1"]].mean().add_suffix("_strong").loc[order],
                   head.groupby("model")[["accuracy", "macro_f1", "weighted_f1"]].mean().add_suffix("_headline").loc[order]], axis=1)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 4)
    print("\nTest metrics, mean over 3 seeds (strong audio vs headline):")
    print(t[["weighted_f1_strong", "weighted_f1_headline", "macro_f1_strong", "macro_f1_headline",
             "accuracy_strong", "accuracy_headline"]].to_string())

    # 2. faithfulness, primary only
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    full_old = pd.read_csv(RESULTS / "resampling_full_log.csv")
    emb_head = {"clf": S["clf"]["audio"], "test": S["test"]["audio"]}
    chk = faithfulness(D, emb_head, iv04.load_models, "resample_full")
    for m in MODELS:   # reproduction check on the headline checkpoints
        a = chk[chk.model == m].sort_values(["seed", "row"])
        bb = full_old[full_old.model == m].sort_values(["seed", "row"])
        assert np.allclose(a.dp_text.values, bb.dp_text.values, atol=1e-5), "headline reproduction failed"
    print("check passed: headline checkpoints reproduce results/resampling_full_log.csv", flush=True)
    log = faithfulness(D, {"clf": A["clf"], "test": A["test"]}, load_models2, "resample_full")

    order_ = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order_.dialogue_id.values == g) for g in np.unique(order_.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]
    rows = []
    for model, signal, label in an06.PAIRS:
        s, e = an06.arrays(log, model, signal, "resample_full")
        pt = an06.stats(s, e)
        bs = np.array([an06.stats(s[:, i], e[:, i]) for i in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for k, n in enumerate(["agreement", "always_text_baseline", "gain", "spearman"]):
            rows.append({"pair": label, "metric": n, "strong_audio": pt[k], "ci_low": lo[k], "ci_high": hi[k]})
    res = pd.DataFrame(rows)
    hv = pd.read_csv(RESULTS / "resampling_full_faithfulness.csv")
    hv = hv[hv.method == "resample_full"][["pair", "metric", "estimate"]].rename(columns={"estimate": "headline"})
    res = res.merge(hv, on=["pair", "metric"])
    res.to_csv(RESULTS / "better_audio_faithfulness.csv", index=False)
    print("\nFaithfulness, full-pool swap (primary): strong audio vs headline")
    for n in ["agreement", "always_text_baseline", "gain", "spearman"]:
        tt = res[res.metric == n].copy()
        tt["strong"] = tt.apply(lambda r: f"{r.strong_audio:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
        print(f"\n== {n} ==")
        print(tt[["pair", "strong", "headline"]].to_string(index=False))


if __name__ == "__main__":
    main()
