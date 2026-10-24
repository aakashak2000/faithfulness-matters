"""Step 25: audio-model quality ladder, gate-input control, and a wider regularisation grid.

Three questions, one script (all VARIANTS; the headline model is unchanged):

  Item 2  Is the step 22 optimum limited by the grid? The chosen L2 value (0.1) was the largest tried.
  Item 1  Was the step 23/24 change in faithfulness caused by the stronger audio head, or by the changed
          GATE INPUT (8,192 standardised numbers, which pushed the gate's weights to the extremes)?
  Item 3  How do the faithfulness numbers move along a ladder of audio-model strength?

Pre-registered before running (nothing below is tuned on test):

  Item 2 grid extension (audio head only, selection on DEV weighted-F1, same LBFGS recipe as step 22):
      concat of layers {12,16,20,24}, mean+std : wd in {0.3, 1, 3, 10}
      concat of layers {8,10,...,24} (9 layers), mean+std : wd in {0.1, 0.3, 1, 3, 10}
      all 25 layers, mean+std : wd in {1, 3, 10}
    The step 22 optimum (concat4, mean+std, wd 0.1, dev 0.4623) is replaced only if a new configuration
    beats it on dev by at least 0.005. Otherwise it stays.

  Ladder rungs (each is: this audio head + headline text head + a gate whose INPUT IS THE HEADLINE INPUT,
  i.e. the text embedding and the headline 1,024-d audio embedding; only the audio head's votes change):
      R0  headline audio head (last layer, step 2 head)           [existing checkpoints]
      R1  layer 20, mean+std, wd 0.1
      R2  concat {12,16,20,24}, mean only, wd 0.1
      R3  concat {12,16,20,24}, mean+std, wd 0.1   (the step 22 optimum)
      R4  the item 2 winner, if it replaces R3 under the rule above
    Implementation: the audio embedding passed through the swaps is [rung features ; headline embedding];
    the audio head reads only the first block and the gate only the second (the other weights are zero).
    The text head and the gate are trained exactly as in 02_train.py (3 seeds, gate on the gate split,
    early stopping on dev loss).

  Item 1 control: compare R3 (gate input = headline input) with the step 24 result for the same audio
    head with the gate fed the 8,192 standardised features (results/strong_audio_checks.csv).
    Reading rule, fixed now, on the gate-weight gain over the MAJORITY modality:
      R3 within 0.10 of +0.294  -> the change is mostly due to the stronger audio head;
      R3 below +0.15            -> it is mostly due to the gate's input;
      in between                -> mixed.

  Item 3: no direction is predicted. For every rung report: audio-only and fused (gate) weighted-F1,
    P(text mattered more), gate agreement, gain over the majority modality (the right trivial rule once
    audio can dominate), Spearman, the dev-fitted cutoff (per seed, frozen, applied to test, as in 19)
    with its test gain, the share of gate weights below 0.1 or above 0.9.
  Primary method only: full-pool swap, |dp| effect, dialogue-cluster bootstrap (1000 draws, rng 0,
  3 seeds averaged).

Usage: python scripts/25_quality_ladder.py   (about 30 to 40 minutes)
Outputs: results/audio_grid_extension.csv, results/quality_ladder.csv
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, RESULTS, Head, LearnedGate, class_weights, fuse_average,  # noqa: E402
                    fuse_confidence, fuse_gate, load_dataset, metrics, set_determinism, train_gate)

b22 = importlib.import_module("22_better_audio")
s24 = importlib.import_module("24_strong_audio_checks")
iv04 = importlib.import_module("04_interventions")
an06 = importlib.import_module("06_analysis")
SEEDS = [0, 1, 2]
GRID = s24.GRID
REF_DEV = 0.4623
NINE = [8, 10, 12, 14, 16, 18, 20, 22, 24]


def std_feats(mp, layers, pool, tr, dv, te):
    mtr, str_ = tr
    feat = {}
    for sp in ("clf", "gate"):
        rows = torch.tensor(mp[mp.split == sp].sort_values("row").file_index.values)
        feat[sp] = b22.build(mtr[rows], str_[rows], layers, pool)
    feat["dev"] = b22.build(dv[0], dv[1], layers, pool)
    feat["test"] = b22.build(te[0], te[1], layers, pool)
    mu, sd = feat["clf"].mean(0), feat["clf"].std(0) + 1e-6
    return {k: ((v - mu) / sd).contiguous() for k, v in feat.items()}


def grid_extension(mp, tr, dv, te, y, w):
    cfgs = [("concat4", [12, 16, 20, 24], "mean+std", [0.3, 1, 3, 10]),
            ("concat9", NINE, "mean+std", [0.1, 0.3, 1, 3, 10]),
            ("all25", list(range(25)), "mean+std", [1, 3, 10])]
    rows = []
    for name, layers, pool, wds in cfgs:
        F_ = std_feats(mp, layers, pool, tr, dv, te)
        for wd in wds:
            W, b = b22.fit(F_["clf"], y["clf"], w, wd)
            dw, dm = b22.score(F_["dev"], y["dev"], W, b)
            tw, tm = b22.score(F_["test"], y["test"], W, b)
            rows.append({"features": name, "pool": pool, "wd": wd, "dim": F_["clf"].shape[1],
                         "dev_wF1": dw, "dev_macroF1": dm, "test_wF1": tw, "test_macroF1": tm})
            print(f"grid {name:8s} wd={wd:<5} dev {dw:.4f} test {tw:.4f}", flush=True)
        del F_
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "audio_grid_extension.csv", index=False)
    return res


def build_rung(D, S, mp, tr, dv, te, layers, pool, wd, y, w_clf, w_gate):
    """Return (models by seed, audio features by split with the headline embedding appended, audio-only wF1)."""
    F_ = std_feats(mp, layers, pool, tr, dv, te)
    W, b = b22.fit(F_["clf"], y["clf"], w_clf, wd)
    d = F_["clf"].shape[1]
    E = {s: torch.cat([F_[s], S[s]["audio"]], 1).contiguous() for s in ("clf", "gate", "dev", "test")}
    ah = Head(d + 1024)
    ah.lin.weight.data = torch.cat([W.T.contiguous(), torch.zeros(7, 1024)], 1)
    ah.lin.bias.data = b.clone()
    ah.eval()
    models = {}
    for seed in SEEDS:
        ck0 = torch.load(CKPT / f"seed{seed}.pt", map_location="cpu", weights_only=False)
        th = Head(384)
        th.load_state_dict(ck0["text_head"])
        th.eval()
        set_determinism(seed)
        with torch.no_grad():
            pr = {s: (torch.softmax(th(S[s]["text"]), 1), torch.softmax(ah(E[s]), 1)) for s in ("gate", "dev", "test")}
        g = train_gate(S["gate"]["text"], S["gate"]["audio"], *pr["gate"], y["gate"],
                       (S["dev"]["text"], S["dev"]["audio"], *pr["dev"], y["dev"]), w_gate, seed)
        G = LearnedGate(384, d + 1024)
        w1 = g.net[0].weight.data
        G.net[0].weight.data = torch.cat([w1[:, :384], torch.zeros(w1.shape[0], d), w1[:, 384:]], 1)
        G.net[0].bias.data = g.net[0].bias.data.clone()
        G.net[2].load_state_dict(g.net[2].state_dict())
        G.eval()
        models[seed] = (th, ah, G)
    return models, E


def fused_f1(models, E, S, y):
    rows = []
    for seed in SEEDS:
        th, ah, G = models[seed]
        with torch.no_grad():
            p_t, p_a = torch.softmax(th(S["test"]["text"]), 1), torch.softmax(ah(E["test"]), 1)
            outs = {"audio_only": p_a, "fusion_learned_gate": fuse_gate(G, S["test"]["text"], E["test"], p_t, p_a)[0],
                    "fusion_average": fuse_average(p_t, p_a)[0], "fusion_confidence": fuse_confidence(p_t, p_a)[0]}
        for k, p in outs.items():
            rows.append({"model": k, "seed": seed, **metrics(y["test"], p)})
    return pd.DataFrame(rows).groupby("model").weighted_f1.mean()


def evaluate(name, S, load_fn, E, f1, draws):
    test_log = s24.swap_log(S["test"]["text"], E["test"], S["clf"]["text"], E["clf"], load_fn, "resample_full")
    dev_log = s24.swap_log(S["dev"]["text"], E["dev"], S["clf"]["text"], E["clf"], load_fn, "dev")
    return summarise(name, S, test_log, dev_log, f1, draws)


def summarise(name, S, test_log, dev_log, f1, draws):
    s, e = an06.arrays(test_log, "fusion_learned_gate", "gate_w_t", "resample_full")
    pt = s24.stats3(s, e)
    bs = np.array([s24.stats3(s[:, i], e[:, i]) for i in draws])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    cuts = []
    for seed in SEEDS:
        dd = dev_log[(dev_log.model == "fusion_learned_gate") & (dev_log.seed == seed)].sort_values("row")
        ee = dd.dp_text.abs().values - dd.dp_audio.abs().values
        v = ee != 0
        ag = np.array([np.mean(np.sign(dd.gate_w_t.values[v] - t) == np.sign(ee[v])) for t in GRID])
        best = np.flatnonzero(ag == ag.max())
        cuts.append(float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]]))
    st = (s + 1) / 2 - np.array(cuts)[:, None]
    ptau = s24.stats3(st, e)
    bt = np.array([s24.stats3(st[:, i], e[:, i]) for i in draws])
    lot, hit = np.percentile(bt, [2.5, 97.5], axis=0)
    w = test_log[test_log.model == "fusion_learned_gate"].gate_w_t.values
    return {"rung": name, "audio_only_wF1": f1.get("audio_only", np.nan), "gate_wF1": f1.get("fusion_learned_gate", np.nan),
            "p_text_mattered_more": pt[1], "agreement": pt[0], "agreement_lo": lo[0], "agreement_hi": hi[0],
            "gain_over_majority": pt[3], "gain_lo": lo[3], "gain_hi": hi[3], "spearman": pt[4],
            "dev_cutoff": float(np.mean(cuts)), "agreement_at_cutoff": ptau[0],
            "gain_majority_at_cutoff": ptau[3], "gain_cut_lo": lot[3], "gain_cut_hi": hit[3],
            "share_weights_extreme": float(np.mean((w < 0.1) | (w > 0.9))), "mean_weight": float(w.mean())}


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    tr, dv, te = b22.feats("train"), b22.feats("dev"), b22.feats("test")
    y = {k: S[k]["y"] for k in ("clf", "gate", "dev", "test")}
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(an06.N_BOOT)]

    # ---- item 2: grid extension
    ext = grid_extension(mp, tr, dv, te, y, w_clf)
    top = ext.loc[ext.dev_wF1.idxmax()]
    replace = bool(top.dev_wF1 >= REF_DEV + 0.005)
    print(f"\nbest new config: {top.features} wd={top.wd} dev {top.dev_wF1:.4f} (reference {REF_DEV}); replaces R3: {replace}"
          f"  [{time.time() - t0:.0f}s]", flush=True)

    # ---- item 3 + item 1: ladder with the headline gate input
    rungs = [("R1 layer20 mean+std", [20], "mean+std", 0.1),
             ("R2 concat4 mean", [12, 16, 20, 24], "mean", 0.1),
             ("R3 concat4 mean+std", [12, 16, 20, 24], "mean+std", 0.1)]
    if replace:
        lay = {"concat4": [12, 16, 20, 24], "concat9": NINE, "all25": list(range(25))}[top.features]
        rungs.append((f"R4 {top.features} mean+std wd{top.wd}", lay, "mean+std", float(top.wd)))
    out = []

    # R0: the headline model (existing checkpoints, headline input)
    old_log = pd.read_csv(RESULTS / "resampling_full_log.csv")
    base = pd.read_csv(RESULTS / "baselines_per_seed.csv")
    f1_0 = base.groupby("model").weighted_f1.mean().rename({"audio_only": "audio_only"}).to_dict()
    dev_log0 = s24.swap_log(S["dev"]["text"], S["dev"]["audio"], S["clf"]["text"], S["clf"]["audio"], iv04.load_models, "dev")
    out.append(summarise("R0 headline", S, old_log, dev_log0, f1_0, draws))
    print("R0 done", flush=True)

    for name, layers, pool, wd in rungs:
        models, E = build_rung(D, S, mp, tr, dv, te, layers, pool, wd, y, w_clf, w_gate)
        f1 = fused_f1(models, E, S, y).to_dict()
        out.append(evaluate(name, S, lambda seed, m=models: m[seed], E, f1, draws))
        print(f"{name} done: audio {f1['audio_only']:.4f} gate {f1['fusion_learned_gate']:.4f} "
              f"agreement {out[-1]['agreement']:.3f} gain_major {out[-1]['gain_over_majority']:.3f} [{time.time() - t0:.0f}s]", flush=True)
        pd.DataFrame(out).to_csv(RESULTS / "quality_ladder.csv", index=False)

    # item 1 reference row: same audio head as R3, gate fed the 8,192 standardised features (step 23/24)
    chk = pd.read_csv(RESULTS / "strong_audio_checks.csv")
    c = chk[(chk.variant == "strong_audio_real_labels") & (chk.pair == "Learned gate ← gate weight")].set_index("metric")
    bf = pd.read_csv(RESULTS / "better_audio_baselines.csv").groupby("model").weighted_f1.mean()
    cut = pd.read_csv(RESULTS / "strong_audio_dev_cutoff.csv")
    cut_t = cut[cut.seed == "test, dev-fitted cutoff"].set_index("metric")
    out.append({"rung": "R3 with gate fed strong features (steps 23/24)", "audio_only_wF1": bf["audio_only"],
                "gate_wF1": bf["fusion_learned_gate"], "p_text_mattered_more": c.loc["p_text_mattered_more", "estimate"],
                "agreement": c.loc["agreement", "estimate"], "agreement_lo": c.loc["agreement", "ci_low"],
                "agreement_hi": c.loc["agreement", "ci_high"], "gain_over_majority": c.loc["gain_over_majority", "estimate"],
                "gain_lo": c.loc["gain_over_majority", "ci_low"], "gain_hi": c.loc["gain_over_majority", "ci_high"],
                "spearman": c.loc["spearman", "estimate"], "dev_cutoff": float(cut_t.tau.iloc[0]),
                "agreement_at_cutoff": cut_t.loc["agreement", "estimate"],
                "gain_majority_at_cutoff": cut_t.loc["gain_over_majority", "estimate"],
                "gain_cut_lo": cut_t.loc["gain_over_majority", "ci_low"], "gain_cut_hi": cut_t.loc["gain_over_majority", "ci_high"],
                "share_weights_extreme": 0.657, "mean_weight": np.nan})
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "quality_ladder.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    print("\n== QUALITY LADDER (gate input = headline input unless stated) ==")
    print(res[["rung", "audio_only_wF1", "gate_wF1", "p_text_mattered_more", "agreement", "gain_over_majority",
               "gain_lo", "gain_hi", "spearman", "dev_cutoff", "gain_majority_at_cutoff", "share_weights_extreme"]].to_string(index=False))
    r3 = res[res.rung == "R3 concat4 mean+std"].gain_over_majority.iloc[0]
    print(f"\nItem 1 reading: R3 (headline gate input) gain over majority {r3:.3f}; step 24 reference 0.294. "
          + ("mostly the audio head" if abs(r3 - 0.294) <= 0.10 else ("mostly the gate input" if r3 < 0.15 else "mixed")))
    print(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
