"""Step 32: does the weight's edge depend on BALANCE (how often each modality decides), not on model strength?

Hypothesis H_bal (stated before running): the gate's weight beats the best trivial guess when text and audio
each decide a sizeable share of the utterances, and adds nothing when one modality dominates.
Evidence so far (steps 25, 31): edge +2 points at 77% text-decides, +23 to +26 at 44 to 46%, 0 at 93%.

Two controlled experiments (VARIANTS; the headline model is unchanged):

  Experiment 1, temperature dial. Fixed heads: the RoBERTa-large text head and the all-layer WavLM audio head
    of step 29 (trained on clf2). The text head's logits are divided by T in {1, 1.5, 2, 3, 5}: this makes its
    votes less extreme without changing which class it picks (so text-only accuracy is unchanged) and moves
    influence toward audio. For each T the gate is retrained (3 seeds, gate2, headline gate inputs) and the swap
    test is rerun.
  Experiment 2, text ladder. Same audio head; text heads of different strength: from the step 28 grid,
    the configurations (excluding the selected best) whose DEV weighted-F1 is closest to 0.50, 0.53, 0.56 and 0.58,
    refit on clf2 (same LBFGS recipe), plus the selected best (T = 1 above is the same model).

  For every setting: text-only and fused (gate) weighted-F1 on test, P(text mattered more), the gate weight's
  agreement, gain over the majority modality with a dialogue-cluster bootstrap (1000 draws, rng 0), the share of the
  possible edge (gain / (1 - majority share)), Spearman, the dev-fitted cutoff and its test gain.
  Swap: full-pool, as in step 31 (effects on the fused probability before any prior correction).

  Reading rule, fixed now: H_bal is SUPPORTED if (a) at settings with P(text mattered more) between 0.35 and 0.65
  the gain over the majority modality is at least 0.10, and (b) at settings with P(text mattered more) at or above
  0.90 it is at most 0.05. It is NOT supported if the edge is similar across the range or does not rise as the
  split approaches 50/50. Intermediate outcomes are reported as such.

Usage: python scripts/32_balance_experiments.py     Output: results/balance_experiments.csv
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, RESULTS, Head, LearnedGate, class_weights, fuse_average, fuse_confidence,  # noqa: E402
                    fuse_gate, load_dataset, metrics, set_determinism, train_gate)

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
s23 = importlib.import_module("23_better_audio_fusion")
s25 = importlib.import_module("25_quality_ladder")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
iv04 = importlib.import_module("04_interventions")
an06 = importlib.import_module("06_analysis")
SEEDS = [0, 1, 2]
TEMPS = [1, 1.5, 2, 3, 5]
TARGETS = [0.50, 0.53, 0.56, 0.58]


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    rows, ids, Wb = P["rows"], P["ids"], P["Wb"]
    sel = json.loads((RESULTS / "text_head_selected.json").read_text())["selected"]
    best_layers = m29.LAYERSETS[sel["features"]]
    assert best_layers == list(range(25)), "script assumes the selected text head uses all 25 layers"
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    emo = {e: i for i, e in enumerate(D["emotions"])}
    lab = {(a, b): emo[e] for a, b, e in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Emotion)}
    ytr = torch.tensor([lab[tuple(i)] for i in ids["train"]])
    y = {"clf": ytr[rows["clf"]], "gate": ytr[rows["gate"]], "dev": S["dev"]["y"], "test": S["test"]["y"]}
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    txt_head = torch.zeros(len(ids["train"]), 384)
    aud_head = torch.zeros(len(ids["train"]), 1024)
    for sp in ("clf", "gate"):
        m = mp[mp.split == sp].sort_values("row")
        txt_head[torch.tensor(m.file_index.values)] = S[sp]["text"]
        aud_head[torch.tensor(m.file_index.values)] = S[sp]["audio"]
    H = {"clf": (txt_head[rows["clf"]], aud_head[rows["clf"]]), "gate": (txt_head[rows["gate"]], aud_head[rows["gate"]]),
         "dev": (S["dev"]["text"], S["dev"]["audio"]), "test": (S["test"]["text"], S["test"]["audio"])}

    rt = {sp: torch.load(DATA / f"roberta_layers_{sp}.pt", map_location="cpu", weights_only=False)["mean"] for sp in ("train", "dev", "test")}
    wl = {sp: torch.load(DATA / f"wavlm_layers_{sp}.pt", map_location="cpu", weights_only=False) for sp in ("train", "dev", "test")}
    bt = lambda m, idx=None: torch.cat([(m if idx is None else m[idx])[:, l].float() for l in range(25)], 1)
    ba = lambda sp, idx=None: b22.build(wl[sp]["mean"] if idx is None else wl[sp]["mean"][idx],
                                        wl[sp]["std"] if idx is None else wl[sp]["std"][idx], m29.AUDIO_LAYERS, m29.AUDIO_POOL)
    rawT = {"clf": bt(rt["train"], rows["clf"]), "gate": bt(rt["train"], rows["gate"]), "dev": bt(rt["dev"]), "test": bt(rt["test"])}
    rawA = {"clf": ba("train", rows["clf"]), "gate": ba("train", rows["gate"]), "dev": ba("dev"), "test": ba("test")}
    FT = {k: ((v - rawT["clf"].mean(0)) / (rawT["clf"].std(0) + 1e-6)).contiguous() for k, v in rawT.items()}
    FA = {k: ((v - rawA["clf"].mean(0)) / (rawA["clf"].std(0) + 1e-6)).contiguous() for k, v in rawA.items()}
    del rawT, rawA, rt, wl
    dT, dA = FT["clf"].shape[1], FA["clf"].shape[1]
    ET = {k: torch.cat([FT[k], H[k][0]], 1).contiguous() for k in FT}
    EA = {k: torch.cat([FA[k], H[k][1]], 1).contiguous() for k in FA}
    (Wt, bt_), (Wa, ba_) = Wb["text"], Wb["audio"]
    ah = Head(dA + 1024)
    ah.lin.weight.data = torch.cat([Wa.T.contiguous(), torch.zeros(7, 1024)], 1)
    ah.lin.bias.data = ba_.clone()
    ah.eval()
    print(f"features ready ({time.time() - t0:.0f}s)", flush=True)

    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(an06.N_BOOT)]
    masks = {sp: (rs07.valid_masks(H[sp][0], H["clf"][0]), rs07.valid_masks(H[sp][1], H["clf"][1])) for sp in ("dev", "test")}

    def run(name, W_full, b_vec, extra):
        th = Head(dT + 384)
        th.lin.weight.data = torch.cat([W_full, torch.zeros(7, 384)], 1)
        th.lin.bias.data = b_vec.clone()
        th.eval()
        models = {}
        for seed in SEEDS:
            set_determinism(seed)
            with torch.no_grad():
                pr = {s: (torch.softmax(th(ET[s]), 1), torch.softmax(ah(EA[s]), 1)) for s in ("gate", "dev", "test")}
            g = train_gate(H["gate"][0], H["gate"][1], *pr["gate"], y["gate"], (H["dev"][0], H["dev"][1], *pr["dev"], y["dev"]), w_gate, seed)
            w1 = g.net[0].weight.data
            Gp = LearnedGate(dT + 384, dA + 1024)
            Gp.net[0].weight.data = torch.cat([torch.zeros(w1.shape[0], dT), w1[:, :384], torch.zeros(w1.shape[0], dA), w1[:, 384:]], 1)
            Gp.net[0].bias.data = g.net[0].bias.data.clone()
            Gp.net[2].load_state_dict(g.net[2].state_dict())
            Gp.eval()
            models[seed] = (th, ah, Gp)
        f1 = {}
        for rule in ("text_only", "audio_only", "fusion_learned_gate"):
            v = []
            for seed in SEEDS:
                t_, a_, G_ = models[seed]
                with torch.no_grad():
                    pt, pa = torch.softmax(t_(ET["test"]), 1), torch.softmax(a_(EA["test"]), 1)
                    p = {"text_only": pt, "audio_only": pa, "fusion_learned_gate": fuse_gate(G_, ET["test"], EA["test"], pt, pa)[0]}[rule]
                v.append(metrics(y["test"], p)["weighted_f1"])
            f1[rule] = float(np.mean(v))

        def swaps(split, tag):
            old = iv04.load_models
            iv04.load_models = lambda seed: models[seed]
            try:
                fr = []
                for seed in SEEDS:
                    orig, _, _, cls, pb, _, _ = rs08.exact_average(seed, ET[split], EA[split], ET["clf"], EA["clf"], *masks[split])
                    fr.append(s23.frames(seed, tag, orig, cls, pb))
            finally:
                iv04.load_models = old
            return pd.concat(fr, ignore_index=True)

        row = s25.summarise(name, S, swaps("test", "resample_full"), swaps("dev", "dev"), {"audio_only": f1["audio_only"], "fusion_learned_gate": f1["fusion_learned_gate"]}, draws)
        row.update(extra)
        row["text_only_wF1"] = f1["text_only"]
        maj = max(row["p_text_mattered_more"], 1 - row["p_text_mattered_more"])
        row["share_of_possible_edge"] = row["gain_over_majority"] / (1 - maj)
        print(f"{name}: text {f1['text_only']:.3f} audio {f1['audio_only']:.3f} gate {f1['fusion_learned_gate']:.3f} | "
              f"P(text) {row['p_text_mattered_more']:.3f} agreement {row['agreement']:.3f} gain {row['gain_over_majority']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        return row

    out = []
    # ---- experiment 1: temperature dial on the text head
    for T in TEMPS:
        out.append(run(f"E1 text temperature T={T}", Wt.T.contiguous() / T, bt_ / T, {"experiment": "1 dial", "setting": T}))
        pd.DataFrame(out).to_csv(RESULTS / "balance_experiments.csv", index=False)

    # ---- experiment 2: text ladder (rungs picked by DEV weighted-F1)
    grid = pd.read_csv(RESULTS / "text_head_grid.csv")
    grid = grid[~grid.selected_on_dev]
    chosen = []
    for tgt in TARGETS:
        r = grid.iloc[(grid.dev_wF1 - tgt).abs().argsort()].iloc[0]
        while (r.features, r.wd) in chosen:
            grid = grid[~((grid.features == r.features) & (grid.wd == r.wd))]
            r = grid.iloc[(grid.dev_wF1 - tgt).abs().argsort()].iloc[0]
        chosen.append((r.features, r.wd))
        layers = m29.LAYERSETS[r.features]
        cols = torch.cat([torch.arange(l * 1024, (l + 1) * 1024) for l in layers])
        Xc = FT["clf"][:, cols]
        W, b = b22.fit(Xc, y["clf"], w_clf, float(r.wd))
        Wfull = torch.zeros(7, dT)
        Wfull[:, cols] = W.T
        out.append(run(f"E2 text {r.features} wd={r.wd} (dev {r.dev_wF1:.3f})", Wfull, b, {"experiment": "2 ladder", "setting": f"{r.features} wd={r.wd}"}))
        pd.DataFrame(out).to_csv(RESULTS / "balance_experiments.csv", index=False)

    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "balance_experiments.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    print("\n", res[["rung", "text_only_wF1", "gate_wF1", "p_text_mattered_more", "agreement", "gain_over_majority", "gain_lo",
                     "gain_hi", "share_of_possible_edge", "spearman", "dev_cutoff", "gain_majority_at_cutoff"]].to_string(index=False))
    mid = res[(res.p_text_mattered_more >= 0.35) & (res.p_text_mattered_more <= 0.65)]
    top = res[res.p_text_mattered_more >= 0.90]
    a = bool(len(mid) and (mid.gain_over_majority >= 0.10).all())
    b = bool(len(top) and (top.gain_over_majority <= 0.05).all())
    print(f"\nReading rule: balanced settings {len(mid)} (all gain >= 0.10: {a}); text-dominated settings {len(top)} "
          f"(all gain <= 0.05: {b}) -> H_bal {'SUPPORTED' if a and b else 'not cleanly supported; see the table'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
