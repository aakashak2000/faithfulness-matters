"""Step 31: the primary swap test on the strong-text + strong-audio model (S1 of step 30, no context).

Question: when BOTH modalities are strong, does the gate's weight still agree with the swap test, and does its
edge over the best trivial guess persist? (The ladder of step 25 varied only the audio side.)

Pre-registered before running:
  - Model: RoBERTa-large text head (config of results/text_head_selected.json) and the all-layer WavLM audio head
    (R4), both trained on clf2 (the v2 dialogue-level split of step 29), gates trained on gate2 with the HEADLINE
    gate inputs (MiniLM text embedding, 1,024-d audio embedding), seeds 0 to 2. No conversation context.
  - Swap: full-pool, exactly as 08_resampling_full.py: donors = the clf2 utterances, byte-identical headline
    embeddings excluded; the replaced modality is swapped for a donor's FULL feature vector (the head's features
    and the gate's headline embedding move together); effects are measured on the fused probability BEFORE the class
    prior correction (the mixture the gate controls); the decision rule's prior correction is not part of the swap.
  - Reported (same definitions as step 25): P(text mattered more), the gate weight's agreement, gain over the majority
    modality with a dialogue-cluster bootstrap (1000 draws, rng 0, 3 seeds averaged), Spearman, the dev-fitted
    cutoff (per seed, frozen) with its test gain, the share of extreme weights; plus fused weighted-F1 at alpha = 0.
  - No direction is predicted. Reference rows: the headline model (R0) and the best audio-only ladder rung (R4).

Usage: python scripts/31_swap_strong_text.py
Output: results/swap_strong_text.csv
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


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    rows, ids, Wb = P["rows"], P["ids"], P["Wb"]
    sel = json.loads((RESULTS / "text_head_selected.json").read_text())["selected"]
    text_layers = m29.LAYERSETS[sel["features"]]
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    emo = {e: i for i, e in enumerate(D["emotions"])}
    lab = {(a, b): emo[e] for a, b, e in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Emotion)}
    ytr = torch.tensor([lab[tuple(i)] for i in ids["train"]])
    y = {"clf": ytr[rows["clf"]], "gate": ytr[rows["gate"]], "dev": S["dev"]["y"], "test": S["test"]["y"]}
    w_gate = class_weights(y["gate"])
    txt_head = torch.zeros(len(ids["train"]), 384)
    aud_head = torch.zeros(len(ids["train"]), 1024)
    for sp in ("clf", "gate"):
        m = mp[mp.split == sp].sort_values("row")
        txt_head[torch.tensor(m.file_index.values)] = S[sp]["text"]
        aud_head[torch.tensor(m.file_index.values)] = S[sp]["audio"]
    H = {"clf": (txt_head[rows["clf"]], aud_head[rows["clf"]]), "gate": (txt_head[rows["gate"]], aud_head[rows["gate"]]),
         "dev": (S["dev"]["text"], S["dev"]["audio"]), "test": (S["test"]["text"], S["test"]["audio"])}

    # standardised features (statistics from clf2, exactly as in step 29)
    rt = {sp: torch.load(DATA / f"roberta_layers_{sp}.pt", map_location="cpu", weights_only=False)["mean"] for sp in ("train", "dev", "test")}
    wl = {sp: torch.load(DATA / f"wavlm_layers_{sp}.pt", map_location="cpu", weights_only=False) for sp in ("train", "dev", "test")}
    bt = lambda m, idx=None: torch.cat([(m if idx is None else m[idx])[:, l].float() for l in text_layers], 1)
    ba = lambda sp, idx=None: b22.build(wl[sp]["mean"] if idx is None else wl[sp]["mean"][idx],
                                        wl[sp]["std"] if idx is None else wl[sp]["std"][idx], m29.AUDIO_LAYERS, m29.AUDIO_POOL)
    rawT = {"clf": bt(rt["train"], rows["clf"]), "gate": bt(rt["train"], rows["gate"]), "dev": bt(rt["dev"]), "test": bt(rt["test"])}
    rawA = {"clf": ba("train", rows["clf"]), "gate": ba("train", rows["gate"]), "dev": ba("dev"), "test": ba("test")}
    FT = {k: ((v - rawT["clf"].mean(0)) / (rawT["clf"].std(0) + 1e-6)).contiguous() for k, v in rawT.items()}
    FA = {k: ((v - rawA["clf"].mean(0)) / (rawA["clf"].std(0) + 1e-6)).contiguous() for k, v in rawA.items()}
    del rawT, rawA, rt, wl
    dT, dA = FT["clf"].shape[1], FA["clf"].shape[1]
    print(f"features ready: text {dT}, audio {dA}  ({time.time() - t0:.0f}s)", flush=True)

    # heads (trained on clf2 in step 29), zero-padded to read only their own block
    (Wt, bt_), (Wa, ba_) = Wb["text"], Wb["audio"]
    th = Head(dT + 384)
    th.lin.weight.data = torch.cat([Wt.T.contiguous(), torch.zeros(7, 384)], 1)
    th.lin.bias.data = bt_.clone()
    ah = Head(dA + 1024)
    ah.lin.weight.data = torch.cat([Wa.T.contiguous(), torch.zeros(7, 1024)], 1)
    ah.lin.bias.data = ba_.clone()
    th.eval(); ah.eval()
    ET = {k: torch.cat([FT[k], H[k][0]], 1).contiguous() for k in FT}
    EA = {k: torch.cat([FA[k], H[k][1]], 1).contiguous() for k in FA}

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
    print(f"gates trained ({time.time() - t0:.0f}s)", flush=True)

    # fused weighted-F1 at alpha = 0 for reference
    f1 = {}
    for rule in ("text_only", "audio_only", "fusion_learned_gate", "fusion_average", "fusion_confidence"):
        vals = []
        for seed in SEEDS:
            t_, a_, G_ = models[seed]
            with torch.no_grad():
                pt, pa = torch.softmax(t_(ET["test"]), 1), torch.softmax(a_(EA["test"]), 1)
                p = {"text_only": pt, "audio_only": pa, "fusion_learned_gate": fuse_gate(G_, ET["test"], EA["test"], pt, pa)[0],
                     "fusion_average": fuse_average(pt, pa)[0], "fusion_confidence": fuse_confidence(pt, pa)[0]}[rule]
            vals.append(metrics(y["test"], p)["weighted_f1"])
        f1[rule] = float(np.mean(vals))
    print("test weighted-F1 at alpha 0:", {k: round(v, 4) for k, v in f1.items()}, flush=True)

    # swaps: masks from the headline embeddings (identical clips give identical vectors)
    def swaps(split, tag):
        ok_t = rs07.valid_masks(H[split][0], H["clf"][0])
        ok_a = rs07.valid_masks(H[split][1], H["clf"][1])
        old = iv04.load_models
        iv04.load_models = lambda seed: models[seed]
        try:
            fr = []
            for seed in SEEDS:
                orig, _, _, cls, pb, _, _ = rs08.exact_average(seed, ET[split], EA[split], ET["clf"], EA["clf"], ok_t, ok_a)
                fr.append(s23.frames(seed, tag, orig, cls, pb))
                print(f"  {split} seed {seed} swaps done ({time.time() - t0:.0f}s)", flush=True)
        finally:
            iv04.load_models = old
        return pd.concat(fr, ignore_index=True)

    test_log = swaps("test", "resample_full")
    dev_log = swaps("dev", "dev")
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(an06.N_BOOT)]
    row = s25.summarise("S1 strong text + strong audio (gate: headline input)", S, test_log, dev_log, f1, draws)
    lad = pd.read_csv(RESULTS / "quality_ladder.csv")
    keep = lad[lad.rung.isin(["R0 headline", "R4 all25 mean+std wd1.0"])]
    res = pd.concat([keep, pd.DataFrame([row])], ignore_index=True)
    res.to_csv(RESULTS / "swap_strong_text.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    print(res[["rung", "audio_only_wF1", "gate_wF1", "p_text_mattered_more", "agreement", "agreement_lo", "agreement_hi",
               "gain_over_majority", "gain_lo", "gain_hi", "spearman", "dev_cutoff", "gain_majority_at_cutoff",
               "share_weights_extreme"]].to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
