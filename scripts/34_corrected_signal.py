"""Step 34: a corrected attribution signal (step 2 of the "better solution" plan).

Idea (derived, not fitted): in decision-level fusion p = w * p_text + (1 - w) * p_audio, replacing one modality by
a typical donor changes the probability of the predicted class c by about
      text  :  w       * | p_text[c]  - m_text[c]  |
      audio : (1 - w)  * | p_audio[c] - m_audio[c] |
where m_text, m_audio are the average head probabilities over the donor pool (the "typical vote"). The weight w alone
ignores how DISTINCTIVE each head's vote is. So the corrected signal is
      s_corr = w * |p_text[c] - m_text[c]|  -  (1 - w) * |p_audio[c] - m_audio[c]|     (positive: text mattered more).
It needs one forward pass and the pool means (computed once); it does not run any swap. For the learned gate it
ignores the gate's own re-weighting after a swap (step 15: that flips the winner for about 7% of utterances), so it is
an approximation. Because it uses the same donor averages the swap uses, it explains the mismatch; it is not an
independent confirmation.

Pre-registered before running:
  - Models: M0b, M1, M2, M3 of step 33 (same heads, split, 10 gates each, headline gate inputs). c = the gate model's
    predicted class on the original input. m = mean of the head's softmax over the clf2 donors.
  - Targets: the primary swap test of step 33 (full-pool, |dp| effect difference), same bootstrap (1000 draws, rng 0,
    10 gates averaged inside each draw).
  - Compared: the plain weight (2w - 1) and s_corr, on agreement, gain over the majority modality, share of the
    possible edge, and Spearman with the effect difference.
  - Success rule, fixed now: s_corr is a useful correction if, in every model with headroom (1 - majority share) of
    at least 0.05, (a) its gain over the majority modality exceeds the plain weight's, and (b) it reaches at least
    50% of the possible edge. Models with less headroom are reported but not judged. If it only matches or lowers the
    plain weight, we say so.

Usage: python scripts/34_corrected_signal.py        Output: results/corrected_signal.csv
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
from common import (DATA, RESULTS, Head, LearnedGate, class_weights, fuse_gate, load_dataset,  # noqa: E402
                    set_determinism, train_gate, train_head)

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
m33 = importlib.import_module("33_eval_harness")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
iv04 = importlib.import_module("04_interventions")
an06 = importlib.import_module("06_analysis")
SEEDS = list(range(10))


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    rows, ids, Wb = P["rows"], P["ids"], P["Wb"]
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

    def padded(W_blocks, b, dim_total, offset, n_in):
        h = Head(dim_total)
        w = torch.zeros(7, dim_total)
        w[:, offset:offset + n_in] = W_blocks
        h.lin.weight.data, h.lin.bias.data = w, b.clone()
        h.eval()
        return h

    set_determinism(0)
    th_mini = train_head(H["clf"][0], y["clf"], H["dev"][0], y["dev"], w_clf, 0)
    ah_head = train_head(H["clf"][1], y["clf"], H["dev"][1], y["dev"], w_clf, 0)
    T_MINI = padded(th_mini.lin.weight.data, th_mini.lin.bias.data, dT + 384, dT, 384)
    T_ROB = padded(Wt.T.contiguous(), bt_, dT + 384, 0, dT)
    A_HEAD = padded(ah_head.lin.weight.data, ah_head.lin.bias.data, dA + 1024, dA, 1024)
    A_STR = padded(Wa.T.contiguous(), ba_, dA + 1024, 0, dA)
    specs = {"M0b headline recipe, v2 split": (T_MINI, A_HEAD), "M1 stronger audio": (T_MINI, A_STR),
             "M2 stronger text": (T_ROB, A_HEAD), "M3 stronger text + audio": (T_ROB, A_STR)}
    masks = {sp: (rs07.valid_masks(H[sp][0], H["clf"][0]), rs07.valid_masks(H[sp][1], H["clf"][1])) for sp in ("test",)}

    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(an06.N_BOOT)]
    print(f"ready ({time.time() - t0:.0f}s)", flush=True)

    out = []
    for name, (th, ah) in specs.items():
        with torch.no_grad():
            m_t = torch.softmax(th(ET["clf"]), 1).mean(0)     # typical text vote
            m_a = torch.softmax(ah(EA["clf"]), 1).mean(0)     # typical audio vote
        gates, corr_rows = {}, []
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
            gates[seed] = (th, ah, Gp)
            with torch.no_grad():
                pt, pa = pr["test"]
                p0, w = fuse_gate(Gp, ET["test"], EA["test"], pt, pa)
                c = p0.argmax(1)
                idx = torch.arange(len(c))
                dt, da = (pt[idx, c] - m_t[c]).abs(), (pa[idx, c] - m_a[c]).abs()
                corr = w.squeeze(1) * dt - (1 - w.squeeze(1)) * da
            corr_rows.append(pd.DataFrame({"seed": seed, "row": range(len(c)), "s_corr": corr.numpy()}))
        corr_df = pd.concat(corr_rows, ignore_index=True)
        old = iv04.load_models
        iv04.load_models = lambda seed: gates[seed]
        try:
            fr = []
            for seed in SEEDS:
                orig, _, _, cls, pb, _, _ = rs08.exact_average(seed, ET["test"], EA["test"], ET["clf"], EA["clf"], *masks["test"])
                fr.append(m33.frames(seed, orig, cls, pb))
        finally:
            iv04.load_models = old
        log = pd.concat(fr, ignore_index=True)
        s_w, e = m33.effect_arrays(log, "fusion_learned_gate", "gate_w_t", "dp")
        s_c = np.stack([corr_df[corr_df.seed == k].sort_values("row").s_corr.values for k in SEEDS])
        # stats() treats the sign of s as the call and ranks s for Spearman; s_corr is already signed
        row = {"model": name}
        for tag, s in (("weight", s_w), ("corrected", s_c)):
            pt_ = m33.stats(s, e)
            b = np.array([m33.stats(s[:, i], e[:, i]) for i in draws])
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            row.update({f"{tag}_agreement": pt_[0], f"{tag}_gain_majority": pt_[2], f"{tag}_gain_lo": lo[2], f"{tag}_gain_hi": hi[2],
                        f"{tag}_spearman": pt_[3]})
            if tag == "weight":
                row["p_text_mattered_more"] = pt_[1]
        maj = max(row["p_text_mattered_more"], 1 - row["p_text_mattered_more"])
        row["headroom"] = 1 - maj
        for tag in ("weight", "corrected"):
            row[f"{tag}_share_of_possible_edge"] = row[f"{tag}_gain_majority"] / row["headroom"]
        out.append(row)
        pd.DataFrame(out).to_csv(RESULTS / "corrected_signal.csv", index=False)
        print(f"{name}: P(text) {row['p_text_mattered_more']:.3f} | weight agreement {row['weight_agreement']:.3f} gain {row['weight_gain_majority']:.3f} "
              f"spearman {row['weight_spearman']:.3f} | corrected agreement {row['corrected_agreement']:.3f} gain {row['corrected_gain_majority']:.3f} "
              f"[{row['corrected_gain_lo']:.3f},{row['corrected_gain_hi']:.3f}] spearman {row['corrected_spearman']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "corrected_signal.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    print("\n", res[["model", "p_text_mattered_more", "headroom", "weight_agreement", "weight_gain_majority", "weight_share_of_possible_edge",
                     "corrected_agreement", "corrected_gain_majority", "corrected_gain_lo", "corrected_gain_hi",
                     "corrected_share_of_possible_edge", "weight_spearman", "corrected_spearman"]].to_string(index=False))
    judged = res[res.headroom >= 0.05]
    ok_a = bool((judged.corrected_gain_majority > judged.weight_gain_majority).all())
    ok_b = bool((judged.corrected_share_of_possible_edge >= 0.5).all())
    print(f"\nSuccess rule on {len(judged)} models with headroom >= 0.05: (a) beats the plain weight in all: {ok_a}; "
          f"(b) reaches 50% of the possible edge in all: {ok_b}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
