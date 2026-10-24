"""Step 37: the primary swap test on the BEST model (S2 of step 30: strong text + strong audio + conversation context).

PROTOCOL, written before running (the 63.6 weighted-F1 model had no faithfulness result until now):
  - Model: exactly S2 of step 30 (RoBERTa-large text head, all-layer WavLM audio head, stage-2 context heads chosen on
    dev, gate trained on gate2 with the headline inputs, seeds 0 to 2). Refit here deterministically; the refit must
    reproduce the saved stage-2 logits and the step 30 gate test weighted-F1 (0.6357 at dev alpha 0.5).
  - Swap = the same definition as 08/31, applied to the utterance being explained ONLY: its text (or audio) is replaced
    by a donor utterance's, its dialogue neighbours are left untouched. Technically: the utterance's stage-1 logits for
    that modality are replaced by the donor's and the headline gate embedding moves with them; the stage-2 context part
    (neighbours' stage-1 logits, unchanged) is kept; stage-2 and the gate are recomputed. Exact over the full donor pool.
  - Donors = the clf2 utterances, byte-identical headline embeddings excluded. Donor stage-1 logits are the OUT-OF-FOLD
    ones (those are the logits stage 2 was trained on, so they are in-distribution for it).
  - Effects on the fused probability BEFORE the class-prior correction (the mixture the gate controls); predicted class
    = argmax of that mixture. Everything else as 31: P(text mattered more), agreement, gain over the majority modality
    (dialogue-cluster bootstrap, 1000 draws, rng 0), dev-fitted cutoff, share of extreme weights; plus the one-pass
    corrected signal of 34 (m = mean donor stage-2 vote, with the utterance's own context).
  - No direction predicted. Reference rows: M3 (same encoders, no context) from results/harness_models.csv.

Usage: python scripts/37_swap_with_context.py
Output: results/swap_with_context.csv
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, RESULTS, class_weights, fuse_gate, load_dataset, set_determinism, train_gate)  # noqa: E402

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
s23 = importlib.import_module("23_better_audio_fusion")
s25 = importlib.import_module("25_quality_ladder")
rs07 = importlib.import_module("07_resampling")
an06 = importlib.import_module("06_analysis")
SEEDS, MODELS, CHUNK = [0, 1, 2], rs07.MODELS, 100


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    P2 = torch.load(DATA / "v2_pipeline_v2.pt", weights_only=False)
    rows, Z1, ids, dl, ut = P["rows"], P["Z1"], P["ids"], P["dlg"], P["utt"]
    cfg, alpha = P2["cfg"], P2["alpha"]["S2 + context, bidirectional"]
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    emo = {e: i for i, e in enumerate(D["emotions"])}
    lab = {(a, b): emo[e] for a, b, e in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Emotion)}
    ytr = torch.tensor([lab[tuple(i)] for i in ids["train"]])
    y = {"clf": ytr[rows["clf"]], "gate": ytr[rows["gate"]], "dev": S["dev"]["y"], "test": S["test"]["y"]}
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    pi = torch.bincount(y["clf"], minlength=7).float()
    lp = (pi / pi.sum()).log()
    txt_head = torch.zeros(len(ids["train"]), 384)
    aud_head = torch.zeros(len(ids["train"]), 1024)
    for sp in ("clf", "gate"):
        m = mp[mp.split == sp].sort_values("row")
        txt_head[torch.tensor(m.file_index.values)] = S[sp]["text"]
        aud_head[torch.tensor(m.file_index.values)] = S[sp]["audio"]
    G = {"clf": (txt_head[rows["clf"]], aud_head[rows["clf"]]), "gate": (txt_head[rows["gate"]], aud_head[rows["gate"]]),
         "dev": (S["dev"]["text"], S["dev"]["audio"]), "test": (S["test"]["text"], S["test"]["audio"])}

    # ---- refit stage 2 (deterministic) and split each logit into own part + context part
    own, ctxpart, Z2 = {}, {}, {}
    for kind in ("text", "audio"):
        c = cfg[kind]
        C = {s: m29.ctx_features(Z1[kind][s], dl[s], ut[s], c["k"], c["mode"]) for s in ("clf", "gate", "dev", "test")}
        W, b = b22.fit(C["clf"], y["clf"], w_clf, float(c["wd"]))
        own[kind] = W[:7]
        ctxpart[kind] = {s: C[s][:, 7:] @ W[7:] + b for s in ("dev", "test")}
        Z2[kind] = {s: C[s] @ W + b for s in ("gate", "dev", "test")}
        err = (Z2[kind]["test"] - P2["Z2"][kind]["test"]).abs().max().item()
        print(f"{kind}: stage-2 refit max |diff| to saved logits {err:.2e}", flush=True)
        assert err < 1e-3
    donor_logit = {k: Z1[k]["clf"] @ own[k] for k in ("text", "audio")}   # own part for each donor (pool, 7)

    models = {}
    for seed in SEEDS:
        set_determinism(seed)
        Pg = {s: (torch.softmax(Z2["text"][s], 1), torch.softmax(Z2["audio"][s], 1)) for s in ("gate", "dev", "test")}
        g = train_gate(G["gate"][0], G["gate"][1], *Pg["gate"], y["gate"],
                       (G["dev"][0], G["dev"][1], *Pg["dev"], y["dev"]), w_gate, seed)
        g.eval()
        models[seed] = g
    f1, f1a = {}, {}
    for rule in ("text_only", "audio_only", "fusion_learned_gate", "fusion_average", "fusion_confidence"):
        v0, va = [], []
        for seed in SEEDS:
            pt, pa = torch.softmax(Z2["text"]["test"], 1), torch.softmax(Z2["audio"]["test"], 1)
            with torch.no_grad():
                w = {"fusion_learned_gate": models[seed](*G["test"]), "fusion_average": torch.full((len(pt), 1), 0.5),
                     "fusion_confidence": (pt.max(1, keepdim=True).values / (pt.max(1, keepdim=True).values + pa.max(1, keepdim=True).values)),
                     "text_only": torch.ones(len(pt), 1), "audio_only": torch.zeros(len(pt), 1)}[rule]
            p = w * pt + (1 - w) * pa
            v0.append(f1_score(y["test"].numpy(), p.argmax(1).numpy(), average="weighted", zero_division=0))
            va.append(f1_score(y["test"].numpy(), (p.log() + alpha * lp).argmax(1).numpy(), average="weighted", zero_division=0))
        f1[rule], f1a[rule] = float(np.mean(v0)), float(np.mean(va))
    print("test weighted-F1 alpha 0:", {k: round(v, 4) for k, v in f1.items()}, flush=True)
    print(f"test weighted-F1 dev alpha {alpha}:", {k: round(v, 4) for k, v in f1a.items()}, "(step 30 gate: 0.6357)", flush=True)

    def swaps(split, tag):
        n = len(y[split])
        ok = {"text": rs07.valid_masks(G[split][0], G["clf"][0]), "audio": rs07.valid_masks(G[split][1], G["clf"][1])}
        pt, pa = torch.softmax(Z2["text"][split], 1), torch.softmax(Z2["audio"][split], 1)
        ct, ca = pt.max(1).values, pa.max(1).values
        idx = torch.arange(n)
        frames, corr = [], []
        for seed in SEEDS:
            g = models[seed]
            lin1, lin2 = g.net[0], g.net[2]
            Wt, Wa, b1 = lin1.weight[:, :384], lin1.weight[:, 384:], lin1.bias
            w2, b2 = lin2.weight[0], lin2.bias[0]
            with torch.no_grad():
                w0 = g(*G[split])
                ws = {"fusion_learned_gate": w0, "fusion_confidence": (ct / (ct + ca))[:, None], "fusion_average": torch.full((n, 1), 0.5)}
                orig = {m: (ws[m] * pt + (1 - ws[m]) * pa, ws[m]) for m in MODELS}
                cls = {m: orig[m][0].argmax(1) for m in MODELS}
                At_pool, Aa_pool = G["clf"][0] @ Wt.T, G["clf"][1] @ Wa.T
                At_i, Aa_i = G[split][0] @ Wt.T, G[split][1] @ Wa.T
                Sm = {(m, k): torch.zeros(n, 7) for m in MODELS for k in ("text", "audio")}
                Vote = {k: torch.zeros(n, 7) for k in ("text", "audio")}
                for kind in ("text", "audio"):
                    okm = ok[kind].float()
                    for lo in range(0, G["clf"][0].shape[0], CHUNK):
                        sl = slice(lo, lo + CHUNK)
                        m_ = okm[:, sl]
                        Pd = torch.softmax(donor_logit[kind][sl][None] + ctxpart[kind][split][:, None], 2)   # (n, chunk, 7)
                        Vote[kind] += (m_[:, :, None] * Pd).sum(1)
                        if kind == "text":
                            pre = At_pool[sl][None] + Aa_i[:, None]
                            wd = torch.sigmoid(torch.relu(pre + b1) @ w2 + b2)
                            ct_, ca_ = Pd.max(2).values, ca[:, None]
                        else:
                            pre = At_i[:, None] + Aa_pool[sl][None]
                            wd = torch.sigmoid(torch.relu(pre + b1) @ w2 + b2)
                            ct_, ca_ = ct[:, None], Pd.max(2).values
                        wm = {"fusion_learned_gate": wd, "fusion_confidence": ct_ / (ct_ + ca_), "fusion_average": torch.full_like(wd, 0.5)}
                        for m in MODELS:
                            w = wm[m] * m_
                            if kind == "text":
                                Sm[(m, kind)] += (w[:, :, None] * Pd).sum(1) + pa * (m_ - w).sum(1, keepdim=True)
                            else:
                                Sm[(m, kind)] += pt * w.sum(1, keepdim=True) + ((m_ - w)[:, :, None] * Pd).sum(1)
                n_t, n_a = ok["text"].sum(1).float(), ok["audio"].sum(1).float()
                pb = {m: (Sm[(m, "text")] / n_t[:, None], Sm[(m, "audio")] / n_a[:, None]) for m in MODELS}
                mt, ma = Vote["text"] / n_t[:, None], Vote["audio"] / n_a[:, None]
                c = cls["fusion_learned_gate"]
                w = w0.squeeze(1)
                sc = w * (pt[idx, c] - mt[idx, c]).abs() - (1 - w) * (pa[idx, c] - ma[idx, c]).abs()
            frames.append(s23.frames(seed, tag, orig, cls, pb))
            corr.append(sc.numpy())
            print(f"  {split} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
        return pd.concat(frames, ignore_index=True), np.stack(corr)

    test_log, s_corr = swaps("test", "resample_full")
    dev_log, _ = swaps("dev", "dev")
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(an06.N_BOOT)]
    row = s25.summarise("S2 best model: strong text + strong audio + context", S, test_log, dev_log, f1, draws)
    row["gate_wF1_dev_alpha"] = f1a["fusion_learned_gate"]
    _, e = an06.arrays(test_log, "fusion_learned_gate", "gate_w_t", "resample_full")
    s24 = importlib.import_module("24_strong_audio_checks")
    pt_ = s24.stats3(s_corr, e)
    bs = np.array([s24.stats3(s_corr[:, i], e[:, i]) for i in draws])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    row.update({"corrected_agreement": pt_[0], "corrected_agreement_lo": lo[0], "corrected_agreement_hi": hi[0],
                "corrected_gain_majority": pt_[3], "corrected_gain_lo": lo[3], "corrected_gain_hi": hi[3], "corrected_spearman": pt_[4]})
    # where audio won (the rare case): how often does the weight say so, and how often is a 'audio' call right?
    s_w, _ = an06.arrays(test_log, "fusion_learned_gate", "gate_w_t", "resample_full")
    ra, rt, pr_a = [], [], []
    for sk, ek in zip(s_w, e):
        v = ek != 0
        ra.append(np.mean(sk[v & (ek < 0)] < 0))
        rt.append(np.mean(sk[v & (ek > 0)] > 0))
        pr_a.append(np.mean(ek[v & (sk < 0)] < 0) if np.any(v & (sk < 0)) else np.nan)
    row.update({"weight_recall_audio_won": float(np.mean(ra)), "weight_recall_text_won": float(np.mean(rt)),
                "weight_precision_audio_call": float(np.nanmean(pr_a)),
                "share_audio_won": float(np.mean(e[e != 0] < 0))})
    # confidence / average rules as extra reference
    for m in ("fusion_confidence", "fusion_average"):
        s_, e_ = an06.arrays(test_log, m, "conf_w_t" if m == "fusion_confidence" else "gate_w_t", "resample_full")
        if m == "fusion_average":
            s_ = np.zeros_like(s_)
        st = s24.stats3(s_, e_)
        row[f"{m}_agreement"], row[f"{m}_gain_majority"] = st[0], st[3]
    res = pd.DataFrame([row])
    res.to_csv(RESULTS / "swap_with_context.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    print(res.T.to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
