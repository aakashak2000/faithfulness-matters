"""Step 35: calibrated attribution with abstention (step 3 of the "better solution" plan).

Goal: turn a modality signal into a PROBABILITY that "text mattered more" for each utterance, and let it say
"inconclusive" when that probability is near 0.5. Fit on DEV, evaluate on TEST.

Pre-registered before running:
  - Models and gates: M0b, M1, M2, M3 of step 33 (10 gates each). Swap targets: full-pool, |dp| effect difference e;
    the label is T = 1 if e > 0, ties (e = 0) excluded (as in all earlier metrics).
  - Two signals: the plain weight, x = logit(w) with w clipped to [0.001, 0.999]; and the corrected signal of step 34,
    x = s_corr divided by its dev standard deviation (per model).
  - Calibrator: one-dimensional logistic regression P(T = 1 | x) fitted on the dev rows of all 10 gates pooled
    (sklearn LogisticRegression, C = 1). No other features. No test data used for fitting.
  - Reported on test, for each model and signal: Brier score, expected calibration error (10 equal-width bins),
    log loss, accuracy of the call at probability 0.5, and the abstention curve: accuracy on the utterances where
    max(p, 1 - p) is at least a threshold chosen so that the kept share is about 100%, 80%, 60% and 40% of the
    utterances. Reference for each: the accuracy of always guessing the majority modality (computed on the same
    kept set is NOT possible, so the majority share on all test rows is shown).
  - Reading rule, fixed now: the corrected signal is "better calibrated" if its Brier score and ECE are lower than the
    plain weight's in every model with headroom of at least 0.05 (M0b, M1, M3); abstention "helps" if accuracy rises
    monotonically as coverage falls.

Usage: python scripts/35_calibrated_attribution.py      Outputs: results/calibrated_attribution.csv,
       results/calibrated_attribution_abstention.csv
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, RESULTS, Head, LearnedGate, class_weights, fuse_gate, load_dataset,  # noqa: E402
                    set_determinism, train_gate, train_head)

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
m33 = importlib.import_module("33_eval_harness")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
iv04 = importlib.import_module("04_interventions")
SEEDS = list(range(10))
COVERAGES = [1.0, 0.8, 0.6, 0.4]


def ece(p, t, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(t[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


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
    masks = {sp: (rs07.valid_masks(H[sp][0], H["clf"][0]), rs07.valid_masks(H[sp][1], H["clf"][1])) for sp in ("dev", "test")}
    print(f"ready ({time.time() - t0:.0f}s)", flush=True)

    out, curves = [], []
    for name, (th, ah) in specs.items():
        with torch.no_grad():
            m_t = torch.softmax(th(ET["clf"]), 1).mean(0)
            m_a = torch.softmax(ah(EA["clf"]), 1).mean(0)
        gates = {}
        sig = {sp: {"w": [], "corr": [], "e": []} for sp in ("dev", "test")}
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
            for sp in ("dev", "test"):
                with torch.no_grad():
                    pt, pa = pr[sp]
                    p0, w = fuse_gate(Gp, ET[sp], EA[sp], pt, pa)
                    c = p0.argmax(1)
                    idx = torch.arange(len(c))
                    corr = w.squeeze(1) * (pt[idx, c] - m_t[c]).abs() - (1 - w.squeeze(1)) * (pa[idx, c] - m_a[c]).abs()
                sig[sp]["w"].append(w.squeeze(1).numpy())
                sig[sp]["corr"].append(corr.numpy())
        for sp in ("dev", "test"):
            old = iv04.load_models
            iv04.load_models = lambda seed: gates[seed]
            try:
                for seed in SEEDS:
                    orig, _, _, cls, pb, _, _ = rs08.exact_average(seed, ET[sp], EA[sp], ET["clf"], EA["clf"], *masks[sp])
                    fr = m33.frames(seed, orig, cls, pb)
                    d = fr[fr.model == "fusion_learned_gate"].sort_values("row")
                    sig[sp]["e"].append(d.dp_text.abs().values - d.dp_audio.abs().values)
            finally:
                iv04.load_models = old
        # flatten gates x utterances; drop ties
        flat = {sp: {k: np.concatenate(v) for k, v in sig[sp].items()} for sp in sig}
        keep = {sp: flat[sp]["e"] != 0 for sp in flat}
        T = {sp: (flat[sp]["e"][keep[sp]] > 0).astype(int) for sp in flat}
        maj_test = max(T["test"].mean(), 1 - T["test"].mean())
        feats = {"plain weight": lambda sp, sc=None: np.log(np.clip(flat[sp]["w"][keep[sp]], 1e-3, 1 - 1e-3) / (1 - np.clip(flat[sp]["w"][keep[sp]], 1e-3, 1 - 1e-3))),
                 "corrected signal": None}
        sd_dev = flat["dev"]["corr"][keep["dev"]].std()
        feats["corrected signal"] = lambda sp, sc=sd_dev: flat[sp]["corr"][keep[sp]] / sc
        for sname, fn in feats.items():
            Xd, Xt = fn("dev").reshape(-1, 1), fn("test").reshape(-1, 1)
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xd, T["dev"])
            p = clf.predict_proba(Xt)[:, 1]
            brier = float(np.mean((p - T["test"]) ** 2))
            ll = float(-np.mean(T["test"] * np.log(np.clip(p, 1e-6, 1)) + (1 - T["test"]) * np.log(np.clip(1 - p, 1e-6, 1))))
            acc = float(np.mean((p > 0.5) == T["test"]))
            out.append({"model": name, "signal": sname, "p_text_test": float(T["test"].mean()), "majority_share": float(maj_test),
                        "brier": brier, "brier_of_constant": float(T["test"].mean() * (1 - T["test"].mean())), "ece": ece(p, T["test"]),
                        "log_loss": ll, "accuracy_at_0.5": acc})
            conf = np.maximum(p, 1 - p)
            for cov in COVERAGES:
                thr = np.quantile(conf, 1 - cov) if cov < 1 else 0.0
                k = conf >= thr
                curves.append({"model": name, "signal": sname, "coverage": cov, "kept_share": float(k.mean()),
                               "accuracy_kept": float(np.mean((p[k] > 0.5) == T["test"][k])), "majority_share_all": float(maj_test)})
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(out).to_csv(RESULTS / "calibrated_attribution.csv", index=False)
        pd.DataFrame(curves).to_csv(RESULTS / "calibrated_attribution_abstention.csv", index=False)

    res, cur = pd.DataFrame(out), pd.DataFrame(curves)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 4)
    print("\n", res.to_string(index=False))
    print("\nAbstention (accuracy on the kept utterances):")
    print(cur.pivot_table(index=["model", "signal"], columns="coverage", values="accuracy_kept").round(3).to_string())
    judged = res[res.model.isin(["M0b headline recipe, v2 split", "M1 stronger audio", "M3 stronger text + audio"])]
    a = judged[judged.signal == "plain weight"].set_index("model")
    b = judged[judged.signal == "corrected signal"].set_index("model")
    print("\nReading rule: corrected has lower Brier and ECE in all judged models:",
          bool(((b.brier < a.brier) & (b.ece < a.ece)).all()))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
