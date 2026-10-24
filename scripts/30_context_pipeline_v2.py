"""Step 30: corrected version of the F1 push (replaces the S2 and S3 rows of step 29).

CORRECTION NOTICE (disclosed, written before this script was run): step 29 chose the conversation-context
settings by dev weighted-F1 BEFORE applying any class-prior correction, and compared the stages without giving
each its own prior correction. Class-balanced training lowers weighted-F1 because it favours rare classes, so the
stage-2 refit looked worse than it is (a stage-2 model that sees no context at all also dropped 3 to 10 points).
The comparison of stages was therefore unfair. This script fixes the order and gives every stage its own
dev-selected prior correction. The step 29 stage-1 logits (data/v2_pipeline.pt) are reused unchanged.

Fixed before running:
  - Same v2 split, features, stage-1 heads (and their out-of-fold logits), gate, seeds and metrics as step 29.
  - Prior correction: log p + alpha * log(pi), pi = clf2 class frequencies, alpha in {0, 0.25, 0.5, 0.75, 1}.
  - Stage 2 (conversation context), per modality: same grid as step 29 (mode prev or both; k in {1,2,3,5};
    wd in {0.001, 0.01, 0.1}; balanced class-weighted CE). A configuration's score = its best dev weighted-F1
    over alpha. The configuration with the highest score is chosen; the best "prev only" (causal) one is
    reported too. No context = the stage-1 logits themselves.
  - Fusion: gate (3 seeds, as 02_train.py, headline inputs), average, confidence. The prior correction alpha is
    applied to the FUSED probabilities, chosen per stage on dev (mean over the 3 seeds, gate rule) and applied to
    all three rules; alpha = 0 is reported too.
  - Stages (test weighted-F1, macro-F1, accuracy, mean over 3 seeds; each with alpha = 0 and with its dev alpha):
      S0 MiniLM text head (Adam, as 02) + audio head, no context        (control on the v2 split)
      S1 RoBERTa text head + audio head, no context
      S2 + conversation context (bidirectional, chosen on dev)
      S2c + conversation context, previous utterances only (causal)
  - Test is scored once per stage; nothing is chosen on test.

Usage: python scripts/30_context_pipeline_v2.py
Outputs: results/f1_push_v2.csv, results/context_grid_v2.csv, results/f1_push_v2_summary.json,
         data/v2_pipeline_v2.pt (local)
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
from common import (DATA, RESULTS, class_weights, fuse_average, fuse_confidence, fuse_gate,  # noqa: E402
                    load_dataset, metrics, set_determinism, train_gate, train_head)

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
SEEDS = [0, 1, 2]
ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    rows, Z1, ids, dl, ut = P["rows"], P["Z1"], P["ids"], P["dlg"], P["utt"]
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    emo = {e: i for i, e in enumerate(D["emotions"])}
    lab = {(a, b): emo[e] for a, b, e in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Emotion)}
    ytr = torch.tensor([lab[tuple(i)] for i in ids["train"]])
    y = {"clf": ytr[rows["clf"]], "gate": ytr[rows["gate"]], "dev": S["dev"]["y"], "test": S["test"]["y"]}
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    pi = torch.bincount(y["clf"], minlength=7).float()
    pi = pi / pi.sum()
    lp = pi.log()

    txt_head = torch.zeros(len(ids["train"]), 384)
    aud_head = torch.zeros(len(ids["train"]), 1024)
    for sp in ("clf", "gate"):
        m = mp[mp.split == sp].sort_values("row")
        txt_head[torch.tensor(m.file_index.values)] = S[sp]["text"]
        aud_head[torch.tensor(m.file_index.values)] = S[sp]["audio"]
    G = {"clf": (txt_head[rows["clf"]], aud_head[rows["clf"]]), "gate": (txt_head[rows["gate"]], aud_head[rows["gate"]]),
         "dev": (S["dev"]["text"], S["dev"]["audio"]), "test": (S["test"]["text"], S["test"]["audio"])}

    def wf(logits, yy, alpha=0.0):
        pred = (torch.log_softmax(logits, 1) + alpha * lp).argmax(1).numpy()
        return f1_score(yy.numpy(), pred, average="weighted", zero_division=0)

    # ---- stage 2 selection with alpha applied inside the selection
    grid, Z2, cfg, Z2c, cfgc = [], {}, {}, {}, {}
    for kind in ("text", "audio"):
        best = best_c = None
        for mode in ("prev", "both"):
            for k in (1, 2, 3, 5):
                C = {s: m29.ctx_features(Z1[kind][s], dl[s], ut[s], k, mode) for s in ("clf", "gate", "dev", "test")}
                for wd in (0.001, 0.01, 0.1):
                    W, b = b22.fit(C["clf"], y["clf"], w_clf, wd)
                    lg = {s: C[s] @ W + b for s in ("gate", "dev", "test")}
                    sc = {a: wf(lg["dev"], y["dev"], a) for a in ALPHAS}
                    a_best = max(sc, key=sc.get)
                    rec = {"modality": kind, "mode": mode, "k": k, "wd": wd, "alpha": a_best, "dev_wF1": sc[a_best],
                           "test_wF1_at_dev_alpha": wf(lg["test"], y["test"], a_best), "logits": lg}
                    grid.append({kk: vv for kk, vv in rec.items() if kk != "logits"})
                    if best is None or rec["dev_wF1"] > best["dev_wF1"]:
                        best = rec
                    if mode == "prev" and (best_c is None or rec["dev_wF1"] > best_c["dev_wF1"]):
                        best_c = rec
        Z2[kind], Z2c[kind] = best["logits"], best_c["logits"]
        cfg[kind] = {k: v for k, v in best.items() if k != "logits"}
        cfgc[kind] = {k: v for k, v in best_c.items() if k != "logits"}
        s1 = {a: wf(Z1[kind]["dev"], y["dev"], a) for a in ALPHAS}
        a1 = max(s1, key=s1.get)
        print(f"{kind}: no context dev {s1[a1]:.4f} (alpha {a1}) test {wf(Z1[kind]['test'], y['test'], a1):.4f} | "
              f"context {best['mode']} k={best['k']} wd={best['wd']} dev {best['dev_wF1']:.4f} (alpha {best['alpha']}) "
              f"test {best['test_wF1_at_dev_alpha']:.4f} | causal k={best_c['k']} dev {best_c['dev_wF1']:.4f} "
              f"test {best_c['test_wF1_at_dev_alpha']:.4f}", flush=True)
    pd.DataFrame(grid).to_csv(RESULTS / "context_grid_v2.csv", index=False)

    # ---- fusion stages: train gates once per stage, then apply alphas to the fused probabilities
    def run_stage(name, ZT, ZA):
        P_ = {s: (torch.softmax(ZT[s], 1), torch.softmax(ZA[s], 1)) for s in ("gate", "dev", "test")}
        fused = {}
        for seed in SEEDS:
            set_determinism(seed)
            gate = train_gate(G["gate"][0], G["gate"][1], *P_["gate"], y["gate"],
                              (G["dev"][0], G["dev"][1], *P_["dev"], y["dev"]), w_gate, seed)
            for split in ("dev", "test"):
                e_t, e_a = G[split]
                with torch.no_grad():
                    fused[(seed, split)] = {"average": fuse_average(*P_[split])[0], "confidence": fuse_confidence(*P_[split])[0],
                                            "gate": fuse_gate(gate, e_t, e_a, *P_[split])[0],
                                            "text_only": P_[split][0], "audio_only": P_[split][1]}
        # alpha chosen on dev for the gate rule (mean over seeds)
        def score(alpha, split, rule):
            return np.mean([f1_score(y[split].numpy(), (fused[(s, split)][rule].log() + alpha * lp).argmax(1).numpy(),
                                     average="weighted", zero_division=0) for s in SEEDS])
        dev_scores = {a: score(a, "dev", "gate") for a in ALPHAS}
        a_best = max(dev_scores, key=dev_scores.get)
        out = []
        for tag, alpha in (("alpha=0", 0.0), (f"dev alpha={a_best}", a_best)):
            for rule in ("text_only", "audio_only", "average", "confidence", "gate"):
                ms = []
                for s in SEEDS:
                    p = fused[(s, "test")][rule]
                    p = torch.softmax(p.log() + alpha * lp, 1)
                    ms.append(metrics(y["test"], p))
                out.append({"stage": name, "calibration": tag, "rule": rule,
                            **{k: float(np.mean([m[k] for m in ms])) for k in ("weighted_f1", "macro_f1", "accuracy")},
                            "dev_gate_wF1_at_alpha": dev_scores[alpha]})
        return pd.DataFrame(out), a_best

    out = []
    set_determinism(0)
    th = train_head(G["clf"][0], y["clf"], G["dev"][0], y["dev"], w_clf, 0)
    with torch.no_grad():
        Z0 = {s: th(G[s][0]) for s in ("gate", "dev", "test")}
    stages = {}
    for name, (ZT, ZA) in {"S0 control (MiniLM text + audio, no context)": (Z0, Z1["audio"]),
                           "S1 RoBERTa text + audio, no context": (Z1["text"], Z1["audio"]),
                           "S2 + context, bidirectional": (Z2["text"], Z2["audio"]),
                           "S2c + context, previous only (causal)": (Z2c["text"], Z2c["audio"])}.items():
        r, a = run_stage(name, ZT, ZA)
        out.append(r)
        stages[name] = a
        g = r[(r.rule == "gate")]
        print(f"{name}: gate test wF1 alpha0 {g.iloc[0].weighted_f1:.4f} | dev alpha {a}: {g.iloc[1].weighted_f1:.4f} "
              f"(macro {g.iloc[1].macro_f1:.4f})  [{time.time() - t0:.0f}s]", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(RESULTS / "f1_push_v2.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 4)
    print("\nTEST (mean over 3 seeds):")
    print(res[res.calibration != "alpha=0"][["stage", "calibration", "rule", "weighted_f1", "macro_f1", "accuracy"]].to_string(index=False))
    (RESULTS / "f1_push_v2_summary.json").write_text(json.dumps({"stage2": cfg, "stage2_causal": cfgc, "alpha_by_stage": stages}, indent=2, default=str))
    torch.save({"Z2": Z2, "Z2c": Z2c, "cfg": cfg, "cfgc": cfgc, "alpha": stages, "pi": pi}, DATA / "v2_pipeline_v2.pt")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
