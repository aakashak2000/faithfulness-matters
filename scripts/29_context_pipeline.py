"""Step 29: the F1 push, stage by stage: stronger text head (item 1), conversation context (item 2),
calibration and class priors (item 3). A VARIANT ("v2"); the headline model is unchanged.

DESIGN, fixed before running
  Splits ("v2" protocol). The 9,988 training utterances are re-split BY DIALOGUE: dialogues are shuffled with
  numpy rng 3000 and the first 80% go to clf2, the rest to gate2 (so a dialogue is never split between them and
  context never crosses splits). dev (1,108) and test (2,610) are the official ones. Context is always taken
  within one dialogue.
  Heads (stage 1), each trained on clf2 only, LBFGS to convergence, class-weighted CE + 0.5 * wd * ||W||^2:
    text  : RoBERTa-large, features from the configuration selected on dev in step 28
            (results/text_head_selected.json);
    audio : WavLM, all 25 layers, mean + std, wd 1 (the step 25 winner, R4).
  Context (stage 2), one per modality: input = [own logits; mean of the previous k logits; mean of the next k
    logits (mode "both" only); has_prev; has_next], k in {1, 2, 3, 5}, mode in {prev, both},
    wd in {0.001, 0.01, 0.1}; multinomial LR trained on clf2 using OUT-OF-FOLD stage-1 logits (5 folds grouped by
    dialogue); for gate2, dev and test the stage-1 logits come from the head trained on all of clf2.
    Chosen per modality on dev weighted-F1. "both" uses following utterances, like bcLSTM (offline); the best
    "prev"-only (causal) configuration is reported next to it.
  Calibration (item 3), in this order: (a) one temperature per head, T in {0.5, 0.75, 1, 1.5, 2, 3}, chosen to
    minimise dev NLL, applied before the gate is trained; (b) a class-prior correction on the fused probabilities,
    p ~ p * pi^alpha with pi = clf2 class frequencies, alpha in {-1, -0.5, 0, 0.5, 1}, chosen on dev weighted-F1.
  Gate: exactly as in 02_train.py (MLP on the HEADLINE inputs: MiniLM text embedding and the 1,024-d audio embedding;
    trained on gate2 with the heads' out-of-sample probabilities; early stopping on dev loss), seeds 0 to 2.
  Fusion rules reported: average, confidence, learned gate.
  Stages reported (test weighted-F1, macro-F1, accuracy; mean over 3 seeds):
    S0  control: MiniLM text head (Adam, as in 02) + the stage-1 audio head, no context, new split
    S1  + RoBERTa text head
    S2  + context on both heads
    S3  + temperature scaling + prior correction
  Test is scored once per stage and never used to choose anything. Nothing here is tuned on test.
  Sanity check: S0 should land within about 1 point of the step 25 R4 result (gate 54.8).

Usage: python scripts/29_context_pipeline.py     (about 20 to 30 minutes)
Outputs: results/f1_push_stages.csv, results/context_grid.csv, results/f1_push_summary.json,
         data/v2_pipeline.pt (local; everything needed to rerun the swap tests later)
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, RESULTS, Head, class_weights, fuse_average, fuse_confidence, fuse_gate,  # noqa: E402
                    load_dataset, metrics, set_determinism, train_gate, train_head)

b22 = importlib.import_module("22_better_audio")
SEEDS = [0, 1, 2]
AUDIO_LAYERS, AUDIO_POOL, AUDIO_WD = list(range(25)), "mean+std", 1.0
KS, MODES, WDS2 = [1, 2, 3, 5], ["prev", "both"], [0.001, 0.01, 0.1]
TEMPS = [0.5, 0.75, 1, 1.5, 2, 3]
ALPHAS = [-1, -0.5, 0, 0.5, 1]
NINE = [8, 10, 12, 14, 16, 18, 20, 22, 24]
LAYERSETS = {"layer8": [8], "layer12": [12], "layer16": [16], "layer20": [20], "layer24": [24],
             "concat4": [12, 16, 20, 24], "concat9": NINE, "all25": list(range(25))}


def softmax_T(z, T):
    return torch.softmax(z / T, 1)


def ctx_features(Z, dlg, utt, k, mode):
    """[own, mean of previous k, mean of next k (both only), has_prev, has_next] within each dialogue."""
    n, c = Z.shape
    prev, nxt = torch.zeros(n, c), torch.zeros(n, c)
    hp, hn = torch.zeros(n, 1), torch.zeros(n, 1)
    for d in np.unique(dlg):
        idx = np.flatnonzero(dlg == d)
        idx = idx[np.argsort(utt[idx])]
        for pos, i in enumerate(idx):
            lo = idx[max(0, pos - k):pos]
            hi = idx[pos + 1:pos + 1 + k]
            if len(lo):
                prev[i] = Z[lo].mean(0); hp[i] = 1
            if len(hi):
                nxt[i] = Z[hi].mean(0); hn[i] = 1
    parts = [Z, prev, hp] if mode == "prev" else [Z, prev, nxt, hp, hn]
    return torch.cat(parts, 1)


def wf1(logits, y):
    return f1_score(y.numpy(), logits.argmax(1).numpy(), average="weighted", zero_division=0)


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    sel = json.loads((RESULTS / "text_head_selected.json").read_text())["selected"]
    text_layers = LAYERSETS[sel["features"]]
    text_wd = float(sel["wd"])
    print(f"text head config: {sel['features']} wd={text_wd}", flush=True)

    # ---- row bookkeeping in "file index" space (train rows 0..9987), dev and test as given
    ids = {sp: torch.load(DATA / f"wavlm_layers_{sp}.pt", map_location="cpu", weights_only=False)["ids"]
           for sp in ("train", "dev", "test")}
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    emo = {e: i for i, e in enumerate(D["emotions"])}
    lab = {(a, b): emo[e] for a, b, e in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Emotion)}
    ytr = torch.tensor([lab[tuple(i)] for i in ids["train"]])
    dlg = {sp: np.array([i[0] for i in ids[sp]]) for sp in ids}
    utt = {sp: np.array([i[1] for i in ids[sp]]) for sp in ids}
    rng = np.random.default_rng(3000)
    dialogues = np.unique(dlg["train"])
    perm = rng.permutation(dialogues)
    clf_d = set(perm[: int(0.8 * len(perm))].tolist())
    is_clf = np.array([d in clf_d for d in dlg["train"]])
    rows = {"clf": np.flatnonzero(is_clf), "gate": np.flatnonzero(~is_clf)}
    y = {"clf": ytr[rows["clf"]], "gate": ytr[rows["gate"]], "dev": S["dev"]["y"], "test": S["test"]["y"]}
    print("v2 split: clf2", len(rows["clf"]), "gate2", len(rows["gate"]),
          "| class share clf2", (torch.bincount(y["clf"], minlength=7).float() / len(y["clf"])).round(decimals=3).tolist(), flush=True)
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    D_ = {"clf": dlg["train"][rows["clf"]], "gate": dlg["train"][rows["gate"]], "dev": dlg["dev"], "test": dlg["test"]}
    U_ = {"clf": utt["train"][rows["clf"]], "gate": utt["train"][rows["gate"]], "dev": utt["dev"], "test": utt["test"]}

    # ---- headline-style gate inputs for every utterance (MiniLM text embedding, 1,024-d audio embedding)
    txt_head = torch.zeros(len(ids["train"]), 384)
    aud_head = torch.zeros(len(ids["train"]), 1024)
    for sp in ("clf", "gate"):
        m = mp[mp.split == sp].sort_values("row")
        txt_head[torch.tensor(m.file_index.values)] = S[sp]["text"]
        aud_head[torch.tensor(m.file_index.values)] = S[sp]["audio"]
    G = {"clf": (txt_head[rows["clf"]], aud_head[rows["clf"]]), "gate": (txt_head[rows["gate"]], aud_head[rows["gate"]]),
         "dev": (S["dev"]["text"], S["dev"]["audio"]), "test": (S["test"]["text"], S["test"]["audio"])}

    # ---- standardised features for the two stage-1 heads
    def feats(kind):
        if kind == "text":
            tr = torch.load(DATA / "roberta_layers_train.pt", map_location="cpu", weights_only=False)["mean"]
            dv = torch.load(DATA / "roberta_layers_dev.pt", map_location="cpu", weights_only=False)["mean"]
            te = torch.load(DATA / "roberta_layers_test.pt", map_location="cpu", weights_only=False)["mean"]
            build = lambda m, idx=None: torch.cat([(m if idx is None else m[idx])[:, l].float() for l in text_layers], 1)
            raw = {"clf": build(tr, rows["clf"]), "gate": build(tr, rows["gate"]), "dev": build(dv), "test": build(te)}
        else:
            L = {sp: torch.load(DATA / f"wavlm_layers_{sp}.pt", map_location="cpu", weights_only=False) for sp in ids}
            bld = lambda sp, idx=None: b22.build(L[sp]["mean"] if idx is None else L[sp]["mean"][idx],
                                                 L[sp]["std"] if idx is None else L[sp]["std"][idx], AUDIO_LAYERS, AUDIO_POOL)
            raw = {"clf": bld("train", rows["clf"]), "gate": bld("train", rows["gate"]), "dev": bld("dev"), "test": bld("test")}
        mu, sd = raw["clf"].mean(0), raw["clf"].std(0) + 1e-6
        return {k: ((v - mu) / sd).contiguous() for k, v in raw.items()}

    Z1, Wb = {}, {}   # stage-1 logits per modality/split, and head weights
    for kind, wd in (("text", text_wd), ("audio", AUDIO_WD)):
        X = feats(kind)
        # out-of-fold logits on clf2 (5 folds grouped by dialogue), full-head logits elsewhere
        oof = torch.zeros(len(y["clf"]), 7)
        for tr_i, va_i in GroupKFold(5).split(X["clf"], groups=D_["clf"]):
            W, b = b22.fit(X["clf"][tr_i], y["clf"][tr_i], class_weights(y["clf"][tr_i]), wd)
            oof[va_i] = X["clf"][va_i] @ W + b
        W, b = b22.fit(X["clf"], y["clf"], w_clf, wd)
        Z1[kind] = {"clf": oof, **{s: X[s] @ W + b for s in ("gate", "dev", "test")}}
        Wb[kind] = (W, b)
        full_clf = X["clf"] @ W + b
        print(f"stage-1 {kind}: OOF clf2 wF1 {wf1(oof, y['clf']):.4f} | dev {wf1(Z1[kind]['dev'], y['dev']):.4f} "
              f"test {wf1(Z1[kind]['test'], y['test']):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        del X

    # ---- stage 2: conversation context, chosen per modality on dev
    grid_rows, Z2, cfg2, causal = [], {}, {}, {}
    for kind in ("text", "audio"):
        best, best_causal = None, None
        for mode in MODES:
            for k in KS:
                C = {s: ctx_features(Z1[kind][s], D_[s], U_[s], k, mode) for s in ("clf", "gate", "dev", "test")}
                for wd in WDS2:
                    W, b = b22.fit(C["clf"], y["clf"], w_clf, wd)
                    dv, dm = b22.score(C["dev"], y["dev"], W, b)
                    tw, tm = b22.score(C["test"], y["test"], W, b)
                    rec = {"modality": kind, "mode": mode, "k": k, "wd": wd, "dev_wF1": dv, "dev_macro": dm,
                           "test_wF1": tw, "test_macro": tm}
                    grid_rows.append(rec)
                    if best is None or dv > best["dev_wF1"]:
                        best = rec
                    if mode == "prev" and (best_causal is None or dv > best_causal["dev_wF1"]):
                        best_causal = rec
        cfg2[kind], causal[kind] = best, best_causal
        C = {s: ctx_features(Z1[kind][s], D_[s], U_[s], best["k"], best["mode"]) for s in ("clf", "gate", "dev", "test")}
        W, b = b22.fit(C["clf"], y["clf"], w_clf, best["wd"])
        Z2[kind] = {s: C[s] @ W + b for s in ("gate", "dev", "test")}
        print(f"stage-2 {kind}: best {best['mode']} k={best['k']} wd={best['wd']}: dev {best['dev_wF1']:.4f} "
              f"test {best['test_wF1']:.4f}; best causal (prev only): dev {best_causal['dev_wF1']:.4f} "
              f"test {best_causal['test_wF1']:.4f}", flush=True)
    pd.DataFrame(grid_rows).to_csv(RESULTS / "context_grid.csv", index=False)

    # ---- helpers for fusion
    def fuse_all(pt, pa, split, seed, gate):
        e_t, e_a = G[split]
        with torch.no_grad():
            outs = {"average": fuse_average(pt, pa)[0], "confidence": fuse_confidence(pt, pa)[0],
                    "gate": fuse_gate(gate, e_t, e_a, pt, pa)[0]}
        return outs

    def run_stage(name, ZT, ZA, T=(1.0, 1.0), alpha=0.0, pi=None):
        """Train a gate per seed on gate2 and score test. ZT, ZA: dicts of logits per split."""
        res = []
        P = {s: (softmax_T(ZT[s], T[0]), softmax_T(ZA[s], T[1])) for s in ("gate", "dev", "test")}
        for seed in SEEDS:
            set_determinism(seed)
            gate = train_gate(G["gate"][0], G["gate"][1], *P["gate"], y["gate"],
                              (G["dev"][0], G["dev"][1], *P["dev"], y["dev"]), w_gate, seed)
            for split in ("dev", "test"):
                outs = fuse_all(*P[split], split, seed, gate)
                for rule, p in outs.items():
                    if pi is not None and alpha != 0:
                        p = p * (pi ** alpha)
                        p = p / p.sum(1, keepdim=True)
                    res.append({"stage": name, "seed": seed, "split": split, "rule": rule, **metrics(y[split], p)})
                if split == "test":
                    for nm, pp in (("text_only", P[split][0]), ("audio_only", P[split][1])):
                        res.append({"stage": name, "seed": seed, "split": split, "rule": nm, **metrics(y[split], pp)})
        return pd.DataFrame(res)

    out = []
    # S0: control (MiniLM text head, Adam as in 02, no context) + stage-1 audio head
    set_determinism(0)
    th = train_head(G["clf"][0], y["clf"], G["dev"][0], y["dev"], w_clf, 0)
    with torch.no_grad():
        Z0 = {s: th(G[s][0]) for s in ("gate", "dev", "test")}
    out.append(run_stage("S0 control: MiniLM text + audio head, no context", Z0, Z1["audio"]))
    out.append(run_stage("S1 + RoBERTa text head", Z1["text"], Z1["audio"]))
    out.append(run_stage("S2 + conversation context", Z2["text"], Z2["audio"]))
    print(f"stages S0 to S2 done ({time.time() - t0:.0f}s)", flush=True)

    # S3: temperature scaling on dev (NLL), then the prior correction on dev wF1
    Ts = []
    for kind in ("text", "audio"):
        nll = {T: F.cross_entropy(Z2[kind]["dev"] / T, y["dev"]).item() for T in TEMPS}
        Ts.append(min(nll, key=nll.get))
    pi = torch.bincount(y["clf"], minlength=7).float()
    pi = pi / pi.sum()
    best_alpha, best_dev = 0, -1
    for a in ALPHAS:
        r = run_stage("tmp", Z2["text"], Z2["audio"], T=tuple(Ts), alpha=a, pi=pi)
        d = r[(r.split == "dev") & (r.rule == "gate")].weighted_f1.mean()
        print(f"  alpha {a:+.1f}: dev gate wF1 {d:.4f}", flush=True)
        if d > best_dev:
            best_alpha, best_dev = a, d
    print(f"temperatures {Ts}, prior alpha {best_alpha} (dev {best_dev:.4f})", flush=True)
    out.append(run_stage("S3 + temperature scaling + prior correction", Z2["text"], Z2["audio"], T=tuple(Ts), alpha=best_alpha, pi=pi))

    res = pd.concat(out, ignore_index=True)
    res.to_csv(RESULTS / "f1_push_stages.csv", index=False)
    tab = res[res.split == "test"].groupby(["stage", "rule"])[["weighted_f1", "macro_f1", "accuracy"]].mean().round(4)
    pd.set_option("display.width", 200)
    print("\nTEST, mean over 3 seeds:")
    print(tab.to_string())
    summary = {"text_head": {"features": sel["features"], "wd": text_wd}, "stage2": cfg2, "stage2_causal": causal,
               "temperatures": Ts, "alpha": best_alpha, "split_sizes": {"clf2": len(rows["clf"]), "gate2": len(rows["gate"])}}
    (RESULTS / "f1_push_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save({"rows": rows, "Wb": Wb, "Z1": Z1, "Z2": Z2, "cfg2": cfg2, "Ts": Ts, "alpha": best_alpha, "pi": pi,
                "ids": ids, "dlg": D_, "utt": U_}, DATA / "v2_pipeline.pt")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
