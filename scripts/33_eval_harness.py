"""Step 33: a fixed model set and ONE evaluation harness (step 1 of the "better solution" plan).

Purpose: every later experiment (the corrected attribution signal, calibrated attribution, the conflict test, robustness
on new models) reuses the same models and the same measurements, so results are comparable and cheap to add.

Pre-registered before running:
  Model set (all share the dialogue-level split "v2" of step 29, except M0):
    M0   headline model, original split and original checkpoints (reference; heads also vary by seed)
    M0b  headline RECIPE on the v2 split: MiniLM text head + last-layer audio head (Adam, early stopping on dev)
    M1   M0b with the stronger audio head (all-layer WavLM, step 25 R4)
    M2   M0b with the stronger text head (RoBERTa-large, all layers, step 28)
    M3   stronger text AND stronger audio (S1 of step 30)
    M5   M3 + conversation context + class-prior correction (F1 only; the swap test with context is not yet defined)
  Heads are trained once (seed 0) and are shared across seeds; 10 gates (seeds 0 to 9), always on the HEADLINE gate
  inputs (MiniLM text embedding, 1,024-d audio embedding), trained on gate2 as in 02_train.py. M0 uses its own
  per-seed checkpoints.
  Measurements for every model (test set, 2,610 utterances):
    - weighted-F1, macro-F1 and accuracy of text-only, audio-only, average, confidence and gate, at alpha = 0 and with
      a class-prior correction alpha in {0, 0.25, 0.5, 0.75, 1} chosen on dev (gate rule, mean over seeds);
    - primary swap test (full-pool, effects on the fused probability before any prior correction): for the four pairs of
      06: agreement, P(text mattered more), gain over the majority modality, Spearman, dialogue-cluster bootstrap
      (1000 draws, rng 0, 10 gates averaged inside each draw), and the share of the possible edge;
    - for the gate's own weight: the dev-fitted cutoff (per gate, frozen) and its test gain, and the per-gate spread;
    - the same gain under three other effect measures: whole-distribution change (TVD), class flips, and signed drop.
  No tuning on test. Nothing here changes the headline model.

Usage: python scripts/33_eval_harness.py     (about 15 minutes)
Outputs: results/harness_models.csv, results/harness_alt_effects.csv, results/harness_per_gate.csv
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, DATA, RESULTS, Head, LearnedGate, class_weights, fuse_average, fuse_confidence,  # noqa: E402
                    fuse_gate, load_dataset, metrics, set_determinism, train_gate, train_head)

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
iv04 = importlib.import_module("04_interventions")
an06 = importlib.import_module("06_analysis")
SEEDS = list(range(10))
ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]
GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)
MODELS = ["fusion_learned_gate", "fusion_confidence", "fusion_average"]


def frames(seed, orig, cls, pb):
    n = len(cls["fusion_learned_gate"])
    out = []
    for m in MODELS:
        p0, _ = orig[m]
        dpt, flt, tvt = iv04.effect(pb[m][0], p0, cls[m])
        dpa, fla, tva = iv04.effect(pb[m][1], p0, cls[m])
        out.append(pd.DataFrame({
            "seed": seed, "row": range(n), "model": m,
            "gate_w_t": orig["fusion_learned_gate"][1].squeeze(1).numpy(), "conf_w_t": orig["fusion_confidence"][1].squeeze(1).numpy(),
            "dp_text": dpt.numpy(), "dp_audio": dpa.numpy(), "tvd_text": tvt.numpy(), "tvd_audio": tva.numpy(),
            "flip_text": flt.numpy().astype(float), "flip_audio": fla.numpy().astype(float)}))
    return pd.concat(out, ignore_index=True)


def effect_arrays(log, model, signal, kind):
    d = log[log.model == model].sort_values(["seed", "row"])
    seeds = sorted(d.seed.unique())
    s = np.stack([2 * d[d.seed == k][signal].values - 1 for k in seeds])
    f = {"dp": lambda x: x.dp_text.abs() - x.dp_audio.abs(), "tvd": lambda x: x.tvd_text - x.tvd_audio,
         "flip": lambda x: x.flip_text - x.flip_audio, "signed": lambda x: x.dp_audio - x.dp_text}[kind]
    e = np.stack([f(d[d.seed == k]).values for k in seeds])
    return s, e


def stats(s, e):
    """agreement, P(text), gain over majority, spearman, gain over always-text; averaged over gates."""
    ag, pt, rho = [], [], []
    for sk, ek in zip(s, e):
        v = (sk != 0) & (ek != 0)
        ag.append((np.sign(sk[v]) == np.sign(ek[v])).mean())
        pt.append((ek[v] > 0).mean())
        rho.append(np.corrcoef(rankdata(sk), rankdata(ek))[0, 1])
    a, p = np.mean(ag), np.mean(pt)
    return np.array([a, p, a - max(p, 1 - p), np.mean(rho), a - p])


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    rows, ids, Wb = P["rows"], P["ids"], P["Wb"]
    sel = json.loads((RESULTS / "text_head_selected.json").read_text())["selected"]
    assert m29.LAYERSETS[sel["features"]] == list(range(25))
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

    # heads on the padded input spaces
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
    print(f"data and heads ready ({time.time() - t0:.0f}s)", flush=True)

    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(an06.N_BOOT)]

    def ci(fn_s, fn_e):
        b = np.array([stats(fn_s[:, i], fn_e[:, i]) for i in draws])
        return np.percentile(b, [2.5, 97.5], axis=0)

    def report(name, log, fused_by_seed, pi_, yy, per_gate_rows):
        """fused_by_seed[(seed, split)][rule] -> probabilities; builds the model-level row and the alt-effect rows."""
        lp_ = pi_.log()

        def f1_at(split, rule, alpha):
            return np.mean([f1_score(yy[split].numpy(), (fused_by_seed[(s, split)][rule].log() + alpha * lp_).argmax(1).numpy(),
                                     average="weighted", zero_division=0) for s in SEEDS])
        dev_sc = {a: f1_at("dev", "gate", a) for a in ALPHAS}
        a_best = max(dev_sc, key=dev_sc.get)
        row = {"model": name, "alpha_dev": a_best}
        for rule in ("text_only", "audio_only", "average", "confidence", "gate"):
            for tag, alpha in (("a0", 0.0), ("adev", a_best)):
                ms = [metrics(yy["test"], torch.softmax(fused_by_seed[(s, "test")][rule].log() + alpha * lp_, 1)) for s in SEEDS]
                for k in ("weighted_f1", "macro_f1", "accuracy"):
                    row[f"{rule}_{k}_{tag}"] = float(np.mean([m[k] for m in ms]))
        for model, signal, label in an06.PAIRS:
            s, e = effect_arrays(log, model, signal, "dp")
            pt, (lo, hi) = stats(s, e), ci(s, e)
            tag = {"Learned gate ← gate weight": "gateW", "Confidence fusion ← confidence": "confW",
                   "Learned gate ← confidence": "gateCONF", "Average fusion ← confidence": "avgCONF"}[label]
            row.update({f"{tag}_agreement": pt[0], f"{tag}_p_text": pt[1], f"{tag}_gain_majority": pt[2], f"{tag}_gain_lo": lo[2],
                        f"{tag}_gain_hi": hi[2], f"{tag}_spearman": pt[3], f"{tag}_gain_always_text": pt[4]})
            if tag == "gateW":
                maj = max(pt[1], 1 - pt[1])
                row["gateW_share_of_possible_edge"] = pt[2] / (1 - maj)
                per = [stats(s[k:k + 1], e[k:k + 1]) for k in range(len(SEEDS))]
                row["gateW_gain_sd_across_gates"] = float(np.std([p[2] for p in per], ddof=1))
                for k, p in zip(SEEDS, per):
                    per_gate_rows.append({"model": name, "gate_seed": k, "agreement": p[0], "p_text": p[1], "gain_majority": p[2]})
        w = log[log.model == "fusion_learned_gate"].gate_w_t.values
        row["mean_weight"], row["share_weights_extreme"] = float(w.mean()), float(np.mean((w < 0.1) | (w > 0.9)))
        alt = []
        for kind in ("tvd", "flip", "signed"):
            s, e = effect_arrays(log, "fusion_learned_gate", "gate_w_t", kind)
            pt, (lo, hi) = stats(s, e), ci(s, e)
            alt.append({"model": name, "effect": kind, "p_text": pt[1], "agreement": pt[0], "gain_majority": pt[2],
                        "gain_lo": lo[2], "gain_hi": hi[2], "spearman": pt[3]})
        return row, alt, row

    results, alts, per_gate = [], [], []

    # ---------------- M0: original headline checkpoints (10 seeds)
    def m0():
        ck_models = {s: iv04.load_models(s) for s in SEEDS}
        pi0 = torch.bincount(S["clf"]["y"], minlength=7).float()
        pi0 = pi0 / pi0.sum()
        ok_t = rs07.valid_masks(S["test"]["text"], S["clf"]["text"]); ok_a = rs07.valid_masks(S["test"]["audio"], S["clf"]["audio"])
        fr, fused = [], {}
        for seed in SEEDS:
            orig, p_t, p_a = iv04.run_all_models(S["test"]["text"], S["test"]["audio"], *ck_models[seed])
            o2, _, _, cls, pb, _, _ = rs08.exact_average(seed, S["test"]["text"], S["test"]["audio"], S["clf"]["text"], S["clf"]["audio"], ok_t, ok_a)
            fr.append(frames(seed, o2, cls, pb))
            for split in ("dev", "test"):
                od, pt_, pa_ = iv04.run_all_models(S[split]["text"], S[split]["audio"], *ck_models[seed])
                fused[(seed, split)] = {"gate": od["fusion_learned_gate"][0], "average": od["fusion_average"][0],
                                        "confidence": od["fusion_confidence"][0], "text_only": pt_, "audio_only": pa_}
        yy = {"dev": S["dev"]["y"], "test": S["test"]["y"]}
        log = pd.concat(fr, ignore_index=True)
        # dev cutoff
        old = pd.concat([frames(seed, *(lambda r: (r[0], r[3], r[4]))(rs08.exact_average(seed, S["dev"]["text"], S["dev"]["audio"], S["clf"]["text"], S["clf"]["audio"],
                                                                                          rs07.valid_masks(S["dev"]["text"], S["clf"]["text"]), rs07.valid_masks(S["dev"]["audio"], S["clf"]["audio"])))) for seed in SEEDS],
                        ignore_index=True)
        return log, old, fused, pi0, yy

    def finish(name, log, devlog, fused, pi_, yy):
        row, alt, _ = report(name, log, fused, pi_, yy, per_gate)
        s, e = effect_arrays(log, "fusion_learned_gate", "gate_w_t", "dp")
        cuts = []
        for seed in SEEDS:
            dd = devlog[(devlog.model == "fusion_learned_gate") & (devlog.seed == seed)].sort_values("row")
            ee = dd.dp_text.abs().values - dd.dp_audio.abs().values
            v = ee != 0
            ag = np.array([np.mean(np.sign(dd.gate_w_t.values[v] - t) == np.sign(ee[v])) for t in GRID])
            best = np.flatnonzero(ag == ag.max())
            cuts.append(float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]]))
        st = (s + 1) / 2 - np.array(cuts)[:, None]
        row["dev_cutoff"], row["gateW_gain_majority_at_cutoff"] = float(np.mean(cuts)), float(stats(st, e)[2])
        results.append(row)
        alts.extend(alt)
        pd.DataFrame(results).to_csv(RESULTS / "harness_models.csv", index=False)
        pd.DataFrame(alts).to_csv(RESULTS / "harness_alt_effects.csv", index=False)
        pd.DataFrame(per_gate).to_csv(RESULTS / "harness_per_gate.csv", index=False)
        print(f"{name}: text {row['text_only_weighted_f1_a0']:.3f} audio {row['audio_only_weighted_f1_a0']:.3f} gate {row['gate_weighted_f1_a0']:.3f} "
              f"(alpha {row['alpha_dev']}: {row['gate_weighted_f1_adev']:.3f}) | P(text) {row['gateW_p_text']:.3f} agreement {row['gateW_agreement']:.3f} "
              f"gain {row['gateW_gain_majority']:.3f} [{row['gateW_gain_lo']:.3f},{row['gateW_gain_hi']:.3f}] ({time.time() - t0:.0f}s)", flush=True)

    log0, dev0, fused0, pi0, yy0 = m0()
    finish("M0 headline (original)", log0, dev0, fused0, pi0, yy0)

    # ---------------- M0b to M3
    yy = {"dev": y["dev"], "test": y["test"]}
    for name, (th, ah) in specs.items():
        gates, fused = {}, {}
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
            for split in ("dev", "test"):
                with torch.no_grad():
                    pt_, pa_ = pr[split]
                    fused[(seed, split)] = {"gate": fuse_gate(Gp, ET[split], EA[split], pt_, pa_)[0], "average": fuse_average(pt_, pa_)[0],
                                            "confidence": fuse_confidence(pt_, pa_)[0], "text_only": pt_, "audio_only": pa_}

        def swaps(split):
            old = iv04.load_models
            iv04.load_models = lambda seed: gates[seed]
            try:
                fr = []
                for seed in SEEDS:
                    orig, _, _, cls, pb, _, _ = rs08.exact_average(seed, ET[split], EA[split], ET["clf"], EA["clf"], *masks[split])
                    fr.append(frames(seed, orig, cls, pb))
            finally:
                iv04.load_models = old
            return pd.concat(fr, ignore_index=True)
        finish(name, swaps("test"), swaps("dev"), fused, pi, yy)

    # ---------------- M5: F1 only, from step 30
    f = pd.read_csv(RESULTS / "f1_push_v2.csv")
    f = f[f.stage.str.startswith("S2 + context, bidirectional") & (f.rule == "gate")]
    m5 = {"model": "M5 + context + prior correction (F1 only)"}
    for _, r in f.iterrows():
        tag = "a0" if r.calibration == "alpha=0" else "adev"
        for k in ("weighted_f1", "macro_f1", "accuracy"):
            m5[f"gate_{k}_{tag}"] = r[k]
    results.append(m5)
    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "harness_models.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    cols = ["model", "text_only_weighted_f1_a0", "audio_only_weighted_f1_a0", "gate_weighted_f1_a0", "gate_weighted_f1_adev", "gate_macro_f1_adev",
            "gateW_p_text", "gateW_agreement", "gateW_gain_majority", "gateW_gain_lo", "gateW_gain_hi", "gateW_share_of_possible_edge",
            "gateW_gain_sd_across_gates", "dev_cutoff", "confW_gain_majority", "gateCONF_gain_majority"]
    print("\n", res[cols].to_string(index=False))
    print("\nAlternative effect measures (gate weight):")
    print(pd.DataFrame(alts).round(3).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
