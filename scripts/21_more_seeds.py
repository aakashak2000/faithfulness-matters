"""Step 21: ten trained models instead of three (action item 5).

Why: every interval so far is "conditional on the 3 trained models": the bootstrap resamples
dialogues, but the heads and the gate come from only 3 training seeds. This script trains 7 more
seeds (3 to 9), reruns the same interventions for all 10, and reports how much the headline
numbers move from seed to seed and whether the conclusions change.

Definitions, fixed before running:
  - Seeds 0 to 9, fixed now. Seeds 0 to 2 are the existing checkpoints; seeds 3 to 9 are trained
    exactly as in 02_train.py (clf-split heads, gate on the gate split, class-weighted CE, early
    stopping on dev, CPU, set_determinism(seed)). No seed is dropped or replaced after seeing results.
  - Check: retraining seed 0 here must reproduce checkpoints/seed0.pt (heads and gate) to 1e-6, and
    the intervention code on seeds 0 to 2 must reproduce results/interventions_log.csv (mean, blank)
    and results/resampling_full_log.csv (dp, weights) to 1e-5.
  - Interventions: mean, blank and full-pool resampling (primary), exactly as 04 and 08. Effect
    e = |dp_text| - |dp_audio| (primary); tvd difference as the alternative of 14.
  - Reported for each pair of 06 and each method:
      (a) 10-seed point estimate (metrics averaged over seeds, as in 06) with the same
          dialogue-cluster bootstrap (1000 draws, rng 0, 10 seeds averaged inside each draw);
      (b) the 10 per-seed values: mean, standard deviation, min, max, and a seed-level 95%
          t-interval of the mean (this is the part the 3-seed intervals could not show);
      (c) the old 3-seed point estimate next to it.
  - Also: gain of the gate under the tvd effect (primary method), the paired difference
    confidence minus gate (as 12), the dev-fitted cutoff result (as 19: cutoff fitted per seed on
    dev, frozen, applied to test), and baseline F1 (mean +- sd over 10 seeds).
  - Reading rule: a headline claim is "stable across seeds" if its 10-seed bootstrap interval leads to
    the same reading as the 3-seed one (excludes 0 or not, same sign) and every per-seed value of
    the gate gain lies within +-0.03 of the 10-seed mean; otherwise it is reported as seed-dependent.

Usage: python scripts/21_more_seeds.py   (~10 min on CPU)
Outputs: results/more_seeds_baselines.csv, more_seeds_faithfulness.csv, more_seeds_per_seed.csv,
         more_seeds_checks.csv; checkpoints/seed3..9.pt (local only)
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, DATA, RESULTS, class_weights, fuse_average, fuse_confidence,  # noqa: E402
                    fuse_gate, load_dataset, metrics, set_determinism, train_gate, train_head)

iv04 = importlib.import_module("04_interventions")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
an06 = importlib.import_module("06_analysis")
m19 = importlib.import_module("19_dev_calibration")
ALL = list(range(10))
OLD = [0, 1, 2]
MODELS = rs07.MODELS
METHODS = ["resample_full", "mean", "blank"]


def train_seed(D, seed, save):
    S = D["splits"]
    clf, gate_s, dev = S["clf"], S["gate"], S["dev"]
    set_determinism(seed)
    w_clf, w_gate = class_weights(clf["y"]), class_weights(gate_s["y"])
    th = train_head(clf["text"], clf["y"], dev["text"], dev["y"], w_clf, seed)
    ah = train_head(clf["audio"], clf["y"], dev["audio"], dev["y"], w_clf, seed)
    with torch.no_grad():
        pr = {s: (torch.softmax(th(S[s]["text"]), 1), torch.softmax(ah(S[s]["audio"]), 1)) for s in ["gate", "dev", "test"]}
    gate = train_gate(gate_s["text"], gate_s["audio"], *pr["gate"], gate_s["y"],
                      (dev["text"], dev["audio"], *pr["dev"], dev["y"]), w_gate, seed)
    ck = {"text_head": th.state_dict(), "audio_head": ah.state_dict(), "gate": gate.state_dict(), "seed": seed}
    if save:
        torch.save(ck, CKPT / f"seed{seed}.pt")
    return ck


def baselines(D, seed):
    test = D["splits"]["test"]
    th, ah, gate = iv04.load_models(seed)
    with torch.no_grad():
        p_t, p_a = torch.softmax(th(test["text"]), 1), torch.softmax(ah(test["audio"]), 1)
        outs = {"text_only": p_t, "audio_only": p_a, "fusion_average": fuse_average(p_t, p_a)[0],
                "fusion_confidence": fuse_confidence(p_t, p_a)[0],
                "fusion_learned_gate": fuse_gate(gate, test["text"], test["audio"], p_t, p_a)[0]}
    return [{"model": k, "seed": seed, **metrics(test["y"], p)} for k, p in outs.items()]


def frames(seed, tag, orig, cls, no_text, no_audio):
    n = len(cls["fusion_learned_gate"])
    out = []
    for m in MODELS:
        p0, _ = orig[m]
        dpt, _, tvt = iv04.effect(no_text[m], p0, cls[m])
        dpa, _, tva = iv04.effect(no_audio[m], p0, cls[m])
        out.append(pd.DataFrame({
            "seed": seed, "row": range(n), "model": m, "method": tag,
            "gate_w_t": orig["fusion_learned_gate"][1].squeeze(1).numpy(),
            "conf_w_t": orig["fusion_confidence"][1].squeeze(1).numpy(),
            "dp_text": dpt.numpy(), "dp_audio": dpa.numpy(),
            "tvd_text": tvt.numpy(), "tvd_audio": tva.numpy()}))
    return pd.concat(out, ignore_index=True)


def build_logs(D, seeds):
    S = D["splits"]
    test, clf = S["test"], S["clf"]
    e_t, e_a = test["text"], test["audio"]
    n = len(test["y"])
    iv = torch.load(DATA / "intervention_embeddings.pt", map_location="cpu", weights_only=False)["test"]
    rep = {"mean": (clf["text"].mean(0, keepdim=True).expand(n, -1), clf["audio"].mean(0, keepdim=True).expand(n, -1)),
           "blank": (iv["text_empty"][None].expand(n, -1), iv["audio_silence"])}
    ok_t, ok_a = rs07.valid_masks(e_t, clf["text"]), rs07.valid_masks(e_a, clf["audio"])
    fr = []
    for seed in seeds:
        heads = iv04.load_models(seed)
        orig, _, _ = iv04.run_all_models(e_t, e_a, *heads)
        cls = {m: orig[m][0].argmax(1) for m in MODELS}
        for tag, (rt, ra) in rep.items():
            nt, _, _ = iv04.run_all_models(rt, e_a, *heads)
            na, _, _ = iv04.run_all_models(e_t, ra, *heads)
            fr.append(frames(seed, tag, orig, cls, {m: nt[m][0] for m in MODELS}, {m: na[m][0] for m in MODELS}))
        o2, _, _, cls2, pb, _, _ = rs08.exact_average(seed, e_t, e_a, clf["text"], clf["audio"], ok_t, ok_a)
        fr.append(frames(seed, "resample_full", o2, cls2, {m: pb[m][0] for m in MODELS}, {m: pb[m][1] for m in MODELS}))
        print(f"interventions seed {seed} done", flush=True)
    return pd.concat(fr, ignore_index=True)


def arrays(log, model, signal, method, kind):
    d = log[(log.model == model) & (log.method == method)].sort_values(["seed", "row"])
    seeds = sorted(d.seed.unique())
    s = np.stack([2 * d[d.seed == k][signal].values - 1 for k in seeds])
    if kind == "dp":
        e = np.stack([d[d.seed == k].dp_text.abs().values - d[d.seed == k].dp_audio.abs().values for k in seeds])
    else:
        e = np.stack([d[d.seed == k].tvd_text.values - d[d.seed == k].tvd_audio.values for k in seeds])
    return s, e


def main():
    D = load_dataset()
    checks = []

    # 1. retraining seed 0 reproduces the saved checkpoint
    ck = train_seed(D, 0, save=False)
    old = torch.load(CKPT / "seed0.pt", map_location="cpu", weights_only=False)
    worst = max(float((ck[k][n] - old[k][n]).abs().max()) for k in ("text_head", "audio_head", "gate") for n in ck[k])
    assert worst < 1e-6, f"retraining seed 0 differs from checkpoint by {worst}"
    checks.append({"check": "retrain seed 0 vs checkpoints/seed0.pt, max abs weight diff", "value": worst})
    print(f"check passed: seed 0 retrains identically (max diff {worst:.2e})", flush=True)

    # 2. train seeds 3..9
    for seed in ALL[3:]:
        train_seed(D, seed, save=True)
        print(f"trained seed {seed}", flush=True)

    # 3. baselines
    base = pd.DataFrame([r for s in ALL for r in baselines(D, s)])
    base.to_csv(RESULTS / "more_seeds_baselines.csv", index=False)
    summ = base.groupby("model")[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    print(summ.round(4).to_string(), flush=True)

    # 4. interventions for all 10 seeds, with the reproduction check on seeds 0..2
    log = build_logs(D, ALL)
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    full_old = pd.read_csv(RESULTS / "resampling_full_log.csv")
    for tag, src in (("mean", ref), ("blank", ref), ("resample_full", full_old)):
        for m in MODELS:
            a = log[(log.model == m) & (log.method == tag) & log.seed.isin(OLD)].sort_values(["seed", "row"])
            b = src[(src.model == m) & (src.method == tag)].sort_values(["seed", "row"])
            for c in ["dp_text", "dp_audio", "gate_w_t", "conf_w_t"]:
                d = float(np.abs(a[c].values - b[c].values).max())
                assert d < 1e-5, f"reproduction failed {tag} {m} {c}: {d}"
    checks.append({"check": "interventions on seeds 0..2 reproduce the committed logs, max abs diff < 1e-5", "value": 1e-5})
    print("check passed: interventions on seeds 0..2 reproduce the committed logs", flush=True)

    # 5. faithfulness: 10-seed bootstrap + per-seed spread
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]
    old3 = pd.concat([pd.read_csv(RESULTS / "bootstrap_ci.csv"), pd.read_csv(RESULTS / "resampling_full_faithfulness.csv")])
    names = ["agreement", "always_text_baseline", "gain", "spearman"]
    rows, per_seed = [], []
    for model, signal, label in an06.PAIRS:
        for method in METHODS:
            s, e = arrays(log, model, signal, method, "dp")
            pt = np.array(an06.stats(s, e))
            b = np.array([an06.stats(s[:, i], e[:, i]) for i in draws])
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            ps = np.array([an06.stats(s[k:k + 1], e[k:k + 1]) for k in range(len(ALL))])
            for k, nme in enumerate(names):
                tci = sps.t.interval(0.95, len(ALL) - 1, loc=ps[:, k].mean(), scale=ps[:, k].std(ddof=1) / np.sqrt(len(ALL)))
                o = old3[(old3.pair == label) & (old3.method == method) & (old3.metric == nme)]
                rows.append({"pair": label, "method": method, "metric": nme, "estimate_10": pt[k],
                             "ci_low": lo[k], "ci_high": hi[k], "per_seed_mean": ps[:, k].mean(),
                             "per_seed_sd": ps[:, k].std(ddof=1), "per_seed_min": ps[:, k].min(),
                             "per_seed_max": ps[:, k].max(), "seed_t_ci_low": tci[0], "seed_t_ci_high": tci[1],
                             "estimate_3": float(o.estimate.iloc[0]) if len(o) else np.nan})
            for sd_, vals in zip(ALL, ps):
                per_seed.append({"pair": label, "method": method, "seed": sd_, **dict(zip(names, vals))})
        print(f"done {label}", flush=True)
    fa = pd.DataFrame(rows)
    fa.to_csv(RESULTS / "more_seeds_faithfulness.csv", index=False)
    pd.DataFrame(per_seed).to_csv(RESULTS / "more_seeds_per_seed.csv", index=False)

    # 6. extras on the primary method: tvd gain for the gate, paired confidence minus gate, dev-fitted cutoff
    extra = []
    sg, eg = arrays(log, "fusion_learned_gate", "gate_w_t", "resample_full", "tvd")
    pt = an06.stats(sg, eg)[2]
    b = np.array([an06.stats(sg[:, i], eg[:, i])[2] for i in draws])
    extra.append({"quantity": "gate gain, tvd effect, resample_full", "estimate": pt,
                  "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)})
    s_g, e = arrays(log, "fusion_learned_gate", "gate_w_t", "resample_full", "dp")
    s_c, _ = arrays(log, "fusion_learned_gate", "conf_w_t", "resample_full", "dp")
    pt = an06.stats(s_c, e)[2] - an06.stats(s_g, e)[2]
    b = np.array([an06.stats(s_c[:, i], e[:, i])[2] - an06.stats(s_g[:, i], e[:, i])[2] for i in draws])
    extra.append({"quantity": "paired gain, confidence minus gate, resample_full", "estimate": pt,
                  "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)})

    m19.SEEDS = ALL
    sil = torch.load(DATA / "intervention_embeddings_dev.pt", map_location="cpu", weights_only=False)["audio_silence"]
    dev = m19.dev_effects(D, sil)
    taus = np.array([m19.fit_tau(*(lambda d: (d.w.values, d.e.values))(
        dev[(dev.seed == k) & (dev.method == "resample_full")].sort_values("row")))[0] for k in ALL])[:, None]
    s_t = (s_g + 1) / 2 - taus
    pt = an06.stats(s_t, e)[2]
    b = np.array([an06.stats(s_t[:, i], e[:, i])[2] for i in draws])
    extra.append({"quantity": "gate gain with dev-fitted cutoff, resample_full", "estimate": pt,
                  "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)})
    per_seed_cut = [an06.stats(s_t[k:k + 1], e[k:k + 1])[2] for k in range(len(ALL))]
    extra.append({"quantity": "dev-fitted cutoffs (mean, min, max over seeds)", "estimate": float(taus.mean()),
                  "ci_low": float(taus.min()), "ci_high": float(taus.max())})
    ex = pd.DataFrame(extra)
    ex.to_csv(RESULTS / "more_seeds_extras.csv", index=False)
    pd.DataFrame(checks).to_csv(RESULTS / "more_seeds_checks.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 4)
    g = fa[(fa.metric == "gain") & fa.pair.isin(["Learned gate ← gate weight", "Confidence fusion ← confidence"])]
    print("\n== GAIN over always-text: 10 seeds vs old 3 seeds ==")
    print(g[["pair", "method", "estimate_3", "estimate_10", "ci_low", "ci_high", "per_seed_sd",
             "per_seed_min", "per_seed_max", "seed_t_ci_low", "seed_t_ci_high"]].to_string(index=False))
    sp = fa[(fa.metric == "spearman") & (fa.pair == "Learned gate ← gate weight")]
    print("\n== SPEARMAN, gate ==")
    print(sp[["method", "estimate_3", "estimate_10", "ci_low", "ci_high", "per_seed_sd"]].to_string(index=False))
    print("\n== extras ==")
    print(ex.to_string(index=False))
    print(f"\nper-seed gate gain with dev-fitted cutoff: {np.round(per_seed_cut, 4).tolist()}")
    gp = pd.DataFrame(per_seed)
    gg = gp[(gp.pair == "Learned gate ← gate weight") & (gp.method == "resample_full")]
    print(f"per-seed gate gain at 0.5, resample_full: {np.round(gg.gain.values, 4).tolist()}")


if __name__ == "__main__":
    main()
