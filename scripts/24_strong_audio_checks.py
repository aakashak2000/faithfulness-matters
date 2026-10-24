"""Step 24: checks on the strong-audio variant before anyone quotes its faithfulness numbers.

Why: in step 23 the always-text baseline fell to 0.364 (audio now mattered more in about 64% of
utterances) and the gate's agreement rose to 0.930. Two things must be checked first:
  (1) the trivial baseline is no longer "always text": the strongest trivial rule is the majority
      modality, so the gain must be reported over max(P(text), P(audio));
  (2) how much of the new agreement is structural, i.e. appears even when the models carry no label
      information (the shuffled-label control, as in step 18).

Pre-registered before running:
  - Same strong-audio features, heads, gates and checkpoints as step 23 (checkpoints/audio2/).
  - Reported for the strong variant, primary method (full-pool swap, |dp| effect), 3 seeds, dialogue
    cluster bootstrap (1000 draws, rng 0, same draws as 06):
      * P(text mattered more) and the majority baseline max(P, 1 - P);
      * agreement, gain over always-text (as before), and gain over the MAJORITY baseline (new);
      * calibration by gate weight bin;
      * dev-fitted cutoff (procedure of 19: grid 0.05 to 0.95, best dev agreement, per seed, frozen)
        and its test gain over the majority baseline.
  - Shuffled-label control for the strong features: permute clf, gate and dev labels (numpy rng
    1000 + seed), train the audio head (same LBFGS, wd 0.1), the text head (common.train_head) and the
    gate (common.train_gate), run the same swaps, report agreement and both gains.
  - Reading rule: if the shuffled-label gain over the majority baseline is within 0.10 of the real
    one, the agreement is mostly structural; if it is far below, the real agreement reflects what the
    models learned. Nothing here changes the headline model.

Usage: python scripts/24_strong_audio_checks.py
Outputs: results/strong_audio_checks.csv, results/strong_audio_calibration.csv,
         results/strong_audio_dev_cutoff.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, RESULTS, Head, LearnedGate, class_weights, fuse_gate, load_dataset,  # noqa: E402
                    set_determinism, train_gate, train_head)

iv04 = importlib.import_module("04_interventions")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
an06 = importlib.import_module("06_analysis")
b22 = importlib.import_module("22_better_audio")
s23 = importlib.import_module("23_better_audio_fusion")
SEEDS = [0, 1, 2]
SHUF = CKPT / "audio2_shuffled"
GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def load_shuf(seed):
    ck = torch.load(SHUF / f"seed{seed}.pt", map_location="cpu", weights_only=False)
    th, ah, gate = Head(384), Head(ck["audio_dim"]), LearnedGate(384, ck["audio_dim"])
    th.load_state_dict(ck["text_head"]); ah.load_state_dict(ck["audio_head"]); gate.load_state_dict(ck["gate"])
    for m in (th, ah, gate):
        m.eval()
    return th, ah, gate


def swap_log(e_t, e_a, pool_t, pool_a, load_fn, tag):
    ok_t, ok_a = rs07.valid_masks(e_t, pool_t), rs07.valid_masks(e_a, pool_a)
    old = iv04.load_models
    iv04.load_models = load_fn
    try:
        fr = []
        for seed in SEEDS:
            orig, _, _, cls, pb, _, _ = rs08.exact_average(seed, e_t, e_a, pool_t, pool_a, ok_t, ok_a)
            fr.append(s23.frames(seed, tag, orig, cls, pb))
    finally:
        iv04.load_models = old
    return pd.concat(fr, ignore_index=True)


def stats3(s, e):
    """agreement, P(text), gain over always-text, gain over majority, spearman; averaged over seeds."""
    ag, pt, rho = [], [], []
    for sk, ek in zip(s, e):
        v = (sk != 0) & (ek != 0)
        ag.append((np.sign(sk[v]) == np.sign(ek[v])).mean())
        pt.append((ek[v] > 0).mean())
        rho.append(np.corrcoef(rankdata(sk), rankdata(ek))[0, 1])
    a, p = np.mean(ag), np.mean(pt)
    return np.array([a, p, a - p, a - max(p, 1 - p), np.mean(rho)])


def train_shuffled(D, A, seed):
    S = D["splits"]
    rng = np.random.default_rng(1000 + seed)
    y = {k: S[k]["y"][torch.as_tensor(rng.permutation(len(S[k]["y"])))] for k in ("clf", "gate", "dev")}
    set_determinism(seed)
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    W, b = b22.fit(A["clf"], y["clf"], w_clf, s23.WD)
    ah = Head(A["clf"].shape[1])
    ah.lin.weight.data, ah.lin.bias.data = W.T.contiguous(), b.clone()
    ah.eval()
    th = train_head(S["clf"]["text"], y["clf"], S["dev"]["text"], y["dev"], w_clf, seed)
    with torch.no_grad():
        pr = {s: (torch.softmax(th(S[s]["text"]), 1), torch.softmax(ah(A[s]), 1)) for s in ["gate", "dev"]}
    gate = train_gate(S["gate"]["text"], A["gate"], *pr["gate"], y["gate"],
                      (S["dev"]["text"], A["dev"], *pr["dev"], y["dev"]), w_gate, seed)
    SHUF.mkdir(parents=True, exist_ok=True)
    torch.save({"text_head": th.state_dict(), "audio_head": ah.state_dict(), "gate": gate.state_dict(),
                "audio_dim": A["clf"].shape[1], "seed": seed}, SHUF / f"seed{seed}.pt")


def main():
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    A = s23.build_audio(S, mp)
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]
    names = ["agreement", "p_text_mattered_more", "gain_over_always_text", "gain_over_majority", "spearman"]

    def summarise(log, tag):
        rows = []
        for model, signal, label in an06.PAIRS:
            s, e = an06.arrays(log, model, signal, "resample_full")
            pt = stats3(s, e)
            b = np.array([stats3(s[:, i], e[:, i]) for i in draws])
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            for k, n in enumerate(names):
                rows.append({"variant": tag, "pair": label, "metric": n, "estimate": pt[k], "ci_low": lo[k], "ci_high": hi[k]})
        return pd.DataFrame(rows)

    # 1. strong variant, test swaps (and dev swaps for the cutoff)
    log = swap_log(S["test"]["text"], A["test"], S["clf"]["text"], A["clf"], s23.load_models2, "resample_full")
    real = summarise(log, "strong_audio_real_labels")
    print("strong-audio swaps done", flush=True)

    # calibration by weight bin (gate model, test)
    d = log[log.model == "fusion_learned_gate"].copy()
    d["text_bigger"] = d.dp_text.abs() > d.dp_audio.abs()
    d["bin"] = pd.cut(d.gate_w_t, np.linspace(0, 1, 11), include_lowest=True)
    cal = d.groupby("bin", observed=False).text_bigger.agg(["mean", "size"]).reset_index()
    cal["bin"] = cal["bin"].astype(str)
    cal.to_csv(RESULTS / "strong_audio_calibration.csv", index=False)

    # dev-fitted cutoff, applied to test; gain over the majority baseline
    dev_log = swap_log(S["dev"]["text"], A["dev"], S["clf"]["text"], A["clf"], s23.load_models2, "dev")
    cuts, cut_rows = {}, []
    for seed in SEEDS:
        dd = dev_log[(dev_log.model == "fusion_learned_gate") & (dev_log.seed == seed)].sort_values("row")
        e = dd.dp_text.abs().values - dd.dp_audio.abs().values
        v = e != 0
        w, ee = dd.gate_w_t.values[v], e[v]
        ag = np.array([np.mean(np.sign(w - t) == np.sign(ee)) for t in GRID])
        best = np.flatnonzero(ag == ag.max())
        tau = float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]])
        cuts[seed] = tau
        cut_rows.append({"seed": seed, "tau": tau, "dev_agreement": float(ag.max()), "dev_p_text": float(np.mean(ee > 0))})
    s0, e0 = an06.arrays(log, "fusion_learned_gate", "gate_w_t", "resample_full")
    tau = np.array([cuts[k] for k in SEEDS])[:, None]
    st = (s0 + 1) / 2 - tau
    pt = stats3(st, e0)
    bt = np.array([stats3(st[:, i], e0[:, i]) for i in draws])
    lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
    for k, n in enumerate(names):
        cut_rows.append({"seed": "test, dev-fitted cutoff", "tau": float(np.mean(list(cuts.values()))), "metric": n,
                         "estimate": pt[k], "ci_low": lo[k], "ci_high": hi[k]})
    pd.DataFrame(cut_rows).to_csv(RESULTS / "strong_audio_dev_cutoff.csv", index=False)
    print("dev cutoff done", flush=True)

    # 2. shuffled-label control with the strong features
    for seed in SEEDS:
        train_shuffled(D, A, seed)
        print(f"shuffled models trained, seed {seed}", flush=True)
    slog = swap_log(S["test"]["text"], A["test"], S["clf"]["text"], A["clf"], load_shuf, "resample_full")
    shuf = summarise(slog, "strong_audio_shuffled_labels")

    out = pd.concat([real, shuf], ignore_index=True)
    out.to_csv(RESULTS / "strong_audio_checks.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    for n in names:
        t = out[out.metric == n].copy()
        t["v"] = t.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
        print(f"\n== {n} ==")
        print(t.pivot(index="pair", columns="variant", values="v").to_string())
    print("\nCalibration by gate weight bin (strong audio, test):")
    print(cal.round(3).to_string(index=False))
    print("\nDev-fitted cutoffs and test result:")
    print(pd.DataFrame(cut_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
