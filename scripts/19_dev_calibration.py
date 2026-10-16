"""Step 19: gate diagnosis with a DEV-fitted threshold (action item 16).

Background: results/gate_calibration.csv shows the share of utterances where text mattered more
crosses 50% at a gate text weight of about 0.35 to 0.4, not 0.5. That break-even was read off the
TEST set, so it cannot be used as a result. Question: if the gate's "text weight > threshold" rule
uses a threshold fitted on DEV and frozen, does the gate beat the always-text baseline on TEST?
This separates "the gate's ranking carries information but is miscalibrated" from "it carries none".

Definitions, fixed before running:
  - Dev interventions for the current model (3 seeds): mean replacement (clf mean embeddings),
    blank (empty text, zero waveform of the clip's own length in 0.1 s buckets; needs a new
    WavLM pass on dev), and full-pool resampling (exact, as 08_resampling_full.py). Effects are
    dp text/audio, the primary effect difference e = |dp_text| - |dp_audio|.
  - Per seed and per method: threshold tau = the gate text weight in {0.05, 0.06, ..., 0.95}
    that maximises dev agreement between sign(w_t - tau) and sign(e) (exact ties in e excluded;
    ties between thresholds broken toward 0.5). Fitted on dev only, then frozen.
  - Test: signal s = w_t - tau (per seed). Metrics on test with the same effects as before:
    agreement, always-text baseline (unchanged), gain, per method (resample_full primary).
    Spearman is unchanged by a shift and is not reported again.
  - Bootstrap: dialogue clusters, 1000 draws, rng 0, 3 seeds averaged; tau is held fixed, so
    intervals are conditional on the fitted thresholds (they do not include dev sampling noise).
  - Reading rule: gain CI above 0 on test with the dev-fitted tau -> the gate's ranking carries
    usable information that its 0.5 cut-off wastes; gain CI includes 0 -> no usable edge even
    after recalibration. Either way, matching the test-set break-even is not used as evidence.
  - Check: the dev silence embeddings must be 1024-d, finite, and the dev clip lengths must match
    results of 09_extract_layers (data/wavlm_layers_dev.pt).

Usage: python scripts/19_dev_calibration.py
Outputs: data/intervention_embeddings_dev.pt (local), results/dev_calibration_thresholds.csv,
         results/dev_calibration_test.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, load_dataset  # noqa: E402

iv04 = importlib.import_module("04_interventions")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
an06 = importlib.import_module("06_analysis")
s03 = importlib.import_module("03_silence_embeddings")
SEEDS = [0, 1, 2]
GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)
METHODS = ["resample_full", "mean", "blank"]


def dev_silence(D):
    """Zero-waveform WavLM embeddings for the dev clips, same recipe as 03 (0.1 s buckets)."""
    L = torch.load(DATA / "wavlm_layers_dev.pt", map_location="cpu", weights_only=False)
    meta = D["meta"]["dev"]
    assert list(zip(meta.dialogue_id, meta.utterance_id)) == [tuple(i) for i in L["ids"]]
    lengths = L["clip_length_s"].numpy()
    buckets = np.maximum(np.round(lengths / s03.BUCKET_S) * s03.BUCKET_S, s03.BUCKET_S)
    unique = np.unique(buckets)
    print(f"dev clips {len(lengths)}, {len(unique)} length buckets", flush=True)
    from transformers import AutoFeatureExtractor, AutoModel
    device = s03.pick_device()
    ex = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    enc = AutoModel.from_pretrained("microsoft/wavlm-large").to(device).eval()
    silent = [np.zeros(int(round(b * s03.SR)), dtype=np.float32) for b in unique]
    emb = s03.wavlm_embed(silent, ex, enc, device, batch_size=1)
    assert emb.shape[1] == 1024 and torch.isfinite(emb).all()
    lookup = {round(float(b), 3): emb[k] for k, b in enumerate(unique)}
    sil = torch.stack([lookup[round(float(b), 3)] for b in buckets])
    torch.save({"audio_silence": sil, "clip_length_s": torch.tensor(lengths)}, DATA / "intervention_embeddings_dev.pt")
    return sil


def dev_effects(D, sil):
    S = D["splits"]
    dev, clf = S["dev"], S["clf"]
    mean_t, mean_a = clf["text"].mean(0), clf["audio"].mean(0)
    empty_t = torch.load(DATA / "intervention_embeddings.pt", map_location="cpu", weights_only=False)["test"]["text_empty"]
    n = len(dev["y"])
    e_t, e_a = dev["text"], dev["audio"]
    ok_t, ok_a = rs07.valid_masks(e_t, clf["text"]), rs07.valid_masks(e_a, clf["audio"])
    rows = []
    for seed in SEEDS:
        heads = iv04.load_models(seed)
        orig, p_t, p_a = iv04.run_all_models(e_t, e_a, *heads)
        c = orig["fusion_learned_gate"][0].argmax(1)
        p0 = orig["fusion_learned_gate"][0]
        w0 = orig["fusion_learned_gate"][1].squeeze(1).numpy()
        outs = {}
        for tag, rt, ra in (("mean", mean_t[None].expand(n, -1), mean_a[None].expand(n, -1)),
                            ("blank", empty_t[None].expand(n, -1), sil)):
            nt, _, _ = iv04.run_all_models(rt, e_a, *heads)
            na, _, _ = iv04.run_all_models(e_t, ra, *heads)
            outs[tag] = (nt["fusion_learned_gate"][0], na["fusion_learned_gate"][0])
        o2, _, _, cls2, pb, _, _ = rs08.exact_average(seed, e_t, e_a, clf["text"], clf["audio"], ok_t, ok_a)
        assert (cls2["fusion_learned_gate"] == c).all()
        outs["resample_full"] = pb["fusion_learned_gate"]
        for tag, (pt_, pa_) in outs.items():
            dpt = iv04.effect(pt_, p0, c)[0].numpy()
            dpa = iv04.effect(pa_, p0, c)[0].numpy()
            rows.append(pd.DataFrame({"seed": seed, "method": tag, "row": range(n), "w": w0,
                                      "e": np.abs(dpt) - np.abs(dpa)}))
    return pd.concat(rows, ignore_index=True)


def fit_tau(w, e):
    v = e != 0
    w, e = w[v], e[v]
    ag = np.array([np.mean(np.sign(w - t) == np.sign(e)) for t in GRID])
    best = np.flatnonzero(ag == ag.max())
    return float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]]), float(ag.max()), float(np.mean(e > 0))


def main():
    D = load_dataset()
    sil = dev_silence(D)
    dev = dev_effects(D, sil)

    taus, rows = {}, []
    for seed in SEEDS:
        for method in METHODS:
            d = dev[(dev.seed == seed) & (dev.method == method)].sort_values("row")
            tau, ag, base = fit_tau(d.w.values, d.e.values)
            taus[(seed, method)] = tau
            rows.append({"seed": seed, "method": method, "tau": tau, "dev_agreement": ag,
                         "dev_always_text": base, "dev_gain_at_tau": ag - base})
    th = pd.DataFrame(rows)
    th.to_csv(RESULTS / "dev_calibration_thresholds.csv", index=False)
    print(th.to_string(index=False), flush=True)

    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    log = pd.concat([ref, pd.read_csv(RESULTS / "resampling_full_log.csv")], ignore_index=True)
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]

    out = []
    for method in METHODS:
        s0, e = an06.arrays(log, "fusion_learned_gate", "gate_w_t", method)   # s0 = 2 w - 1, seeds sorted
        w = (s0 + 1) / 2
        tau = np.array([taus[(k, method)] for k in SEEDS])[:, None]
        for label, s in (("gate at 0.5 (as before)", s0), ("gate at dev-fitted tau", w - tau)):
            pt = np.array(an06.stats(s, e))[:3]
            b = np.array([an06.stats(s[:, i], e[:, i])[:3] for i in draws])
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            for k, name in enumerate(["agreement", "always_text_baseline", "gain"]):
                out.append({"method": method, "rule": label, "metric": name, "estimate": pt[k],
                            "ci_low": lo[k], "ci_high": hi[k]})
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "dev_calibration_test.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)
    t = res[res.metric == "gain"].copy()
    t["v"] = t.apply(lambda r: f"{r.estimate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]", axis=1)
    print("\nTEST gain over always-text:")
    print(t.pivot(index="method", columns="rule", values="v").to_string())


if __name__ == "__main__":
    main()
