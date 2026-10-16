"""Step 7 (addendum): resampling intervention, a third and more in-distribution way to
"remove" a modality. Written and fixed BEFORE running. It does not change any definition in
the header of 04_interventions.py (signals, effect, agreement, near-zero 0.02, Spearman).

RESAMPLING INTERVENTION
-----------------------
Idea: instead of one stand-in (mean or blank), replace one modality of a test utterance with
that modality from many REAL other utterances ("donors"), rerun the whole model each time,
and average the OUTPUT probabilities.

Definitions
  donors     50 rows drawn at random, without replacement, from the clf split only
             (numpy default_rng(0); natural label mix, no balancing by emotion). The SAME 50
             donors are used for every test utterance, every seed and both modalities.
             A donor supplies its own text embedding (text removal) or audio embedding
             (audio removal); the two removals are separate, never combined.
  per target donors are skipped for a given test utterance when the donor's embedding of the
             modality being replaced is byte-identical to the test utterance's own embedding
             (self-swap; covers the audio_shared duplicates and repeated short texts). The
             average is over the remaining donors; the count is logged.
  same dialogue  test and clf rows come from different MELD dialogues (official splits are
             dialogue-disjoint), so same-dialogue donors cannot occur. clf rows carry no
             dialogue metadata, so this is an assumption, not something checked here.
  forward    for each donor, replace the modality, rerun the WHOLE model with its own fusion
             rule (the gate recomputes its weight for each donor; the confidence rule too).
  averaging  p_bar = mean over valid donors of the fused probability vector. Donor embeddings
             are NEVER averaged (that would collapse to the mean stand-in).
  effect     same as the protocol: with c = the originally predicted class,
             dp_text = p_bar(text removed)[c] - p_orig[c], dp_audio likewise; effect = |dp|.
             flip = argmax of p_bar differs from c. tvd = 0.5 * sum |p_bar - p_orig|.
             Secondary, logged only: mean over donors of |dp| (mean_abs_dp_text/audio).
  signals    read on the ORIGINAL natural input, exactly as in 04.
  metrics    agreement, always-text baseline, gain, Spearman exactly as in 05/06, 3 seeds,
             dialogue-level bootstrap with the SAME 1,000 draws as 06 (rng 0), so intervals
             are paired with the mean and blank results.
  stability  a second, independent donor draw (default_rng(1), method "resample_alt") is run
             only to check donor-set variance. It is never used to choose between draws.

Relation to H2 (conflict test). Donors have their own emotion, so a swapped utterance can
carry conflicting emotion. The two experiments stay distinct:
  - here the swap is a MEASUREMENT TOOL: signals are read on the original natural utterance,
    donors are drawn blind to emotion, and the result is averaged over 50 donors;
  - H2 changes the INPUT (text paired with audio of a chosen different emotion, same-label
    control matched on speaker, duration and donor confidence) and reads the signals on that
    conflicted input;
  - results here are never sliced by donor emotion or by donor-versus-target label match. Any
    such slice would be exploratory and cannot count as evidence for or against H2.

PRE-REGISTERED USE OF THE RESULT (fixed before seeing it)
  1. Resampling becomes the PRIMARY intervention for the report, because it is the most
     in-distribution of the three (every input is a real utterance). Mean and blank stay as
     sensitivity checks. The old primary (mean) was set before any results; this change is made
     after seeing mean and blank results but before seeing any resampling result, and is
     justified on in-distribution grounds, not on outcome.
  2. Headline claim = gate gain over always-text under resampling with its 95% CI. If the CI
     excludes 0 we report that sign; if it includes 0 we report "not distinguishable". All
     three methods are reported in every case.
  3. Matching mean or blank does NOT validate that method; it only shows the conclusion is
     the same under another removal.
  4. We do not change donor count, donor seed, pool or exclusions after seeing results.
  5. Slides: this goes on the midterm slides only if the run is finished, checked and
     reviewed by Oct 17 AND the resampling gate-gain CI excludes 0 (it then breaks the tie
     between mean and blank). If the CI includes 0, the existing takeaway stands and the
     result is a one-line Q&A answer plus report material. Bucket: before midterm report.

Sanity checks (asserted): original-input probabilities and gate/confidence weights match
results/interventions_log.csv; averaged outputs sum to 1; every utterance keeps at least 45
donors; the mean and blank bootstrap rows reproduce results/bootstrap_ci.csv.

Output: results/resampling_log.csv (same schema as interventions_log plus extras),
        results/resampling_faithfulness.csv, results/resampling_gain_differences.csv
Usage:  python scripts/07_resampling.py
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_dataset  # noqa: E402

iv04 = importlib.import_module("04_interventions")
an06 = importlib.import_module("06_analysis")
SEEDS = [0, 1, 2]
N_DONORS = 50
DRAWS = {"resample": 0, "resample_alt": 1}
MODELS = ["fusion_learned_gate", "fusion_confidence", "fusion_average"]
MIN_DONORS = 45


def valid_masks(target, donors):
    """(n_target, n_donors) bool: True unless the donor embedding is byte-identical to the target's."""
    by_bytes = {}
    for k, row in enumerate(donors):
        by_bytes.setdefault(row.numpy().tobytes(), []).append(k)
    mask = torch.ones(len(target), len(donors), dtype=torch.bool)
    for i, row in enumerate(target):
        for k in by_bytes.get(row.numpy().tobytes(), []):
            mask[i, k] = False
    return mask


def run_draw(seed, method, e_t, e_a, d_t, d_a, ok_t, ok_a, test, meta, ref):
    heads = iv04.load_models(seed)
    n = e_t.shape[0]
    idx = torch.arange(n)
    orig, p_t, p_a = iv04.run_all_models(e_t, e_a, *heads)
    acc = {(m, k): torch.zeros(n, 7) for m in MODELS for k in ("text", "audio")}
    acc_abs = {(m, k): torch.zeros(n) for m in MODELS for k in ("text", "audio")}
    cls = {m: orig[m][0].argmax(1) for m in MODELS}
    for j in range(d_t.shape[0]):
        no_text, _, _ = iv04.run_all_models(d_t[j][None].expand(n, -1), e_a, *heads)
        no_audio, _, _ = iv04.run_all_models(e_t, d_a[j][None].expand(n, -1), *heads)
        for m in MODELS:
            for kind, res, ok in (("text", no_text, ok_t), ("audio", no_audio, ok_a)):
                w = ok[:, j].float()
                acc[(m, kind)] += w[:, None] * res[m][0]
                acc_abs[(m, kind)] += w * (res[m][0][idx, cls[m]] - orig[m][0][idx, cls[m]]).abs()
    n_t, n_a = ok_t.sum(1).float(), ok_a.sum(1).float()
    assert min(n_t.min(), n_a.min()) >= MIN_DONORS, "too few valid donors for some utterance"
    pb = {m: (acc[(m, "text")] / n_t[:, None], acc[(m, "audio")] / n_a[:, None]) for m in MODELS}
    mabs = {m: (acc_abs[(m, "text")] / n_t, acc_abs[(m, "audio")] / n_a) for m in MODELS}
    return make_frames(seed, method, orig, p_t, p_a, cls, pb, mabs, (n_t, n_a), test, meta, ref)


def make_frames(seed, method, orig, p_t, p_a, cls, pb, mabs, counts, test, meta, ref):
    """Log rows (same schema as interventions_log plus extras) from donor-averaged outputs.
    pb[m] = (p_bar text removed, p_bar audio removed); mabs[m] = mean |dp| over donors."""
    n = p_t.shape[0]
    n_t, n_a = counts
    frames = []
    for m in MODELS:
        p0, w0 = orig[m]
        c = cls[m]
        pb_t, pb_a = pb[m]
        assert torch.allclose(pb_t.sum(1), torch.ones(n), atol=1e-4) and torch.allclose(pb_a.sum(1), torch.ones(n), atol=1e-4)
        dpt, flt, tvt = iv04.effect(pb_t, p0, c)
        dpa, fla, tva = iv04.effect(pb_a, p0, c)
        r = ref[(ref.seed == seed) & (ref.model == m) & (ref.method == "mean")].sort_values("row")
        assert np.allclose(r.p_pred.values, p0.max(1).values.numpy(), atol=1e-5), "original probs differ from log"
        assert np.allclose(r.gate_w_t.values, orig["fusion_learned_gate"][1].squeeze(1).numpy(), atol=1e-5)
        frames.append(pd.DataFrame({
            "seed": seed, "row": range(n), "model": m, "method": method,
            "label": test["y"].numpy(), "pred": c.numpy(), "correct": (c == test["y"]).numpy(),
            "p_pred": p0.max(1).values.numpy(),
            "gate_w_t": orig["fusion_learned_gate"][1].squeeze(1).numpy(),
            "conf_w_t": orig["fusion_confidence"][1].squeeze(1).numpy(),
            "text_pred": p_t.argmax(1).numpy(), "audio_pred": p_a.argmax(1).numpy(),
            "dp_text": dpt.numpy(), "dp_audio": dpa.numpy(),
            "flip_text": flt.numpy(), "flip_audio": fla.numpy(),
            "tvd_text": tvt.numpy(), "tvd_audio": tva.numpy(),
            "mean_abs_dp_text": mabs[m][0].numpy(), "mean_abs_dp_audio": mabs[m][1].numpy(),
            "n_donors_text": n_t.int().numpy(), "n_donors_audio": n_a.int().numpy(),
            "audio_shared": meta["audio_shared"].values,
            "audio_shared_with_train": meta["audio_shared_with_train"].values,
            "dialogue_id": meta["dialogue_id"].values, "utterance_id": meta["utterance_id"].values}))
    return pd.concat(frames, ignore_index=True)


def analyse(log, ref, methods=("mean", "blank", "resample", "resample_alt"),
            primary="resample", others=("mean", "blank", "resample_alt")):
    """Same bootstrap draws as 06 (rng 0, same dialogue order). Point estimates and CIs per
    pair and method, plus paired gain differences for the primary pairs."""
    rng = np.random.default_rng(0)
    order = ref[(ref.seed == ref.seed.min()) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]
    allog = pd.concat([ref, log], ignore_index=True)
    rows, gains = [], {}
    for model, signal, label in an06.PAIRS:
        for method in methods:
            s, e = an06.arrays(allog, model, signal, method)
            point = an06.stats(s, e)
            boots = np.array([an06.stats(s[:, i], e[:, i]) for i in draws])
            gains[(label, method)] = boots[:, 2]
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            for k, name in enumerate(["agreement", "always_text_baseline", "gain", "spearman"]):
                rows.append({"pair": label, "model": model, "signal": signal, "method": method,
                             "metric": name, "estimate": point[k], "ci_low": lo[k], "ci_high": hi[k]})
    res = pd.DataFrame(rows)
    diffs = []
    for model, signal, label in an06.PAIRS:
        for other in others:
            d = gains[(label, primary)] - gains[(label, other)]
            est = res[(res.pair == label) & (res.method == primary) & (res.metric == "gain")].estimate.item() \
                - res[(res.pair == label) & (res.method == other) & (res.metric == "gain")].estimate.item()
            lo, hi = np.percentile(d, [2.5, 97.5])
            diffs.append({"pair": label, f"gain_{primary}_minus": other, "estimate": est, "ci_low": lo, "ci_high": hi})
    return res, pd.DataFrame(diffs)


def main():
    D = load_dataset()
    test, meta, clf = D["splits"]["test"], D["meta"]["test"], D["splits"]["clf"]
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    e_t, e_a = test["text"], test["audio"]
    frames = []
    for method, dseed in DRAWS.items():
        pick = torch.as_tensor(np.random.default_rng(dseed).choice(len(clf["y"]), N_DONORS, replace=False))
        d_t, d_a = clf["text"][pick], clf["audio"][pick]
        ok_t, ok_a = valid_masks(e_t, d_t), valid_masks(e_a, d_a)
        print(f"{method}: donor labels {np.bincount(clf['y'][pick].numpy(), minlength=7).tolist()}, "
              f"min valid donors text {int(ok_t.sum(1).min())} audio {int(ok_a.sum(1).min())}")
        for seed in SEEDS:
            frames.append(run_draw(seed, method, e_t, e_a, d_t, d_a, ok_t, ok_a, test, meta, ref))
            print(f"  {method} seed {seed} done")
    log = pd.concat(frames, ignore_index=True)
    log.to_csv(RESULTS / "resampling_log.csv", index=False)

    res, diffs = analyse(log, ref)
    # sanity: mean/blank rows must reproduce the committed bootstrap results
    old = pd.read_csv(RESULTS / "bootstrap_ci.csv")
    chk = res[res.method.isin(["mean", "blank"])].merge(old, on=["pair", "method", "metric"], suffixes=("", "_old"))
    assert len(chk) == 4 * 2 * 4 and np.allclose(chk.estimate, chk.estimate_old, atol=1e-6) \
        and np.allclose(chk.ci_low, chk.ci_low_old, atol=1e-6), "mean/blank do not reproduce bootstrap_ci.csv"
    res.to_csv(RESULTS / "resampling_faithfulness.csv", index=False)
    diffs.to_csv(RESULTS / "resampling_gain_differences.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    prim = res[res.pair.isin(["Learned gate ← gate weight", "Confidence fusion ← confidence"])]
    print("\nPrimary pairs (gain over always-text, agreement, Spearman):")
    print(prim.pivot_table(index=["pair", "method"], columns="metric", values="estimate")
          [["agreement", "always_text_baseline", "gain", "spearman"]])
    print("\nGain with 95% CI:")
    print(prim[prim.metric == "gain"][["pair", "method", "estimate", "ci_low", "ci_high"]].to_string(index=False))
    print("\nPaired gain differences (resample minus other):")
    print(diffs.to_string(index=False))


if __name__ == "__main__":
    main()
