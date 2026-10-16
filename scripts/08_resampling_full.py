"""Step 8 (addendum): resampling with the FULL clf split as the donor pool (exact average,
no donor sampling). Written and fixed BEFORE running. Follow-up to 07_resampling.py, which
found that the 50-donor result depends on which 50 donors are drawn (gate gain +0.018 for
draw 0, +0.029 for draw 1). It changes no definition in 04 or 07 except the donor set.

DEFINITIONS (identical to 07 except the donors)
  donors     ALL clf-split rows (7,990). Nothing is sampled, so there is no donor variance.
  per target a donor is skipped for a test utterance when its embedding of the replaced
             modality is byte-identical to the test utterance's own (as in 07).
  averaging  p_bar = mean over valid donors of the fused probability vector; the gate (or
             confidence rule) recomputes its weight for every (utterance, donor) pair.
  effect, signals, metrics, bootstrap draws: as in 07 (same 1,000 dialogue draws as 06).
  same dialogue: clf and test rows are from different MELD dialogues (official splits are
             dialogue-disjoint); clf rows carry no dialogue metadata, so this is assumed.

HOW IT IS COMPUTED (exact, just cheaper)
  Heads are linear, so each donor's head probabilities are computed once. The gate's first
  layer is Linear(1408 -> 128) on [e_t; e_a], so it splits exactly into
  W_t e_t + W_a e_a + b; for each (utterance, donor) pair only the ReLU and the second layer
  are evaluated. Check: the same code restricted to the 50 donors of draw 0 must reproduce
  the "resample" rows of results/resampling_log.csv (dp, mean |dp|, donor counts) to 1e-5.

PRE-REGISTERED USE (fixed before seeing the result)
  1. The full-pool estimate SUPERSEDES the 50-donor draws as the primary resampling number.
     Both 50-donor draws (methods "resample", "resample_alt") stay in the report as a
     sensitivity check, unchanged.
  2. Headline claim for the gate = gain over always-text under full-pool resampling with its
     95% CI. CI excludes 0 -> report that sign; includes 0 -> "not distinguishable".
     Mean and blank are reported as sensitivity checks; matching one does not validate it.
  3. No change to the exclusion rule, effect definition or metrics after seeing results.
  4. Bucket: before midterm report. Slide condition unchanged: only if finished, checked and
     reviewed by Oct 17 AND the full-pool gate-gain CI excludes 0.
  5. Wording for slides and Q&A (recorded after the run, numbers taken from this run): the
     gate's honest reading is "at most a small edge over always-text", and the negative
     result under blank stand-ins is treated as a stand-in artifact, not a finding.

Output: results/resampling_full_log.csv, resampling_full_faithfulness.csv,
        resampling_full_gain_differences.csv
Usage:  python scripts/08_resampling_full.py
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
rs07 = importlib.import_module("07_resampling")
SEEDS, MODELS = rs07.SEEDS, rs07.MODELS
CHUNK = 100
METHOD = "resample_full"


@torch.no_grad()
def exact_average(seed, e_t, e_a, pool_t, pool_a, ok_t, ok_a):
    """Donor-averaged outputs over every pool row, exact. Returns the pieces make_frames needs."""
    text_head, audio_head, gate = iv04.load_models(seed)
    n = e_t.shape[0]
    idx = torch.arange(n)
    orig, p_t, p_a = iv04.run_all_models(e_t, e_a, text_head, audio_head, gate)
    cls = {m: orig[m][0].argmax(1) for m in MODELS}
    P_t, P_a = torch.softmax(text_head(pool_t), 1), torch.softmax(audio_head(pool_a), 1)
    lin1, lin2 = gate.net[0], gate.net[2]
    Wt, Wa, b1 = lin1.weight[:, :e_t.shape[1]], lin1.weight[:, e_t.shape[1]:], lin1.bias
    w2, b2 = lin2.weight[0], lin2.bias[0]
    A_t_pool, A_a_pool = pool_t @ Wt.T, pool_a @ Wa.T          # (pool, 128)
    A_t_test, A_a_test = e_t @ Wt.T, e_a @ Wa.T                # (n, 128)
    ct_pool, ca_pool = P_t.max(1).values, P_a.max(1).values
    ct_test, ca_test = p_t.max(1).values, p_a.max(1).values

    S = {(m, k): torch.zeros(n, 7) for m in MODELS for k in ("text", "audio")}
    SA = {(m, k): torch.zeros(n) for m in MODELS for k in ("text", "audio")}
    for kind, ok in (("text", ok_t), ("audio", ok_a)):
        for lo in range(0, pool_t.shape[0], CHUNK):
            sl = slice(lo, lo + CHUNK)
            m_ = ok[:, sl].float()                                         # (n, chunk)
            if kind == "text":      # donor text, original audio
                pre = A_t_pool[sl][None] + A_a_test[:, None]               # (n, chunk, 128)
                ct, ca = ct_pool[sl][None], ca_test[:, None]
                Pd = P_t[sl]                                               # donor probs of removed modality
            else:                   # original text, donor audio
                pre = A_t_test[:, None] + A_a_pool[sl][None]
                ct, ca = ct_test[:, None], ca_pool[sl][None]
                Pd = P_a[sl]
            w_gate = torch.sigmoid(torch.relu(pre + b1) @ w2 + b2)         # (n, chunk)
            weights = {"fusion_learned_gate": w_gate, "fusion_confidence": ct / (ct + ca),
                       "fusion_average": torch.full_like(w_gate, 0.5)}
            for m in MODELS:
                w = weights[m]
                if kind == "text":  # fused = w * P_donor + (1 - w) * p_a[i]
                    S[(m, kind)] += (m_ * w) @ Pd + p_a * (m_ * (1 - w)).sum(1, keepdim=True)
                    fc = w * Pd[:, cls[m]].T + (1 - w) * p_a[idx, cls[m]][:, None]
                else:               # fused = w * p_t[i] + (1 - w) * P_donor
                    S[(m, kind)] += p_t * (m_ * w).sum(1, keepdim=True) + (m_ * (1 - w)) @ Pd
                    fc = w * p_t[idx, cls[m]][:, None] + (1 - w) * Pd[:, cls[m]].T
                SA[(m, kind)] += (m_ * (fc - orig[m][0][idx, cls[m]][:, None]).abs()).sum(1)
    n_t, n_a = ok_t.sum(1).float(), ok_a.sum(1).float()
    pb = {m: (S[(m, "text")] / n_t[:, None], S[(m, "audio")] / n_a[:, None]) for m in MODELS}
    mabs = {m: (SA[(m, "text")] / n_t, SA[(m, "audio")] / n_a) for m in MODELS}
    return orig, p_t, p_a, cls, pb, mabs, (n_t, n_a)


def main():
    D = load_dataset()
    test, meta, clf = D["splits"]["test"], D["meta"]["test"], D["splits"]["clf"]
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    e_t, e_a = test["text"], test["audio"]

    # 1. verification: restrict to the 50 donors of draw 0 and compare with 07's log
    pick = torch.as_tensor(np.random.default_rng(0).choice(len(clf["y"]), rs07.N_DONORS, replace=False))
    d_t, d_a = clf["text"][pick], clf["audio"][pick]
    old = pd.read_csv(RESULTS / "resampling_log.csv")
    old = old[old.method == "resample"]
    cols = ["dp_text", "dp_audio", "mean_abs_dp_text", "mean_abs_dp_audio", "n_donors_text", "n_donors_audio", "flip_text", "tvd_text"]
    for seed in SEEDS:
        out = exact_average(seed, e_t, e_a, d_t, d_a, rs07.valid_masks(e_t, d_t), rs07.valid_masks(e_a, d_a))
        new = rs07.make_frames(seed, "check", *out[:2], out[2], out[3], out[4], out[5], out[6], test, meta, ref)
        o = old[old.seed == seed].reset_index(drop=True)
        assert (new.model.values == o.model.values).all()
        for c in cols:
            assert np.allclose(new[c].values.astype(float), o[c].values.astype(float), atol=1e-5), f"verification failed: {c}"
    print("verification passed: exact code on the 50 draw-0 donors reproduces results/resampling_log.csv")

    # 2. full pool
    pool_t, pool_a = clf["text"], clf["audio"]
    ok_t, ok_a = rs07.valid_masks(e_t, pool_t), rs07.valid_masks(e_a, pool_a)
    floor = 0.98 * pool_t.shape[0]
    assert min(ok_t.sum(1).min(), ok_a.sum(1).min()) >= floor, "too many donors excluded"
    print(f"pool {pool_t.shape[0]}; min valid donors text {int(ok_t.sum(1).min())}, audio {int(ok_a.sum(1).min())}")
    frames = []
    for seed in SEEDS:
        out = exact_average(seed, e_t, e_a, pool_t, pool_a, ok_t, ok_a)
        frames.append(rs07.make_frames(seed, METHOD, *out[:2], out[2], out[3], out[4], out[5], out[6], test, meta, ref))
        print(f"  {METHOD} seed {seed} done")
    log = pd.concat(frames, ignore_index=True)
    log.to_csv(RESULTS / "resampling_full_log.csv", index=False)

    prev = pd.read_csv(RESULTS / "resampling_log.csv")
    res, diffs = rs07.analyse(pd.concat([prev, log], ignore_index=True), ref,
                              methods=("mean", "blank", "resample", "resample_alt", METHOD),
                              primary=METHOD, others=("mean", "blank", "resample", "resample_alt"))
    oldci = pd.read_csv(RESULTS / "bootstrap_ci.csv")
    chk = res[res.method.isin(["mean", "blank"])].merge(oldci, on=["pair", "method", "metric"], suffixes=("", "_old"))
    assert len(chk) == 32 and np.allclose(chk.estimate, chk.estimate_old, atol=1e-6), "mean/blank do not reproduce bootstrap_ci.csv"
    res.to_csv(RESULTS / "resampling_full_faithfulness.csv", index=False)
    diffs.to_csv(RESULTS / "resampling_full_gain_differences.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    prim = res[res.pair.isin(["Learned gate ← gate weight", "Confidence fusion ← confidence"])]
    print("\nPrimary pairs, gain over always-text with 95% CI:")
    print(prim[prim.metric == "gain"][["pair", "method", "estimate", "ci_low", "ci_high"]].to_string(index=False))
    print("\nAgreement, always-text baseline, Spearman (full-pool):")
    print(prim[prim.method == METHOD].pivot_table(index="pair", columns="metric", values="estimate")
          [["agreement", "always_text_baseline", "gain", "spearman"]])
    print("\nPaired gain differences (full pool minus other), primary pairs:")
    print(diffs[diffs.pair.isin(["Learned gate ← gate weight", "Confidence fusion ← confidence"])].to_string(index=False))


if __name__ == "__main__":
    main()
