"""Step 18: label-shuffled control (action item 6).

Question: how much of each signal's agreement with the interventions is a property of the
architecture, not of the heads having learned anything? Train the whole pipeline on labels that
carry no information, run the same full-pool resampling interventions, and score the same signals.

Definitions, fixed before running:
  - For each seed, permute the labels of the clf, gate and dev splits independently (numpy rng
    1000 + seed). No real label is used anywhere in training, gating or early stopping.
    Test labels are real and used only for F1 (to show the heads are at chance) and for
    nothing else; faithfulness metrics never use labels.
  - Same architecture, loss, class weights (from the permuted labels), optimiser and early
    stopping as 02_train.py. Seeds 0, 1, 2.
  - Interventions: full-pool resampling, exactly as 08_resampling_full.py (same donors, same
    exclusion rule), effect = |dp| difference (primary), signals and 4 pairs as in 06.
  - Bootstrap: dialogue clusters, 1000 draws, rng 0, 3 seeds averaged (same draws as 06).
  - Check: the same code run on the REAL checkpoints must reproduce dp_text, dp_audio and the
    weights of results/resampling_full_log.csv to 1e-5.
  Reading rule: if a signal's gain under shuffled labels is within 0.05 of its gain under real
  labels, we say its agreement does not depend on the heads carrying label information
  (structural). Gain well below the real value means the real result relies on learned structure.
  This provides the source file for the "confidence agreement is partly structural" claim.

Usage: python scripts/18_shuffled_control.py
Outputs: results/shuffled_control.csv, results/shuffled_control_f1.csv
         checkpoints/shuffled/seed{N}.pt (local only)
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, RESULTS, class_weights, load_dataset, set_determinism,  # noqa: E402
                    train_gate, train_head, fuse_gate)

iv04 = importlib.import_module("04_interventions")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
an06 = importlib.import_module("06_analysis")
SEEDS, MODELS = rs07.SEEDS, rs07.MODELS
SHUF = CKPT / "shuffled"


def train_shuffled(D, seed):
    S = D["splits"]
    rng = np.random.default_rng(1000 + seed)
    y = {k: S[k]["y"][torch.as_tensor(rng.permutation(len(S[k]["y"])))] for k in ("clf", "gate", "dev")}
    set_determinism(seed)
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    th = train_head(S["clf"]["text"], y["clf"], S["dev"]["text"], y["dev"], w_clf, seed)
    ah = train_head(S["clf"]["audio"], y["clf"], S["dev"]["audio"], y["dev"], w_clf, seed)
    with torch.no_grad():
        pr = {s: (torch.softmax(th(S[s]["text"]), 1), torch.softmax(ah(S[s]["audio"]), 1)) for s in ["gate", "dev", "test"]}
    gate = train_gate(S["gate"]["text"], S["gate"]["audio"], *pr["gate"], y["gate"],
                      (S["dev"]["text"], S["dev"]["audio"], *pr["dev"], y["dev"]), w_gate, seed)
    SHUF.mkdir(parents=True, exist_ok=True)
    torch.save({"text_head": th.state_dict(), "audio_head": ah.state_dict(), "gate": gate.state_dict(),
                "seed": seed}, SHUF / f"seed{seed}.pt")
    with torch.no_grad():
        p_t, p_a = pr["test"]
        pg, _ = fuse_gate(gate, S["test"]["text"], S["test"]["audio"], p_t, p_a)
    yt = S["test"]["y"].numpy()
    f1 = {k: f1_score(yt, p.argmax(1).numpy(), average="weighted", zero_division=0)
          for k, p in [("text_only", p_t), ("audio_only", p_a), ("fusion_learned_gate", pg)]}
    return f1


def build_log(D, ckpt_dir, tag):
    """Full-pool resampling log (the columns the metrics need) for the checkpoints in ckpt_dir."""
    iv04.CKPT = ckpt_dir
    test, meta, clf = D["splits"]["test"], D["meta"]["test"], D["splits"]["clf"]
    e_t, e_a = test["text"], test["audio"]
    ok_t, ok_a = rs07.valid_masks(e_t, clf["text"]), rs07.valid_masks(e_a, clf["audio"])
    frames = []
    for seed in SEEDS:
        orig, p_t, p_a, cls, pb, mabs, counts = rs08.exact_average(seed, e_t, e_a, clf["text"], clf["audio"], ok_t, ok_a)
        for m in MODELS:
            p0, w0 = orig[m]
            dpt, flt, tvt = iv04.effect(pb[m][0], p0, cls[m])
            dpa, fla, tva = iv04.effect(pb[m][1], p0, cls[m])
            frames.append(pd.DataFrame({
                "seed": seed, "row": range(len(e_t)), "model": m, "method": tag,
                "gate_w_t": orig["fusion_learned_gate"][1].squeeze(1).numpy(),
                "conf_w_t": orig["fusion_confidence"][1].squeeze(1).numpy(),
                "dp_text": dpt.numpy(), "dp_audio": dpa.numpy()}))
    iv04.CKPT = CKPT
    return pd.concat(frames, ignore_index=True)


def main():
    D = load_dataset()
    real_ref = pd.read_csv(RESULTS / "resampling_full_log.csv")

    # check: real checkpoints reproduce the logged full-pool run
    real = build_log(D, CKPT, "resample_full")
    for m in MODELS:
        a = real[real.model == m].sort_values(["seed", "row"])
        b = real_ref[real_ref.model == m].sort_values(["seed", "row"])
        for c in ["dp_text", "dp_audio", "gate_w_t", "conf_w_t"]:
            assert np.allclose(a[c].values, b[c].values, atol=1e-5), f"check failed: {m} {c}"
    print("check passed: real checkpoints reproduce results/resampling_full_log.csv", flush=True)

    f1s = []
    for seed in SEEDS:
        f1 = train_shuffled(D, seed)
        f1s.append({"seed": seed, **f1})
        print(f"seed {seed}: shuffled-label test weighted-F1 {({k: round(v, 3) for k, v in f1.items()})}", flush=True)
    pd.DataFrame(f1s).to_csv(RESULTS / "shuffled_control_f1.csv", index=False)
    shuf = build_log(D, SHUF, "shuffled")

    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]
    both = pd.concat([real, shuf], ignore_index=True)
    rows = []
    names = ["agreement", "always_text_baseline", "gain", "spearman"]
    for model, signal, label in an06.PAIRS:
        for tag, lab in (("resample_full", "real labels"), ("shuffled", "shuffled labels")):
            s, e = an06.arrays(both, model, signal, tag)
            point = an06.stats(s, e)
            b = np.array([an06.stats(s[:, i], e[:, i]) for i in draws])
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            for k, n in enumerate(names):
                rows.append({"pair": label, "labels": lab, "metric": n, "estimate": point[k],
                             "ci_low": lo[k], "ci_high": hi[k]})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "shuffled_control.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    for n in names:
        t = res[res.metric == n].copy()
        t["v"] = t.apply(lambda r: f"{r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" if n == "gain"
                         else f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
        print(f"\n== {n} ==")
        print(t.pivot(index="pair", columns="labels", values="v")[["real labels", "shuffled labels"]].to_string())


if __name__ == "__main__":
    main()
