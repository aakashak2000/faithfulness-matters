"""Step 20: H2, does agreement drop when text and audio carry conflicting emotion?
(action items 18 and 19)

H2 (proposal): importance signals agree less with intervention effects when the text and the
audio of an utterance point to different emotions.

DESIGN, fixed before running
  Constructed pairs. Every test utterance i keeps its own TEXT. Its AUDIO is swapped for the audio
  of another test utterance j:
    control   j has the SAME emotion label as i       (matched same-label control)
    conflict  j has a DIFFERENT label: first a class c' != label(i) is drawn uniformly among the
              other classes, then j uniformly among test utterances of class c'.
  Both conditions use the same text, so the only thing that differs is whether the audio's
  emotion label matches the text's. j must be from a different dialogue than i, must not be
  flagged audio_shared, and its audio embedding must differ from i's. Pairings are drawn with
  numpy rng 2000 (primary) and rng 2001 (sensitivity), identical for all three model seeds.
  The audio's emotion is the label of its own utterance (MELD labels are per utterance, so this is
  a noisy proxy for what the audio expresses; stated as a limit).
  Interventions on each constructed pair: full-pool resampling exactly as 08_resampling_full.py
  (the constructed pair is the "original input"; donors are the 7,990 clf rows). Effect and
  signals as in 04/06; effect difference e = |dp_text| - |dp_audio|.
  Resampling is NOT sliced by donor emotion; the conflict is created in the input pair only.
  Statistic. For each pair of 06 (primary: Learned gate <- gate weight), gain over the always-text
  baseline within each condition, and the paired difference
        D = gain(conflict) - gain(control)
  with a dialogue-cluster bootstrap (1000 draws, rng 0, same draws for both conditions, 3 seeds
  averaged; clusters = dialogue of the text utterance). Secondary: the same difference for
  agreement and Spearman; alternative effect measure tvd for gain.
  Reading rule (written now): H2 is SUPPORTED for the gate if the 95% interval of D for
  Learned gate <- gate weight lies below 0; NOT SUPPORTED if it includes 0 ("agreement does not
  differ detectably"); the OPPOSITE direction if it lies above 0. Same rule for the other pairs,
  reported with the caveat that the confidence rule is partly structural.
  Manipulation checks (reported, not used to decide): share of constructed pairs where the text
  head and the audio head predict different classes, per condition; fused accuracy against the
  text utterance's label per condition.
  Limits to state in the report: constructed pairs are not natural utterances (the control is
  equally constructed, which is why it is the comparison); one random pairing per rng, with a
  second as sensitivity; intervals are conditional on the 3 trained models.

Usage: python scripts/20_conflict_h2.py
Outputs: results/conflict_pairs.csv, results/conflict_h2_metrics.csv, results/conflict_h2_diff.csv,
         results/conflict_h2_checks.csv
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
rs08 = importlib.import_module("08_resampling_full")
an06 = importlib.import_module("06_analysis")
SEEDS, MODELS = rs07.SEEDS, rs07.MODELS
DRAWS = {"primary": 2000, "alt": 2001}


def make_pairs(D, rng_seed):
    meta, y = D["meta"]["test"], D["splits"]["test"]["y"].numpy()
    audio = D["splits"]["test"]["audio"]
    dia = meta.dialogue_id.values
    ok = ~meta.audio_shared.values
    rng = np.random.default_rng(rng_seed)
    n = len(y)
    classes = np.unique(y)
    by_class = {c: np.flatnonzero((y == c) & ok) for c in classes}
    ctrl, conf = np.zeros(n, int), np.zeros(n, int)
    for i in range(n):
        def pick(c):
            cand = by_class[c]
            cand = cand[(dia[cand] != dia[i]) & (cand != i)]
            cand = np.array([j for j in cand if not torch.equal(audio[j], audio[i])])
            return int(rng.choice(cand))
        ctrl[i] = pick(y[i])
        conf[i] = pick(int(rng.choice([c for c in classes if c != y[i]])))
    return ctrl, conf


def frames_for(D, e_a_cond, tag, y_text):
    test, clf = D["splits"]["test"], D["splits"]["clf"]
    e_t = test["text"]
    ok_t, ok_a = rs07.valid_masks(e_t, clf["text"]), rs07.valid_masks(e_a_cond, clf["audio"])
    out, checks = [], []
    for seed in SEEDS:
        orig, p_t, p_a, cls, pb, mabs, counts = rs08.exact_average(seed, e_t, e_a_cond, clf["text"], clf["audio"], ok_t, ok_a)
        for m in MODELS:
            p0, w0 = orig[m]
            dpt, flt, tvt = iv04.effect(pb[m][0], p0, cls[m])
            dpa, fla, tva = iv04.effect(pb[m][1], p0, cls[m])
            out.append(pd.DataFrame({
                "seed": seed, "row": range(len(e_t)), "model": m, "method": tag,
                "gate_w_t": orig["fusion_learned_gate"][1].squeeze(1).numpy(),
                "conf_w_t": orig["fusion_confidence"][1].squeeze(1).numpy(),
                "dp_text": dpt.numpy(), "dp_audio": dpa.numpy(),
                "tvd_text": tvt.numpy(), "tvd_audio": tva.numpy()}))
        checks.append({"seed": seed, "condition": tag,
                       "heads_disagree": float((p_t.argmax(1) != p_a.argmax(1)).float().mean()),
                       "fused_acc_vs_text_label": float((cls["fusion_learned_gate"].numpy() == y_text).mean()),
                       "text_head_acc": float((p_t.argmax(1).numpy() == y_text).mean()),
                       "audio_head_acc_vs_text_label": float((p_a.argmax(1).numpy() == y_text).mean())})
    return pd.concat(out, ignore_index=True), checks


def main():
    D = load_dataset()
    test = D["splits"]["test"]
    y = test["y"].numpy()
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    order = ref[(ref.seed == 0) & (ref.model == an06.PAIRS[0][0]) & (ref.method == "mean")].sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(an06.N_BOOT)]

    all_checks, all_rows, all_diffs, pair_rows = [], [], [], []
    for name, rs in DRAWS.items():
        ctrl, conf = make_pairs(D, rs)
        pair_rows.append(pd.DataFrame({"draw": name, "row": range(len(y)), "label": y,
                                       "control_donor": ctrl, "control_label": y[ctrl],
                                       "conflict_donor": conf, "conflict_label": y[conf]}))
        logs = {}
        for cond, idx in (("control", ctrl), ("conflict", conf)):
            logs[cond], chk = frames_for(D, test["audio"][torch.as_tensor(idx)], cond, y)
            all_checks += [{**c, "draw": name} for c in chk]
            print(f"{name} / {cond} done", flush=True)
        both = pd.concat(logs.values(), ignore_index=True)
        for model, signal, label in an06.PAIRS:
            res = {}
            for cond in ("control", "conflict"):
                s, e = an06.arrays(both, model, signal, cond)
                et = np.stack([(both[(both.model == model) & (both.method == cond) & (both.seed == k)]
                                .sort_values("row").assign(t=lambda d: d.tvd_text - d.tvd_audio).t.values)
                               for k in SEEDS])
                pt = np.array(an06.stats(s, e))
                pt_tvd = np.array(an06.stats(s, et))
                b = np.array([an06.stats(s[:, i], e[:, i]) for i in draws])
                b_tvd = np.array([an06.stats(s[:, i], et[:, i]) for i in draws])
                res[cond] = (pt, b, pt_tvd, b_tvd)
                lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
                for k, mname in enumerate(["agreement", "always_text_baseline", "gain", "spearman"]):
                    all_rows.append({"draw": name, "pair": label, "condition": cond, "metric": mname,
                                     "estimate": pt[k], "ci_low": lo[k], "ci_high": hi[k]})
            for mname, k, eff in (("gain", 2, "dp"), ("agreement", 0, "dp"), ("spearman", 3, "dp"), ("gain", 2, "tvd")):
                i_pt, i_b = (0, 1) if eff == "dp" else (2, 3)
                est = res["conflict"][i_pt][k] - res["control"][i_pt][k]
                d = res["conflict"][i_b][:, k] - res["control"][i_b][:, k]
                lo, hi = np.percentile(d, [2.5, 97.5])
                all_diffs.append({"draw": name, "pair": label, "effect": eff, "metric": mname,
                                  "conflict_minus_control": est, "ci_low": lo, "ci_high": hi,
                                  "excludes_zero": bool(lo > 0 or hi < 0)})
    pd.concat(pair_rows).to_csv(RESULTS / "conflict_pairs.csv", index=False)
    pd.DataFrame(all_rows).to_csv(RESULTS / "conflict_h2_metrics.csv", index=False)
    diffs = pd.DataFrame(all_diffs)
    diffs.to_csv(RESULTS / "conflict_h2_diff.csv", index=False)
    chk = pd.DataFrame(all_checks)
    chk.to_csv(RESULTS / "conflict_h2_checks.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    print("\nManipulation checks (mean over seeds):")
    print(chk.groupby(["draw", "condition"])[["heads_disagree", "fused_acc_vs_text_label", "text_head_acc",
                                              "audio_head_acc_vs_text_label"]].mean().to_string())
    m = pd.DataFrame(all_rows)
    g = m[(m.draw == "primary") & (m.metric.isin(["gain", "agreement", "always_text_baseline"]))]
    g = g.assign(v=g.apply(lambda r: f"{r.estimate:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}]", axis=1))
    print("\nPrimary draw, per condition:")
    print(g.pivot_table(index=["pair", "metric"], columns="condition", values="v", aggfunc="first").to_string())
    print("\nD = conflict minus control (primary draw unless stated):")
    print(diffs.to_string(index=False))


if __name__ == "__main__":
    main()
