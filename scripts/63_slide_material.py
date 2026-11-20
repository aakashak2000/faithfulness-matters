"""Step 63: material for the final presentation. Selects the example clip by the rule in docs/slides/final/example_clip_rule.md (written
before any clip was looked at), saves the few slide numbers that are not already in a results file, and draws three charts from
committed results. No new experiment.

Usage:  python scripts/63_slide_material.py
Output: results/slide_numbers.csv, figures/final/fig{1,2,3}_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import load_rescoring_inputs  # noqa: E402
from common import DATA, FIGURES, RESULTS, load_dataset  # noqa: E402

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated pair (dataviz validator: all checks pass)
OUT = FIGURES / "final"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=13, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def bar_pair(ax, labels, a, b, a_ci, b_ci, a_name, b_name, fmt, ylim, ref=None):
    x = np.arange(len(labels))
    wdt = 0.36
    for off, vals, ci, col, name in ((-wdt / 2 - 0.01, a, a_ci, BLUE, a_name), (wdt / 2 + 0.01, b, b_ci, ORANGE, b_name)):
        ax.bar(x + off, vals, wdt, color=col, label=name, edgecolor=SURF, linewidth=2)
        if ci is not None:
            ax.errorbar(x + off, vals, yerr=[vals - ci[0], ci[1] - vals], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
        tops = vals if ci is None else np.maximum(vals, ci[1])
        for xi, v, t in zip(x + off, vals, tops):
            ax.text(xi, t + (ylim[1] - ylim[0]) * 0.02, fmt(v), ha="center", va="bottom", fontsize=14, color=INK)
    if ref is not None:
        ax.axhline(ref[0], color=MUTED, linewidth=1.2, linestyle="--")
        ax.text(-0.55, ref[0] + (ylim[1] - ylim[0]) * 0.01, ref[1], ha="left", va="bottom", fontsize=12, color=MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=13, loc="upper left", labelcolor=INK2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- example clip (rule fixed beforehand)
    D = load_dataset()
    emo = D["emotions"]
    meta = D["meta"]["test"]
    X = load_rescoring_inputs("M1")["test"]
    A = np.load(DATA / "corrected_rule_arrays_M1.npz")
    pt, pa, w, y = X["pt"], X["pa"], X["w"][0], X["y"]
    e = A["e_cor_real"][0]
    nwords = meta["utterance"].str.split().str.len().values
    ok = (~meta["audio_shared"].values.astype(bool)) & (pt.argmax(1) != pa.argmax(1)) & (w >= 0.6) & (e < 0) & (nwords >= 3) & (nwords <= 15)
    i = int(np.flatnonzero(ok)[0])
    ex = {"row": i, "speaker": meta["speaker"].iloc[i], "utterance": meta["utterance"].iloc[i], "gold": emo[y[i]],
          "text_head_top": emo[pt[i].argmax()], "text_head_prob": float(pt[i].max()), "audio_head_top": emo[pa[i].argmax()],
          "audio_head_prob": float(pa[i].max()), "gate_text_weight": float(w[i]), "fused_prediction": emo[int(A["c_cor"][0][i])],
          "text_swap_effect": float(abs(A["cor_real_text_mean"][0][i])), "audio_swap_effect": float(abs(A["cor_real_audio_mean"][0][i])),
          "n_qualifying_rows": int(ok.sum())}
    for k, v in ex.items():
        rows.append({"item": "example_clip", "key": k, "value": v})
        print(f"EXAMPLE {k}: {v}", flush=True)

    # ---- mean text weight per phase 2 arm (not saved elsewhere)
    Z = np.load(DATA / "controlled_series_arrays.npz")
    arms = {"Control": ["C0"], "Over-\nconfident\nvoice": ["C1_seed0", "C1_seed1", "C1_seed2"], "Sharper\nvoice": ["S1"],
            "Flatter\nvoice": ["S2"], "30%\nwrong\naudio": ["P"]}
    refw = {"Control": None}
    meanw = {}
    for name, runs in arms.items():
        meanw[name] = float(np.concatenate([Z[f"{r}__w"] for r in runs]).mean())
        rows.append({"item": "phase2_mean_text_weight", "key": name.replace("-\n", "").replace("\n", " "), "value": meanw[name]})
    rows.append({"item": "phase2_mean_text_weight", "key": "S0 (reference for sharper/flatter)", "value": float(Z["S0__w"].mean())})
    pd.DataFrame(rows).to_csv(RESULTS / "slide_numbers.csv", index=False)

    # ---- fig 1: real vs shuffled dial (raw rule, dial held fixed during swaps)
    n2 = pd.read_csv(RESULTS / "decomposition_null_v2.csv").set_index(["model", "metric"])
    g = lambda m, k: n2.loc[(m, k)]
    real = np.array([g(m, "agreement_actual").value for m in ("M1", "M5")]) * 100
    shuf = np.array([g(m, "agreement_null_v2").value for m in ("M1", "M5")]) * 100
    rci = np.array([[g(m, "agreement_actual").lo for m in ("M1", "M5")], [g(m, "agreement_actual").hi for m in ("M1", "M5")]]) * 100
    sci = np.array([[g(m, "agreement_null_v2").lo for m in ("M1", "M5")], [g(m, "agreement_null_v2").hi for m in ("M1", "M5")]]) * 100
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURF)
    style(ax)
    bar_pair(ax, ["Balanced model", "Words-heavy model"], real, shuf, rci, sci, "Learned dial", "Shuffled dial", lambda v: f"{v:.0f}%", (0, 115))
    ax.set_ylabel("Agrees with swap test (%)", fontsize=13, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_shuffled_dial.png", dpi=200, facecolor=SURF)

    # ---- fig 2: winner flips, similar-clip donors vs random-50 noise floor
    v4 = pd.read_csv(RESULTS / "reference_sensitivity_v4.csv").set_index(["model", "metric"])
    h = lambda m, k: v4.loc[(m, k)]
    nn = np.array([h(m, "flip_full_nn50_recomputed_gate").value for m in ("M1", "M5")]) * 100
    nf = np.array([h(m, "noise_full_global50_recomputed_gate").value for m in ("M1", "M5")]) * 100
    nnci = np.array([[h(m, "flip_full_nn50_recomputed_gate").lo for m in ("M1", "M5")], [h(m, "flip_full_nn50_recomputed_gate").hi for m in ("M1", "M5")]]) * 100
    nfci = np.array([[h(m, "noise_full_global50_recomputed_gate").lo for m in ("M1", "M5")], [h(m, "noise_full_global50_recomputed_gate").hi for m in ("M1", "M5")]]) * 100
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURF)
    style(ax)
    bar_pair(ax, ["Balanced model", "Words-heavy model"], nn, nf, nnci, nfci, "Swap in similar clips", "Swap in 50 random clips (noise)",
             lambda v: f"{v:.0f}%", (0, 24))
    ax.set_ylabel("Answers that flip vs all clips (%)", fontsize=13, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_reference.png", dpi=200, facecolor=SURF)

    # ---- fig 3: overall level (left) vs per-clip ranking (right)
    ph = pd.read_csv(RESULTS / "controlled_series_posthoc.csv").set_index("metric")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURF, gridspec_kw={"width_ratios": [1.35, 1]})
    style(a1)
    names = list(arms)
    vals = np.array([meanw[k] for k in names]) * 100
    a1.bar(range(len(names)), vals, 0.6, color=[MUTED] + [BLUE] * (len(names) - 1), edgecolor=SURF, linewidth=2)
    for xi, v in enumerate(vals):
        a1.text(xi, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=14, color=INK)
    a1.set_xticks(range(len(names)), names, fontsize=12)
    a1.set_ylim(0, 100)
    a1.set_ylabel("Average dial setting: share on words (%)", fontsize=12, color=INK2)
    a1.set_title("Overall: the dial turns the voice down", fontsize=14, color=INK, loc="left")
    style(a2)
    labs = ["Which listener\nis right", "Which clips have\nwrong audio"]
    dial = np.array([ph.loc["gate_auroc_which_head_correct_C0", "value"], ph.loc["gate_auroc_planted_P", "value"]]) * 100
    prb = np.array([ph.loc["probe_auroc_which_head_correct_C0", "value"], ph.loc["probe_auroc_planted_P", "value"]]) * 100
    dci = np.array([[ph.loc["gate_auroc_which_head_correct_C0", "lo"], ph.loc["gate_auroc_planted_P", "lo"]],
                    [ph.loc["gate_auroc_which_head_correct_C0", "hi"], ph.loc["gate_auroc_planted_P", "hi"]]]) * 100
    pci = np.array([[ph.loc["probe_auroc_which_head_correct_C0", "lo"], ph.loc["probe_auroc_planted_P", "lo"]],
                    [ph.loc["probe_auroc_which_head_correct_C0", "hi"], ph.loc["probe_auroc_planted_P", "hi"]]]) * 100
    bar_pair(a2, labs, dial, prb, dci, pci, "Dial", "Detector trained for it", lambda v: f"{v:.0f}", (40, 82), ref=(50, "coin flip"))
    a2.legend(frameon=False, fontsize=13, loc="upper right", labelcolor=INK2)
    a2.set_ylabel("Ranking score (50 = coin flip)", fontsize=12, color=INK2)
    a2.set_title("Per clip: weak signal, mostly used", fontsize=14, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_what_dial_learns.png", dpi=200, facecolor=SURF)
    print("wrote results/slide_numbers.csv and figures/final/fig1-3", flush=True)


if __name__ == "__main__":
    main()
