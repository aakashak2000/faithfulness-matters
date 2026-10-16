"""Step 17: figures on the PRIMARY intervention (full-pool resampling), plus mean and blank as
sensitivity panels. 06_analysis.py's figures (mean and blank only) are left as they were.

  fig2b_agreement_primary.png   agreement vs always-text baseline, 3 methods
  fig3b_gate_calibration_primary.png   share where text mattered more, by gate weight
  fig4b_gate_scatter_primary.png       gate signal vs measured effect, seed 0
Numbers come from results/resampling_full_faithfulness.csv (agreement, baseline, CI) and
results/resampling_full_log.csv / interventions_log.csv (calibration, scatter). The calibration
table is also saved to results/gate_calibration_primary.csv.

Usage: python scripts/17_figures_primary.py
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, RESULTS  # noqa: E402

an06 = importlib.import_module("06_analysis")  # palette and rcParams
BLUE, ORANGE, INK2, MUTED, GRID, AXIS, SURFACE = (an06.BLUE, an06.ORANGE, an06.INK2, an06.MUTED,
                                                   an06.GRID, an06.AXIS, an06.SURFACE)
PANELS = [("resample_full", "Full-pool resampling (primary)"),
          ("mean", "Mean replacement"), ("blank", "Empty text / silent audio")]
GREEN = "#2f8f5b"


def fig_agreement(ci):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
    labels = [p[2] for p in an06.PAIRS]
    y = np.arange(len(labels))[::-1]
    for ax, (method, title) in zip(axes, PANELS):
        for yi, label in zip(y, labels):
            r = ci[(ci.pair == label) & (ci.method == method)].set_index("metric")
            a, b = r.loc["agreement"], r.loc["always_text_baseline"]
            ax.plot([b.estimate, a.estimate], [yi, yi], color=GRID, linewidth=2, zorder=1)
            ax.errorbar(a.estimate, yi, xerr=[[a.estimate - a.ci_low], [a.ci_high - a.estimate]],
                        fmt="o", color=BLUE, ms=7, capsize=3, zorder=3,
                        label="Signal's agreement" if yi == y[0] else None)
            ax.plot(b.estimate, yi, marker="|", color=INK2, ms=14, mew=2, linestyle="none", zorder=2,
                    label='"Always text" baseline' if yi == y[0] else None)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0.6, 1.0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncols=2)
    fig.tight_layout()
    fig.supxlabel("Share of utterances where the signal picks the modality whose removal changed the prediction more",
                  fontsize=9, color=INK2, y=-0.04)
    fig.savefig(FIGURES / "fig2b_agreement_primary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def calibration(log, method):
    d = log[(log.model == "fusion_learned_gate") & (log.method == method)].copy()
    d["text_bigger"] = d.dp_text.abs() > d.dp_audio.abs()
    bins = np.linspace(0, 1, 11)
    d["bin"] = pd.cut(d.gate_w_t, bins, include_lowest=True)
    g = d.groupby("bin", observed=False).text_bigger.agg(["mean", "size"])
    return (bins[:-1] + bins[1:]) / 2, g


def fig_calibration(ref, full):
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    rows = []
    for method, log, color, lw, ls, lab in [("resample_full", full, BLUE, 2.4, "-", "Full-pool resampling (primary)"),
                                            ("mean", ref, MUTED, 1.4, "--", "Mean"),
                                            ("blank", ref, ORANGE, 1.4, "--", "Blank")]:
        centers, g = calibration(log, method)
        ok = g["size"] >= 30
        ax.plot(centers[ok.values], g["mean"][ok].values, marker="o", ms=5 if lw > 2 else 3, lw=lw, ls=ls,
                color=color, label=lab)
        for c, (m, n) in zip(centers, g.itertuples(index=False)):
            rows.append({"method": method, "gate_w_t_bin_center": c, "share_text_bigger": m, "n": n})
    ax.axvspan(0, 0.5, color=GRID, alpha=0.45, lw=0)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.02, 0.06, 'gate says "audio\nmattered more"', color=INK2, fontsize=8, transform=ax.transAxes)
    ax.text(0.99, 0.03, "bins with under 30 utterance-rows not drawn", color=MUTED, fontsize=7,
            ha="right", transform=ax.transAxes)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Share where removing text\nchanged the prediction more")
    ax.set_title("Does the gate's text weight predict which modality matters?", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncols=3, fontsize=8)
    w = full[(full.model == "fusion_learned_gate")].gate_w_t
    axh.hist(w, bins=np.linspace(0, 1, 11), color=BLUE, edgecolor=SURFACE, linewidth=2)
    axh.set_ylabel("Utterances\n(3 seeds)")
    axh.set_xlabel("Learned gate's text weight on the original input")
    axh.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3b_gate_calibration_primary.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS / "gate_calibration_primary.csv", index=False)
    return pd.DataFrame(rows)


def fig_scatter(full):
    d = full[(full.model == "fusion_learned_gate") & (full.seed == 0)]
    s = 2 * d.gate_w_t - 1
    e = d.dp_text.abs() - d.dp_audio.abs()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.scatter(s, e, s=9, color=BLUE, alpha=0.35, linewidths=0)
    ax.set_xlabel("Gate signal: text weight − audio weight")
    ax.set_ylabel("Measured effect: |Δp| text removed − |Δp| audio removed")
    ax.set_title("Learned gate, full-pool resampling, seed 0 (2,610 utterances)", loc="left", fontsize=10)
    for (x, yv, txt) in [(0.95, 0.95, "both say text"), (0.05, 0.05, "both say audio"),
                         (0.05, 0.95, "gate: audio\neffect: text")]:
        ax.text(x, yv, txt, transform=ax.transAxes, ha="right" if x > 0.5 else "left",
                va="top" if yv > 0.5 else "bottom", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4b_gate_scatter_primary.png", dpi=200)
    plt.close(fig)


def main():
    ci = pd.read_csv(RESULTS / "resampling_full_faithfulness.csv")
    ref = pd.read_csv(RESULTS / "interventions_log.csv")
    full = pd.read_csv(RESULTS / "resampling_full_log.csv")
    fig_agreement(ci)
    cal = fig_calibration(ref, full)
    fig_scatter(full)
    pd.set_option("display.precision", 3)
    print(cal[cal.method == "resample_full"].to_string(index=False))
    print("saved fig2b, fig3b, fig4b and results/gate_calibration_primary.csv")


if __name__ == "__main__":
    main()
