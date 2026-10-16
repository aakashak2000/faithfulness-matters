"""Step 6: confidence intervals and figures for the midterm.

Bootstrap: resample test DIALOGUES (1,000 times, cluster bootstrap); each resample's
metric is averaged over the 3 seeds. Reports 95% intervals for agreement, the
always-text baseline, gain = agreement - baseline, and Spearman, plus the counts of
tied / near-zero utterances behind each agreement number.

Figures (figures/):
  fig1_baselines.png              weighted and macro F1 per model
  fig2_agreement_vs_baseline.png  each signal's agreement vs the always-text baseline
  fig3_gate_calibration.png       when the gate says "audio", does audio actually matter more?
  fig4_gate_scatter.png           gate signal difference vs effect difference

Usage: python scripts/06_analysis.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import FIGURES, RESULTS

N_BOOT = 1000
PAIRS = [("fusion_learned_gate", "gate_w_t", "Learned gate ← gate weight"),
         ("fusion_confidence", "conf_w_t", "Confidence fusion ← confidence"),
         ("fusion_learned_gate", "conf_w_t", "Learned gate ← confidence"),
         ("fusion_average", "conf_w_t", "Average fusion ← confidence")]
METHODS = {"mean": "Mean-embedding replacement", "blank": "Empty text / silent audio"}

# reference palette (validated: tools/dataviz validate_palette, light mode)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.axisbelow": True, "axes.titleweight": "bold", "legend.frameon": False,
})


def arrays(log, model, signal, method):
    d = log[(log.model == model) & (log.method == method)].sort_values(["seed", "row"])
    seeds = sorted(d.seed.unique())
    s = np.stack([2 * d[d.seed == k][signal].values - 1 for k in seeds])
    e = np.stack([d[d.seed == k].dp_text.abs().values - d[d.seed == k].dp_audio.abs().values
                  for k in seeds])
    return s, e


def stats(s, e):
    """Metrics averaged over seeds. s, e: (seeds, n)."""
    agree, base, rho = [], [], []
    for sk, ek in zip(s, e):
        v = (sk != 0) & (ek != 0)
        agree.append((np.sign(sk[v]) == np.sign(ek[v])).mean())
        base.append((ek[v] > 0).mean())
        rho.append(np.corrcoef(rankdata(sk), rankdata(ek))[0, 1])
    a, b = np.mean(agree), np.mean(base)
    return a, b, a - b, np.mean(rho)


def bootstrap(log):
    """Cluster bootstrap: resample whole DIALOGUES, since utterances in one
    conversation are not independent (and shared-audio clips sit inside dialogues)."""
    rng = np.random.default_rng(0)
    order = log[(log.seed == log.seed.min()) & (log.model == PAIRS[0][0]) & (log.method == "mean")]
    order = order.sort_values("row")
    groups = [np.flatnonzero(order.dialogue_id.values == g) for g in np.unique(order.dialogue_id)]
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(N_BOOT)]
    rows, counts = [], []
    for model, signal, label in PAIRS:
        for method in METHODS:
            s, e = arrays(log, model, signal, method)
            point = stats(s, e)
            boots = np.array([stats(s[:, idx], e[:, idx]) for idx in draws])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            for k, name in enumerate(["agreement", "always_text_baseline", "gain", "spearman"]):
                rows.append({"pair": label, "model": model, "signal": signal, "method": method,
                             "metric": name, "estimate": point[k], "ci_low": lo[k], "ci_high": hi[k],
                             "bootstrap_fraction_gain_le_0": float((boots[:, 2] <= 0).mean()) if name == "gain" else np.nan})
            counts.append({"pair": label, "method": method, "n_total_per_seed": s.shape[1],
                           "n_signal_tied": int((s == 0).sum() / s.shape[0]),
                           "n_effect_tied": int((e == 0).sum() / e.shape[0]),
                           "n_effect_below_0.02": int((np.abs(e) < 0.02).sum() / e.shape[0])})
    print(f"(cluster bootstrap over {len(groups)} dialogues)")
    return pd.DataFrame(rows), pd.DataFrame(counts)


def fig_baselines():
    s = pd.read_csv(RESULTS / "baselines_per_seed.csv")
    order = ["text_only", "audio_only", "fusion_average", "fusion_confidence", "fusion_learned_gate"]
    names = ["Text only", "Audio only", "Average\nfusion", "Confidence\nfusion", "Learned\ngate"]
    g = s.groupby("model")[["weighted_f1", "macro_f1"]].agg(["mean", "std"]).loc[order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for off, col, color, lab in [(-0.19, "weighted_f1", BLUE, "Weighted F1"),
                                 (0.19, "macro_f1", ORANGE, "Macro F1")]:
        bars = ax.bar(x + off, g[(col, "mean")], 0.36, yerr=g[(col, "std")], color=color,
                      label=lab, error_kw={"ecolor": INK2, "elinewidth": 1}, edgecolor=SURFACE, linewidth=2)
        for b, sd in zip(bars, g[(col, "std")]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + sd + 0.012, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("Test F1 (mean of 3 seeds)")
    ax.set_title("Test F1 by model (mean and std over 3 seeds)", loc="left")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_baselines.png", dpi=200)
    plt.close(fig)


def fig_agreement(ci):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    labels = [p[2] for p in PAIRS]
    y = np.arange(len(labels))[::-1]
    for ax, (method, title) in zip(axes, METHODS.items()):
        for yi, label in zip(y, labels):
            r = ci[(ci.pair == label) & (ci.method == method)].set_index("metric")
            a, b = r.loc["agreement"], r.loc["always_text_baseline"]
            ax.plot([b.estimate, a.estimate], [yi, yi], color=GRID, linewidth=2, zorder=1)
            ax.errorbar(a.estimate, yi, xerr=[[a.estimate - a.ci_low], [a.ci_high - a.estimate]],
                        fmt="o", color=BLUE, ms=7, capsize=3, zorder=3,
                        label="Signal's agreement" if yi == y[0] else None)
            ax.plot(b.estimate, yi, marker="|", color=INK2, ms=14, mew=2, linestyle="none", zorder=2,
                    label='"Always text" baseline' if yi == y[0] else None)
        ax.set_title(title, loc="left")
        lo = min(ci[ci.metric.isin(["agreement", "always_text_baseline"])].ci_low.min(),
                 ci[ci.metric == "always_text_baseline"].estimate.min())
        ax.set_xlim(max(0.0, np.floor((lo - 0.03) * 20) / 20), 1.0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncols=2)
    fig.tight_layout()
    fig.supxlabel("Share of utterances where the signal picks the modality whose removal changed the prediction more",
                  fontsize=9, color=INK2, y=-0.04)
    fig.savefig(FIGURES / "fig2_agreement_vs_baseline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(log):
    d = log[log.model == "fusion_learned_gate"].copy()
    d["text_bigger"] = d.dp_text.abs() > d.dp_audio.abs()
    bins = np.linspace(0, 1, 11)
    d["bin"] = pd.cut(d.gate_w_t, bins, include_lowest=True)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    rows = []
    for method, color in [("mean", BLUE), ("blank", ORANGE)]:
        g = d[d.method == method].groupby("bin", observed=False).text_bigger.agg(["mean", "size"])
        ok = g["size"] >= 30
        ax.plot(centers[ok.values], g["mean"][ok].values, marker="o", ms=6, lw=2, color=color,
                label=METHODS[method])
        for c, (m, n) in zip(centers, g.itertuples(index=False)):
            rows.append({"method": method, "gate_w_t_bin_center": c, "share_text_bigger": m, "n": n})
    ax.axvspan(0, 0.5, color=GRID, alpha=0.45, lw=0)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(0.02, 0.06, 'gate says "audio\nmattered more"', color=INK2, fontsize=8, transform=ax.transAxes)
    ax.text(0.99, 0.03, "bins with under 30 utterance-rows not drawn (0.0 to 0.1: n=4)", color=MUTED,
            fontsize=7, ha="right", transform=ax.transAxes)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Share where removing text\nchanged the prediction more")
    ax.set_title("Does the gate's text weight predict which modality matters?", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncols=2)
    w = d[(d.method == "mean")].gate_w_t
    axh.hist(w, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=2)
    axh.set_ylabel("Utterances\n(3 seeds)")
    axh.set_xlabel("Learned gate's text weight on the original input")
    axh.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_gate_calibration.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS / "gate_calibration.csv", index=False)


def fig_scatter(log):
    d = log[(log.model == "fusion_learned_gate") & (log.method == "mean") & (log.seed == 0)]
    s = 2 * d.gate_w_t - 1
    e = d.dp_text.abs() - d.dp_audio.abs()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.scatter(s, e, s=9, color=BLUE, alpha=0.35, linewidths=0)
    ax.set_xlabel("Gate signal: text weight − audio weight")
    ax.set_ylabel("Measured effect: |Δp| text removed − |Δp| audio removed")
    ax.set_title("Learned gate, mean replacement, seed 0 (2,610 utterances)", loc="left")
    for (x, yv, txt) in [(0.95, 0.95, "both say text"), (0.05, 0.05, "both say audio"),
                         (0.05, 0.95, "gate: audio\neffect: text")]:
        ax.text(x, yv, txt, transform=ax.transAxes, ha="right" if x > 0.5 else "left",
                va="top" if yv > 0.5 else "bottom", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_gate_scatter.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    log = pd.read_csv(RESULTS / "interventions_log.csv")
    print(f"bootstrapping ({N_BOOT} resamples)...")
    ci, counts = bootstrap(log)
    ci.to_csv(RESULTS / "bootstrap_ci.csv", index=False)
    counts.to_csv(RESULTS / "agreement_counts.csv", index=False)
    fig_baselines()
    fig_agreement(ci)
    fig_calibration(log)
    fig_scatter(log)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    view = ci.assign(value=lambda t: t.apply(
        lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1))
    print("\n=== 95% cluster-bootstrap intervals (dialogues resampled, averaged over 3 seeds; conditional on the trained models) ===")
    print(view.pivot_table(index=["pair", "method"], columns="metric", values="value", aggfunc="first")
          [["agreement", "always_text_baseline", "gain", "spearman"]])
    print("\nShare of bootstrap resamples with gain <= 0 (a tail fraction, not a formal p-value):")
    print(ci[ci.metric == "gain"].set_index(["pair", "method"])["bootstrap_fraction_gain_le_0"].round(3))
    print("\n=== Denominators per seed (agreement excludes exact ties) ===")
    print(counts.set_index(["pair", "method"]))
    cal = pd.read_csv(RESULTS / "gate_calibration.csv")
    print("\n=== Gate calibration: share where text removal mattered more, by gate weight ===")
    print(cal.pivot_table(index="gate_w_t_bin_center", columns="method",
                          values=["share_text_bigger", "n"]).round(3))
    print(f"\nSaved results/bootstrap_ci.csv, results/gate_calibration.csv and 4 figures in {FIGURES}")


if __name__ == "__main__":
    main()
