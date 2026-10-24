"""Step 26: the quality-ladder figure for slide 2.

x = audio-only model strength (weighted-F1, from results/quality_ladder.csv); y = share of the 2,610
test utterances where a guess about "which modality mattered more" matches the swap test.
Series: the gate's weight (read at 0.5, 95% dialogue-cluster interval) and the best trivial guess
(always the majority modality, = agreement minus the gain over the majority). Form: emphasis (accent
colour for the weight, neutral ink for the trivial guess), shaded gap = the weight's edge.
Rungs R0 to R4 use the headline gate input; the step 23 row (gate fed 8,192 features) is excluded.

Usage: python scripts/26_figure_ladder.py     Output: figures/fig5_quality_ladder.png
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
BLUE, INK, INK2, MUTED, SURFACE, GRID = an06.BLUE, an06.INK, an06.INK2, an06.MUTED, an06.SURFACE, an06.GRID


def main():
    r = pd.read_csv(RESULTS / "quality_ladder.csv")
    r = r[r.rung.str.match(r"R[0-4] ") & ~r.rung.str.contains("steps 23")].copy()
    x = r.audio_only_wF1.values * 100
    ag, lo, hi = r.agreement.values * 100, r.agreement_lo.values * 100, r.agreement_hi.values * 100
    triv = (r.agreement - r.gain_over_majority).values * 100
    gain = r.gain_over_majority.values * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(x, triv, ag, color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(x, triv, color=INK2, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, zorder=3)
    ax.errorbar(x, ag, yerr=[ag - lo, hi - ag], color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                capsize=3, elinewidth=1.4, zorder=4)
    lo_e, hi_e = gain[1:].min(), gain[1:].max()
    ax.text(x[1:].mean() + 0.3, (ag[1:].mean() + triv[1:].mean()) / 2, f"weight's edge:\n+{lo_e:.0f} to +{hi_e:.0f} points",
            ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(x[-1] + 0.5, ag[-1], "Weight read at 0.5", color=BLUE, fontsize=9, va="center", fontweight="bold")
    ax.text(x[-1] + 0.5, triv[-1], "Best trivial guess", color=INK2, fontsize=9, va="center", fontweight="bold")
    ax.annotate(f"headline model\nedge +{gain[0]:.0f} points", (x[0], ag[0] + 1.2), (x[0] + 0.2, 87), fontsize=8.5,
                color=INK2, ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(35.5, 52)
    ax.set_ylim(40, 92)
    ax.set_xlabel("Audio model strength (audio-only weighted F1)")
    ax.set_ylabel("Share of utterances where the guess\nmatches the swap test (%)")
    ax.set_title("Weight vs trivial guess: +2 points with the weaker audio model, +23 to +26 with stronger ones", loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, "Shaded gap = the weight's edge, in percentage points. Trivial guess = always the modality that mattered more most often.\n"
             "Bars: 95% interval (dialogue-level bootstrap, 3 trained models). Same text model and gate input for every point.",
             fontsize=7.2, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_quality_ladder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(pd.DataFrame({"audio_F1": x.round(1), "weight_agreement": ag.round(1), "trivial": triv.round(1), "edge": gain.round(1)}).to_string(index=False))


if __name__ == "__main__":
    main()
