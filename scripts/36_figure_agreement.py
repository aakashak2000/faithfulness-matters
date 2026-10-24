"""Step 36: slide 2 figure. For four models (10 gates each), how often each guess matches the swap test:
the best trivial guess (always the modality that mattered more most often), the gate's weight read at 0.5, and the
corrected signal of step 34. Data: results/corrected_signal.csv (agreements and P(text mattered more)).
Usage: python scripts/36_figure_agreement.py     Output: figures/fig6_agreement_by_balance.png
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

an06 = importlib.import_module("06_analysis")
BLUE, ORANGE, INK, INK2, MUTED, SURFACE = an06.BLUE, an06.ORANGE, an06.INK, an06.INK2, an06.MUTED, an06.SURFACE
GRAYBAR = "#9a9992"


def main():
    r = pd.read_csv(RESULTS / "corrected_signal.csv").sort_values("p_text_mattered_more").reset_index(drop=True)
    triv = np.maximum(r.p_text_mattered_more, 1 - r.p_text_mattered_more) * 100
    wt, co = r.weight_agreement * 100, r.corrected_agreement * 100
    labels = [f"text mattered more\nin {p * 100:.0f}%" for p in r.p_text_mattered_more]
    names = {"M1 stronger audio": "stronger audio", "M0b headline recipe, v2 split": "first model",
             "M3 stronger text + audio": "stronger text + audio", "M2 stronger text": "stronger text"}
    sub = [names[m] for m in r.model]
    x = np.arange(len(r))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for off, vals, col, lab in ((-w, triv, GRAYBAR, "Best trivial guess"), (0, wt, BLUE, "Gate weight, read at 0.5"),
                                (w, co, ORANGE, "Corrected signal")):
        bars = ax.bar(x + off, vals, w * 0.92, color=col, label=lab, edgecolor=SURFACE, linewidth=2)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{l}\n({s})" for l, s in zip(labels, sub)], fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Matches the swap test (% of utterances)")
    ax.set_title("The weight beats a trivial guess only when text and audio are comparably strong", loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3, fontsize=8.5, frameon=False)
    fig.text(0.01, -0.08, "2,610 test utterances; 10 trained gates per model. Corrected signal = weight x how distinctive each model's vote is.",
             fontsize=7.2, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_agreement_by_balance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(pd.DataFrame({"model": sub, "trivial": triv.round(1), "weight": wt.round(1), "corrected": co.round(1)}).to_string(index=False))


if __name__ == "__main__":
    main()
