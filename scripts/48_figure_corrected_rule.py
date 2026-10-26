"""Step 48: figure for the midterm report. Reads results/corrected_rule_swap_{M1,M5}.csv only (corrected decision rule, primary effect, gate recomputed).
Panel (a): how often a call (text or audio) matches the swap-test winner, always-text versus the gate weight, with 95% dialogue-bootstrap intervals.
Panel (b): the weight's gain over always-text with its 95% interval. Output: figures/fig7_gain_corrected_rule.{png,pdf}. No new computation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rows = {}
for tag in ("M1", "M5"):
    d = pd.read_csv(ROOT / "results" / f"corrected_rule_swap_{tag}.csv")
    r = d[(d.rule == "cor") & (d.effect == "primary_abs_of_mean")].iloc[0]
    rows[tag] = r
GREY, BLUE, RED = "#8c8c8c", "#1f5fa8", "#c0392b"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
labels = ["M1 balanced\n(58.8 F1)", "M5 text-heavy\n(63.6 F1)"]
w = 0.36
for i, tag in enumerate(("M1", "M5")):
    r = rows[tag]
    t, tl, th = r.p_text, r.p_text_lo, r.p_text_hi
    ag, al, ah = r.agreement, r.agreement_lo, r.agreement_hi
    t, tl, th = [max(x, 1 - y) if False else x for x, y in ((t, 0), (tl, 0), (th, 0))]
    maj = max(r.p_text, 1 - r.p_text)
    a.bar(i - w / 2, 100 * maj, w, color=GREY, label="always text" if i == 0 else None)
    a.bar(i + w / 2, 100 * ag, w, color=BLUE, label="gate weight" if i == 0 else None)
    a.errorbar(i + w / 2, 100 * ag, yerr=[[100 * (ag - al)], [100 * (ah - ag)]], color="black", lw=0.9, capsize=2)
    a.text(i - w / 2, 100 * maj + 1.2, f"{100 * maj:.1f}", ha="center", fontsize=7.5)
    a.text(i + w / 2, 100 * ah + 1.2, f"{100 * ag:.1f}", ha="center", fontsize=7.5, color=BLUE, fontweight="bold")
a.set_xticks([0, 1])
a.set_xticklabels(labels)
a.set_ylim(0, 112)
a.set_yticks([0, 25, 50, 75, 100])
a.set_ylabel("matches the swap-test winner (%)")
a.set_title("(a) Agreement with the swap test", fontsize=8.5, loc="left")
a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02))
for i, tag in enumerate(("M1", "M5")):
    r = rows[tag]
    g, lo, hi = 100 * r.gain_dev_constant, 100 * r.gain_dev_constant_lo, 100 * r.gain_dev_constant_hi
    col = BLUE if g > 5 else RED
    b.bar(i, g, 0.5, color=col)
    b.errorbar(i, g, yerr=[[g - lo], [hi - g]], color="black", lw=0.9, capsize=3)
    b.text(i, (hi + 1.5) if g > 0 else (lo - 3.2) if False else hi + 1.5, f"{g:+.1f}\n[{lo:.1f}, {hi:.1f}]", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=col)
b.axhline(0, color="black", lw=0.8)
b.set_xticks([0, 1])
b.set_xticklabels(labels)
b.set_ylim(-4, 31)
b.set_ylabel("gain over always-text (points)")
b.set_title("(b) What the weight adds", fontsize=8.5, loc="left")
plt.tight_layout(w_pad=2.0)
out = ROOT / "figures"
fig.savefig(out / "fig7_gain_corrected_rule.png", dpi=300)
fig.savefig(out / "fig7_gain_corrected_rule.pdf")
print({k: (round(float(v.agreement), 4), round(float(v.gain_dev_constant), 4)) for k, v in rows.items()})
