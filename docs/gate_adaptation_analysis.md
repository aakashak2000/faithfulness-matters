# Gate-adaptation analysis (2026-10-23), from existing row-level arrays; no new training or swaps
Script `scripts/44_gate_adaptation.py` (questions fixed in its header before running); outputs `results/gate_adaptation_M1.csv`, `results/gate_adaptation_M5.csv`, `results/gate_adaptation_crosstab.csv`; gate-input diagnostic `scripts/45_gate_input_diagnostic.py` -> `results/gate_input_diagnostic.csv`. "Frozen" = gate weight held at its original value w0 for every donor; "adaptive" = gate recomputed per donor (the primary experiment). Intervals: dialogue-cluster bootstrap (1,000 draws), gates averaged within a draw; they do not include head-training variance.

## 1. Frozen versus adaptive (gain = weight agreement minus the majority-winner rule for the SAME procedure)
| Model, rule | Text won, frozen / adaptive | Agreement, frozen / adaptive | Gain, frozen [CI] | Gain, adaptive [CI] | Paired change in gain, adaptive minus frozen [CI] |
|---|---|---|---|---|---|
| M1 raw | 53.3 / 54.2% | 78.3 / 75.4% | +25.0 [22.7, 27.5] | +21.1 [18.8, 23.6] | **-3.9 [-5.2, -2.5]** |
| M1 corrected | 52.2 / 52.8% | 75.2 / 73.6% | +23.0 [20.6, 25.4] | +20.8 [18.5, 23.1] | **-2.2 [-3.6, -0.7]** |
| M5 raw | 93.3 / 95.2% | 94.3 / 93.6% | +1.0 [0.1, 2.0] | -1.6 [-2.6, -0.6] | **-2.6 [-4.0, -1.4]** |
| M5 corrected | 91.6 / 92.9% | 93.0 / 91.9% | +1.3 [0.3, 2.4] | -0.9 [-1.9, -0.0] | **-2.3 [-3.7, -0.9]** |
Each procedure's majority-winner rule is "always text" in all eight cells. Paired change in agreement: M1 raw -2.9 [-3.7, -2.1], M1 corrected -1.6 [-2.6, -0.6], M5 raw -0.8 [-1.5, -0.0], M5 corrected -1.0 [-1.9, -0.2].

## 2. How often adaptation reverses the winner
| | Reversed share [CI] | Frozen text -> adaptive audio | Frozen audio -> adaptive text | On reversed rows the weight agrees with the frozen winner / adaptive winner |
|---|---|---|---|---|
| M1 raw | 6.1% [5.2, 6.9] | 2.5% | 3.5% | 73.7% / 26.3% |
| M1 corrected | 8.0% [7.0, 9.1] | 3.7% | 4.3% | 59.9% / 40.1% |
| M5 raw | 4.2% [3.4, 5.2] | 1.2% | 3.0% | 59.4% / 40.6% |
| M5 corrected | 5.5% [4.7, 6.4] | 2.2% | 3.4% | 59.5% / 40.5% |
Reversals go both ways but more often from frozen-audio to adaptive-text in all four cases; the weight's call usually matches the frozen winner on the reversed rows, which is why adaptation lowers agreement.

## 3. Signed decomposition: effect (adaptive) = effect (frozen) + gate term
| | Mean abs frozen effect, text / audio | Mean abs gate term, text / audio | Gate term opposes the frozen change, text / audio (share of rows) | Mean change of the text-minus-audio effect gap caused by the gate term [CI] |
|---|---|---|---|---|
| M1 raw | 0.131 / 0.099 | 0.014 / 0.038 | 53.7% / 47.4% | -0.0134 [-0.0162, -0.0109] |
| M1 corrected | 0.130 / 0.100 | 0.015 / 0.039 | 52.2% / 47.4% | -0.0137 [-0.0165, -0.0112] |
| M5 raw | 0.222 / 0.022 | 0.010 / 0.029 | 54.1% / 47.2% | -0.0115 [-0.0135, -0.0094] |
| M5 corrected | 0.227 / 0.025 | 0.011 / 0.028 | 59.1% / 47.2% | -0.0106 [-0.0128, -0.0083] |
Reading: the gate term is larger for audio than for text in every case. In M5 the audio gate term (0.029) is larger on average than the audio frozen effect (0.022): most of what an audio swap does in M5 comes through the gate re-weighting rather than through the audio head's own vote. The gate term, on average, reduces the text advantage in the effect gap by about 0.011 to 0.014 in all four settings (intervals exclude 0).

## 4. What these results support, and what they do not
- **Supported:** the choice between a frozen and an adaptive gate changes the weight's apparent gain by -2 to -4 points in all four settings (paired intervals exclude 0), flips the sign of M5's gain in the raw rule (+1.0 frozen, -1.6 adaptive), and reverses the winner on 4 to 8% of utterances.
- **Not supported: gate adaptation as the main cause of the weight's failure on the text-dominated model.** Even with the gate frozen, M5's gain is only +1.0 to +1.3 points, close to zero, because text wins about 92 to 93% of utterances with a frozen effect ten times the audio one (0.222 vs 0.022). Adaptation moves it further down by about 2.3 to 2.6 points. For M1, the weight's edge is already +23 to +25 points with a frozen gate, so adaptation lowers a large edge by 2 to 4 points.
- **Not tested:** whether adaptation is caused by what the gate reads (the gate sees the MiniLM embedding and the audio embedding, so swapping a modality changes the gate input), versus other properties of the gate. This is separable only with a controlled gate-input experiment (H4).

## 5. Gate-input mismatch, descriptive diagnostic (test set, 10 cached gates per model; `results/gate_input_diagnostic.csv`)
| Model | Text head reads | Spearman(w, confidence-rule weight) | Spearman(w, text confidence) | AUC of w for "only text head correct" vs "only audio head correct" |
|---|---|---|---|---|
| M0b | MiniLM (same as gate) | 0.16 | 0.31 | 0.592 |
| M1 | MiniLM (same as gate) | 0.07 | 0.25 | 0.588 |
| M2 | RoBERTa (gate sees MiniLM only) | 0.08 | 0.16 | 0.541 |
| M3 | RoBERTa (gate sees MiniLM only) | 0.01 | 0.16 | 0.555 |
All gates have weak reliability information (AUC 0.54 to 0.59, SD across gates 0.001 to 0.002). The two models whose gate cannot see the RoBERTa input have lower AUC (0.54, 0.56) and a lower correlation with the text head's confidence (0.16 vs 0.25 to 0.31) than the two where the gate reads the text head's own input. The comparison is confounded (text and audio strengths differ across models), so **this does not show that the mismatch matters**; it motivates the controlled test H4.

## 6. Does prior correction change agreement only where the predicted class changes? (cross-tabulation, adaptive gate, averaged over gates; shares of all test utterances)
| Model | Class changed | Winner changed | Agreement gained | Agreement lost | Unchanged |
|---|---|---|---|---|---|
| M1 | no | no | 0 | 0 | 81.8% |
| M1 | no | yes | 0.7% | 0.7% | 0 |
| M1 | yes | no | 0 | 0 | 9.1% |
| M1 | yes | yes | 3.0% | 4.8% | 0 |
| M5 | no | no | 0 | 0 | 68.9% |
| M5 | no | yes | 0.9% | 0.4% | 0 |
| M5 | yes | no | 0 | 0 | 24.7% |
| M5 | yes | yes | 1.5% | 3.6% | 0 |
- **The assertion is false as worded.** On rows where the predicted class does not change, the swap winner still changes for 1.3 to 1.4% of all rows, and agreement changes in those rows (M1: +0.7 gained, -0.7 lost, net about 0; M5: +0.9 gained, -0.4 lost, net +0.45 point).
- **The net loss is concentrated in class-changed rows:** M1 net -1.8 points (+3.0 - 4.8), M5 net -2.1 points (+1.5 - 3.6). Total net change: M1 -1.8, M5 -1.6, equal to the paired differences reported earlier.
- **Why same-class rows change:** the correction multiplies each donor's fused probability vector by the prior and renormalises, so the probability of the original class changes by different amounts under different donors, and the text and audio effects (averages over donors) can reorder even when the argmax of the unswapped input is unchanged.
