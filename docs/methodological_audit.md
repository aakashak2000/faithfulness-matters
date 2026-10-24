# Methodological audit (2026-10-21)
"Verified" = computed by a script in this repo and stored in results/. "Interpretation" = our reading, not tested.

## 1. The corrected signal
- Formula (scripts/34_corrected_signal.py): s = w |p_t[c] - m_t[c]| - (1 - w) |p_a[c] - m_a[c]|. w = gate weight on the original input, c = class predicted by the original fused probability, p_t and p_a = the two models' original probabilities, m = mean of the model's probability over the donor pool. "Distinctive" = how far a model's vote for c is from the typical donor vote. It uses the donor MEAN only, not the full distribution. (verified, code)
- Identity (scripts/38_frozen_gate.py, 39; results/frozen_gate_audit.csv): the swap with the gate frozen at w, computed by looping over donors with no formula, equals w (m_t[c] - p_t[c]) and (1 - w)(m_a[c] - p_a[c]) to 5e-7 (float32) in all 5 models. The sign of s equals the sign of the frozen-gate effect difference in 100% of utterances; the largest effect-size difference is 0.0025, from step 34 using the whole pool while the swap excludes byte-identical donors (the masked version differs by 0). The primary effect is |mean over donors of fused[c] - original fused[c]| (donor-averaged vector first, then absolute value), which is what the identity uses. (verified)

## 2. Frozen-gate experiment (same utterances, same donors; 10 gates for M0b to M3, 3 for M5)
| Model | real vs frozen agreement | real vs s agreement | frozen vs s agreement | Spearman real vs s | weight vs real | share reversed by gate |
|---|---|---|---|---|---|---|
| M0b headline recipe | 93.5% | 93.5% | 100% | 0.960 | 78.5% | 6.5% |
| M1 balanced | 93.9% | 93.9% | 100% | 0.974 | 75.4% | 6.1% |
| M2 stronger text | 98.8% | 98.8% | 100% | 0.983 | 96.7% | 1.2% |
| M3 stronger text + audio | 98.1% | 98.1% | 100% | 0.970 | 92.7% | 1.9% |
| M5 63.6 model (context) | 95.8% | 95.8% | 100% | 0.962 | 93.6% | 4.2% |
My recomputed-gate effect equals the pipeline's rs08.exact_average effect to 2.4e-7 (validation). (verified)

## 3. Residual disagreement
- Real effect = frozen effect + a gate term, where gate term = mean over donors of (w_d - w)(P_d[c] - p_other[c]); exact because the fused probability is linear in the weight. Since s equals the frozen effect, ALL disagreement between s and the real swap is the gate term reversing the winner. It happens in 6.5%, 6.1%, 1.2%, 1.9% and 4.2% of utterances (M0b, M1, M2, M3, M5; 95% intervals in results/audit_bootstrap.csv, e.g. M1 [5.2, 6.9]). (verified)
- Those utterances are near-ties: median frozen margin 0.013 to 0.044 against 0.11 to 0.23 over all utterances. In 58% to 72% of them the frozen effect says audio won (except M2: 48%). The audio gate term is larger than the text gate term (median 0.017 to 0.025 vs 0.002 to 0.008). Adding only the audio gate term raises agreement with the real swap more than adding only the text term. (verified)

## 4. Robustness of the main result (weight's gain over the majority-modality guess, points; 95% interval from the dialogue bootstrap)
| Check | M0 / M0b | M1 balanced | M2 text-strong | M3 | M5 63.6 |
|---|---|---|---|---|---|
| Random-donor swap, primary (harness_models.csv, swap_with_context.csv) | +1.8 [-0.4, 4.0] | +21.1 [18.8, 23.6] | -0.4 [-0.7, -0.2] | -0.1 [-0.6, 0.4] | -1.6 [-2.6, -0.6] |
| TVD effect (harness_alt_effects.csv) | +7.2 | +29.2 | -0.5 | 0.0 | not run |
| Class flips | +5.4 | +23.5 | -0.3 | -0.1 | not run |
| Signed drop | -0.3 | +19.8 | -0.6 | -0.5 | not run |
| Mean replacement (resampling_full_faithfulness.csv, M0 only) | +1.4 [-0.7, 3.6] | not run | not run | not run | not run |
| Blank replacement (M0 only) | -3.2 [-5.4, -1.0] | not run | not run | not run | not run |
| Cutoff fitted on dev (harness_models.csv) | +4.8 | +21.9 | -0.2 | +0.1 | not run |
All available checks agree: a clear edge only in M1; none or slightly negative in M2, M3, M5. Weaker or missing evidence: mean and blank replacement were run only on the original headline model; the alternative effect measures were not run on M5; M0b/M0 show an edge under TVD and flips (+5 to +7) but none under signed drop and the primary effect, so for the headline model the edge depends on the metric. No ground-truth-class probability effect is implemented. (verified)

## 5. Is modality dominance isolated?
| Model | Text encoder | Audio | Context | Gate weighted-F1 (dev prior correction) | Text mattered more | Weight gain over majority |
|---|---|---|---|---|---|---|
| M0b | MiniLM | last layer | no | 57.5 | 76.7% | +1.8 |
| M1 | MiniLM | all-layer WavLM | no | 58.8 | 54.2% | +21.1 |
| M2 | RoBERTa-large all layers | last layer | no | 63.5 | 97.1% | -0.4 |
| M3 | RoBERTa-large | all-layer WavLM | no | 62.7 | 92.8% | -0.1 |
| M5 | RoBERTa-large | all-layer WavLM | yes | 63.6 | 95.2% | -1.6 |
(harness_models.csv, swap_with_context.csv). Supported: across five models the weight's edge tracks the text share (about 54%: large edge; 93% to 97%: none); context is not required for the loss of the edge (M2, M3 have none). Balance by temperature flattening (balance_experiments.csv): text share 72% gives +9.9 (T=2), 42% gives only +2.3 (T=3), and text-weak models (12% to 15% text) give -12.7: so balance is not sufficient, and the edge is not monotone in text share. Remaining causal hypotheses: that balance itself causes the edge (M1 differs from M0b only by the audio encoder, but M2 vs M0b differs by the text encoder, and headroom shrinks as one modality dominates: at 95% the largest possible gain is 4.8 points); that the audio encoder's quality, not balance, drives M1. Confidence-weight fusion was fitted on the same heads in each model.

## 6. Donors and leakage
- Pool: the clf split of the training data, never the test or dev split: 8,004 utterances in the dialogue-level split v2 (M0b to M5); 7,990 in the original split (M0). Donors come from training dialogues only, so a swapped modality can never come from the same dialogue. (verified: scripts 33, 37, 38; 07_resampling.valid_masks)
- Duplicates: for each test utterance, donors whose headline embedding of the swapped modality is byte-identical are excluded (identical audio clips, and identical short texts such as "Yeah" that embed identically). The 39 test rows that share audio with another test row cannot be donors (test is never a donor pool); the 3 that share audio with the training split are excluded by this mask. results/audio_duplicates.json. (verified)
- Labels: not used in donor selection or in the effect. The predicted class c comes from the model's own fused probability. (verified, code)
- Interpretation: donors follow the training class distribution (neutral-heavy), so m is the average training utterance and an effect measures departure from the typical utterance; random donors also produce text and audio that do not belong together.

## 7. Statistical reliability
- Resampling unit: test dialogues, with replacement; 1,000 draws, rng 0; the gate-seed average is taken inside each draw; all comparisons paired on the same draws (06, 33, 37, 40). (verified)
- results/audit_bootstrap.csv (paired): agreement, corrected minus weight: M0b +15.0 [13.7, 16.4], M1 +18.6 [17.1, 19.9], M2 +2.1 [1.5, 2.8], M3 +5.4 [4.5, 6.4], M5 +2.2 [1.4, 3.0] points. Spearman, corrected minus weight: M1 +0.253 [0.236, 0.270] (0.721 to 0.974), M5 +0.528 [0.493, 0.564] (0.433 to 0.962), M0b +0.279, M2 +0.653, M3 +0.574 (intervals in the file, all above zero). Corrected-signal agreement intervals: M1 [93.1, 94.8], M5 [94.8, 96.7]. Comparisons across models are not paired (different models). (verified)

## 8. Metric dependence
Primary: |change in the probability of the predicted class| with the donor-averaged vector. Sensitivity (results/harness_alt_effects.csv): the winner changes for few utterances (text-wins share M1: 54.2% primary, 55.7% TVD, 53.3% flips, 55.0% signed; M0b: 76.7, 78.6, 71.9, 77.8), but the headline model's weight edge depends on the metric (+1.8 primary, +7.2 TVD, +5.4 flips, -0.3 signed). For M1, M2 and M3 the conclusion is the same under every metric. Not run: M5, ground-truth-class probability.

## 9. What is new
- Established in the literature: gates used as modality importance (Liu et al. 2020, He and Xiao 2025); swap-and-average marginalisation (Hessel and Lee 2020, for cross-modal interactions); fusion arithmetic.
- Mathematical consequences: the fixed-weight swap effect is w times the distance of the model's vote from the donor mean (the corrected signal); the gate term is exact; confidence-weight agreement is partly structural (its weight is built from the probabilities it mixes; results/shuffled_control.csv).
- New empirical observations (our models, MELD only): the weight's agreement is high in every model but its gain over the majority guess is +21 points only in the balanced model and zero or negative when text dominates; context is not needed for the loss; the edge is not monotone in text share; the gate re-weighting reverses the winner in 1% to 7% of utterances, near-ties.
- Hypotheses: balance as the cause; generalisation beyond MELD, other fusion types, other gates; swap realism (matched donors).
- Learned gate: tested on all models. Confidence rule: gain +25 to +28 points in M0b, M1, M3 and +5 in M2 (harness_models.csv), not interpretable as a faithful explanation because it is partly structural.

## 10. Strongest defensible conclusion
- Finding: In the text + audio models we built on MELD, a gate's weight agrees with the swap test mainly because one modality usually wins, and it beats the majority-modality guess by 21 points only when text and audio are comparably strong; once text dominates it adds nothing, and a signal built from the weight and how far each model's vote is from a typical vote explains the gap (94% to 99% agreement with the real swap, exact when the gate is frozen).
- Support: (1) +21.1 [18.8, 23.6] in M1 against -0.4 to -1.6 in M2, M3, M5 (all four effect measures in M1 to M3). (2) The loss of the edge occurs without context (M2, M3). (3) With the gate frozen the corrected signal equals the swap exactly; the gate's re-weighting explains all residual disagreement (1% to 7%, near-ties).
- Limitations: one dataset and our own models; swap donors are random real utterances (realism untested); only the primary metric was run on M5.
- Strongest criticism: "Your explanation of the gap is arithmetic, and the swap test is not ground truth, so you have shown how a gate behaves under one intervention, not that weights are unfaithful."
- Experiment: a matched-donor swap (donors with similar speaker or duration, not emotion), plus a second dataset or architecture, with the conflict test on the balanced and text-heavy models.

## 11. Order of mean and absolute value (added 2026-10-21; scripts 38, 39, 41; results/mean_of_abs_check.csv)
- Primary (verified, code): effect = | mean over donors of fused_d[c] - fused_0[c] |, i.e. donor-average first, then the change in the predicted class, then the absolute value (iv04.effect; the abs is taken in 06.arrays and 33.effect_arrays). The corrected signal and the frozen-gate identity refer to this order. The spoken phrase "see how much the prediction changes" is ambiguous; say "average the donors' predictions, then see how far the predicted class's probability moved".
- Variant: mean over donors of | fused_d[c] - fused_0[c] |. Always at least as large; differs when donors push in opposite directions.
| Model | text mattered more (primary / variant) | weight agreement | weight gain over majority, points [95% interval] | corrected-signal agreement | winner differs |
|---|---|---|---|---|---|
| M0b | 76.7 / 82.0 | 78.5 / 80.4 | +1.8 [-0.4, 4.0] / -1.5 [-3.7, 0.8] | 93.5 / 90.5 | 5.9% |
| M1 balanced | 54.2 / 53.6 | 75.4 / 80.3 | +21.1 [18.8, 23.6] / +26.7 [24.2, 29.2] | 93.9 / 93.5 | 5.5% |
| M2 | 97.1 / 99.5 | 96.7 / 98.8 | -0.4 / -0.7 [-1.0, -0.4] | 98.8 / 97.6 | 2.4% |
| M3 | 92.8 / 97.4 | 92.7 / 96.9 | -0.1 / -0.5 [-1.0, 0.0] | 98.1 / 94.7 | 4.6% |
| M5 (63.6) | 95.2 / 97.5 | 93.6 / 94.9 | -1.6 [-2.6, -0.6] / -2.5 [-3.6, -1.5] | 95.8 / 95.0 | 2.3% |
Reading: the headline numbers (54.2, 75.4, 21.1, 95.2, 93.6, -1.6) are the primary-order numbers and reproduce exactly. Under the variant the pattern is the same or stronger (balanced edge +26.7, text-dominated models zero or negative) except for the headline model M0b, whose small edge (+1.8) turns slightly negative (-1.5). The winner changes in 2% to 6% of utterances. The corrected signal is exact only for the primary order; under the variant its agreement is 0.5 to 3.4 points lower (still 90% to 98%) and its Spearman 0.94 to 0.98.
