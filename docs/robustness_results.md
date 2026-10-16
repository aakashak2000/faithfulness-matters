# Robustness checks on the current model (run 2026-10-15)

Plain-language summary of the checks that do not depend on the stronger audio head. Every number
is in the named file in `results/`. Each script's header states the definitions and the reading rule
that were written before it ran. The primary intervention is full-pool resampling; "gain" means
agreement minus the always-text baseline (how much better than always saying "text").

| # | Question | Script | Result file | Answer |
|---|---|---|---|---|
| 7 | Can the gate and the confidence rule be told apart on the gate's own model? | `12_paired_gate_vs_confidence.py` | `paired_gate_vs_confidence.csv` | No. Gain +0.025 (confidence) vs +0.019 (gate); difference +0.006 [-0.017, +0.028]. The gate ranks effect sizes slightly better (Spearman 0.705 vs 0.674; difference -0.031 [-0.062, +0.002]). |
| 11, 12 | Does the verdict depend on how "effect" is measured? | `14_alt_effect_metrics.py` | `alt_effect_metrics.csv` | **Yes for the gate.** Gate gain is +0.019 [-0.003, +0.041] with the primary effect (change in the predicted class's probability) and +0.000 with the signed version, but +0.090 [+0.069, +0.112] when effect is the change in the whole output distribution (TVD) and +0.061 [+0.033, +0.092] when effect is whether the predicted class flips. On its own model and on the average-fusion model the confidence rule stays far above its baseline under all four measures (lower bound at least +0.100). On the gate's model it falls below its baseline under TVD (-0.071). |
| 10 | Is the gate's gap partly caused by the gate re-weighting after we remove a modality? | `15_gate_fixed.py` | `gate_fixed_diagnostic.csv`, `gate_fixed_summary.json` | Partly. With the weight held fixed, the gate's gain is +0.090 [+0.067, +0.116] vs +0.019 recomputed (difference +0.072 [+0.053, +0.091]). Re-weighting changes which modality "mattered more" for 6.9% of utterances. Removing audio moves the gate's weight about as much as it moves the audio head's output (median 0.022 vs 0.025); removing text mostly moves the text head (0.116 vs 0.005). Caveat: with the weight held fixed, agreement is partly structural. |
| 17 | Does weighting the signal by how much the two heads disagree rank effects better? | `16_weight_disagreement.py` | `weight_disagreement.csv` | Modestly. Gate Spearman 0.705 -> 0.741 (+0.036 [+0.027, +0.047]) with the primary variant. Agreement and gain cannot change (the sign is unchanged). |
| 6 | How much of each signal's performance is structural? | `18_shuffled_control.py` | `shuffled_control.csv`, `shuffled_control_f1.csv` | The confidence rule's gain is +0.225 with heads trained on shuffled labels, identical to +0.225 with real labels: structural. The gate's gain is +0.197 on shuffled labels vs +0.019 real, because the always-text baseline is 0.40 there and 0.77 for the real model; its rank correlation falls from 0.705 to 0.281. |
| 16 | Does the gate beat always-text if its cut-off is fitted on dev instead of fixed at 0.5? | `19_dev_calibration.py` | `dev_calibration_thresholds.csv`, `dev_calibration_test.csv` | Yes, modestly. Dev-fitted thresholds are 0.36 to 0.38 for the primary method. Test gain with the frozen threshold is +0.049 [+0.035, +0.062] (vs +0.019 at 0.5). Intervals are conditional on the fitted threshold. |
| 18, 19 | H2: does agreement drop when text and audio conflict? | `20_conflict_h2.py` | `conflict_h2_*.csv`, `conflict_pairs.csv` | **Not supported.** Primary pairing: gate gain conflict minus same-label control = +0.008 [-0.018, +0.034]. A second random pairing gives +0.041 [+0.014, +0.068] (gain higher under conflict). The two pairings disagree, so pairing noise is larger than the bootstrap interval; more pairings are needed before any claim about direction. |
| 8 | Where do the "distance from real data" numbers come from? | `diag_distances.py` | `embedding_distances.json` | They are cosine similarities (higher = closer to real embeddings): silent audio 0.30, mean audio 0.68, empty text 0.18, mean text 0.37. |
| 15 | Do the shared-audio and seed claims hold? | `diag_wording_checks.py` | `wording_checks.json` | Dropping the 42 shared-audio rows changes agreement and gain by at most 0.002, the baseline by 0.003 and Spearman by 0.004. Agreement across seeds has a standard deviation up to 0.011 and a widest range of 0.020. |

## What changes in the story
1. **The gate verdict is not one number.** Under the pre-registered primary effect the gate is not
   distinguishable from always-text (+0.019). Under TVD and flips it is clearly better (+0.09, +0.06),
   and with a dev-fitted threshold it is +0.049. The honest summary is "a small, measure-dependent edge,
   mostly lost by the 0.5 cut-off and by re-weighting". The primary stays the pre-registered one; TVD is
   not promoted because it happens to look better.
2. **The confidence rule's large gain is structural** (same with shuffled labels). It cannot be told
   apart from the gate on the gate's own model.
3. **H2 has no support yet.** The manipulation worked (audio head accuracy against the text label 10%
   vs 33%; heads disagree on 88% vs 78% of pairs), but gain does not drop under conflict.

## Still open
- Strong audio head variant: steps 3.3c to 3.5 (retrain on the layer-averaged features, rerun the
  interventions; `docs/plans/step-03-strong-audio-head.md`).
- H2: repeat with more random pairings (for example 10) and report the spread across pairings.
- Report: add these checks as sensitivity analyses; keep the pre-registered primary as the headline.
