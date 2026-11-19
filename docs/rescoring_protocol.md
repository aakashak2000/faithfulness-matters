# Re-scoring of the existing swap results (item 0 of the revised plan)
Written 2026-11-17, before `scripts/49_rescoring.py` was run on any split. Nothing here changes the swap definition, the donor pool, alpha, or any earlier result; it only computes new summaries of arrays that already exist.

## Status
Exploratory. The metrics below were chosen after the midterm results were seen (they answer critiques of those results), so nothing here is a confirmatory test. Every number is reported with its 95% dialogue-cluster bootstrap interval and labelled exploratory in any write-up.

## Inputs (unchanged)
- Models M1 (alpha 0.25, 10 gates) and M5 (alpha 0.5, 3 gates), the two models of the corrected-rule swap test (`docs/corrected_rule_swap_protocol.md`).
- Test arrays `data/corrected_rule_arrays_{M1,M5}.npz` (step 43); dev arrays `data/audit_dev_arrays_{M1,M5}.npz` (step 42).
- Head probabilities p_text, p_audio and gold labels, rebuilt with `scripts/audit_common.py` exactly as in step 43 (the heads are shared by all gates of a model).

## Fixed definitions
- Weight signal: the gate's text weight w (s = 2w - 1, as in step 43).
- Effect difference e: corrected rule, absolute value of the donor-averaged change, gate recomputed (`e_cor_real`), the primary effect of step 43. "Text won" means e > 0. Rows with e = 0 or s = 0 are dropped, as in step 43.
- Sensitivity effects: frozen gate (`e_cor_fro`) and raw rule (`e_raw_real`), metric block A only.
- Every metric is computed per gate and averaged over gates. Intervals: 1,000 dialogue-cluster bootstrap draws, numpy rng seed 0, built exactly as in step 43.

## Metrics
A. Threshold-free and fixed-threshold summaries (all valid test rows): AUROC of w for "text won"; at w = 0.5: balanced accuracy, Cohen's kappa, recall on audio-won rows, share of rows with w < 0.5; number of audio-won rows. Repeated without the 42 `audio_shared` rows.

B. Dev-fitted threshold: for each gate, the threshold t on w that maximises balanced accuracy on dev (candidates: 201 evenly spaced quantiles of that gate's dev w). Applied unchanged to test: balanced accuracy, kappa, audio-won recall.

C. Meaningful effects: tau = median of |e| on dev for each gate (primary) and the dev 75th percentile (secondary). On test rows with |e| > tau: agreement at w = 0.5, balanced accuracy at 0.5, AUROC, share text won, number of rows.

D. Reliability versus influence: rows where exactly one unimodal head is correct (head argmax against the gold label) and e is not 0. On these same rows: AUROC of w for "the text head is the correct one" (reliability) and AUROC of w for "text won the swap" (influence); their paired difference; the share of rows where the correct head is also the one that won the swap.

E. Disagreement strata: Jensen-Shannon divergence (base 2) between p_text and p_audio. Quartile cut points fixed on dev and applied to test. Per test quartile: agreement at 0.5, AUROC, balanced accuracy, share text won, mean |e|, number of rows.

## Reading rules, fixed now
- If M5's test AUROC (block A) has a lower bound above 0.55, the midterm statement "M5's weight does no better than always-text" is reported as specific to the 0.5 threshold, not as "the weight carries no information".
- The weight is said to track reliability more than influence (or the reverse) only if the block D paired-difference interval excludes 0.
- Block E is descriptive only. It does not test H2; H2 stays with the frozen conflict protocol.

## Validation, run before any new number is read
- Agreement recomputed from the arrays must match `results/corrected_rule_swap_{M1,M5}.csv` (cor, primary_abs_of_mean) to 1e-9.
- The gate weight rebuilt on dev and test must match the saved s within 1e-5 (this checks row order).
- M1's reliability AUROC on all only-one-correct rows (without dropping e = 0) must match `results/gate_input_diagnostic.csv` (0.58797) to 1e-4.

## Output
`results/rescoring.csv` (one row per model, block, metric and subset).

## Post-run note (2026-11-17, after the run; no definition above was changed)
Block D is not a fair comparison as designed. The influence label (sign of e) is built partly from w itself: with the gate frozen, e is close to w * d_text - (1 - w) * d_audio. So the AUROC of w for influence is raised by that identity, while the AUROC for reliability (which head is correct) is not. The block D difference therefore must not be read as "the weight tracks influence more than reliability". A fair version needs an influence target that does not contain w (for example, which head is more distinctive, d_text against d_audio). That would be a new analysis with its own written definition before running. The same caveat applies, more mildly, to every metric that compares w with e (blocks A to C and E): they show how far the variation in distinctiveness spoils the ranking w already implies, not independent evidence.

## Version 2 (written 2026-11-19 after the cloud review, before the re-run)
Two changes; `results/rescoring.csv` is kept unchanged and the new output is `results/rescoring_v2.csv`.
1. Without `audio_shared`: blocks B to E are also reported without the 42 `audio_shared` test rows (block A already was). Same draws, restricted to the remaining rows.
2. Degenerate subsets: a gate's class-dependent metrics (AUROC, balanced accuracy, kappa, audio-won recall and the block D AUROCs) are set to missing when that gate has fewer than 5 rows of either class in the subset (5 rows of either reliability class for block D). Kappa is no longer set to 0 in that case. Each row now reports `n_valid_draws` (bootstrap draws where at least one gate had the metric) and `underpowered` (point estimate missing, or fewer than 950 of 1,000 valid draws). Expected effect: only sparse subsets change (for example M5 block C above the dev 75th percentile, where the audio-won count was 3); all other values must equal `results/rescoring.csv` (checked after the run).
