# Phase 1: reference sensitivity and swap-free selective attribution
Written 2026-11-19, before `scripts/53_reference_pools.py` and `scripts/54_selective_attribution.py` were run on any split. Status: exploratory analysis of existing models (no training). Nothing earlier changes.

## Questions
A. Reference sensitivity: does the modality that "won" the swap test depend on which donors we swap in?
B. Selective attribution (objective 2): can a score computed WITHOUT running swaps tell in advance which modality claims will be contradicted under a donor pool it was never fitted on, better than simply abstaining when the weight is close to 0.5?

## Fixed
Models M1 (alpha 0.25, 10 gates) and M5 (alpha 0.5, 3 gates); heads, gates, splits, the step 43 swap routine (`audit_common.core2`), corrected rule with gate recomputed as primary; dialogue-cluster bootstrap (1,000 draws, seed 0). Donors are always the 8,004 clf-split training utterances, excluding byte-identical ones (step 43 valid-donor masks).

## Donor pools (per utterance, same donors for every gate; 50 donors each, matched)
- full: all valid donors (step 43; existing arrays, not rerun).
- global50: 50 valid donors drawn at random (numpy seed 544, rows in split order).
- speaker50: 50 valid donors drawn at random (same generator, continued) from the utterance's own speaker; utterances whose speaker has fewer than 50 valid donors are excluded from every comparison involving this pool (82.6% of test rows qualify).
- nn50: the 50 valid donors nearest by cosine similarity in the swapped modality's embedding (MiniLM 384-d for text swaps, pooled WavLM 1,024-d for audio swaps, the gate's inputs). Nearest neighbours give small, local substitutions; this is a different reference, not a less noisy one.
Dev: global50 and nn50 (used only to fit part B). Test: global50, speaker50, nn50.

## Part A outcomes (test)
For each pool and each influence definition, kept separate: |E[delta]| (primary, step 43) and E[|delta|] (secondary). Winner = sign of the text-minus-audio effect.
- Pairwise winner disagreement between pools, averaged over gates; global50 versus full is the sampling-noise floor (same distribution, 50 donors).
- Share of utterances whose winner is not the same in all four pools (on rows where speaker50 exists).
- Gate agreement (w > 0.5 versus the winner) in each pool.
Reading rule: reference dependence is called substantive if the share of rows where speaker50 or nn50 disagrees with full exceeds the noise floor by at least 3 percentage points, with the bootstrap interval of the difference excluding 0. Otherwise part A is reported as a robustness result.

## Part B: selective attribution
Claims (both swap-free): the gate claim (text if w > 0.5) and the readout claim R2 (text if w d_t > (1 - w) d_a, d = distinctiveness at the raw fused class against the full-pool donor means).
Confidence scores, each ranking which claims to keep:
- margin: |w - 0.5| for the gate claim, |R2 - 0.5| for R2;
- learned: a logistic regression fitted on dev (pooled over gates), target = the claim agrees with the winner under BOTH dev pools (global50 and nn50); features, all swap-free: w, |w - 0.5|, R2, |R2 - 0.5|, tau, |w - tau|, d_t, d_a, gate-free d0_t and d0_a, max probability and entropy of each head, whether the heads' argmax agree, JS divergence between the heads. Features standardised on dev; sklearn LogisticRegression with default settings (C = 1), no tuning;
- random (numpy seed 7).
Primary evaluation, test, held-out pool speaker50 (never used in fitting): risk = share of kept claims that disagree with the speaker50 winner (|E[delta]|); risk-coverage curve over coverage 5% to 100% in 5% steps; area under it (AURC); risk at 50%, 75% and 90% coverage. Per gate, averaged over gates.
Secondary: the same against global50, nn50 and full on test; and, on utterances where exactly one head is correct, the share of kept claims that name the correct head at 50% coverage (a reliability check that does not depend on any swap).
Reading rule, fixed now: the learned score is a useful method if its AURC is lower than the margin's for the R2 claim in BOTH models, with the paired bootstrap interval of the difference below 0. If not, we report that the margin already captures what the swap-free features know.

## Validation (before any new number is read)
- Rerunning gate 0 on test with the full-pool masks reproduces `e_cor_real` in `data/corrected_rule_arrays_{M}.npz` to 1e-6.
- Every pool mask has exactly 50 donors on included rows, and none is invalid.

## Outputs
`data/reference_pools_{M1,M5}.npz` (arrays, local), `results/reference_sensitivity.csv` (part A), `results/selective_attribution.csv` (part B).

## Results against the reading rules (recorded 2026-11-19, after the run; files `results/reference_sensitivity.csv`, `results/selective_attribution.csv`)
- Part A, |E[delta]|: share of test rows where speaker50 or nn50 disagrees with full, minus the noise floor (full vs global50): M1 0.140 [0.124, 0.155], M5 0.065 [0.053, 0.078]. Rule met: reference dependence is substantive in both models. It comes from nn50 (excess 0.106 M1, 0.044 M5); speaker50 is within noise (0.008 [-0.001, 0.018] M1, 0.000 M5). Gate agreement rates barely move across pools (M1 0.726 to 0.735), so the aggregate agreement is robust while the per-utterance winner is not.
- Part B, primary (R2 claim, held-out speaker50): AURC learned minus margin M1 -0.024 [-0.032, -0.016], M5 -0.009 [-0.014, -0.005]. Rule met in both models.
- Limits noted after the run: (1) speaker50 behaves like global50, so the "held-out" pool is close to a fitted one in distribution; nn50 was used in fitting (on dev rows). (2) The swap-free reliability check at 50% coverage does not improve for the R2 claim (M1 learned 0.671 vs random 0.666), so the score predicts stability of the claim, not whether it names the head that is correct. (3) Post hoc, not pre-registered: the fitted score is dominated by |w - tau| (distance of the weight from its break-even point). A single-feature |w - tau| abstention rule would need its own written comparison before being claimed. (4) The logistic-regression fit prints spurious BLAS overflow warnings on macOS; inputs are finite, the fit converges (37 iterations, M5 R2), and an independent scipy float64 fit agrees within 0.002 in probability.
