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

## Amendment 1: version 2 of part B, and part A additions (written 2026-11-19 after review, before any of this was run)
Why: the score's main feature, |w - tau|, is the distance from the fusion identity's own decision boundary, so it is the baseline the equation gives for free, and it was never compared. Also, the held-out pool (speaker50) behaves like a pool used in fitting; nn50 is the only pool that behaves differently. Version 1 outputs stay as they are.

Part B version 2:
- Fit on dev using global50 and a new dev speaker50 pool (same construction as on test; dev rows without 50 same-speaker donors are dropped from the fit). nn50 is NOT used in fitting.
- Held-out evaluation on test against nn50 (primary), on all 2,610 test rows (nn50 exists for every row). Secondary: speaker50 rows, as before.
- Scores: learned (refitted as above, same features and settings), identity, margin, random. Identity score: for the R2 claim, |w - tau|; for the gate claim, |w - tau| when w and tau put the claim on the same side (that is, (w > 0.5) equals (w > tau)), and -|w - tau| when they disagree. Margin and random as in version 1.
- Reading rule: the learned score adds something beyond the identity only if AURC(learned) minus AURC(identity) has a bootstrap interval below 0 for the R2 claim in BOTH models. Otherwise the conclusion is: confidence in a modality claim should be measured from the break-even point tau, and that follows from the fusion identity.
- Note: the identity score was suggested by looking at version 1's fitted coefficients after the test results were seen; it is pre-registered here as a comparator, not as a discovery.
- "Swap-free" means: no per-utterance swaps at inference, given precomputed donor-pool averages.

Part A additions (descriptive):
- Direction of flips between full and nn50: share of rows going text to audio and audio to text, with the mean change in d_t and d_a under nn50 donors.
- global50 against nn50 and full against nn50 on all 2,610 test rows (not only the 2,156 speaker-eligible rows).
- Noise floor over four random draws: global50 with seeds 544 (existing), 545, 546, 547; report the mean and range of full-versus-global50 disagreement.
- Note: nearest neighbours for audio are found in the pooled 1,024-d gate embedding, while M1's audio head reads 51,200-d all-layer features.
Outputs: `data/reference_pools_v2_{M1,M5}.npz`, `results/reference_sensitivity_v2.csv`, `results/selective_attribution_v2.csv` (scripts 55 and 56).

## Amendment 1 results (recorded 2026-11-19 after the run; `results/reference_sensitivity_v2.csv`, `results/selective_attribution_v2.csv`)
- Reading rule for part B version 2 NOT met: learned minus identity AURC, R2 claim, held-out nn50, all 2,610 rows: M1 -0.008 [-0.018, 0.001], M5 -0.005 [-0.011, 0.001]; both intervals include 0. Conclusion as fixed in advance: confidence in a modality claim should be measured from the break-even point tau, which follows from the fusion identity; the learned score adds no reliable gain on a pool it was not fitted on. (On speaker50, which resembles a fitted pool, learned beats identity: M1 -0.012 [-0.019, -0.006], M5 -0.009 [-0.013, -0.006].)
- For the gate claim (text if w > 0.5) on nn50, the identity score beats the margin |w - 0.5| in M1 (AURC difference -0.025 [-0.034, -0.016]; risk at 50% coverage 0.111 vs 0.146) but not in M5 (+0.007 [0.001, 0.012]).
- Part A: flips under nn50 lean one way: text to audio 0.116 vs audio to text 0.051 (M1), 0.054 vs 0.030 (M5). Similar-utterance donors shrink text distinctiveness more than audio (M1 d_text 0.199 to 0.114, d_audio 0.226 to 0.188; M5 d_text 0.268 to 0.191, d_audio 0.106 to 0.085). Noise floor over four random draws: M1 0.052 to 0.056, M5 0.037 to 0.046; nn50 excess over its mean on all rows: M1 0.113 [0.100, 0.126], M5 0.043 [0.033, 0.054].
- Step 54 rerun after moving helpers to audit_common: outputs byte-identical.
