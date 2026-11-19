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

## Amendment 2: is the text-to-audio flip direction an artifact of the neighbour search? (written 2026-11-19, before the check was run)
Concern (review): in M1, text neighbours are found in the MiniLM embedding that the text head itself reads, while audio neighbours are found in the pooled 1,024-d embedding, not in the 51,200-d features the audio head reads. Neighbours in the head's own input space will get near-identical predictions, so the larger drop in text distinctiveness may come from the search, not from the modality.
Check: a new pool nnp50, the 50 valid donors nearest to the utterance in each head's OUTPUT space (Euclidean distance between the head's probability vectors, text head for text swaps, audio head for audio swaps; for M5 the donor probabilities include the utterance's own context part, as in the swap). "Similar" then means the same thing for both heads. Same swap routine, all 2,610 test rows.
Reported: winner disagreement full vs nnp50 and its excess over the four-draw noise floor; flips text to audio and audio to text, and their difference; mean d_text and d_audio under nnp50.
Reading rule: the direction claim ("similar-utterance references move influence away from text") is kept only if, under nnp50, text-to-audio flips exceed audio-to-text flips with the bootstrap interval of the difference above 0 in BOTH models. Otherwise the direction claim is dropped and only "local and global references give different answers" is kept.
Also reported (descriptive, from existing arrays): the share of rows where the gate claim and the identity disagree, (w > 0.5) != (w > tau), per model, to explain why measuring from tau helps M1 and not M5.
Output: `data/reference_pools_v3_{M1,M5}.npz`, `results/reference_sensitivity_v3.csv` (script 57).

## Amendment 2 results (recorded 2026-11-19 after the run; `results/reference_sensitivity_v3.csv`)
- Pre-registered rule, formally MET: under output-space neighbours (nnp50), text-to-audio flips exceed audio-to-text flips in both models (difference M1 0.236 [0.210, 0.260], M5 0.408 [0.387, 0.429]).
- But the mechanism behind the original claim does not survive. With output-space neighbours both heads' distinctiveness collapses about equally (M1 d_text 0.199 to 0.012, d_audio 0.226 to 0.013; M5 0.268 to 0.010 and 0.106 to 0.010), so "text distinctiveness falls more" was a property of the embedding-space search.
- Post hoc (added to script 57 after its first run, not pre-registered): with the gate held fixed, the nnp50 direction reverses in M1 (text to audio 0.158 vs audio to text 0.231) and shrinks in M5 (0.132 vs 0.037); recomputing the gate makes the effects 4.1 times (M1: 0.0335 vs 0.0081) and 2.1 times (M5: 0.0212 vs 0.0101) larger. Under minimal counterfactuals the primary swap mostly measures the gate's own reaction, not the heads.
- Decision: the direction claim is dropped from slides and report headlines. Kept: local and global references give different answers (nn50 excess over noise, M1 0.113, M5 0.043, version 2). New observation, reported as exploratory: when head outputs barely change, the recomputed-gate swap is dominated by the gate's re-weighting.
- Why measuring from tau helps M1 and not M5: the gate claim and the identity disagree, (w > 0.5) != (w > tau), on 21.7% [20.3, 23.2] of M1 rows and 5.7% [4.9, 6.5] of M5 rows.

## Correction to the Amendment 2 reading (2026-11-19, after review)
nnp50 donors leave both heads' outputs almost unchanged (distinctiveness about 0.01), and its winners disagree with the full pool on about half of rows, close to chance. So (1) the passed direction rule is uninformative, and (2) "the swap mostly measures the gate's re-weighting" under nnp50 holds by construction (if the head outputs do not change, any effect must pass through the gate's input) and is NOT reported as a finding. The real methods point: the primary swap lets a modality act through two paths, its head's output and the gate's input; the report states this definition explicitly.

## Amendment 3: does the nn50 headline survive with the gate held fixed during the swap? (written 2026-11-19, before the check was run)
Terms: "head-path-only" = the gate weight is held at its original value during the swap (no gate path); "gate path" = the recomputed-gate effect minus the head-path-only effect, per modality, on the signed donor-averaged change (computed before absolute values, so cancellation is visible).
Check: rerun the global50 (seed 544) and nn50 pools on test saving the head-path-only effects; compare winners with the full-pool head-path-only winners (`e_cor_fro`, step 43).
Reported: flip rate full vs nn50 and its excess over the global50 noise floor, both for the recomputed gate (as before) and head-path-only; mean absolute head-path and gate-path terms per modality under full and nn50.
Reading rule: the headline "with similar-utterance donors about 1 in 6 (M1) answers flip" is kept as a statement about the heads if the head-path-only excess over noise is at least half of the recomputed-gate excess, in both models. Otherwise the headline becomes "much of what changes with the reference is the gate reacting to its own input".
Output: `results/reference_sensitivity_v4.csv` (script 58).

## Amendment 3 results (recorded 2026-11-19; `results/reference_sensitivity_v4.csv`)
- Rule MET in both models: the head-path-only excess of nn50 over the noise floor is 0.096 [0.081, 0.111] (M1) and 0.034 [0.025, 0.042] (M5), that is 0.85 [0.74, 0.96] and 0.74 [0.56, 0.97] of the recomputed-gate excess (0.114 and 0.045). The headline "with similar-utterance donors about 1 in 6 (M1) / 1 in 12 (M5) answers flip" stands as a statement about the heads.
- Size of the gate path (mean absolute signed term, full pool): M1 text 0.015 vs head path 0.130; M1 audio 0.039 vs 0.100; M5 text 0.011 vs 0.227; M5 audio 0.028 vs 0.025. In M5 the audio swap acts about as much through the gate's input as through the audio head. The gate path opposes the head path on about half of rows (0.47 to 0.59), so the two partly cancel.
