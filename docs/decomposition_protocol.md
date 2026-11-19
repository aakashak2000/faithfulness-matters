# Decomposition of the weight's agreement with the swap (structural versus learned)
Written 2026-11-17, before `scripts/50_decomposition.py` was run. Follows the post-run note in `docs/rescoring_protocol.md`. No swap is rerun, no model is trained, no earlier definition changes.

## Status
Exploratory, chosen after seeing the re-scoring results. Predictions are written below so the result can contradict them.

## Why
With the raw rule and the gate held fixed, swapping text changes the predicted class's probability by w * d_t on average and swapping audio by (1 - w) * d_a, where d is a head's distinctiveness (absolute difference between its probability for the predicted class c and the donor-pool mean). So text wins exactly when w > tau, with tau = d_a / (d_t + d_a), a break-even weight per utterance. Any score of w against the winner is then set by the spread of w, the spread of tau, and how w and tau move together. Only the last part, and how w relates to reliability, says anything about what the gate learned.

## Inputs
M1 (10 gates) and M5 (3 gates); `data/corrected_rule_arrays_{M1,M5}.npz` (step 43); head probabilities and labels rebuilt as in `scripts/49_rescoring.py`.

## Definitions
- d_t = |raw_fro_text_mean| / w and d_a = |raw_fro_audio_mean| / (1 - w), per gate (raw rule, frozen gate, the gate's own predicted class c, valid donors only). tau = d_a / (d_t + d_a).
- Target of the decomposition: text won under the raw rule with the frozen gate (e_raw_fro > 0), where the identity is exact. Also reported: the share of rows where 1[w > tau] matches the primary effect of step 43 (corrected rule, gate recomputed).
- Reliability R: the text head is correct and the audio head is wrong (1) or the reverse (0); rows where exactly one head is correct. Binary, so temperature scaling cannot change it. Secondary: r = log p_t[y] - log p_a[y] on all rows.
- Limitation, stated now: tau is free of the gate weight but not entirely free of the gate, because c is the fused prediction.

## Metrics (per gate, averaged over gates; 1,000 dialogue-cluster bootstrap draws, rng seed 0, as in step 43)
1. Permutation null: w shuffled across test rows, label recomputed as 1[w' > tau]. Reported: actual AUROC and agreement at 0.5, their null means (200 permutations for the point estimate, 20 inside each bootstrap draw), and the excess (actual minus null).
2. Couplings: AUROC of w for R (w with reliability); Spearman of w with tau (w with distinctiveness); AUROC of the text distinctiveness share 1 - tau for R (are reliable heads distinctive); Spearman of w with r.
3. Heads (test, gate-free, point values): expected calibration error (15 equal-width bins on max probability), mean entropy, mean max probability, per head; mean and median d_t and d_a averaged over gates; share of rows with tau > 0.5.

## Predictions (from the thesis: a likelihood-trained gate learns reliability at best, not tau)
- P1: excess AUROC is about 0; the lower bound of its interval is not above 0.02 in either model.
- P2: AUROC of w for R is above 0.5 but below 0.65 (already near-known: 0.588 M1, step 45).
- P3: AUROC of 1 - tau for R is above 0.5 (reliable heads tend to be more distinctive).
- P4: median tau is above 0.5 in both models (audio more distinctive), consistent with the dev cutoffs of step 49.
## Reading rules, fixed now
- If P1 fails (excess lower bound above 0.02), the gate encodes distinctiveness beyond chance and the thesis is weakened; say so.
- If P2 fails below (AUROC interval includes 0.5), the gate learned neither reliability nor tau; the thesis becomes "the gate learns little".
- If P3 fails, faithfulness of w cannot be explained through reliability, and the thesis loses its mechanism.

## Validation
- 1[w > tau] must equal 1[e_raw_fro > 0] on every row with e_raw_fro not 0.
- step 43 closed-form error (`closed_form_err_*`) below 1e-4.
- Agreement of w against e_raw_fro must match `results/corrected_rule_swap_{M}.csv` (rule raw, effect frozen_gate) to 1e-9.

## Output
`results/decomposition.csv`

## Amendment 1 (2026-11-17, written after the run above, before the check below was run)
Result that prompted it: P1 failed (excess AUROC above 0.02 in both models), driven by a negative Spearman between w and tau. The limitation stated above can produce exactly that without any learning: tau is measured at the fused predicted class c, and a higher w makes c more likely to be the text head's own preferred class, which raises d_t at c and lowers tau. So the coupling may be mechanical.
Check: recompute distinctiveness at a gate-free class c0 = argmax of 0.5 p_text + 0.5 p_audio, with the same donor pools and valid-donor masks: d0 = |p_head[c0] - mean over valid donors of p_donor[c0]|, tau0 = d0_a / (d0_t + d0_a). Report Spearman(w, tau0) with the same bootstrap, the share of rows where c0 equals c, and AUROC of 1 - tau0 for R.
Validation: on rows where c0 equals c, d0 must equal the d recovered above to 1e-4.
Reading rule: if Spearman(w, tau0) is near 0 (interval includes 0 or its absolute value is below 0.1), the P1 excess is attributed to the class channel, not to learning. If it stays clearly negative, the gate puts more weight on the head that is more distinctive, independently of its own effect on c, and P1's failure stands.
Output: results/decomposition_gate_free_tau.csv (script 51).
Validation change (2026-11-17, after the first attempt failed, before any Amendment 1 number was read): the check on d failed (max 1.7e-3) only because d is recovered by dividing a float32 effect by the weight, and one M1 row has an audio weight of 0.00017. At the same class, the difference on the effect scale (d times the weight) is at most 4.4e-7. The check is now made on the effect scale, tolerance 1e-5. The same division affects d and tau from step 50 on such rows; its effect on rank statistics is negligible but is noted here.

## Amendment 2: recomputed-prediction null, version 2 (2026-11-17, written before script 52 was run)
The step 50 null is a fixed-reference-class null: w is shuffled, but tau (and so the predicted class it is measured at) stays the one produced by the original w. No model with the shuffled weight would explain that class, so the null mixes a shuffled weight with a target built from the real one. Its numbers stay as they are, renamed in reading only ("fixed-class null").
Version 2 adds a recomputed-prediction null. For each permutation w' of a gate's weights: c' = argmax(w' p_text + (1 - w') p_audio); d'_t, d'_a = distinctiveness of each head at c' (same donor pools and masks as Amendment 1); label' = 1[w' d'_t > (1 - w') d'_a] (raw rule, frozen gate: exactly the swap winner a model with weight w' would have). Score w' against label'.
Questions answered:
- fixed-class null: given the class each utterance was actually explained at, how much agreement would weights unrelated to the utterance produce? Its excess includes the gate's influence on its own predicted class.
- recomputed null: if a gate with the same weight distribution assigned weights at random, and the model's prediction and swap winner followed from those weights, how faithful would the weight be? Its excess is the faithfulness added by which utterance gets which weight. This is the null for "what the gate learned".
Reported (same bootstrap, 200 permutations for the point value, 20 per draw): actual and null agreement and AUROC, excess, and accuracy of the actual and null fused predictions (the random gate also changes predictions).
Validation: with the actual weights, the recomputed class must equal c_raw and the recomputed label must equal 1[e_raw_fro > 0] on rows with e_raw_fro not 0.
Reading rule: "agreement is mostly structural" survives if, in each model, the recomputed null's agreement is at least 80% of actual agreement and the excess is below 0.10. Otherwise it is withdrawn.
Interpretation limit (also applies to Amendment 1): a drop in correlation after a change of class is not a decomposition; it does not give a fraction of mechanical versus learned influence.
Output: results/decomposition_null_v2.csv (script 52). Step 50 and 51 outputs are not modified.

## Interpretation note on version 2 (2026-11-17, after script 52 was run; no number changed)
Results: `results/decomposition_null_v2.csv`. A random-assignment gate with the same weight distribution matches the learned gate's agreement with the swap (excess -0.006 [-0.017, 0.005] for M1, -0.001 [-0.006, 0.004] for M5).
What this shows: agreement between the weight and the model's own swap behaviour (self-consistency) is reproduced by any assignment with the same weight distribution, so this agreement is not evidence that the learned gate carries per-utterance information about faithfulness.
What it does not show: that the learned weights carry no such information. Under the recomputed null the explanation and the behaviour it explains both change, so the test measures self-consistency, not whether w predicts an independent notion of modality importance. That question needs a target that does not depend on the gate (planted modality, human-verified conflict pairs) and is open.
Scope: raw rule, frozen gate; the threshold rule 1[w > tau] matches the primary effect (corrected rule, gate recomputed) on 87.3% (M1) and 91.7% (M5) of rows (`results/decomposition.csv`).

## Amendment 1 verdict, per model (recorded 2026-11-19 after the cloud review; numbers from `results/decomposition_gate_free_tau.csv`)
Applying the reading rule fixed in Amendment 1:
- M1: Spearman(w, tau0) = -0.104 [-0.143, -0.066]. The interval excludes 0 and the absolute value is above 0.1, so by the rule P1's failure STANDS for M1: the gate puts somewhat more weight on the head that is more distinctive at a gate-free class. The coupling is weak (just past the 0.1 line).
- M5: Spearman(w, tau0) = -0.060 [-0.100, -0.021]. Absolute value below 0.1, so the P1 excess is attributed to the class channel.
How this sits with version 2: the recomputed-prediction null (`results/decomposition_null_v2.csv`) shows that this M1 coupling does not add self-consistency agreement (excess -0.006 [-0.017, 0.005]). Both statements hold: M1's gate weakly tracks distinctiveness, and that tracking does not show up as extra agreement. Neither says whether the gate tracks an independent notion of modality importance.

## Source of the numbers in "Validation change" above
The figures 1.7e-3 (largest gap in d), 0.00017 (the audio weight on that row) and 4.4e-7 (largest gap on the effect scale) are printed by `scripts/51_gate_free_tau.py` (lines starting "DIAGNOSTIC") from 2026-11-19 on.

## Version 3: audio_shared sensitivity (written 2026-11-19, before the re-run)
Per CLAUDE.md, every step 50 to 52 analysis is repeated without the 42 `audio_shared` test rows: bootstrap draws are the same dialogue-cluster draws restricted to the remaining rows, permutations are within the remaining rows, and the permutation generator is re-seeded (seed 1) per subset. The all-row outputs are regenerated by the same code and must be byte-identical to the committed files (checked with git diff). New files: `results/decomposition_no_audio_shared.csv`, `results/decomposition_gate_free_tau_no_audio_shared.csv`, `results/decomposition_null_v2_no_audio_shared.csv`. Shared metric helpers moved to `scripts/common.py` and the loaders to `scripts/audit_common.py`; no formula changed.
