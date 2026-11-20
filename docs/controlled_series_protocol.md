# Phase 2 protocol: what does a likelihood-trained gate adapt to? (revision 2, 2026-11-19, approved with revisions by review; written before any phase 2 run)
Exploratory study on the M1 architecture. Revision 2 folds in the four reviews of the first draft (changes listed at the end).

## Question
When we change, under control, how reliable the audio head is and how sharp its probabilities are, does a retrained gate move its weight with the head's likelihood (reliability in the sense the training objective sees), with sharpness, or with neither? Does it single out audio known to be substituted?

## Terms
- "C0 gate": the control's gates applied, without retraining, to a changed head (a controlled mechanistic baseline; the exact identity holds only for the raw rule with the gate held fixed).
- "Retrained gate": new gates trained on the changed head (the test).
- "Head-path-only effect": swap effect with the gate weight held at its original value during the swap. "Gate path": recomputed-gate effect minus head-path-only effect, on signed donor-averaged changes.

## What stays fixed
Frozen encoders and stored features; the M1 text head (MiniLM, linear, seed 0); the gate architecture and training routine (`audit_common.harness_gate`: gate split, early stopping on dev); splits; the full donor pool for every influence measure; the step 43 swap routine; the dialogue-cluster bootstrap (1,000 draws, seed 0). Alpha refitted on dev per condition (grid 0, 0.25, 0.5, 0.75, 1; dev weighted F1, corrected rule). Nothing selected on test.

## Conditions (audio head only)
| Arm | Audio head | Intended change |
|---|---|---|
| C0 | original M1 routine (`22_better_audio.fit`, original weight decay), full clf split | reference |
| C1 | same routine on a random 25% of the clf split | reliability down |
| C2 | C1 head with one temperature fitted on dev so its mean dev distinctiveness (gate-free class, full pool) equals C0's | reliability down, mean distinctiveness held |
| S0, S1, S2 | C0 head at temperature T*, T*/2 (sharper), 2T* (flatter); T* = the dev-NLL-optimal temperature of the C0 audio head | sharpness changes, head argmax and accuracy identical; S0 is the reference for S1 and S2 |
| P | for a fixed random 30% of utterances in every split (seed 2026), all audio features (head and gate inputs) replaced by those of a random utterance from a different dialogue in the same split; head and gates refitted | known substitution |
Seeds: head fits are deterministic, so head seeds 0, 1, 2 only set the C1 subset. Gate seeds 0, 1, 2 for every head (C0, S0 to S2, P: 3 gates each; C1, C2: 9 each).

## Manipulation check (required before any gate result is read)
Table per arm on dev and test: audio-head accuracy, NLL, calibration error, mean distinctiveness (gate-free class). Required: C1 dev accuracy at least 2 points below C0, or dev NLL clearly higher (interval excluding 0). If 25% does not achieve this, C1 and C2 are rerun with 10% (fixed now as the only fallback). C2 is reported with how close it gets to C0's distinctiveness on test, and with its calibration, since only the mean is matched. Validation: C0 refit reproduces the original M1 audio head's dev probabilities to 1e-4.

## Outcomes, kept separate
1. Prediction: audio-head accuracy, NLL, calibration error; fused weighted F1 (corrected rule, dev alpha).
2. Gate behaviour (primary, retrained gates): shift in mean logit(w) from the arm's reference (C0 for C1, C2, P; S0 for S1, S2); AUROC of w for reliability R (rows where exactly one head is correct); Spearman of w with tau0.
3. Influence (reported as a consequence): agreement of w > 0.5 with the full-pool recomputed-gate winner; excess over the random-assignment null; share of rows where (w > 0.5) and (w > tau) disagree; mean absolute head-path and gate-path terms per modality.
4. Planted arm, separate outcomes on test: (a) provenance, was the audio substituted; (b) was the audio head wrong on that row; (c) did audio cause the fused error (fused wrong, text head right, audio head wrong). AUROC of the retrained w for each of (a), (b), (c), each also for the confidence rule's weight as a baseline. The planted rows' audio swap effects are reported descriptively (no assumption about their size: a gate reacting to mismatch can make them large).

## Hypotheses and how they are read
Interval below means the dialogue-bootstrap 95% interval of the statistic, averaged over all gates of the arm in each draw. "Supported" needs the interval on the predicted side of 0; "contradicted" needs it entirely on the opposite side; otherwise "inconclusive".
- H-A, the gate follows reliability: in C2, the retrained gate gives audio less weight than C0 (shift in mean logit(w) above 0). Also required for "supported": the shift has the same sign for all three C1 subset seeds (each reported).
- H-B, likelihood versus sharpness: if the gate follows the head's likelihood, audio weight falls in BOTH S1 and S2 relative to S0 (both move away from the NLL optimum). If it follows sharpness, audio weight rises in S1 and falls in S2. Reported as which pattern the two intervals show; a mixed or flat result is "inconclusive".
- H-D, the gate detects substituted audio: in P, AUROC of the retrained w for provenance (a) has a lower bound above 0.5. Wording if supported: "the gate detects the planted rows" (it may be detecting text-audio mismatch), not "the gate knows the audio is useless".
- Secondary, one mechanism across arms: gate shift against the change in the audio head's dev NLL for C1, C2, S1, S2 and P; reported as a plot and a rank correlation (five points, descriptive).
- Reported whatever happens: excess over the random-assignment null in every retrained arm (a lower bound above 0.02 would mean the learned assignment adds self-consistency in that arm).

## Uncertainty and cost
Dialogue-cluster bootstrap on test, statistics averaged over the arm's gates within each draw; per-seed values reported alongside. Five audio-head fits (C0, three C1, P; a fallback adds three), 33 gates, full-pool swaps at about 5 to 20 s per gate. About 2 h compute, 4 h implementation.

## Outputs
`scripts/59_controlled_series.py` (training and evaluation arrays), `scripts/60_controlled_series_analysis.py` (statistics), `data/controlled_series_arrays.npz` (local), `results/controlled_series.csv`, `results/controlled_series_manipulation_check.csv`, `results/controlled_series_arm_settings.csv`. (Output list updated 2026-11-19 before any run: the work is split into two scripts.)

## Changes from the first draft (review of 2026-11-19)
Renamed the two meanings of "frozen gate"; frozen analyses relabelled as mechanistic baselines; manipulation check with a fixed fallback; H-B replaced by a directional likelihood-versus-sharpness test centred on the dev-NLL optimum (with an S0 reference); "falsified" replaced by supported / contradicted / inconclusive; same-sign requirement over the three C1 seeds; planted hypothesis reworded as detection of substitution with separate AUROCs, and the assumption that planted audio has smaller swap effects removed; head-path and gate-path terms added; secondary NLL-shift analysis added.

## Results (recorded 2026-11-19 after the run; `results/controlled_series.csv`, `results/controlled_series_manipulation_check.csv`, step 61 printed output)
Validation: the C0 refit reproduces the original M1 audio head's dev probabilities exactly (max difference 0). T* = 1.097.
Manipulation check: C1 (25%) passed on dev NLL (lower bound of the increase 0.012) with small dev accuracy drops (0.011, 0.018, -0.008), so the fallback was not triggered. BUT on test the C1 heads are not less accurate than C0 (0.479 to 0.486 vs 0.451) and have similar NLL. Post hoc (step 61, added after the results): on the gate split, where the gate is trained, the C1 heads have higher NLL (1.480 to 1.522 vs 1.460), similar accuracy (0.455 to 0.477 vs 0.471) and are overconfident (mean max probability 0.534 to 0.554 vs 0.471). S1/S2 change distinctiveness as intended (test 0.222 to 0.416 and 0.093) with identical accuracy; P lowers audio accuracy to 0.369 on test.
- H-A, per the rule: SUPPORTED. C2 shift in mean logit(w) toward text 0.929 [0.895, 0.964], positive for all three seeds (1.101, 0.962, 0.724). Reading limit: because the C1/C2 heads are not less accurate on test, this is not evidence that the gate follows accuracy; it shows the gate moves away from a head whose likelihood on the gate's training data is worse (here, an overconfident head).
- H-B: LIKELIHOOD PATTERN. Relative to S0, audio weight falls under both sharpening (S1, shift 0.338 [0.299, 0.378]) and flattening (S2, 0.560 [0.529, 0.588]); both raise the audio head's dev NLL (0.219 and 0.102). Not monotone in NLL size (S1 has the larger NLL change but the smaller shift).
- H-D: INCONCLUSIVE. AUROC of the retrained w for planted rows 0.522 [0.497, 0.546] (confidence rule 0.512). Yet P's gates shift strongly toward text overall (1.360 [1.292, 1.432]). The gate lowers its trust in audio globally but does not find the planted utterances. Per-utterance reliability AUROC falls to 0.545 in P (0.588 in C0). Planted rows' audio swap effects are similar to clean rows' (0.050 vs 0.053).
- Secondary: rank correlation of gate shift with dev audio-NLL change across C1, C2, S1, S2, P = 0.10 (five points): no single line on dev NLL.
- Excess over the random-assignment null stays at or below 0.011 in every retrained arm (S2 0.011 [0.001, 0.022]); no arm reaches the 0.02 lower-bound flag. Agreement with the swap moves with the gate's overall level (S1 0.585, P 0.849, C0 0.735), as the structural account predicts.
Summary (wording revised 2026-11-19 after review): the retrained gate adapts its overall modality weighting to changes in the audio head's output quality and probability distributions, with behaviour consistent with likelihood-sensitive training (sharper, flatter, overconfident, or partly substituted audio all push weight toward text). This does not isolate likelihood as the unique cause. The gate did not demonstrate reliable detection of individual substituted utterances (AUROC 0.522 [0.497, 0.546]: at most weak detection, not shown to be impossible). Its per-utterance weights track which head is correct only weakly (AUROC 0.55 to 0.61).
Validation added after review: all saved arrays (including donor means) are finite; `scripts/60_controlled_series_analysis.py` now asserts this before any statistic. The runtime warning in the donor-mean product is the macOS BLAS warning seen earlier, not a non-finite value.

## Amendment 1 (post hoc, written 2026-11-19 after the results above and before these checks were run; `scripts/62_posthoc_checks.py`, `results/controlled_series_posthoc.csv`)
Purpose: decide what the per-utterance claim may say.
(a) Is per-utterance information present in the gate's inputs? A probe with the gate's own architecture and inputs (MiniLM 384-d + pooled audio 1,024-d; 128 hidden units), trained with binary cross-entropy on the gate split, early stopping on dev, seeds 0, 1, 2, scored on test: (a1) planted versus clean rows in arm P; (a2) on rows where exactly one C0 head is correct, which one. Compared with the retrained gate's AUROC for the same target (P: 0.522; C0: 0.588), paired over the same dialogue draws.
Reading: if the probe minus gate interval is above 0, the inputs carry more than the gate uses ("information present, not used"). If the probe's own interval includes 0.55, per-utterance reliability is barely recoverable from these inputs, and the slide says so instead.
(b) Is the global shift what a constant weight would do? For each arm, the single weight minimising the gate's training loss (class-weighted fused NLL) on the gate split; its logit shift from C0's (S1, S2 from S0's) compared with the learned gates' mean logit shift.
Reading: the constant reproduces the global shift if its shift is within 25% of the learned mean shift in every arm.
(c) Does likelihood on the gate split explain the shifts? Per arm: the audio head's NLL on the gate split, and audio's marginal value given text (class-weighted NLL of text alone minus NLL at the best constant weight), with rank correlations of the learned shift against each, over C1, C2, S1, S2, P (descriptive, five points). Reported specifically: C2's gate-split NLL against C0's.
