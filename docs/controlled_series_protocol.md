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
`scripts/59_controlled_series.py`, `data/controlled_series_*.npz` (local), `results/controlled_series.csv`, `results/controlled_series_manipulation_check.csv`.

## Changes from the first draft (review of 2026-11-19)
Renamed the two meanings of "frozen gate"; frozen analyses relabelled as mechanistic baselines; manipulation check with a fixed fallback; H-B replaced by a directional likelihood-versus-sharpness test centred on the dev-NLL optimum (with an S0 reference); "falsified" replaced by supported / contradicted / inconclusive; same-sign requirement over the three C1 seeds; planted hypothesis reworded as detection of substitution with separate AUROCs, and the assumption that planted audio has smaller swap effects removed; head-path and gate-path terms added; secondary NLL-shift analysis added.
