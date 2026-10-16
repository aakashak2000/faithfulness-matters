# Step 07: Midterm presentation
Agreed: 2026-10-15 (decisions given by the user in the session; slide work starts 2026-10-16 in a fresh session)

## What exists
- No slides yet. A draft outline and Q&A table are in `MIDTERM_STATUS.md` section 11 (local only). Its numbers are stale (old mean/blank results, unsourced cosine and "95%" claims); sections 5.4 and 5.5 and the Q&A rows 5.x/11 were corrected on 2026-10-15, the slide outline in section 11 was not.
- Format (`docs/RUBRIC.md`): 2 slides, 3 minutes including Q&A, one presenter, slides due 24 h before the slot. Slot unassigned until Oct 16 12:00; treat Oct 17 as the finish line for everything in the "before slides" bucket.
- Evidence now available for the slides (all in `results/`):
  - Primary result, full-pool resampling: gate gain over always-text +0.019 [-0.003, +0.041]; Spearman 0.705 [0.680, 0.726]; confidence rule +0.225 (`resampling_full_faithfulness.csv`).
  - Dev-fitted cutoff, gate on test: +0.049 [+0.035, +0.062] (`dev_calibration_test.csv`; thresholds in `dev_calibration_thresholds.csv`).
  - Paired gate vs confidence: +0.006 [-0.017, +0.028] (`paired_gate_vs_confidence.csv`).
  - Label-shuffled control: confidence gain +0.225 with shuffled labels, same as real (`shuffled_control.csv`).
  - Sensitivity to the effect measure (`alt_effect_metrics.csv`), re-weighting diagnostic (`gate_fixed_diagnostic.csv`), H2 pilot (`conflict_h2_diff.csv`). Summary of all checks: `docs/robustness_results.md`.
  - Figures: `figures/fig3b_gate_calibration_primary.png` (primary calibration, built), `figures/fig2b_agreement_primary.png`.

## Assessment
Independent rubric-grader on the old draft: about 3/5 (Clarity 0.5/1, Novelty 0.5/1, Progress 1.5/2, Q&A 0.5/1). Faults: problem stated as a question with no inputs/outputs; "intervention tests" absent; prior work unnamed; stale numbers; three Q&A answers on stale or unsourced numbers; no answers to the hardest questions. Since then: the unsourced numbers now have files (`embedding_distances.json`, `shuffled_control.csv`, `wording_checks.json`), the figure for the primary intervention exists, and the confidence Q&A has a paired result. Still to do: the slide text itself, the named citations, the script and rehearsal, roles.

## Decisions (with alternatives rejected and why)
- **Slide 2 takeaway** (accepted because the check passed: the +0.049 was computed under the primary intervention, full-pool resampling with the |dp| of the predicted class as effect, and the cutoff was fitted on dev by a procedure written in the script header before test was evaluated; `scripts/19_dev_calibration.py`):
  > The gate's weights are informative but miscalibrated: at the 0.5 cutoff they do not beat always-text (+0.019, not significant), but they rank influence well (Spearman 0.70) and a cutoff fitted on dev beats it on test (+0.049 [0.035, 0.062]).
  - Wording note: Spearman is 0.7047 (`resampling_full_faithfulness.csv`), which rounds to 0.70, not 0.71.
  - Caveats that must stay in the Q&A: the interval is conditional on the fitted cutoff (it does not include dev sampling noise); the idea of a lower cutoff came from looking at the test-set calibration curve, but the cutoff values come from dev only; the verdict depends on how "effect" is measured (the primary is the pre-registered one).
  - Rejected: takeaway A ("keep the planned line") is only the fallback if the check had failed; takeaway B (audio-confident subset) because it was found after the fact and its baseline changes.
- **Primary intervention** is full-pool resampling; mean and blank are sensitivity checks (step 1 decision). It was promoted after seeing mean and blank results but before seeing resampling results; this must be disclosed as a protocol change in the Q&A and the report.
- **Strong audio head** is paused (`step-03-strong-audio-head.md`): before final, optional.
- **What remains** (slide 2): H2 as a pilot needing about 10 random pairings (the two pilot pairings disagreed in direction, `conflict_h2_diff.csv`); the report with the sensitivity checks; strong audio head only if time allows. The gradient signal is dropped from the slide.
- Roles (presenter, Hessel & Lee owner) are still open.

## Actions
| # | Action | Serves | Bucket | Effort (h) | Acceptance check |
| --- | --- | --- | --- | --- | --- |
| 7.8 | Decide the presenter and the owner for Hessel & Lee 2020 | Novelty, Q&A | Before slides | 0.25 | Two names written in this file |
| 7.1 | Rebase every slide and Q&A number on the primary (full-pool resampling), mean and blank in a footnote | Progress, Q&A | Before slides | 0.5 | Each number on the slides has a `results/` file named in the notes; no number from `MIDTERM_STATUS.md` section 11 survives unchecked |
| 7.2 | Slide 1: one problem sentence naming the inputs (an utterance's text and audio) and outputs (an emotion and a text weight; for our test, whether the weight matches the effect of removing each input); the phrase "intervention tests"; named citations (Jacovi & Goldberg 2020; Jain & Wallace 2019 and Wiegreffe & Pinter 2019; Hessel & Lee 2020; He & Xiao 2025); one "what we add" line (always-text baseline, ranking vs "which mattered more", several removal methods, a modality-conflict test) | Clarity, Novelty | Before slides | 1 | rubric-grader gives full marks on the Clarity and Novelty lines |
| 7.4 | Slide 2 figure: `fig3b_gate_calibration_primary.png`, with the takeaway above as the caption | Progress | Before slides | 0.25 (figure already built) | Figure shows the primary method; caption numbers match `dev_calibration_test.csv` and `resampling_full_faithfulness.csv` |
| 7.7 | Slide 2 "what remains": H2 pilot with about 10 pairings; report with sensitivity checks; strong audio head optional | Progress | Before slides | 0.25 | 2 to 4 scoped one-line items; none is the gradient signal |
| 7.5 | One-sentence answers to the likely questions (rows below) | Q&A | Before slides | 1 | Each answer names its results file and states its caveat |
| 7.6 | A script of about 260 words and three timed rehearsals | All four lines | Before slides | 1.5 | Three rehearsals under 2 minutes 10 seconds |
| 7.9 | Re-run the rubric-grader on the final slide text and script | All four lines | Before slides | 0.5 | Grade at least 4.5 of 5 or the named gaps fixed |

## Q&A rows (draft one-sentence answers)
| Question | Answer |
|---|---|
| Does the confidence rule explain the model better than the gate? | On the gate's own model they cannot be told apart: gain +0.025 vs +0.019, paired difference +0.006 [-0.017, +0.028] (`paired_gate_vs_confidence.csv`). |
| Isn't the confidence rule's high agreement circular? | Partly, yes: with every head trained on shuffled labels its gain is still +0.225, identical to real labels (`shuffled_control.csv`), so we do not count it as evidence of a good explanation. |
| Does the gate beat "always text"? | At the 0.5 cutoff no (+0.019 [-0.003, +0.041]); with a cutoff fitted on dev it does on test (+0.049 [0.035, 0.062], conditional on that cutoff). |
| Does that depend on how you measure the effect? | Yes: the gate's gain is +0.019 with the pre-registered effect (change in the predicted class's probability), +0.09 with the change in the whole distribution and +0.06 with label flips (`alt_effect_metrics.csv`); we report the pre-registered one as primary and the others as sensitivity checks. |
| Why did the primary intervention change? | Blank stand-ins sit far from real data (cosine similarity 0.30 vs 0.68 for the mean, `embedding_distances.json`), so we promoted full-pool resampling, which uses real embeddings, after seeing the mean and blank results and before seeing its own; mean and blank stay as sensitivity checks. |
| Is the blank result (-0.032) a finding? | No: we treat it as an artifact of the stand-in being unlike real data (three methods, blank is the outlier). |
| Does "zero" mean no effect or too little data? | The interval rules out gains above about +0.04 at the 0.5 cutoff (upper bound 0.041). |
| What about conflicting text and audio (H2)? | A pilot found no drop in gain under conflict, but two random pairings disagreed in direction, so we need about 10 pairings before saying anything (`conflict_h2_diff.csv`). |
| How is this different from Hessel & Lee 2020? | Owner and wording to be filled in once someone has read it (open question). |
| Your F1 is below published MELD results. | We use frozen encoders and single utterances on purpose; the question is about explanations, and all fusion variants reproduce our earlier weighted-F1 within 0.006. |

## Open questions
- Presenter and Hessel & Lee owner (7.8).
- Final wording of the problem sentence and the "what we add" line (7.2).
- Raw MELD train and dev audio turned out to be local in `data/all_data/`, so that earlier question is closed.

## Effects on other steps
- Step 1 plan: the "at most a small edge" reading is now "a small edge that depends on the cutoff and the effect measure" (see its update note).
- Steps 5 and 6: sensitivity analyses live in `docs/robustness_results.md`; the report should show them as secondary.
- Step 8: H2 becomes a 10-pairing repeat; the strong audio head is optional.
