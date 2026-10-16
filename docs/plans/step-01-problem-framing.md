# Step 01: Problem framing
Agreed: 2026-10-15

## What exists
- Proposal (graded 16.5 / 20; lost points on novelty 3/5, risk 1/2, resources 1.5/2), `docs/PROJECT_REQUIREMENTS.md`, `docs/RUBRIC.md`.
- Question: do a text+audio model's modality weights (learned gate, confidence rule) match what happens to its prediction when we remove the text or the audio? Evaluation of explanations, not accuracy.
- Definitions as used: for each utterance, T = text mattered more (|dp_text| > |dp_audio|, from the intervention); S = the signal says text (weight above 0.5); agreement = share S = T; always-text baseline = share T = text; gain = agreement - baseline = (right "audio" calls - wrong "audio" calls) / N.
- H1 (gate, natural utterances), gain over always-text with 95% CI, from `results/resampling_full_faithfulness.csv`: full-pool resampling +0.019 [-0.003, 0.041] (primary); mean +0.014 [-0.007, 0.036]; blank -0.032 [-0.054, -0.010]; 50-donor draws +0.018 and +0.029.
- H2 (conflict) not started.

## Assessment
Strengths: question and scope are clear; the consistency framing is honest and stated in the proposal; definitions are fixed in script headers; every headline number traces to `results/`.
Weaknesses: novelty is stated as what we test, not what is new and who acts on it; risks are named without warning signs and the decision each changed; the proposal's resource line (one CARC GPU) does not match the real setup (M4 MacBook MPS plus Colab Pro); Hessel & Lee 2020 (named in the rubric as closest multimodal work) has no owner; the H1 status line in the requirements doc predates resampling.

## Decisions (with alternatives rejected and why)
1. The intervention test is ONE component, a consistency check, not a golden test of faithfulness. Rejected: presenting a mismatch as proof the gate is unfaithful (there is no ground truth for "how much a modality mattered").
2. "Intervention tests" is the key phrase for the midterm presentation.
3. Full-pool resampling (every clf row as donor, exact average) is the primary intervention from now on; mean and blank are sensitivity checks; the two 50-donor draws stay as a sensitivity check. Rejected: keeping mean as primary (least in-distribution after blank); picking the better 50-donor draw. Matching mean or blank does not validate it. Pre-registered in `scripts/07_resampling.py` and `scripts/08_resampling_full.py` before running.
4. Honest reading of the gate: "at most a small edge over always-text" (point estimates +0.014 to +0.029; the full-pool CI upper bound is 0.041, so up to about 4 points cannot be ruled out). The negative result under blank is treated as a stand-in artifact, not a finding (interpretation, supported by the blank-vs-resampling difference +0.051 [0.029, 0.071], `results/resampling_full_gain_differences.csv`).
5. The current model stays the headline; a stronger audio head is a robustness variant, not a replacement (any new model reruns steps 4 to 6).
6. Robustness work comes before slide work; the resampling result goes on the slides only if finished, checked and reviewed by Oct 17 and its gate-gain CI excludes 0. It does not (CI includes 0), so it is Q&A and report material.
7. Resampling is a measurement tool on natural utterances and is never sliced by donor emotion; it stays distinct from H2.

## Actions
| # | Action | Serves | Bucket | Effort (h) | Acceptance check |
| --- | --- | --- | --- | --- | --- |
| 1a | Update the H1 line in `docs/PROJECT_REQUIREMENTS.md` section 2 with the resampling result and the new primary intervention | Requirements section 2; traceable numbers | Before midterm slides | 0.25 | Numbers match `results/resampling_full_faithfulness.csv` and `bootstrap_ci.csv` |
| 1b | Write the novelty statement: finding, implication, who acts on it, and positioning against Jacovi & Goldberg, Jain & Wallace, Wiegreffe & Pinter, Hessel & Lee, He & Xiao | Rubric: Novelty (1 pt); proposal lost 2 pts here | Before midterm report | 1 | rubric-grader gives full marks on the Novelty line |
| 1c | Assign an owner for Hessel & Lee 2020 and add the row to `READING_GUIDE.md` | Rubric: Novelty; Related work | Before midterm report | 0.25 | Row exists with an owner named |
| 1d | Risk table: risk, warning sign, decision made, status, for the risks actually hit | Rubric: Risk (proposal lost 1 pt) | Before midterm report | 1 | Each row cites a file in `results/` |
| 1e | Resources paragraph with real hardware and measured runtimes, replacing the CARC wording | Rubric: Resources (proposal lost 0.5 pt) | Before midterm report | 0.5 | Every runtime comes from a timed run |
| 1f | Add steps 7 and 8 (resampling) to the command list in `CLAUDE.md` | Reproducibility | Before midterm report | 0.1 | Both commands run clean from the repo root |

## Open questions
- Slide takeaway: keep the planned line (A) or add "the gate beats always-text where audio is confident" (B)? Not answered. (Evidence: `results/audio_subset_faithfulness.csv`, an after-the-fact subset.)
- Access to raw MELD train and dev audio, which decides whether the stronger audio head (5b) is possible beyond an MLP on cached embeddings.
- Owner for Hessel & Lee 2020.
- Final wording of the gate verdict, given the full-pool CI upper bound of 0.041.

## Effects on other steps
- Step 3: items 3 to 5 done (`results/mean_embedding_*.csv`, `results/audio_subset_faithfulness.csv`); stronger audio head (5b) pending.
- Step 4: resampling is now the primary intervention; `scripts/07_resampling.py`, `scripts/08_resampling_full.py` added; cosine distances still need saving to `results/`; held-fixed gate diagnostic (item 10) pending.
- Step 5: `results/resampling_full_faithfulness.csv` now holds all five methods; AUC and TVD or flips agreement (items 11, 12) pending.
- Step 6: `scripts/06_analysis.py` and fig2 still show mean and blank only; they need a resampling version before the report.
- Step 7: slide condition for resampling not met; Q&A rows added to `MIDTERM_STATUS.md` section 11.
- Step 8: H2 must stay distinct from resampling (no slicing by donor emotion).

## Update 2026-10-15 (after the robustness checks)
The honest gate reading "at most a small edge over always-text (roughly 1 to 3 points)" is qualified: under the pre-registered primary (full-pool resampling, |dp| of the predicted class) the gain is +0.019 [-0.003, +0.041], but it is +0.090 and +0.061 when effect is the change in the whole distribution or a label flip (`results/alt_effect_metrics.csv`), and +0.049 [+0.035, +0.062] when the cutoff is fitted on dev instead of fixed at 0.5 (`results/dev_calibration_test.csv`). The primary stays the pre-registered one; the others are sensitivity checks. Summary: `docs/robustness_results.md`.
