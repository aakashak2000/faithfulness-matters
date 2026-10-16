# Project state: Faithfulness MATTERS

Written 2026-10-15 (phase 1, "Understand"). Nothing in the repo was changed except this file. Sources: CLAUDE.md, docs/PROJECT_REQUIREMENTS.md, docs/RUBRIC.md, docs/proposal.pdf, README.md, MIDTERM_STATUS.md and READING_GUIDE.md (both local-only), plus five read-only code and results audits. Every number below was checked against the file named next to it. Nothing was re-run.

Dates: the midterm slot is not assigned yet (slots are final on Oct 16, 12 PM). Slides could be due as early as Oct 18, so anything in the "before midterm slides" bucket must be done by **Oct 17**. Midterm report, final presentation and final report dates are not announced; left blank.

## 1. Research question (3 sentences)

Multimodal emotion models mix text and audio with a per-utterance weight, and that weight is easy to read as "how much each modality mattered." We train a small text+audio classifier on MELD (frozen MiniLM and WavLM-large encoders) and test whether its learned gate weight and its confidence weight agree with what actually happens to the prediction when we remove the text or the audio. This is an evaluation-of-explanations project, not an accuracy project, and its conclusions apply to our model on MELD only.

## 2. The project as 8 steps

The step numbers below follow the step-review skill. They are not the same as script numbers (script 02 is step 3, and so on).

### Step 1. Problem framing

- **Purpose:** state the question, hypotheses and what counts as done.
- **Files:** docs/proposal.pdf, docs/PROJECT_REQUIREMENTS.md, docs/RUBRIC.md.
- **Outputs:** H1 (midterm): importance signals agree with intervention effects beyond a trivial baseline. H2 (final): agreement drops when text and audio conflict.
- **Key numbers:** H1 status in PROJECT_REQUIREMENTS.md line 29: gate gain +0.014 [-0.007, 0.036] under mean replacement, -0.032 under blank replacement (both match results/bootstrap_ci.csv: 0.01405 [-0.00734, 0.03638] and -0.03218 [-0.05450, -0.01003]). H2 not tested.
- **Unfinished or inconsistent:**
  - The proposal costs the project as "one GPU (CARC Discovery)". The real setup is an M4 MacBook (MPS, step 3 only) plus Colab Pro. The rubric says the proposal lost half a point for vague resources, so real runtimes should replace the CARC wording in later documents.
  - The rubric names Hessel & Lee 2020 as the closest multimodal prior work and says to position against it. READING_GUIDE.md assigns no owner to that paper, and the proposal does not cite it.
  - Requirements doc lists slot and all due dates as `_fill in_`.

### Step 2. Data and audit

- **Purpose:** show the data, alignment and splits can be trusted before any result is read.
- **Files:** scripts/00_inspect_assets.py (prints legacy .pt structure), scripts/01_build_dataset.py (builds data/dataset.pt, writes results/audit_step1.json), scripts/diag_duplicates.py, diag_order.py, diag_source_files.py (one-off diagnostics, write nothing).
- **Outputs:** data/dataset.pt (local), results/audit_step1.json (committed).
- **Key numbers (results/audit_step1.json):**
  - Alignment: dev 1108/1108 and test 2610/2610 rows match the MELD CSVs exactly; fuzzy matches 0; label agreement 1.0.
  - Only `dia110_utt7` (dev) is skipped.
  - clf/gate max class-proportion gap 0.0004.
  - Non-ASCII text: dev 308/1108 (27.8%), test 757/2610 (29.0%). Winning decoding `raw_meld_c1` on 308/308 dev rows and 742/757 test rows.
  - Duplicate audio: `within_test` 22, `test_in_train` 3.
- **Unfinished or inconsistent:**
  - **FIXED 2026-10-15:** wording corrected in CLAUDE.md and MIDTERM_STATUS.md, and the breakdown (39 within test in 17 groups, 3 train-only, 42 flagged) is saved in `results/audio_duplicates.json` by `scripts/diag_duplicates.py`. Original finding follows.
  - **The "42 test rows share audio with a neighbour" number was loose.** The audit JSON never says 42. It says 22 (extra copies, `sum(v-1)`). The auditor's recount from data/dataset.pt gives 17 within-test groups covering 39 rows, plus 3 rows shared only with train, which makes 42 (the `audio_shared` flag, set in scripts/common.py:41-56 at load time). So "42 share with a neighbouring utterance" should read "39 within test plus 3 shared with train". Three numbers (22, 39, 42) describe similar things and no file reconciles them.
  - "17 of 17 source MP4s byte-identical" comes from diag_source_files.py console output. It is not saved in results/.
  - The audit JSON has facts the status doc omits: 7 duplicate audio rows between the clf and gate training splits (`clf_in_gate`), 15 test rows where plain utf8 beat the raw decoding, and the old standalone `*_tensors.pt` files disagree with the bundle (audio cosine about 0.47, label agreement 25 to 29%).
  - The "20 clips re-extracted, cosine 1.00000" check is stored only inside data/intervention_embeddings.pt (min cosine 0.99999666), not in results/.
  - Train-side alignment is not audited (meta is kept for dev and test only).
  - data/ is gitignored, so nothing in this step re-runs from a clean clone; the JSON is the only committed evidence.

### Step 3. Models and baselines

- **Purpose:** build the model whose explanations we test, and show it works about as well as a reasonable baseline.
- **Files:** scripts/02_train.py, scripts/common.py. Text head 384->7, audio head 1024->7 (linear), trained on the clf split (80% of train, class-weighted, early stopping on dev). Learned gate: MLP on [e_t; e_a] (1408->128->1->sigmoid), trained on the gate split (the other 20%) only. Fusion `p = w_t*p_text + (1-w_t)*p_audio` with w_t = 0.5 (average), c_t/(c_t+c_a) (confidence), or the gate. Three seeds, CPU, deterministic.
- **Outputs:** checkpoints/seed{0,1,2}.pt (local), results/baselines_summary.csv, baselines_per_seed.csv, per_class_f1.csv, legacy_reproduction.csv.
- **Key numbers (results/baselines_summary.csv, test, mean over 3 seeds):**

| Model | Acc | Macro-F1 | Weighted-F1 (std) |
|---|---|---|---|
| Text only | 0.452 | 0.353 | 0.4914 (0.0013) |
| Audio only | 0.332 | 0.277 | 0.3680 (0.0024) |
| Average fusion | 0.476 | 0.382 | 0.5148 (0.0008) |
| Confidence fusion | 0.476 | 0.383 | 0.5145 (0.0015) |
| Learned gate | 0.477 | 0.382 | 0.5175 (0.0012) |

  - Gate text weight on test: min 0.0845, max 0.9988, mean 0.6284, std 0.2213 (computed from `gate_w_t` in results/interventions_log.csv; step 3 prints it but saves nothing).
- **Unfinished or inconsistent:**
  - **"Reproduces the old numbers within 0.003" is false for two models.** Weighted-F1 new minus legacy (baselines_summary.csv vs legacy_reproduction.csv): text +0.0010, audio +0.0009, average +0.0032, confidence +0.0051. Confidence is also off by 0.0049 on accuracy and 0.0047 on macro-F1. The claim appears in MIDTERM_STATUS.md 5.1 and in the Q&A row about published F1.
  - "The gate performs no better than a fixed 50/50 average" is arguable: 0.5175 vs 0.5148 is +0.003 with seed std near 0.001. Not tested.
  - The gate and its early stopping use dev; legitimate, but dev is used for both head and gate selection.
  - Checkpoints store no hash of dataset.pt, so steps 4 to 6 silently assume the dataset is unchanged.
  - per_class_f1.csv has no std.
  - Our F1 is below the MELD paper's bcLSTM text+audio 59.25 by design (frozen encoders, no dialogue context). Stated in the docs, as required.

### Step 4. Intervention design

- **Purpose:** define, before seeing results, how "remove a modality" is done and what is measured.
- **Files:** scripts/03_silence_embeddings.py and the protocol header of scripts/04_interventions.py.
- **What it does:**
  - Step 03 builds the "blank" replacements: WavLM on an all-zero waveform of each test clip's length (rounded to 0.1 s, 128 buckets) and MiniLM on the empty string.
  - Step 04 loads the 3 checkpoints and, for each seed and each of 3 fusion models, runs mean and blank replacement of text and of audio (mean = mean clf-split embedding). Signals are read on the original input. Effect is the change in the fused probability of the originally predicted class.
- **Outputs:** data/intervention_embeddings.pt, data/intervention_probs.pt (both local), results/interventions_log.csv (committed).
- **Key numbers:**
  - results/interventions_log.csv has 46,980 rows x 22 columns = 2610 x 3 seeds x 3 models x 2 methods, no duplicate keys, no NaNs.
  - Mean |dp| (results/faithfulness_summary.csv), gate model, mean replacement: text 0.158, audio 0.057; blank: 0.208 and 0.087. Confidence model: 0.120 / 0.057 (mean), 0.167 / 0.100 (blank).
  - Flip rates (same file): gate mean 48% text, 21% audio; gate blank 70%, 47%; confidence mean 44%, 26%; confidence blank 68%, 56%.
- **Protocol vs code:** the header, MIDTERM_STATUS.md section 7 and the code agree. No hard mismatch.
- **Unfinished or inconsistent:**
  - Exact ties are excluded in scripts/05_faithfulness.py:44-48, but the protocol header does not say so (only MIDTERM_STATUS.md does). There were no ties anyway (n_signal_tied and n_effect_tied are 0 in results/agreement_counts.csv).
  - **No silence or empty-text embeddings exist for dev or clf.** The cache has a "test" key only. The gate diagnosis (fit threshold on dev, apply to test) needs a new WavLM pass on dev (about minutes on MPS).
  - No staleness check: 04 only asserts the row count, so a reordered test set of the same size would pass silently.
  - Full 7-class probabilities live only in data/intervention_probs.pt (not committed). Teammates have the CSV but only TVD, flips and dp.
  - The 20-clip cosine check is the only alignment test for the 2610 silence rows.
  - **Cosine-distance numbers (mean audio 0.68, silent audio 0.30, mean text 0.37, empty text 0.18) are printed by scripts/03_silence_embeddings.py:112-123 only.** They are saved in no results file, but they back the main Q&A answer about out-of-distribution inputs.

### Step 5. Faithfulness metrics

- **Purpose:** turn the log into the headline comparison.
- **Files:** scripts/05_faithfulness.py. Signal s = 2*w_t - 1. Effect e = |dp_text| - |dp_audio|. Agreement = share where sign(s) = sign(e). Baseline = share where e > 0 (what "always text" scores). Gain = agreement minus baseline. Spearman of s against e. Primary pairs: gate with its gate weight, confidence with its confidence weight. Secondary: confidence weight applied to the gate model and the average model.
- **Outputs:** results/faithfulness_per_seed.csv, faithfulness_summary.csv, method_robustness.csv.
- **Key numbers (results/bootstrap_ci.csv):**

| Model <- signal | Method | Agreement | Always-text | Gain [95% CI] | Spearman |
|---|---|---|---|---|---|
| Gate <- gate | mean | 0.797 | 0.783 | +0.014 [-0.007, 0.036] | 0.734 [0.713, 0.753] |
| Gate <- gate | blank | 0.773 | 0.805 | -0.032 [-0.054, -0.010] | 0.614 [0.585, 0.641] |
| Confidence <- confidence | mean | 0.927 | 0.688 | +0.239 [0.220, 0.257] | 0.945 [0.939, 0.949] |
| Confidence <- confidence | blank | 0.788 | 0.710 | +0.079 [0.059, 0.101] | 0.577 [0.544, 0.609] |
| Gate <- confidence | mean | 0.793 | 0.783 | +0.010 [-0.012, 0.029] | 0.659 |
| Gate <- confidence | blank | 0.703 | 0.805 | -0.102 [-0.125, -0.081] | 0.297 |
| Average <- confidence | mean | 0.898 | 0.675 | +0.223 [0.205, 0.240] | 0.895 |
| Average <- confidence | blank | 0.738 | 0.661 | +0.077 [0.057, 0.098] | 0.511 |

  - Robustness (results/method_robustness.csv): mean and blank pick the same winning modality on 78% (average), 81% (confidence), 86% (gate); Spearman of effect differences 0.60 to 0.63.
  - Near-zero effects (< 0.02): 241 to 432 of 2610 per setting (results/agreement_counts.csv); excluding them raises agreement by 0.019 to 0.040.
- **Unfinished or inconsistent:**
  - "Dropping the 42 shared-audio rows changes all metrics by 0.002 or less" holds for agreement (max 0.0020) but not for every column of results/faithfulness_summary.csv: always-text baseline 0.0024, flip rates up to 0.0025, Spearman 0.0038, signal_says_text 0.0045.
  - "Agreement varies by 0.01 or less across seeds" holds as a std (max 0.0113) but not as a range: gate<-confidence blank spans 0.0203 and gate<-gate mean spans 0.0103 (results/faithfulness_per_seed.csv).
  - The "confidence agreement is by construction" claim rests on a mock-data score of "about 95 to 100%". **No script, results file or doc contains that experiment.** It currently has no source, which breaks the rule that every number names a file. (The planned random-head control in Tier B would produce it properly.)
  - The "same winner" check counts a zero effect as agreeing with a zero effect (scripts/05_faithfulness.py:87-95); harmless here because there are no ties.
  - `spearman_p` is computed and dropped. `agreement_nonzero` has no CI.
  - Alternative effect metrics (TVD, flips) are logged but not yet used for agreement.

### Step 6. Uncertainty, calibration and figures

- **Purpose:** put confidence intervals on the headline numbers and explain why the gate fails.
- **Files:** scripts/06_analysis.py. Dialogue-level bootstrap: 1000 resamples of the 280 dialogues, rng seed 0, same draws reused for every pair and method, the 3 seeds averaged inside each draw (so intervals are conditional on these 3 trained models), percentile intervals.
- **Outputs:** results/bootstrap_ci.csv, agreement_counts.csv, gate_calibration.csv, figures/fig1..fig4.
- **Key numbers (results/gate_calibration.csv, share where text mattered more, mean / blank, pooled over seeds):** by gate text weight 0.1 to 0.2 (n=157) 20% / 26%, 0.2 to 0.3 (452) 27% / 38%, 0.3 to 0.4 (814) 46% / 56%, 0.4 to 0.5 (1015) 62% / 67%, 0.5 to 0.6 (1054) 78% / 81%, 0.6 to 0.7 (1170) 90% / 92%, 0.9 to 1.0 (1092) 100% / 98%. The curve crosses 50% at a gate weight of about 0.37 (mean) and 0.32 (blank), not 0.5.
- **Unfinished or inconsistent:**
  - MIDTERM_STATUS.md 5.4 writes the 0.4 to 0.5 mean value as 63%; the file says 0.6246, which is 62% (and fig3 shows 62%).
  - MIDTERM_STATUS.md 5.4 omits three bins that are in the CSV: 0.0 to 0.1 (n=4, 0% / 0%), 0.7 to 0.8 (n=1013, 0.947 / 0.940) and 0.8 to 0.9 (n=1059, 0.984 / 0.956). The omission was a hand edit, not a script filter.
  - The doc says the curve crosses 50% "at roughly 0.35 to 0.4"; that fits mean replacement only. Blank crosses near 0.32.
  - fig2: the left panel's x-axis label runs into the right panel's, and the right one is clipped.
  - fig3 drops the n=4 bin (the plot filters bins under 30 rows). fig4 is seed 0, mean replacement only, and neither its title nor the doc says that; the doc's Spearman 0.734 is pooled over 3 seeds.
  - The 0.02 threshold is a constant in 05 but a literal in 06 (06:90); the pair list is duplicated in 05 and 06. Both can drift.
  - No intervals on the calibration table. The paired gate-vs-confidence comparison (same draws) is not implemented.
  - Doc tables are copied by hand from script output, which is how the errors above got in.

### Step 7. Midterm presentation

- **Purpose:** 2 slides, 3 minutes including Q&A, one presenter, slides due 24 h before the slot (possibly Oct 18; do slide work by Oct 17). Graded 5 points by instructors and peers (Clarity 1, Novelty 1, Progress 2, Q&A 1).
- **Files:** none yet in the repo. A draft outline and Q&A table exist in MIDTERM_STATUS.md section 11 (local). Slide 2 is planned around figures/fig3_gate_calibration.png.
- **Key numbers planned for the slides:** gate gain +0.01 (not significant) and -0.03 (results/bootstrap_ci.csv), Spearman 0.73 (same file).
- **Unfinished or inconsistent:**
  - No slides, no timed script, no presenter named.
  - Novelty rubric line needs papers named on the slide (Jacovi & Goldberg 2020, Jain & Wallace 2019, Wiegreffe & Pinter 2019, Hessel & Lee 2020, He & Xiao 2025). Hessel & Lee has no owner (step 1).
  - Several Q&A answers lean on numbers with no results file: the cosine distances (step 4), the "about 95%" mock-data score (step 5), the "within 0.003" reproduction claim (step 3, false), and "42 shared rows" (step 2, loose). They need fixing or rewording before they are spoken.
  - Section 11 lists "gradient-based signal" as a next step while the tracker puts it in Tier C (only if time allows). The slide's remaining-work list should match the tracker.

### Step 8. Plan to the final

- **Purpose:** test H2 and explain the gate finding, then write the report.
- **Files:** MIDTERM_STATUS.md section 9 (tracker); PLAN.md does not exist yet.
- **Tier A:** diagnose the gate (dev-fit threshold, frozen, applied to test; balanced accuracy; split by whether text and audio heads agree); paired gate-vs-confidence comparison; conflict set (text paired with other-emotion audio, same-label swap control, never the same recording on both sides, judged by gain over always-text); bootstrap statistics for the conflict effect.
- **Tier B:** alternative effect metrics (TVD, flips); weight x head-disagreement decomposition; per-class and neutral-excluded results; per-utterance seed stability; random or label-shuffled head control for the confidence signal.
- **Tier C:** gradient-based signal; clean UTF-8 text re-embedding; second gate design; stronger trivial baselines.
- **Unfinished or inconsistent:**
  - Nothing here has a plan file yet (docs/plans/INDEX.md is empty).
  - Gate diagnosis and the conflict set both need new WavLM passes (dev blank audio; swapped-audio embeddings), which the current caches do not contain.
  - Risk section: the proposal lost a point for risks without warning signs or pivots. Risks we hit (out-of-distribution blank inputs, text dominance making the baseline strong, the confidence signal's structural agreement, MELD duplicate clips) and the decision each changed are not yet written up in one place.
  - Midterm feedback will need incorporating once it arrives.

## 3. Gap check: requirements and rubric vs what exists

| Requirement or rubric line | What exists | Gap |
|---|---|---|
| Midterm slides, 2 slides, deadline 24 h before slot | Outline and Q&A in MIDTERM_STATUS.md section 11 | No slides, no script, no presenter, slot unknown |
| Clarity: problem, inputs, outputs | Plain-language framing in MIDTERM_STATUS.md sections 1 and 2 | Needs a one-sentence version on slide 1 |
| Novelty: name specific prior work and what we add | Papers assigned to 5 teammates | Hessel & Lee 2020 unassigned; related-work paragraphs not yet in the repo |
| Progress (2 pts): real number or figure, 2 to 4 scoped remaining items | Gate gain with CI, fig3 | Remaining-work list must match the tracker (gradient signal is Tier C) |
| Q&A: short direct answers | 7 drafted answers | 4 rest on unsourced or incorrect numbers (see steps 3, 5, 7) |
| Every number traceable to results/ | Most are | Unsourced: cosine distances, mock-data 95 to 100%; loose: 42, 0.003, 0.002, 0.01, 63% |
| Midterm report (4 pages) | Nothing drafted in repo | All sections; risk section with real hits and decisions |
| Hypothesis H1 | Tested | Reported as not supported for the gate; held up by CIs |
| Hypothesis H2 (conflict) | Designed in text only | Not started |
| Compare with MELD bcLSTM 59.25 | Stated in docs | Keep stating the reason ours is lower |
| Real resource numbers (rubric: 1.5/2 on resources) | Runtimes in README and CLAUDE.md | Not yet in any deliverable; proposal still says CARC GPU |
| Reproducible from README | Scripts and README exist | data/ is local only; no staleness checks; legacy files not regenerable |
| Alternative effect metric (done criterion for final) | TVD, flips logged | Not run |
| Gate diagnosis with dev-fitted threshold | Not started | Needs dev blank embeddings (new WavLM pass) |
| Subagents, PLAN.md, step plans | .claude/agents present | PLAN.md and docs/plans/step-*.md do not exist |

## 4. Open questions for Aakash

1. Slot and due dates: update docs/PROJECT_REQUIREMENTS.md on Oct 16 after 12 PM. Until then, treat Oct 17 as the finish line for slide-bucket work.
2. Who presents the midterm slides, and who takes Hessel & Lee 2020?
3. MIDTERM_STATUS.md is local-only and holds the errors listed above (0.003, 63%, the omitted calibration bins, 0.002, 0.01, 42). Do you want them corrected in the file, in a later step, before teammates quote them?
4. The "about 95%" mock-data score has no source. Should the slides drop it, or is it worth a short run before Oct 17 that saves a file in results/? (The planned label-shuffled-head control would produce the same number properly.)
5. The cosine distances (0.68, 0.30, 0.37, 0.18) exist only in console output. Do you want them saved to a results file? That means re-running step 3, about 5 minutes on MPS, or extracting them from the saved embeddings.
6. How should "42" be worded: "39 within test plus 3 shared with train"? Which of 22, 39 and 42 do you want as the reported figure?
7. Slide 2 leads with the mean-replacement result (+0.014, not significant) and the blank result (-0.032). Do you want both numbers on the slide, or mean only with blank as a robustness note?
8. Are you comfortable with the "gate performs no better than 50/50 average" wording? The file shows +0.003 weighted-F1 for the gate, which is small but not zero.
9. Should the dev-set blank-embedding pass (needed for the gate diagnosis) be done before the midterm report, or only before the final?
