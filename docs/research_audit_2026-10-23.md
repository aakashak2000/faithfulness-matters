# Research audit, 2026-10-23 (P0, first pass)
Scope: what could be checked from the repo and saved arrays without retraining. Status words: PASS (recomputed here), FAIL (discrepancy found), UNVERIFIED (not checked, or cannot be checked from what is on disk).
Git: HEAD `24a2ae3`; working tree has 63 entries, 60 untracked. Everything from scripts 21 to 41, `results/` for steps 21 to 41, `docs/methodological_audit.md` and `docs/slides/` is NOT committed, so no result below has a commit hash. Fingerprints are file-level only until the user commits.

## (A) Repository inventory and discrepancies
| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Files named in the brief (scripts 30, 34, 37, 38; results harness_models, swap_with_context, corrected_signal, frozen_gate_audit, harness_alt_effects, harness_per_gate, mean_of_abs_check, audio_duplicates, balance_experiments; the two docs) | PASS, all exist | `ls` |
| 2 | Row-level arrays for M0b, M1, M2, M3, M5 | PASS, `data/frozen_gate_arrays_{M0b,M1,M2,M3,M5}.npz` (local only, gitignored): real, frozen and mean-of-abs effects, gate weights, per gate | numpy load |
| 3 | Donor pool | PASS: the classifier-training split ("clf", 7,990 rows), not the test set; a donor is skipped when its replaced-modality embedding is byte-identical to the test row's. For the 63.6 model donor stage-1 logits are out-of-fold | `scripts/08_resampling_full.py` header; `scripts/37_swap_with_context.py` header |
| 4 | Test rows as donors | PASS for M5, M1, M2, M3, M0b (clf-split pool). The brief's "may be legitimate but must be disclosed" does not arise | same |
| 5 | Dialogue counts of the dataset version used | UNVERIFIED (only utterance counts 9,989 / 1,109 / 2,610 are in the repo docs) | |
| 6 | Duplicate audio | PASS the counts: 17 within-test groups, 39 rows, 3 shared only with train, 42 total. UNVERIFIED: with/without-flagged-rows sensitivity of the swap results; not computed | `results/audio_duplicates.json` |
| 7 | Architecture table for M1 to M5 (exact checkpoints, parameter counts, frozen vs tuned) | UNVERIFIED: the repo documents heads over precomputed embeddings (MiniLM, RoBERTa layers, WavLM layers in `data/*_layers_*.pt`) but I have not tabulated checkpoints, parameter counts or confirmed that no encoder was fine-tuned | `scripts/27`, `29`, `30` |
| 8 | F1 recomputation from prediction arrays | UNVERIFIED: the F1 values below are read from CSVs; not recomputed from predictions in this pass | `results/harness_models.csv` |
| 9 | Seeds | Heads trained once (seed 0) and shared across 10 gates (M5: 3 gates). Bootstrap intervals therefore do not include head-training variance | `scripts/33_eval_harness.py` header |
| 10 | Slide Spearman and agreement numbers | PASS (see B) | |

## (B) Reproduced vs reported (from `data/frozen_gate_arrays_*.npz`, recomputed here; saved in `results/audited_master_results.csv`)
| Model | text larger (swap) | weight agreement | gain over majority (CI from `audit_bootstrap.csv`) | Spearman weight | corrected agreement / Spearman |
|---|---|---|---|---|---|
| M1 balanced | 0.5424 (slide 54.2) | 0.7538 (75.4) | +0.2114 [0.1883, 0.2361] (+21.1) | 0.721 (0.72) | 0.9393 / 0.974 (93.9 / 0.97) |
| M5 text-heavy | 0.9516 (95.2) | 0.9356 (93.6) | -0.0160 [-0.0264, -0.0061] (-1.6) | 0.433 | 0.9577 / 0.962 (95.8) |
| M2 | 0.9709 (97.1) | 0.9666 | -0.0043 [-0.0074, -0.0015] (-0.4 [-0.7, -0.2]) | 0.330 | 0.9877 / 0.983 |
Brief item 5 ("21.2 vs 21.1"): 75.38 minus 54.24 is 21.14. The slide shows 75.4 and 54.2 rounded, whose difference reads 21.2; the reported +21.1 is the unrounded gain. Not an error. No exact ties (swap difference 0, or weight 0.5) in any model.
Frozen-gate identity: winner agreement frozen vs corrected = 1.0 for all five models (max abs error about 5e-7), `results/frozen_gate_audit.csv`. Dynamic agreement real vs corrected 0.939 (M1), 0.958 (M5). PASS.
Mean-of-absolute-changes sensitivity (`results/mean_of_abs_check.csv`): M1 and M2/M3/M5 conclusions keep their sign; M0b's small edge (+1.8 [-0.4, +4.0]) becomes -1.5 [-3.7, +0.8].

## (C) The three greatest validity risks, in descending severity
1. **The model whose F1 is quoted is not the model whose weights were tested.** The swap test uses the fused probability BEFORE the dev-fitted class-prior correction (`37_swap_with_context.py` header, line 13; `33_eval_harness.py` header). Without the correction the gate models score: M1 55.66 (not 58.80), M5 57.09 (not 63.57), both below bcLSTM 59.25. The slides and the script say "our balanced model scores 58.8" and "our text-heavy model reaches 63.6" next to the swap results. Fix: state both F1 values in the report, or rerun the swap on the corrected decision rule (a new, separately registered protocol, since the predicted class can change).
2. **The 63.6 model's lead does not come from context.** With the same correction M2 (strong text, no context) scores 63.48 and M3 62.70, against M5's 63.57. Without the correction M3 is 60.87 and M5 57.09. So "text-heavy model = strong text plus neighbouring utterances, 63.6" is a description, not evidence that context helps. Do not claim a context benefit.
3. **The corrected score is the swap's own algebra** (frozen-gate identity, error about 5e-7) and was derived after the test results were seen. Its 93.9% and 0.97 are diagnostics, not validation, and must be labelled post hoc. Related: the "always-text" baseline is the best constant on the observed test swap winners (descriptive, not deployable); no dev-selected constant rule has been scored.
Also: only 3 gates for M5; heads shared across seeds; the gate reads MiniLM text embeddings even in models whose text head is RoBERTa (headline gate inputs), so the gate does not see the strong text representation (`33_eval_harness.py` header).

## (D) Safe to write in the midterm report now
- Task, data, split, seven classes; the swap definition (average then absolute value, original class, gate recomputed, clf-split donors); the 54.2 / 75.4 and 95.2 / 93.6 results with intervals from `audit_bootstrap.csv`, labelled "exploratory, models built by us".
- Both F1 values per model (see C1), with the correction described as dev-fitted.
- The frozen-gate identity as arithmetic, with the 6.1% (M1) and 4.2% (M5) reversed-by-gate shares.
- Related work only as drafted in `docs/slides/slide-01-sources.md`; the matrix is not yet built.
- Not safe: any claim that context improves F1, that the swap-tested model scores 58.8 or 63.6, that duplicates have negligible effect, that blank/mean checks cover M5, that balancing is sufficient, or the architecture details of M1 to M5 beyond what the scripts show.

## (E) Reruns required, and why
| Experiment | Why | Cost |
|---|---|---|
| Swap test on the prior-corrected decision rule for M1 and M5 (new registered protocol) | Closes risk C1 | about 15 min for M1 (harness), longer for M5 |
| F1 recomputation from prediction arrays, both alphas | Items 8, C1 | minutes |
| Duplicate-audio sensitivity (drop the 42 flagged rows) | Item 6; the brief forbids claiming it negligible | seconds from arrays |
| Dev-selected constant rule scored on test | Risk C3 | seconds |
| More head seeds for M1 and M5 | Item 9 | hours |
| Blank/mean and matched-donor checks on M5; the P1 modality-strength series; the conflict study | Brief P1 | hours to days |

## (F) Midterm report outline (4 pages, 10 pt; word budget about 2,600 words plus figures)
1. Page 1 (550): plain-language goal, stakes (a wrong explanation misleads debugging), task, data, research questions, "influence" defined as swap sensitivity.
2. Page 2 (650): related work in four groups (gated fusion; swap/masking analyses incl. Hessel and Lee 2020; faithfulness in explanation; MELD and baselines), gap and honest novelty. Primary-source quotes still to collect.
3. Page 3 (650): methods: models and F1 with and without correction, donors, effect equation, baselines, corrected diagnostic, hypotheses H1 to H3 with provenance, bootstrap, caveats.
4. Page 4 (600): audited results table and figure, threats (C1 to C3), milestone table (promised / done / evidence / deviation / new date).

## (G) Milestones (owners to be set by the team; dates are internal, confirm LMS)
| Window | Task | Depends on | Cost |
|---|---|---|---|
| Oct 23 to 25 | F1 recomputation; duplicate sensitivity; dev-constant baseline; architecture table (all seconds to an hour) | arrays and checkpoints on disk | minutes |
| Oct 23 to 27 | Related-work matrix from primary papers (read the PDFs, record page numbers) | human reading | 6 to 10 h human |
| Oct 26 to 29 | Corrected-rule swap protocol written, then M1 rerun | protocol written before running | about 15 min to 1 h |
| Oct 28 to Nov 2 | Midterm report draft, visual check | the above | |
| Nov 3 to 10 | Conflict-pair protocol and human audit of 50 to 100 pairs | human time | |
| Nov 11 to 24 | P1 controlled-strength series, conflict study, robustness | frozen protocols | days |

## Not done in this pass
Related-work matrix and bibliography (primary-source quotes), the 4-page report, `docs/final_experiment_plan.md`, `docs/result_registry.md`, the synthetic sanity-test suite and the exact signed decomposition per donor (scripts 38 and 39 already compute the gate term; a separate decomposition table was not produced here). None of the brief's lead claims about other papers (e.g. Jin et al., Xiang Ren, 2025 and 2026 preprints, ACL 2027 deadline) was checked against a primary source.

---
# Update 2 (2026-10-23, after step 42): Phase 1 results
Supersedes rows A6 and A8 above. Commands and hashes: `docs/reproducibility_manifest.md`. Log: `logs/42_audit_cheap_checks.log` (local).

## Claim table
| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Reported gate weighted-F1 values (alpha 0 and dev alpha) for M0b, M1, M2, M3, M5 | **PASS**: recomputed with my own sklearn calls from labels and argmax predictions, equal to the CSV to 4 decimals for every model, both alphas (M1 0.5566 / 0.5880; M5 0.5709 / 0.6357). Own dev re-selection of alpha reproduces the stored value (0.25, 0.25, 0.5, 0.25, 0.5). Macro-F1 also in the file (M1 0.4029 / 0.4112; M5 0.4251 / 0.4646) | `results/audit_f1_recompute.csv` |
| 2 | Correction formula | **PASS**: q = softmax(log p + alpha log pi), applied to the FUSED probability, pi = clf-split class frequencies, alpha in {0, .25, .5, .75, 1} chosen on dev (gate rule, mean over gates). q rows sum to 1 (max error 0.0); argmax q equals argmax of the log-score. Not part of any swap result in steps 33 to 41 | `scripts/audit_common.py`, `33_eval_harness.py` |
| 3 | Headline slide numbers (54.2, 75.4, 95.2, 93.6 and the gains) | **PASS** (recomputed from row-level arrays, see section B above) | `results/audited_master_results.csv` |
| 4 | Duplicate-audio sensitivity | **PASS, no conclusion changes**: flags 42 (3 also shared with train; 39 within test). Primary-effect gain over the majority constant, full vs minus 39 vs minus 42: M1 +21.1 / +21.3 / +21.3 [18.9, 23.7]; M5 -1.6 / -1.6 / -1.6; M2 -0.4 / -0.4 / -0.4; M3 -0.1 / -0.1 / -0.1; M0b +1.8 / +1.9 / +1.9. Same pattern under mean-of-abs and frozen-gate effects. Sample sizes 2,610 / 2,571 / 2,568. This does not remove all possible leakage (only byte-identical audio is flagged) | `results/audit_duplicate_sensitivity.csv` |
| 5 | Dev-selected constant baseline | **PASS**: for every model and every gate the dev winners favour text (text won on dev: M0b 78.1%, M1 57.2%, M2 96.3%, M3 93.5%, M5 94.7%), so the dev-selected constant is "always text", the same rule as the descriptive test-majority baseline here (text also won most often on test in every model). Gains over the dev-selected constant therefore equal the earlier gains: M1 +21.1 [18.8, 23.6], M5 -1.6 [-2.6, -0.6]. The test-majority leakage concern does not change any number | `results/audit_dev_constant.csv` |
| 6 | Frozen-gate identity (corrected signal = frozen swap difference) | **PASS** (unchanged): winner agreement 1.0; and in step 42 each dev run re-checked the closed form to 1e-5 | `results/frozen_gate_audit.csv` |
| 7 | Effect of freezing the gate on the headline comparison | **NEW FINDING**: gain over the majority constant under the FROZEN gate (gate weight held at w0): M0b +7.8 [5.7, 9.9], M1 +25.0 [22.7, 27.5], M2 -0.4 [-0.7, -0.1], M3 +0.4 [-0.2, +1.1], M5 +1.0 [+0.1, +2.0]. Under the real recomputed gate: M0b +1.8, M1 +21.1, M2 -0.4, M3 -0.1, M5 -1.6. So the gate's own re-weighting lowers the weight's apparent agreement in M0b, M1 and M5; for M5 the sign differs between the two procedures. Must be reported | `results/audit_duplicate_sensitivity.csv` (effect = frozen_gate, subset = full) |
| 8 | Uncertainty reported: dialogue bootstrap conditional on trained heads; across gates only | **FAIL as stated on slides ("10 seeds")**: heads are trained once (seed 0) and shared by the 10 gates; M5 has 3 gates. The reported intervals reflect (a) resampling of test dialogues, with gates averaged within each draw, and (b) a separate across-gate SD of the gain (M0b 0.010, M1 0.009, M2 0.002, M3 0.001; `harness_models.csv`). Variation across independently trained heads and across training-data splits is NOT measured | `scripts/33_eval_harness.py` header; `results/harness_models.csv` |
| 9 | Test utterances are donors | **PASS (not donors)**: pool = clf split rows | A3 above |
| 10 | Model architectures M0b to M5 | **PASS with UNVERIFIED cells**: see `docs/model_architecture_registry.md`. Encoders are used as frozen feature extractors; the gate reads MiniLM + 1,024-d audio embeddings in all models | registry |
| 11 | Commit hashes for results | **FAIL until committed**: all step 21 to 43 sources and results are untracked at audit time; the audit commit (below) fixes this going forward | `git log` |
| 12 | Context improves F1 | **FAIL (not supported)**: gate weighted-F1 at alpha 0 / dev alpha: M3 (no context) 60.87 / 62.70, M5 (context) 57.09 / 63.57, M2 59.69 / 63.48. Context is lower at alpha 0 and within 0.1 point of M2 with the correction; M5 also differs in gate count (3 vs 10) and stage structure | `results/audit_f1_recompute.csv` |

---
# Update 3 (2026-10-23): corrected-rule swap results (protocol `docs/corrected_rule_swap_protocol.md`, commit 4f4c662 predates every number below)
Validations passed before interpretation (step 43 log `logs/43_corrected_rule_M1.log`, `..._M5.log`): 7 synthetic tests; raw-rule effects reproduce the step 38/39 arrays (max abs diff 1.8e-7 for M1, 2.4e-7 for M5); the classes tracked reproduce the step 42 F1 exactly (M1 0.5566 / 0.5880, M5 0.5709 / 0.6357). Outputs: `results/corrected_rule_swap_M1.csv`, `results/corrected_rule_swap_M5.csv`. No parameter was changed after seeing a result.

## Primary result (effect = |mean over donors of q_d[c'] - q0[c']|, gate recomputed, dialogue bootstrap, gain over the dev-selected constant = always text)
| Model | Rule | text won | weight agreement [95% CI] | gain over dev constant [95% CI] | Spearman [95% CI] |
|---|---|---|---|---|---|
| M1 | raw (original) | 54.2% | 75.4% [73.9, 76.9] | +21.1 [18.8, 23.6] | 0.721 [0.703, 0.739] |
| M1 | corrected | 52.8% | 73.6% [71.9, 75.3] | **+20.8 [18.5, 23.2]** | 0.703 [0.681, 0.723] |
| M5 | raw (original) | 95.2% | 93.6% [92.6, 94.5] | -1.6 [-2.6, -0.6] | 0.433 [0.398, 0.468] |
| M5 | corrected | 92.9% | 91.9% [90.9, 93.0] | **-0.9 [-1.9, -0.0]** | 0.415 [0.378, 0.450] |
Paired differences, corrected minus raw (same bootstrap draws): M1 agreement -1.8 [-2.8, -0.8], gain -0.3 [-1.7, +1.0]; M5 agreement -1.6 [-2.5, -0.8], gain +0.7 [-0.2, +1.5].

## Reading, by the rule fixed in protocol section 9
- **M1:** the corrected-rule gain interval excludes 0 and is positive, so "the weight is informative against a constant baseline when text and audio are balanced" is supported under the corrected rule as well. The paired gain difference is small and its interval includes 0. The agreement is lower by 1.8 points (interval excludes 0).
- **M5:** the corrected-rule gain is negative with an upper bound at 0 (-0.0001); the original conclusion "no better than always guessing text" stands, with the weaker wording "not better, and slightly worse or equal". Under mean-of-absolute-changes the corrected gain is -3.3 [-4.4, -2.3] and the flip-share effect gives -2.4 [-3.4, -1.4], so the sign is stable across effect definitions.
- **No contradiction** of the preliminary findings; magnitude changes are small, but the quoted F1 and the tested rule are now aligned.

## Things the corrected rule changes or exposes
1. The prior correction changes the predicted class for 19.3% (M1) and 30.7% (M5) of test utterances. On those rows agreement falls from 71.6% to 62.2% (M1) and from 90.6% to 83.7% (M5); on unchanged rows it is unchanged (76.3 / 76.3 and 94.9 / 95.6). The loss of agreement sits in the rows where the correction moves the decision.
2. Under the **frozen gate** (weight held at w0) the corrected-rule agreement is 75.2% (M1) and 93.0% (M5), gains over the majority +23.0 and +1.3, against +21.1 / -1.6 for the real recomputed gate (raw rule). The sign of the M5 gain depends on whether the gate may re-weight.
3. The mean-of-absolute-changes variant gives the corrected rule a higher M1 gain (+29.1 vs +26.7 raw; paired +2.4 [1.5, 3.3]), so effect-definition sensitivity exceeds rule sensitivity for M1.
4. The dev-selected constant is "text" in every run, so the new baseline equals the old one. The test-majority concern is closed for these models.

# Phase 3 (interpretation; no new runs)
- **Gate-input mismatch (M2, M3, M5):** the gate reads the MiniLM text embedding and the 1,024-d audio embedding, not the 25,600-d RoBERTa representation that drives the text head (`docs/model_architecture_registry.md`). It cannot see most of what makes the text prediction confident, which could explain why its weight fails to track text-dominated swap effects. This is a hypothesis (H4 in `docs/final_experiment_plan.md`), not a finding. Controlled test proposed there: train gates on a RoBERTa-based input with heads, donors and alpha fixed, on M2 and M3 only, so text strength and swap-win share barely change and the mismatch is separated from dominance.
- **M2 vs M3 vs M5 (context vs strength):** swap-win share of text 97.1 / 92.8 / 95.2%; gain over the majority constant -0.4 / -0.1 / -1.6; gate weighted-F1 at alpha 0 / dev alpha 59.69 / 63.48, 60.87 / 62.70, 57.09 / 63.57. Context is not shown to help F1 and is not separable from the stage-2 refit and the 3-versus-10 gate count. A context-free model with the same stage structure and 3 gates is the missing control.
- Not claimed anywhere: that context improves F1, that balanced modalities guarantee faithful weights, that the corrected-signal agreement validates causal importance.
