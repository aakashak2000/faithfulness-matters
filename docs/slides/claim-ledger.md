# Claim ledger for both midterm slides
Updated 2026-10-20. Every claim on a slide needs a basis. Status: Verified (basis named and checked), Needs check (basis exists but someone must verify), Open (decision or work missing).

## Cross-cutting (do first)
| # | Item | Status |
|---|---|---|
| X1 | Presentation slot: **group 24, Oct 22** (confirmed 2026-10-20); slides due 24 h before, **Oct 21** at the slot time. Slot time of day not recorded. | Done, record the time |
| X2 | Presenter and owner for Hessel & Lee 2020 (plan item 7.8) | Open |
| X3 | `MIDTERM_STATUS.md` section 11 outline is stale; `docs/slides/` replaces it | Open (retire it) |
| X4 | Uncommitted: `docs/slides/`, 10-seed script and results, plan edits | Open |

## Slide 1
| # | Claim or element | Basis | Status |
|---|---|---|---|
| S1.1 | Many multimodal models fuse text and audio with a learned weight | 8 papers, `slide-01-sources.md` | Verified; citations FINAL: Liu et al. 2020; He & Xiao 2025 |
| S1.2 | Papers often read that weight as how much each modality mattered | Exact quotes from the 8 papers | Verified |
| S1.3 | Inputs and outputs (utterance text + audio; 7 emotions + a weight) | `docs/PROJECT_REQUIREMENTS.md` section 2, `scripts/common.py` | Verified |
| S1.4 | "A learned gate that mixes the two models' predictions" | `LearnedGate`, `fuse_gate` in `scripts/common.py` (one scalar weight over the two heads' output probabilities) | Verified; method line FINAL, now ending "using intervention tests" |
| S1.5 | Confidence-based weight c_t / (c_t + c_a) | `scripts/common.py` | Verified |
| S1.6 | "Intervention tests: replace one modality with real examples from other utterances, rerun the model" | `scripts/08_resampling_full.py` (donors are the 7,990 classifier-split utterances) | Verified |
| S1.7 | Prior-work lines: Jacovi & Goldberg, Jain & Wallace, Wiegreffe & Pinter, Hessel & Lee | Abstracts and search summaries only; PDFs not read | Needs check (owner reads) |
| S1.8 | "In the papers we read, none tests the reading by removing a modality" | Keyword search of the 8 full texts | Verified within its limits; keep the hedge |
| S1.9 | Why test: reading is common and untested; a weight can look right and be wrong; late fusion gives a clean test | S1.1 to S1.8; ReH-FUSE quote; our fusion formula | Partly verified (attention-debate line rests on S1.7) |
| S1.10 | What we add: always-text baseline, ranking measure, several removal methods, conflict test | `scripts/05`, `06`, `04`, `07`, `08`; conflict is a pilot (`scripts/20`) | Verified; call the conflict test "planned", not done |
| S1.11 | Pipeline diagram (text and audio, weight, prediction, remove-one arrow) | none yet | Open |
| S1.12 | Script, 185 words, about 74 s | `docs/slides/script.md`, locked 2026-10-20 | Final text; rehearsal and timing still open |

## Slide 2
| # | Claim or element | Basis | Status |
|---|---|---|---|
| S2.1 | The gate's weights rank influence well (Spearman 0.70) | 0.705 [0.680, 0.726], `results/resampling_full_faithfulness.csv` | Verified; "well" is a judgement, test the word with the grader |
| S2.2 | At the 0.5 cutoff the gate does not beat always-text (+0.019, not significant) | +0.019 [-0.003, +0.041], same file; 10-seed rerun +0.020 [-0.000, +0.043] (`results/more_seeds_faithfulness.csv`) is a notes-only check | Verified (3-seed headline) |
| S2.3 | "Miscalibrated": equal influence sits near weight 0.37, not 0.5 | `results/gate_calibration_primary.csv`; dev cutoffs 0.36 to 0.38 (`results/dev_calibration_thresholds.csv`) | Verified; define the word in a clause on the slide |
| S2.4 | A cutoff fitted on dev beats always-text on test: +0.049 [0.035, 0.062] | `results/dev_calibration_test.csv`, `scripts/19_dev_calibration.py` | Verified; notes must say the interval is conditional on the cutoff and that the idea came from looking at the test-set curve |
| S2.5 | Figure: calibration under the primary intervention | `figures/fig3b_gate_calibration_primary.png` | Needs check: legibility at slide size, caption |
| S2.6 | Sensitivity: gain is +0.09 under TVD and +0.06 under label flips | `results/alt_effect_metrics.csv` | Verified; decide: Q&A only, or one footnote |
| S2.7 | Confidence weight scores +0.225 but the same with shuffled labels | `results/shuffled_control.csv` | Verified; planned for Q&A only |
| S2.8 | Remaining work, 2 to 4 scoped lines: (a) H2 conflict test with about 10 pairings (pilot gave +0.008 and +0.041, disagreeing); (b) report with the sensitivity checks; (c) strong audio head, optional | `results/conflict_h2_diff.csv`, `docs/robustness_results.md`, step 3 plan | Verified; (a) needs a measured runtime before it is called "scoped" |
| S2.9 | Disclosure: the primary intervention was promoted after seeing mean and blank results and before the resampling results | step 7 plan | Open: write the two-sentence note |
| S2.10 | Q&A rows (10 drafted) | step 7 plan | Needs check: one pass to confirm every number against its file |
| S2.11 | Script about 150 words; three timed rehearsals with total under 2 min 10 s | none yet | Open |
| S2.12 | Rubric-grader pass on final slide text and script | none yet | Open |
| S1.13 | "We built our own model" (frozen encoders, linear heads, gate, confidence rule) | Presenter built the legacy embeddings and bundle (confirmed 2026-10-20); heads, gate and fusion trained in `scripts/02_train.py`; weighted-F1 51.7 (`results/baselines_summary.csv`) vs bcLSTM 59.25 (MELD paper Table 11) | Verified; wording pending approval (see `docs/slides/script.md`) |
