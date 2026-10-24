# Result registry (2026-10-23)
One row per model x effect definition x rule, test split (2,610 utterances), donors = clf-split training rows with byte-identical embeddings excluded. Gain = weight agreement minus always-text (equal to the dev-selected constant for every row, see `results/audit_dev_constant.csv`). CI = 95% dialogue-cluster bootstrap (1,000 draws, rng 0), gates averaged inside each draw; it does NOT include head-training variance (heads shared across gates). Source file for every row: `results/audited_master_results.csv` (built from `audit_bootstrap.csv`, `mean_of_abs_check.csv`, `corrected_rule_swap_*.csv`).

## Confirmed (recomputed in this audit)
| Model | Outcome definition | Rule | Gates | Text larger | Agreement | Gain over always-text [CI] | Spearman | Gate wF1 a0 / dev a | Status |
|---|---|---|---|---|---|---|---|---|---|
| M0b | primary_abs_of_mean | raw_rule | 10 | 0.767 | 0.785 | +1.8 [-0.4, 4.0] | 0.681 | 51.7 / 57.5 | RECOMPUTED_from_row_arrays |
| M0b | sensitivity_mean_of_abs | raw_rule | 10 | 0.820 | 0.804 | -1.5 [-3.7, 0.8] | 0.818 | 51.7 / 57.5 | from_script41_not_independently_recomputed |
| M1 | primary_abs_of_mean | raw_rule | 10 | 0.542 | 0.754 | +21.1 [18.8, 23.6] | 0.721 | 55.7 / 58.8 | RECOMPUTED_from_row_arrays |
| M1 | sensitivity_mean_of_abs | raw_rule | 10 | 0.536 | 0.803 | +26.7 [24.2, 29.2] | 0.809 | 55.7 / 58.8 | from_script41_not_independently_recomputed |
| M2 | primary_abs_of_mean | raw_rule | 10 | 0.971 | 0.967 | -0.4 [-0.7, -0.1] | 0.330 | 59.7 / 63.5 | RECOMPUTED_from_row_arrays |
| M2 | sensitivity_mean_of_abs | raw_rule | 10 | 0.995 | 0.988 | -0.7 [-1.0, -0.4] | 0.453 | 59.7 / 63.5 | from_script41_not_independently_recomputed |
| M3 | primary_abs_of_mean | raw_rule | 10 | 0.928 | 0.927 | -0.1 [-0.6, 0.4] | 0.396 | 60.9 / 62.7 | RECOMPUTED_from_row_arrays |
| M3 | sensitivity_mean_of_abs | raw_rule | 10 | 0.974 | 0.969 | -0.5 [-1.0, -0.0] | 0.551 | 60.9 / 62.7 | from_script41_not_independently_recomputed |
| M5 | primary_abs_of_mean | raw_rule | 3 | 0.952 | 0.936 | -1.6 [-2.6, -0.6] | 0.433 | 57.1 / 63.6 | RECOMPUTED_from_row_arrays |
| M5 | sensitivity_mean_of_abs | raw_rule | 3 | 0.975 | 0.949 | -2.5 [-3.6, -1.5] | 0.589 | 57.1 / 63.6 | from_script41_not_independently_recomputed |
| M1 | primary_abs_of_mean | raw_rule | 10 | 0.542 | 0.754 | +21.1 [18.8, 23.6] | 0.721 |  | step43_raw_rule_reproduction |
| M1 | primary_abs_of_mean | corrected_rule | 10 | 0.528 | 0.736 | +20.8 [18.5, 23.1] | 0.703 |  | NEW_step43_corrected_rule_protocol |
| M1 | mean_of_abs | raw_rule | 10 | 0.536 | 0.803 | +26.7 [24.2, 29.2] |  |  | step43_raw_rule_reproduction |
| M1 | mean_of_abs | corrected_rule | 10 | 0.522 | 0.813 | +29.1 [26.6, 31.0] |  |  | NEW_step43_corrected_rule_protocol |
| M1 | flip_share | raw_rule | 10 | 0.530 | 0.768 | +23.8 [21.5, 26.0] |  |  | step43_raw_rule_reproduction |
| M1 | flip_share | corrected_rule | 10 | 0.507 | 0.765 | +25.8 [23.0, 26.9] |  |  | NEW_step43_corrected_rule_protocol |
| M1 | frozen_gate | raw_rule | 10 | 0.533 | 0.783 | +25.0 [22.7, 27.5] |  |  | step43_raw_rule_reproduction |
| M1 | frozen_gate | corrected_rule | 10 | 0.522 | 0.752 | +23.0 [20.6, 25.4] |  |  | NEW_step43_corrected_rule_protocol |
| M5 | primary_abs_of_mean | raw_rule | 3 | 0.952 | 0.936 | -1.6 [-2.6, -0.6] | 0.433 |  | step43_raw_rule_reproduction |
| M5 | primary_abs_of_mean | corrected_rule | 3 | 0.929 | 0.919 | -0.9 [-1.9, -0.0] | 0.415 |  | NEW_step43_corrected_rule_protocol |
| M5 | mean_of_abs | raw_rule | 3 | 0.975 | 0.949 | -2.5 [-3.6, -1.5] |  |  | step43_raw_rule_reproduction |
| M5 | mean_of_abs | corrected_rule | 3 | 0.979 | 0.946 | -3.3 [-4.4, -2.3] |  |  | NEW_step43_corrected_rule_protocol |
| M5 | flip_share | raw_rule | 3 | 0.959 | 0.931 | -2.8 [-3.8, -1.7] |  |  | step43_raw_rule_reproduction |
| M5 | flip_share | corrected_rule | 3 | 0.953 | 0.929 | -2.4 [-3.4, -1.4] |  |  | NEW_step43_corrected_rule_protocol |
| M5 | frozen_gate | raw_rule | 3 | 0.933 | 0.943 | +1.0 [0.1, 2.0] |  |  | step43_raw_rule_reproduction |
| M5 | frozen_gate | corrected_rule | 3 | 0.916 | 0.929 | +1.3 [0.3, 2.4] |  |  | NEW_step43_corrected_rule_protocol |

## Corrected-rule class changes (original class under the rule with vs without prior correction)
| Model | Rows where the predicted class changes | Agreement on those rows, raw / corrected | Agreement on unchanged rows, raw / corrected |
|---|---|---|---|
| M1 | 19.3% (503 of 2,610) | 71.6% / 62.2% | 76.3% / 76.3% |
| M5 | 30.7% (801 of 2,610) | 90.6% / 83.7% | 94.9% / 95.6% |

## Old or preliminary figures NOT to cite as results
- The first model, 51.7 weighted F1 (`MIDTERM_STATUS.md`, local): superseded, not on the slides.
- Headline model M0 / M0b gain +2.0 / +1.8: interval includes 0 and the sign flips under mean-of-absolute-changes.
- Any statement that the swap-tested balanced / text-heavy models score 58.8 / 63.6: the original swap tested the uncorrected rule (55.7 / 57.1). The corrected-rule rows above are the aligned version.
- Corrected-signal agreement (93.9, 95.8) and Spearman (0.97, 0.96): exact algebra of the swap with a frozen gate; diagnostic, not validation.
- Blank / mean replacement and conflict-pair (H2 pilot) results exist for the earlier headline model only, not for M1 to M5.
- Duplicate-audio sensitivity, F1 recomputation and dev-selected constant: see `results/audit_duplicate_sensitivity.csv`, `audit_f1_recompute.csv`, `audit_dev_constant.csv` (all PASS, no conclusion changes).
