# Reproducibility manifest (2026-10-23)

Base commit before this audit: `24a2ae379abc9a9dbb81b329c86f7ec3adcc867b`. Files below were hashed (first 16 hex of SHA-256) at audit time; the audit commit that adds them is recorded in `docs/research_audit_2026-10-23.md`. `data/` and `checkpoints/` are NOT committed (large, local); their hashes below let a teammate check they hold the same files.

## Result-to-command map
| Output | Command (repo root, venv active) | Notes |
|---|---|---|
| results/harness_models.csv, harness_alt_effects.csv, harness_per_gate.csv | `python scripts/33_eval_harness.py` | about 15 min; heads trained once (seed 0), 10 gates per model |
| results/corrected_signal.csv | `python scripts/34_corrected_signal.py` | closed-form signal |
| results/swap_with_context.csv | `python scripts/37_swap_with_context.py` | M5, 3 gates |
| results/frozen_gate_audit.csv, data/frozen_gate_arrays_*.npz | `python scripts/38_frozen_gate.py ; python scripts/39_frozen_gate_m5.py` | M0b to M3, then M5 row appended |
| results/audit_bootstrap.csv | `python scripts/40_audit_bootstrap.py` | needs the npz arrays |
| results/mean_of_abs_check.csv | `python scripts/41_mean_of_abs.py` | needs the npz arrays |
| results/audit_f1_recompute.csv, audit_dev_constant.csv, audit_duplicate_sensitivity.csv, data/audit_cache/*, data/audit_dev_arrays_*.npz | `python scripts/42_audit_cheap_checks.py` | about 12 min (log in logs/, local) |
| results/corrected_rule_swap_{M1,M5}.csv | `python scripts/43_corrected_rule_swap.py --model M1 ; --model M5` | only after docs/corrected_rule_swap_protocol.md is committed |

## Hashes: scripts
| File | sha256[:16] |
|---|---|
| scripts/00_inspect_assets.py | 4c60b3d0573a12b0 |
| scripts/01_build_dataset.py | 783c3a68cae43852 |
| scripts/02_train.py | e105a35f03432173 |
| scripts/03_silence_embeddings.py | 224d7b42804cab47 |
| scripts/04_interventions.py | 9ced2c3afbfc59e2 |
| scripts/05_faithfulness.py | 0cb614e24850e759 |
| scripts/06_analysis.py | 24323fff585bbec0 |
| scripts/07_resampling.py | 7e8bae0a88cfd127 |
| scripts/08_resampling_full.py | bbdaa246522043c5 |
| scripts/09_extract_layers.py | 099ec89d0428bea3 |
| scripts/10_strong_audio.py | 8aeadeb621b2c458 |
| scripts/11_strong_audio_reg.py | 9365ab74c004595e |
| scripts/12_paired_gate_vs_confidence.py | 0b0d32975e2097b6 |
| scripts/13_strong_audio_layer_lr.py | d981441cd2bb6877 |
| scripts/14_alt_effect_metrics.py | 0e3f4b722be727c1 |
| scripts/15_gate_fixed.py | 033b7513257685d0 |
| scripts/16_weight_disagreement.py | ffbeac2fd9f15c50 |
| scripts/17_figures_primary.py | 50296563c6ef0dac |
| scripts/18_shuffled_control.py | 6cc4ec0af92e5df1 |
| scripts/19_dev_calibration.py | 6096d9d615c7e097 |
| scripts/20_conflict_h2.py | 764a13fe9db61549 |
| scripts/21_more_seeds.py | 351b4560f978033a |
| scripts/22_better_audio.py | d3e4b50a42222da6 |
| scripts/23_better_audio_fusion.py | b3e65a020fbd8a0f |
| scripts/24_strong_audio_checks.py | 092b72312742fb16 |
| scripts/25_quality_ladder.py | 1df92d920621cc91 |
| scripts/26_figure_ladder.py | 9b21d99fc27d726a |
| scripts/27_extract_text_layers.py | d10edef6ac5f6701 |
| scripts/28_text_head_grid.py | 8a15e5e19e2cd216 |
| scripts/29_context_pipeline.py | fecd16bff04b7794 |
| scripts/30_context_pipeline_v2.py | 3a894fbee9a45572 |
| scripts/31_swap_strong_text.py | a1e08cb14ad01cb2 |
| scripts/32_balance_experiments.py | ff101e29b0b725bf |
| scripts/33_eval_harness.py | 1cf137057e5c4505 |
| scripts/34_corrected_signal.py | 98294e110cc4d913 |
| scripts/35_calibrated_attribution.py | 1516502f481432bb |
| scripts/36_figure_agreement.py | 6e1eda65c6b82dfb |
| scripts/37_swap_with_context.py | 8a960d31a0a16734 |
| scripts/38_frozen_gate.py | a06f264c375fbcb1 |
| scripts/38_frozen_gate_audit.py | 34eb84977654852e |
| scripts/39_frozen_gate_m5.py | 863fb3b7b9707fae |
| scripts/40_audit_bootstrap.py | 1fa578e6258657c8 |
| scripts/41_mean_of_abs.py | 7d75dd513c08e176 |
| scripts/42_audit_cheap_checks.py | 488c68d618c37f09 |
| scripts/43_corrected_rule_swap.py | 8c2fb86680f66f8e |
| scripts/audit_common.py | 10560f09f69f67b8 |
| scripts/common.py | ea75a263eab5db65 |
| scripts/diag_audio_subset.py | 4acaf2876aa6faf2 |
| scripts/diag_distances.py | e17e38bb597406f6 |
| scripts/diag_duplicates.py | e21415f4d49545ae |
| scripts/diag_extraction_check.py | a5f6b069c5c446f5 |
| scripts/diag_mean_embedding.py | f078f1ebc65f9ed6 |
| scripts/diag_order.py | 29a2918d75996b94 |
| scripts/diag_source_files.py | 0e1bc66a18033f18 |
| scripts/diag_wording_checks.py | 50d26de27ef53636 |

## Hashes: result tables used in the audit
| File | sha256[:16] |
|---|---|
| results/harness_models.csv | 7f04b987300ca352 |
| results/harness_alt_effects.csv | eef3885e1a41b805 |
| results/harness_per_gate.csv | 5bba0650396c3322 |
| results/corrected_signal.csv | e875c9a1a7125899 |
| results/swap_with_context.csv | 8b8c073058e3a7af |
| results/frozen_gate_audit.csv | 2c5b5976da3ced3c |
| results/audit_bootstrap.csv | 7c28c154eb8d18a3 |
| results/mean_of_abs_check.csv | a2584a00699aa357 |
| results/audit_f1_recompute.csv | 909705cce29fd503 |
| results/audit_dev_constant.csv | aa35a35449d90ff7 |
| results/audit_duplicate_sensitivity.csv | 0bc0ffb432922010 |
| results/audio_duplicates.json | c96cf920d6166c45 |
| results/balance_experiments.csv | f66d84130214579b |

## Hashes: local data (not committed)
| File | size MB | sha256[:16] |
|---|---|---|
| data/dataset.pt | 77.9 | f1c0f7e169274122 |
| data/dataset_strong.pt | 77.9 | a8bd3b9a32214bd1 |
| data/intervention_embeddings.pt | 10.7 | 6ddc9c5b6abad6ed |
| data/intervention_embeddings_dev.pt | 4.5 | 7e3bbeac94909d8a |
| data/intervention_probs.pt | 6.0 | bf2b18ec5f764732 |
| data/roberta_layers_dev.pt | 56.8 | c6fc1340e14a4097 |
| data/roberta_layers_test.pt | 133.8 | cd5d5d101287cb75 |
| data/roberta_layers_train.pt | 512.0 | 5bfec96891c775d9 |
| data/v2_pipeline.pt | 3.7 | 0808f1f15b2c0335 |
| data/v2_pipeline_v2.pt | 0.5 | 7e37062257a1b2bb |
| data/wavlm_layers_dev.pt | 118.0 | f61caffc7798650b |
| data/wavlm_layers_test.pt | 278.0 | caff75517e7fc02e |
| data/wavlm_layers_train.pt | 1063.8 | 33331b5eeab911ab |
| data/audit_dev_arrays_M0b.npz | 0.3 | 72c09e9e723727ad |
| data/audit_dev_arrays_M1.npz | 0.3 | 46489afdc555d6eb |
| data/audit_dev_arrays_M2.npz | 0.3 | 15c4a8403f2ee3c5 |
| data/audit_dev_arrays_M3.npz | 0.3 | 37dee3166d1986b2 |
| data/audit_dev_arrays_M5.npz | 0.1 | c282591123070d10 |
| data/frozen_gate_arrays_M0b.npz | 1.1 | 3207b9be976b1773 |
| data/frozen_gate_arrays_M1.npz | 1.1 | 214e5256cdf19259 |
| data/frozen_gate_arrays_M2.npz | 1.1 | 5be20c526974282c |
| data/frozen_gate_arrays_M3.npz | 1.1 | 8131965893643cb7 |
| data/frozen_gate_arrays_M5.npz | 0.3 | 8de15df912465791 |
| data/meld_csv/dev_sent_emo.csv | 0.1 | 2e89c6f8aa182d6f |
| data/meld_csv/test_sent_emo.csv | 0.3 | 8d37103938f70676 |
| data/meld_csv/train_sent_emo.csv | 1.1 | d2fa2d6529cf03ca |
| data/audit_cache/gates_M0b.pt | 400.4 | f9484b574442c312 |
| data/audit_cache/gates_M1.pt | 400.4 | 40eb7a4d05f8bfe5 |
| data/audit_cache/gates_M2.pt | 400.4 | 92a00f4d82bb3b0b |
| data/audit_cache/gates_M3.pt | 400.4 | 6286077ba78c9ff8 |
| data/audit_cache/gates_M5.pt | 2.2 | f7bedcdc5ec5a3c9 |

Checkpoints under `checkpoints/` (seed*.pt headline gates and heads) were not hashed in this pass (UNVERIFIED). The M0b to M3 heads are retrained deterministically from the saved features on each run, so they have no stored checkpoint; their reproduction is shown by the F1 match in `results/audit_f1_recompute.csv`.
