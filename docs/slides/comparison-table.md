# Comparison table (verified 2026-10-20 night)

All MELD numbers: official test set, 7 emotions, weighted F1 (%). Each number was read in the paper's own text or table (PDFs in the session scratchpad). NOT a controlled comparison: encoders, context use and tuning differ.

## MELD
| System | Year | Inputs | Uses dialogue context | MELD weighted F1 | Where read |
|---|---|---|---|---|---|
| bcLSTM (MELD baseline) | 2019 | text + audio | yes | 59.25 | Poria et al. 2019 (MELD paper) Table 11 |
| DialogueRNN | 2019 | text + audio | yes | 60.25 | same table |
| **Ours, first model** | 2026 | text + audio | no | 51.7 (58.4 with dev-chosen class-prior correction) | `results/harness_models.csv` (58.4 M0), headline 51.7 in `docs/robustness_results.md` row 22 |
| **Ours, strongest** | 2026 | text + audio | yes | 63.6 | `results/f1_push_v2.csv`, S2, gate, dev alpha 0.5 |
| UniMSE | 2022 | text + audio + video | yes | 65.51 | TelME Table 2 |
| SDT | 2023 | text + audio + video | yes | 66.60 | SDT paper |
| M2FNet | 2022 | text + audio + video | yes | 66.71 | M2FNet paper |
| MultiEMO | 2023 | text + audio + video | yes | 66.74 | MultiEMO paper |
| TelME | 2024 | text + audio + video | yes | 67.37 (text + audio only: 67.19) | TelME Table 3 |
| MERC-PLTAF | 2025 | text + audio | yes | 67.73 | PMC11909257 |
| ReH-FUSE | 2026 | see paper | yes | 68.03 | arXiv 2609.13857 (modalities not re-checked) |
| MiSTER-E | 2026 | speech + text | yes | 69.5 | arXiv 2602.23300 (fine-tuned 8B LLM text encoder, 7B speech LLM) |

## Other datasets, text + audio (not comparable in score: different task and labels)
| System | Dataset | Score | Where read |
|---|---|---|---|
| MiSTER-E | IEMOCAP (weighted F1) | 70.9 | arXiv 2602.23300 |
| MiSTER-E | CMU-MOSI (sentiment) | 87.9 | same |
| MERC-PLTAF | IEMOCAP | 74.80 | PMC11909257 |
| TelME | IEMOCAP | 70.48 | TelME Table 3 |
| LPGNet | IEMOCAP, 4 classes | above 87% (different class set) | arXiv 2508.08925 (exact figure not re-read) |

## Honest reading
- At the published baselines (59.25, 60.25) our strongest is above; the best recent systems are 4 to 6 points above us.
- None of these papers tests whether their fusion weights match removal effects (the 8-paper gating review, `slide-01-sources.md`). Matching the state of the art is not our goal, a sound model to test is.
- Dropped as unverified: BCAF (numbers only in a figure), the 73.1 / 63.9 / OmniVox summaries.
