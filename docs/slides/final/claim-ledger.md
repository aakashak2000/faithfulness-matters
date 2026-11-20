# Claim ledger for the final presentation (2026-11-19)
Every number on a slide or in the script, with its source. "Raw, dial fixed" = raw fusion rule with the gate weight held fixed during swaps; "primary" = corrected rule, gate recomputed after each swap (the midterm definition).

| Claim on slide / script | Value | Source | Definition |
|---|---|---|---|
| Example clip: dial 61% words, words 28% surprise, voice 69% neutral, final neutral, gold neutral, voice swap 0.20 vs words 0.02 (about 9×) | 0.609; 0.277; 0.686; 0.203 vs 0.023 | `results/slide_numbers.csv` (example_clip), rule `docs/slides/final/example_clip_rule.md` | M1, gate 0, primary swap effects |
| Midterm agreement 74% / 92% | 0.7358 / 0.9193 | `results/corrected_rule_swap_M1.csv`, `_M5.csv` (rule cor, primary_abs_of_mean) | primary |
| Learned vs shuffled dial 78% vs 79%; 94% vs 94% | 0.783 vs 0.789; 0.943 vs 0.944 | `results/decomposition_null_v2.csv` | raw, dial fixed; shuffled = recomputed-prediction null |
| Similar clips flip 17% (noise 5%); words-heavy 8% (noise 4%) | 0.167 / 0.054; 0.084 / 0.039 | `results/reference_sensitivity_v4.csv` | primary; nn50 vs full; noise = random 50 vs full |
| Holds with the dial fixed (Q&A) | 0.143 and 0.063 | `results/reference_sensitivity_v4.csv` (head_path_only) | gate held fixed during swap |
| Dial 59% → 77 / 69 / 67 / 77% words | 0.585; 0.775, 0.685, 0.671, 0.770 | `results/slide_numbers.csv` (phase2_mean_text_weight) | mean retrained gate weight on test; sharper/flatter compare with S0 0.581 |
| Single fixed setting moves as far in 4 of 5 | ratios 1.16, 1.16, 0.98, 0.66, 0.81 | `results/controlled_series_posthoc.csv` | constant-gate logit shift / learned mean logit shift; rule "within 25%" met in 4 of 5 |
| Ranking score: dial 59, detector 64 (which listener is right) | 0.588 / 0.642 | `results/controlled_series_posthoc.csv` | AUROC on rows where exactly one head is correct |
| Wrong audio: dial 52, detector 51 | 0.522 / 0.513 | `results/controlled_series_posthoc.csv` | AUROC for planted rows |
| Learned confidence score did not beat break-even rule | -0.008 [-0.018, 0.001] (M1) | `results/selective_attribution_v2.csv` | AURC learned minus identity, R2 claim, held-out nn50 |
| Break-even rule (Q&A): M1 wrong claims at 50% kept, 14.6% → 11.1% | 0.146 / 0.111 | `results/selective_attribution_v2.csv` (gate claim, nn50, risk_at_50) | |
| Since midterm: 15 analysis scripts, 33 dials retrained, 5 reference sets, 4 claims withdrawn | counts | `scripts/49` to `63`; `docs/controlled_series_protocol.md` (C0 3, C1 9, C2 9, S0 3, S1 3, S2 3, P 3); pools full, global50, speaker50, nn50, nnp50; withdrawn claims in `docs/rescoring_protocol.md`, `docs/decomposition_protocol.md`, `docs/reference_pools_protocol.md` | |
| Example clip is the first of 127 qualifying | 127 | `results/slide_numbers.csv` (n_qualifying_rows) | rule in `example_clip_rule.md` |
| 2,610 test clips, 7 emotions | 2,610; 7 | `results/corrected_rule_swap_M1.csv` (n_rows), MELD label set in `scripts/common.py` load_dataset | |
| 8,004 swap-in clips | 8,004 | clf-split donor count printed in this session from `data/v2_pipeline.pt` (rows clf); `docs/reference_pools_protocol.md` | |
| 6 exploratory experiments | list | re-scoring (step 49), structural decomposition and shuffled dial (50 to 52), reference sets (53 to 58), selective attribution (54, 56), controlled series (59 to 61), post-hoc probes (62) | protocols in `docs/*_protocol.md` |
| 4 claims withdrawn | list | "weight ranks influence well" and "biased toward text" (`docs/rescoring_protocol.md` post-run note; `docs/decomposition_protocol.md`); "fraction mechanical vs learned" (`docs/decomposition_protocol.md` Amendment 2); "similar clips move influence away from text" (`docs/reference_pools_protocol.md` Amendment 2 results) | |
| Overall agreement 73% to 74% across swap-in sets | 0.726 to 0.735 (M1) | `results/reference_sensitivity.csv` (gate_agreement_*) | primary |
| Midterm slides 79%, always-text 77%, cutoff 0.37 gives 82% (Q&A) | | `docs/slides/claim-ledger.md` (midterm), `results/resampling_full_faithfulness.csv`, `results/gate_calibration_primary.csv` | earlier model |
| Always-words 53% / 93% vs dial 74% / 92% (Q&A) | 0.528 / 0.929 | `results/corrected_rule_swap_M1.csv`, `_M5.csv` (p_text) | primary |
| Accuracy with intervals (Q&A) | 0.535 [0.516, 0.555] vs 0.508 [0.490, 0.526]; M5 0.529 vs 0.522 | `results/decomposition_null_v2.csv` | raw fused accuracy |
| Raw rule, dial recomputed: 75% (Q&A reconciliation) | 0.754 (M1) | `results/rescoring.csv` (block A, effect raw_real, agreement) | raw, dial recomputed |
