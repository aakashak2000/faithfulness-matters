# Step 03 (follow-up): stronger audio head as a robustness variant
Agreed: 2026-10-15 (user: "run C + D and do it here")

## What exists
- Audio head: Linear(1024 -> 7) on the last WavLM-large layer, mean-pooled over time. Audio-only weighted-F1 0.368 vs text 0.491 (`results/baselines_summary.csv`).
- Seed-0 audio head scores weighted-F1 0.426 on its own training split (clf), 0.361 on dev, 0.367 on test. It underfits; it is not an overfitting problem.
- Raw audio now local in `data/all_data/` (train 9,988, dev 1,108, test 2,610 WAVs; MELD's own missing dia125_utt3 and dia110_utt7 are absent from both the files and our splits).

## Assessment
Most likely cause: last-layer, mean-pooled features carry little emotion signal (not measured until this plan runs). Scratch tests on cached embeddings (one seed, not saved): standardising features gave test 0.330, a 512-unit MLP collapsed to 0.185 because dev loss bottoms out almost immediately. So the fix is better features, not a bigger head.

## Decisions (with alternatives rejected and why)
- C + D: learned weighted sum over all 25 WavLM layers, with mean pooling (C) and mean + std pooling (D). Alternatives rejected: fine-tuning WavLM (out of scope in `docs/PROJECT_REQUIREMENTS.md`, changes the model under test), a new encoder (new dependency, changes the story).
- Run extraction locally on MPS (about 40 min).
- Head stays linear. Same loss and early stopping as the baseline, so only the features change. Macro-F1 early stopping is a labelled sensitivity run.
- The current model stays the headline. The strong audio head is a robustness variant: does the gate-vs-always-text and confidence-rule result survive a better audio side?
- Layer weights are fit on clf and selected on dev. Nothing is tuned on test.

## Actions
Status as of 2026-10-15. The variant is PAUSED: nothing here blocks the midterm slides, and 3.3c to 3.5 are now "before final, optional" (decision by the user: the best variant only gave +0.03 and the layer mix learned nothing).

| # | Action | Serves | Bucket | Effort (h) | Status | Acceptance check |
| --- | --- | --- | --- | --- | --- | --- |
| 3.1 | `scripts/09_extract_layers.py`: per-layer mean and std pooling for train, dev, test | Robustness | Before final | 1 | Done | Layer-24 mean reproduces cached dev and test embeddings (cosine min 0.99986 dev, 0.99995 test; `logs/extract_layers.log`, local) |
| 3.2 | Map train WAVs to clf/gate rows by matching the layer-24 mean to the cached audio rows, label-checked | Robustness | Before final | 0.5 | Done | 9,988 of 9,988 rows matched, each file used once, cosine min 0.9999 (`results/strong_audio_row_map.csv`) |
| 3.3a | `scripts/10_strong_audio.py`: four feature variants, selected on dev | Robustness | Before final | 1.5 | Done | `results/strong_audio_variants.csv`: C_raw selected (dev 0.383, test 0.395 vs 0.367 baseline, one seed) |
| 3.3b | `scripts/11_strong_audio_reg.py`: weight-decay arm | Robustness | Before final | 0.5 | Done, no gain | `results/strong_audio_regularisation.csv`: dev gain over 0 is 0.0000, unregularised head kept |
| 3.3d | `scripts/13_strong_audio_layer_lr.py`: let the layer weights learn | Robustness | Before final | 0.5 | Done, no gain | `results/strong_audio_layer_lr.csv`: best dev gain +0.0002 (bar 0.005); equal-weight average stays |
| 3.3c | Retrain heads and gate on the C_raw features, 3 seeds | Robustness | **Before final, optional** | 0.5 | Paused | Audio-only weighted-F1 reported as mean +- std |
| 3.4 | Rebuild interventions for the variant (mean, blank, resampling); needs an all-layer silence pass for test | Robustness | **Before final, optional** | 2 | Paused | Protocol in `04_interventions.py` unchanged; effects reproduce for the current model first |
| 3.5 | Rerun faithfulness and bootstrap on the variant; add a variant table to the report | Report | **Before final, optional** | 0.5 | Paused | Every number traceable to a `results/` file |

Findings to keep (for the report, not the slides): the equal average of all 25 layers beats the last layer by about +0.03 weighted-F1 (one seed, no interval); mean + std pooling, standardising, weight decay and a learned layer mix gave no gain on dev; the audio head stays far below text (0.49). `data/dataset_strong.pt` (local) holds the C_raw features if the variant is resumed.

## Open questions
- Does the audio-only gain translate into a different gate verdict? Pre-registered reading: the headline stays the current model whatever happens; the variant only says whether the verdict is robust.
- Blank-audio embedding for the variant needs a silence pass through all layers (test only, 128 length buckets).

## Effects on other steps
- Step 4 (interventions): needs layer-weighted silence embeddings for the variant.
- Steps 5/6: results go in a separate variant table; figures unchanged.
- Step 8 (plan to final): adds this variant to the robustness checks.
