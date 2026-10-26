# Model architecture registry (2026-10-23)
Every cell comes from a script or a saved object named in the Source column. UNVERIFIED means I did not find it in code or data. Encoders are never fine-tuned in any model below: every script extracts features once (`27_extract_text_layers.py`, `09_extract_layers.py`, legacy MiniLM and WavLM embeddings) and trains only the heads and the gate on top of the saved features. This is established by reading the scripts; I did not trace every gradient path.

## Shared pieces
| Piece | Specification | Source |
|---|---|---|
| Data | MELD, official train/dev/test; train re-split BY DIALOGUE into clf2 ("clf", 8,004 rows incl. dialogue-level 80%) and gate2 ("gate", 1,984 rows) with numpy rng 3000 for the v2 split; dev 1,108; test 2,610 (the swap pool for M0b to M5 is these 8,004 rows, verified from the saved pipeline and the cached embeddings, 2026-10-25; the 7,990 in the headers of steps 04 to 20 is the original split's clf size) | `29_context_pipeline.py` header, `08_resampling_full.py` header |
| Gate | `LearnedGate`: Linear(384 + 1024 -> 128), ReLU, Linear(128 -> 1), Sigmoid; 180,481 parameters; input = MiniLM text embedding (384) and the 1,024-d audio embedding ("headline gate inputs") in EVERY model below, including those whose text head reads RoBERTa | `common.py` lines 76 to 80; `29_context_pipeline.py` header; `33_eval_harness.py` header |
| Gate training | class-weighted CE on gate2 using the heads' out-of-sample probabilities; Adam lr 1e-3, full batch, early stopping on dev loss (patience 20, check every 50 steps, max 5,000) | `common.py` `train_full_batch`, `train_gate` |
| Fusion | p = w p_text + (1 - w) p_audio on probability vectors (late fusion) | `common.py` |
| Class-prior correction | q = softmax(log p + alpha log pi), pi = class frequencies of the clf split, alpha from {0, .25, .5, .75, 1} chosen on dev weighted-F1 (gate rule, mean over gates), applied to the FUSED probability at the end; used only for F1, NOT in any swap result of steps 33 to 41 | `33_eval_harness.py`, `30_context_pipeline_v2.py` header |
| Frozen text encoder 1 | `sentence-transformers/all-MiniLM-L6-v2`, 384-d sentence embedding | `01_build_dataset.py` line 86 |
| Frozen text encoder 2 | `FacebookAI/roberta-large`, mean-pooled hidden states of all 25 layers, max 128 tokens, stored as half precision; 25 x 1,024 = 25,600 features, standardised with clf mean and std | `27_extract_text_layers.py`; `text_head_selected.json` (features "all25", wd 1.0, dim 25,600) |
| Frozen audio encoder | `microsoft/wavlm-large`; headline model: pooled 1,024-d embedding (last layer, masked mean); stronger audio: all 25 layers, mean and std pooling, 51,200 features, standardised | `03_silence_embeddings.py`; `22_better_audio.py`, `29_context_pipeline.py` (AUDIO_LAYERS 0 to 24, "mean+std", wd 1.0) |
| Stronger heads | linear (multinomial) head, class-weighted CE with 0.5 wd ||W||^2 trained by LBFGS to convergence on clf; weights stored in `data/v2_pipeline.pt` ("Wb") | `22_better_audio.py`, `29_context_pipeline.py` |
| Weak heads | `Head`: one Linear layer; MiniLM head 384 x 7 + 7 = 2,695 parameters; headline audio head 1,024 x 7 + 7 = 7,175; Adam lr 1e-3, early stopping on dev loss; trained ONCE (seed 0) and shared by all gates of a model | `common.py`; `33_eval_harness.py` |

## Per-model table
| Model | Text side | Audio side | Context | Parameters trained (heads only) | Gates (seeds) | Class-prior alpha (dev) | Reported gate weighted F1, alpha 0 / dev alpha |
|---|---|---|---|---|---|---|---|
| M0b headline recipe, v2 split | MiniLM embedding, linear head (2,695) | last-layer WavLM, linear head (7,175) | none | 9,870 | 10 on shared heads | 0.25 | 0.5167 / 0.5751 (`harness_models.csv`) |
| M1 "balanced" | MiniLM, linear head (2,695) | all-layer WavLM mean+std, linear head 51,200 x 7 + 7 = 358,407 | none | 361,102 | 10 | 0.25 | 0.5566 / 0.5880 |
| M2 stronger text | RoBERTa-large all layers, linear head 25,600 x 7 + 7 = 179,207 | headline last-layer head (7,175) | none | 186,382 | 10 | 0.50 | 0.5969 / 0.6348 |
| M3 | RoBERTa all layers (179,207) | all-layer WavLM (358,407) | none | 537,614 | 10 | 0.25 | 0.6087 / 0.6270 |
| M5 "text-heavy" (63.6) | RoBERTa all layers stage 1; stage 2 multinomial LR on [own logits; mean of previous 5 logits; has_prev], wd 0.01 ("prev", k = 5 for text) | all-layer WavLM stage 1; stage 2 on [own logits; mean of previous 3 and next 3 logits; flags] ("both", k = 3, wd 0.01) | within one dialogue, from stage-1 logits of neighbouring utterances (out-of-fold for training rows); the audio stage uses FOLLOWING utterances too (offline, like bcLSTM) | stage 1 as M3 plus stage 2 (not tabulated, UNVERIFIED) | 3 | 0.50 | 0.5709 / 0.6357 (`swap_with_context.csv`) |
Notes
- The gate's text input is the MiniLM embedding in all models, so in M2, M3 and M5 the gate does not see the representation that drives the text prediction (RoBERTa, 25,600-d). This is a mismatch risk (see `docs/research_audit_2026-10-23.md`, Phase 3).
- "Balanced" M1 pairs a weak text encoder (MiniLM, 384-d, one linear layer) with a stronger audio side. Its balance is a property of this particular pairing, not of an architecture-matched family.
- Parameter counts are computed from the shapes above, not read from a model summary. Stage-2 sizes for M5 are UNVERIFIED.
- Heads are shared across gates, so "10 seeds" are 10 gates, not 10 trained models.
