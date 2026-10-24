# Likely questions and answers (v3, 2026-10-21), ordered as a listener's questions would arise
Built from what the professor asked the other 24 teams: task, labels, takeaway, prior work, artifacts. Results come from two models: balanced (58.8) and text-heavy (63.6). Older versions: `qa-archive-v1.md`.

## A. What is the task? (asked of 8 of 24 teams)

| # | Likely question | Answer | Basis |
|---|---|---|---|
| 1 | What is the input, the output and the label? | Input: one utterance, its text and its audio clip. Output: one of seven emotions (anger, disgust, fear, joy, neutral, sadness, surprise). Labels are MELD's own annotations of Friends dialogue; official split, 2,610 test utterances. | MELD paper (Poria et al. 2019) |
| 2 | What exactly is your model? | A text predictor and an audio predictor each give seven emotion probabilities; a learned gate (small network) or a confidence rule sets a weight w, and the final probability is w x text + (1 - w) x audio. | `scripts/common.py` (`fuse_gate`, `fuse_confidence`) |
| 3 | What is the weight, in plain words? | A number between 0 and 1 per utterance: above 0.5 the model leans on text, below 0.5 on audio. Papers describe it as how much each modality mattered. | `docs/slides/slide-01-sources.md` |
| 4 | Why two models? | The balanced one (text and audio comparably strong, no context) is at the baseline level; the text-heavy one (stronger text plus neighbouring utterances) is our most accurate. They show how the weight behaves as the balance changes. | `results/harness_models.csv`, `results/f1_push_v2.csv` |

## B. Is your model any good? Is it new?

| # | Likely question | Answer | Basis |
|---|---|---|---|
| 5 | How does it compare with published work? | The balanced model, 58.8, is within half a point of the 59.25 bcLSTM baseline (and 1.5 below DialogueRNN's 60.25); the text-heavy model, 63.6, is above both. Recent systems reach 67 to 70 (MERC-PLTAF 67.73, MiSTER-E 69.5, TelME 67.37). Not controlled: our encoders are newer and we fitted a class-prior correction on dev data. | `results/harness_models.csv`; `results/f1_push_v2.csv`; `docs/slides/comparison-table.md` |
| 6 | Is 58.8 fair? | It judges each utterance alone and uses a class-prior correction chosen on dev, never on test; without the correction it is 55.7. | `results/harness_models.csv` (M1 gate_weighted_f1_a0 and _adev) |
| 7 | Why not match the state of the art? | Our question is whether weights match removal effects, not accuracy; we needed models near or above the published baseline to test them on. | `docs/PROJECT_REQUIREMENTS.md` section 2 |
| 8 | Is this already studied? Any recent papers? | Papers use gate weights as importance (Liu et al. 2020, He and Xiao 2025); none of the ones we read tests them by removal. The nearest method is Hessel and Lee 2020, who swap and average for a different question. We read a targeted set, not a full survey. | `docs/slides/slide-01-sources.md` |
| 9 | What about other datasets? | Text plus audio models score 70 to 75 on IEMOCAP and 87.9 on CMU-MOSI (sentiment); different tasks, so not comparable to MELD scores. | `docs/slides/comparison-table.md` |

## C. What is the intervention test?

| # | Likely question | Answer | Basis |
|---|---|---|---|
| 10 | Explain the intervention test in one sentence. | Replace one modality of an utterance with that of other real utterances, rerun the whole model including the weight, average, and see how far the predicted emotion's probability moves. | `scripts/37_swap_with_context.py` header |
| 11 | Why not just delete the modality? | Blank input is far from real data, so a drop could be an artifact; real swaps stay on realistic inputs. Blank and mean replacements are kept as checks. | `docs/robustness_results.md` |
| 12 | How do you decide which modality mattered more? | The one whose swap moves the predicted class's probability further; we compare that with the weight (above 0.5 should mean text). | `scripts/37_swap_with_context.py` |
| 13 | With context, what do you swap? | Only the explained utterance's text or audio; its neighbours stay as they are. (The balanced model has no context.) | `scripts/37_swap_with_context.py` protocol |

## D. What did you find? (takeaway)

| # | Likely question | Answer | Basis |
|---|---|---|---|
| 14 | What is the takeaway? | The weight is informative only while text and audio are comparably strong: on the balanced model it agrees with the swap test 75% of the time against 54% for guessing the usual winner; on the text-heavy model it is 94% against 95%, no better than always guessing text. | `results/harness_models.csv`; `results/swap_with_context.csv` |
| 15 | How big is the gain on the balanced model? | +21.1 points over the best trivial guess, interval 18.8 to 23.6, bootstrapped over dialogues. | `results/harness_models.csv` (gateW_gain_majority, _lo, _hi) |
| 16 | What happens on the text-heavy model? | Text mattered more in 95% of utterances; the weight's gain over always guessing text is -1.6 points [-2.6, -0.6]. When audio did win, the weight said audio 54% of the time and was right 38% of the time it said it. | `results/swap_with_context.csv` |
| 17 | If the weight is not importance, what is? | Roughly the weight times how far a model's vote is from a typical vote; this one-pass signal matches the swap test 93.9% on the balanced model and ranks its size with correlation 0.97, against 0.72 for the weight. | `results/corrected_signal.csv` |
| 18 | Is the corrected signal independent confirmation? | No: it is the closed form of the same swap, so it explains the mismatch rather than validating it. | `scripts/34_corrected_signal.py` header |

## E. Could it be an artifact? Controls and honesty

| # | Likely question | Answer | Basis |
|---|---|---|---|
| 19 | Is the balanced result just one lucky model? | The pattern holds across the models we built (balanced: edge +0.21; text-dominated: 0), under three other effect measures, and the spread across 10 trained gates is about 0.01; it is still models we built, so we do not claim it for every system. | `results/harness_models.csv`, `results/harness_alt_effects.csv`, `results/harness_per_gate.csv` |
| 20 | Did you tune on test? | No: class-prior correction, context settings and cutoffs are chosen on dev and frozen; protocols are in the script headers. | `scripts/30_context_pipeline_v2.py`, `scripts/37_swap_with_context.py` |
| 21 | Does the confidence rule do better? | Its agreement is partly structural, since its weight is built from the probabilities it mixes; we do not present it as a good explanation. | `CLAUDE.md` gotchas |
| 22 | Does the answer depend on the effect measure? | The primary measure is the change in the predicted class's probability; the balanced-model edge also appears with change in the whole distribution, label flips and signed drop. | `results/harness_alt_effects.csv` |
| 23 | Duplicate audio? | 42 test rows share audio; the report gives numbers with and without them. | `results/audio_duplicates.json` |

## F. Next steps

| # | Likely question | Answer | Basis |
|---|---|---|---|
| 24 | What is next? | The conflict test: build pairs where text and audio disagree and see whether the weight points to the modality that actually decides; then the full report. | `docs/PROJECT_REQUIREMENTS.md` |
| 25 | What is technically hardest? | Swapping a modality so the effect stays attributable to one input, including when the model also reads neighbouring utterances; we swap only the explained utterance. | `scripts/37_swap_with_context.py` |
| 26 | How do I read the figure? | Each bar is how often the weight's (or the corrected signal's) call, text or audio, agrees with the swap test; the grey bar is just guessing the usual winner. | `figures/fig6_agreement_by_balance.png` (rebuild for the two-model slide) |
