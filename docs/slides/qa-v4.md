# Q&A set v4 (2026-10-21): everything the professor could ask, in the order a listener's questions arise
Built from (1) the question patterns he used on the 24 teams on 2026-10-20 (`lecture-observations-2026-10-20.md`: task/output/label 8, labels/ground truth 5, takeaway/results 5, prior work 2, artifact 3, metric/controls 2, how to read chart 1, hardest part 1) and (2) the weak points of our own project (`methodological_audit.md`).
Format: you get 1 minute for questions, so say the **Short answer** column (1 to 3 sentences). **If pushed** is the follow-up. Numbers come from the files named in **Basis**. Items marked VERIFY are not established in the repo; check before saying them.
Two models only: balanced (58.8) and text-heavy (63.6). Illustrative numbers in examples are made up and labelled so.

## A. Task, input, output, labels (he asked this of 8 of 24 teams)

| # | Likely question | Short answer (say this) | If pushed | Basis |
|---|---|---|---|---|
| 1 | What is the input, output and label? | Input: one utterance, its transcript and its audio clip. Output: one of seven emotions: anger, disgust, fear, joy, neutral, sadness, surprise. Labels are MELD's own annotations of Friends dialogue. | Official split, 2,610 test utterances. | MELD paper (Poria et al. 2019) |
| 2 | Where do the labels come from, who made them? | MELD's authors released them; we did not create labels. | Annotator agreement is a property of MELD; VERIFY the figure in the MELD paper before quoting it. | MELD paper |
| 3 | Is the output per utterance or per dialogue? | Per utterance. The balanced model sees one utterance alone; the text-heavy model also uses its neighbours' predictions, but still predicts one utterance. | Neighbours' outputs only, never their labels. | `scripts/30_context_pipeline_v2.py` |
| 4 | What does weighted F1 mean? | F1 balances precision and recall for each emotion; weighted F1 averages the seven F1 scores, weighting each by how common that emotion is. 58.8 is not "58.8% of utterances correct". | Example (made-up): 90 neutral, 10 anger, always say neutral: accuracy 90%, weighted F1 about 85%, macro F1 about 47%. | `results/harness_models.csv` |
| 5 | Why weighted F1 and not accuracy? | MELD is imbalanced (neutral dominates), and weighted F1 is the metric the published baselines report, so numbers are comparable. | | MELD paper Table 11 |
| 6 | What is the model, in one breath? | A text predictor and an audio predictor each give seven emotion probabilities. A weight w per utterance mixes them: final = w x text + (1 - w) x audio. w comes from a learned gate or a confidence rule. | Mixing happens on probabilities (late fusion), not on embeddings. | `scripts/common.py` |
| 7 | What is the weight, in plain words? | A number from 0 to 1 per utterance. Above 0.5 the model leans on text, below 0.5 on audio. Papers read it as how much each modality mattered. | | `docs/slides/slide-01-sources.md` |

## B. Is your model any good? Is it new? (prior work, 2 of 24 teams)

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 8 | How does it compare with published work? | Balanced 58.8, within half a point of the 59.25 bcLSTM baseline. Text-heavy 63.6, above bcLSTM and DialogueRNN (60.25). Recent systems are 67 to 70, so we do not claim state of the art. | Not controlled: our encoders are newer, and we fit a class-prior correction on dev data. | `docs/slides/comparison-table.md`, `results/f1_push_v2.csv` |
| 9 | Why not chase state of the art? | Our question is whether weights match removal effects. We needed models at or near the published baseline so findings are not an artifact of a weak model. | | `docs/PROJECT_REQUIREMENTS.md` |
| 10 | Is 58.8 fair? | It judges each utterance alone and uses a class-prior correction chosen on dev, never on test. Without that correction it is 55.7. | So it is 0.45 below bcLSTM, not above. | `results/harness_models.csv` (M1) |
| 11 | Is this already studied? Any recent papers? | Papers read gate weights as importance (Liu et al. 2020; He and Xiao 2025). None of the eight gating papers we looked at tests them by removal. The nearest method is Hessel and Lee 2020, who swap and average for a different question. | This was a targeted search, not a full survey. | `docs/slides/slide-01-sources.md` |
| 12 | What did Hessel and Lee ask? | Whether a model uses interactions between modalities or just adds each one's separate contribution. They test it by swapping or marginalising one modality with other examples. | Image plus text tasks. We reuse swap-and-average to test whether the weight matches the effect. | `docs/slides/slide-01-sources.md` |
| 13 | What is new here? | We test whether gate weights match removal effects on a real text+audio emotion model, and show it depends on modality balance. We do not claim a new attribution method. | Balance alone is not sufficient for an edge, which we also report. | `results/balance_experiments.csv` |
| 14 | Other datasets? | Text+audio systems score 70 to 75 on IEMOCAP and 87.9 on CMU-MOSI (sentiment). Different tasks, not comparable to MELD scores. | | `docs/slides/comparison-table.md` |

## C. The swap test, step by step

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 15 | Explain the swap test in one sentence. | Replace one modality of an utterance with that of other real utterances, rerun the whole model including the weight, average the predictions, and see how far the originally predicted emotion's probability moves. | | `scripts/37_swap_with_context.py` |
| 16 | Walk me through one example. | (Made-up numbers.) Model says Anger 0.68. Swap text for many donors: average Anger 0.64, so text effect 0.04. Swap audio: average 0.50, so audio effect 0.18. Audio mattered more, even if the weight favoured text. | Repeat for every utterance. | `scripts/37_swap_with_context.py` |
| 17 | Why not just delete the modality? | Blank input is far from real data, so a change could be an artifact. Real swaps stay on realistic inputs. Blank and mean replacement are kept as checks. | | `docs/robustness_results.md` |
| 18 | Average first or absolute value first? | Average the donors' predictions first, then take the absolute change. Averaging absolute changes instead gives the same pattern. | The headline model's small edge flips sign under mean-of-abs; the balanced and text-heavy conclusions do not change. | `results/mean_of_abs_check.csv` |
| 19 | Does the swap test use the gate? | Yes. The whole model runs, and the gate is recomputed for every donor. So the swap measures real behaviour, not the original weight. | "Hold text fixed" for the swap; "frozen gate" is a separate audit. | `scripts/38_frozen_gate.py` |
| 20 | Is the swap effect causal ground truth? | No. It is net sensitivity of the prediction to replacing a modality with typical real inputs. It is the reference we compare the weight against. | Effects can cancel in the average, which is why we also checked other measures. | `docs/methodological_audit.md` |
| 21 | Swapping makes unnatural pairs (angry audio, calm text). Does that bias it? | Possibly, since donors are random. Our pool is real utterances from the same data, and we check alternative measures and replacements. Matched-donor swaps are planned for the final report. | | `docs/methodological_audit.md` |
| 22 | With context, what do you swap? | Only the explained utterance's text or audio. Its neighbours stay as they are. | | `scripts/37_swap_with_context.py` header |
| 23 | Why that effect measure? | The change in probability of the emotion the model originally predicted. It is what the weight claims to explain, and the balanced result also holds with whole-distribution change, label flips and signed drop. | | `results/harness_alt_effects.csv` |

## D. Results and takeaway (he asked for takeaways of 5 teams)

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 24 | What is the takeaway? | The weight is informative only while text and audio are comparably strong. Balanced model: weight agrees with the swap test 75%, against 54% for guessing the usual winner. Text-heavy: 94% against 95%, no better than always guessing text. | Weights show trust, not necessarily influence. | `results/harness_models.csv`, `results/swap_with_context.csv` |
| 25 | What is the "guess" baseline and why is it there? | Always say the modality that usually wins. It uses nothing about the utterance, so it tells you how much agreement is free. A weight is useful only if it beats it. | Rain analogy: in a desert, "no rain" is 95% right with no skill. | `results/swap_with_context.csv` |
| 26 | How big is the gain, and is it significant? | Balanced: +21.1 points over the guess, interval 18.8 to 23.6, bootstrapped over dialogues. Text-heavy: -1.6 points, interval -2.6 to -0.6. | Dialogue-level bootstrap, 1000 draws. | `results/harness_models.csv`, `results/swap_with_context.csv` |
| 27 | Why is the weight below the guess on the text-heavy model? | Text wins 95% of the time. When audio did win (4.8% of utterances), the weight said audio 54% of the time and was right 38% of the time it said so. The weight cannot find the rare audio cases. | The gate was trained for accuracy, not for faithful explanation. | `results/swap_with_context.csv` |
| 28 | Is it the context or the strong text that causes it? | A strong-text model without context also shows it: text wins 97.1% and the gate's gain over the guess is -0.4 points [-0.7, -0.2]. So context alone does not explain it. | | `results/harness_models.csv` (M2) |
| 29 | Does balance guarantee the weight is useful? | No. Flattening text to a 72% share gave +9.9 points, 42% gave +2.3, and weak-text models (12 to 15% share) gave about -12.7. | We say informative only while comparable strong, not "whenever balanced". | `results/balance_experiments.csv` |
| 30 | What happens with an audio-heavy model? | Not tested on the slides. Weak-audio models are not shown. The symmetric prediction is that "always audio" becomes a strong guess. | A test for the final report. | `docs/methodological_audit.md` |
| 31 | If the weight is not importance, what is? | Roughly the weight times how far a model's vote is from a typical replacement's vote. Example (made-up): text weight 0.8 but its vote is ordinary (gap 0.05), audio weight 0.2 but unusual (gap 0.30): effects 0.04 vs 0.06, audio wins. | Distinctiveness is measured on each model's own output, before fusion. | `results/corrected_signal.csv` |

## E. The corrected score and rank correlation (the hardest part of the talk)

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 32 | Define the corrected score. | S = w x |text vote - typical text vote| - (1 - w) x |audio vote - typical audio vote|, on the originally predicted class. Positive means text should matter more. | "Typical" is the average over the swap donors of each model's own probability. | `scripts/34_corrected_signal.py` |
| 33 | What is rank correlation, here? | Order the utterances by how much more swapping text moved the prediction than audio. Order them again by a score. Rank correlation says how closely the orders match: 1 identical, 0 unrelated. | Weight alone 0.72, corrected 0.97 on the balanced model; text-heavy 0.43 vs 0.96. | `results/corrected_signal.csv`, `results/swap_with_context.csv` |
| 34 | Isn't that circular? It is built from the same donors. | Partly, and we say so. With the gate held fixed it is exactly the swap difference (100% sign agreement). We use it to explain why the weight misleads, not as independent proof. | It needs only the unimodal models' averages, not fused reruns, but still uses the donors. | `scripts/38_frozen_gate.py`, `results/frozen_gate_audit.csv` |
| 35 | Why is it not 100% on the real test? | The real swap lets the gate recompute its weight per donor. The 93.9% (balanced) gap, 1.2 to 6.5% of utterances across models, is exactly that gate term, mostly near-ties. | | `results/frozen_gate_audit.csv` |
| 36 | Agreement vs rank correlation, what is the difference? | Agreement asks who won, text or audio. Rank correlation asks by how much, so it separates a big text win from a near tie. | | `results/corrected_signal.csv` |
| 37 | Would you propose it as a new attribution method? | No. It is a diagnostic. A gate trained to match the swap effects is a possible next step, but it is not started and would need a protocol written before running. | | `docs/methodological_audit.md` |

## F. Controls, artifacts, honesty (3 of 24 teams got this)

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 38 | Is the balanced result one lucky model? | The spread across 10 trained gates is about 0.01, and the pattern holds across models we built under three other effect measures. They are still models we built, so we do not claim it for every system. | | `results/harness_per_gate.csv`, `results/harness_alt_effects.csv` |
| 39 | Did you tune on the test set? | No. The class-prior correction, context settings and cutoffs are chosen on dev and frozen. Protocols are in the script headers. | | `scripts/30_context_pipeline_v2.py`, `scripts/37_swap_with_context.py` |
| 40 | Leakage with neighbouring utterances? | The text-heavy model uses neighbours' predicted outputs, not labels. Only the explained utterance is swapped. | | `scripts/37_swap_with_context.py` |
| 41 | Does the confidence rule do better? | Its agreement is partly structural, since its weight is built from the same probabilities it mixes. We do not present it as a good explanation. | | `CLAUDE.md` |
| 42 | Duplicate audio? | 42 test rows are flagged (39 share byte-identical audio within test, 3 only with train). Report numbers with and without them. | The headline donor pool excludes byte-identical audio. | `results/audio_duplicates.json` |
| 43 | Is your threshold 0.5 arbitrary? | It is how papers read the weight: above 0.5, text. We test that reading as stated. | | `docs/slides/slide-01-sources.md` |

## G. Next steps and scope (he advised fewer variables, clearer results)

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 44 | What is next? | The conflict test: build pairs where text and audio disagree, plus same-emotion controls, and see whether the weight points to the modality that decides. Then the full report. | | `docs/PROJECT_REQUIREMENTS.md` |
| 45 | What is the hypothesis for the conflict test? | That weights become less reliable when modalities disagree. The opposite is also possible, so it stays a hypothesis. | A conflict pair has no single true label, so it is an attribution test, not an accuracy test. | `docs/PROJECT_REQUIREMENTS.md` |
| 46 | Why a control? | A mismatched pair differs in speaker, length and noise too. A same-emotion replacement separates swapping in general from emotional disagreement. | | `docs/PROJECT_REQUIREMENTS.md` |
| 47 | What is technically hardest? | Keeping each effect attributable to one input: swapping a modality so the model with context still changes only the explained utterance's input. | | `scripts/37_swap_with_context.py` |
| 48 | What would you do with more time? | Matched-donor and blank/mean checks on the text-heavy model, an audio-heavy model, and a better audio encoder. | | `docs/methodological_audit.md` |

## H. Models, compute, cost (VERIFY details before answering)

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 49 | Text model? | Balanced: MiniLM sentence embedding (384-d). Text-heavy: RoBERTa-large. Both feed small classifier heads. | VERIFY whether encoders were frozen or fine-tuned in the newer models. | `scripts/27_extract_text_layers.py`, `scripts/29_context_pipeline.py` |
| 50 | Audio model? | WavLM-large embeddings (1,024-d), pooled over time. The balanced model uses all layers. | Audio embeddings may carry words as well as tone. | `scripts/22_better_audio.py` |
| 51 | Fusion techniques? | Average with fixed 0.5, confidence rule c_t/(c_t + c_a), and a learned gate (small network over both embeddings). Weights test only the gate and the confidence rule. | | `scripts/common.py` |
| 52 | How much compute? | Training the heads and gates runs on CPU on purpose (determinism). Extracting audio embeddings took about 5 minutes on MPS for the test set. | VERIFY total for the newer models. | `CLAUDE.md` |
| 53 | How many seeds? | The earlier baseline used 3 seeds; the gate spread uses 10 trained gates. | | `results/more_seeds_*.csv`, `results/harness_per_gate.csv` |

## I. Chart reading

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 54 | How do I read this chart? | Each bar is how often a call, text or audio, agrees with the swap test. Grey is always guessing the usual winner. On the balanced model the gate bar is well above grey; on the text-heavy one it is not. | The corrected-signal bar is a diagnostic, not independent evidence. | `figures/fig6_agreement_by_balance.png` |

## J. Added from a second reviewer's list (kept only what is new and checkable)

| # | Likely question | Short answer | If pushed | Basis |
|---|---|---|---|---|
| 55 | Isn't "weight is not influence" obvious from the fusion equation? | The equation says a high weight can still mean a small effect. Whether learned weights track influence on real data is empirical, and we find it works in a balanced model but not when one modality dominates. | The corrected score is the equation's side of the story; the swap test is the data's. | `results/harness_models.csv`, `results/swap_with_context.csv` |
| 56 | Why not SHAP or Integrated Gradients? | Our claim under test is about the fusion weight itself, so we test that weight directly against interventions. Comparing with other attribution methods is a useful extension. | | `docs/PROJECT_REQUIREMENTS.md` |
| 57 | Isn't the always-text guess picked using test results, a leak? | It is a descriptive yardstick, the best constant rule on the observed swap winners. It trains nothing and changes no prediction. Because it is the best constant on this data, it is a hard baseline to beat. | A deployable version would pick the majority modality on dev and score it on test. | `results/swap_with_context.csv` |
| 58 | Does a different emotion label guarantee a real text-audio conflict? | No. A label can mismatch while the audio is still neutral-sounding. That is why the plan has same-emotion controls and an audio-verified subset. | These are planned, not yet decided in detail. | `docs/PROJECT_REQUIREMENTS.md` |
| 59 | What if the conflict test shows no drop in agreement? | Still informative: emotional disagreement alone would not explain the failures, and we would look at modality dominance and gate re-weighting instead. | | `docs/methodological_audit.md` |
| 60 | Why an audio-heavy model if you already have a text-heavy one? | To see whether the pattern is specific to text or follows from any one modality dominating. | | `docs/methodological_audit.md` |
| 61 | Does a better classifier mean a better explanation? | No. The text-heavy model has higher F1 (63.6 vs 58.8), yet its weight is no better than always guessing text. | | `results/swap_with_context.csv` |
| 62 | If this does not generalise, what did you contribute? | A concrete way to audit a fusion weight against a trivial baseline, and evidence that high agreement alone can mislead. The limits are part of the finding. | | `docs/methodological_audit.md` |

## K. Claims in that second list NOT to repeat until checked
- "We evaluated blank and mean replacements" on the two slide models: the blank/mean checks exist for earlier models; for the 63.6 model they are still a to-do for the final report (`docs/robustness_results.md`). Say "kept as checks, extending to the new model".
- "Alternative measures such as total variation distance": our alternative measures are whole-distribution change, label flips and signed drop (`results/harness_alt_effects.csv`); total variation is not confirmed, and they are not yet run on the 63.6 model.
- "Excluding duplicate-audio rows had negligible effects": not verified for these two models. Say the report will give numbers with and without the 42 flagged rows.
- Frozen MiniLM/WavLM/MLP details and "separate training partitions": documented for the earlier pipeline only. VERIFY for the newer models (rows 49 to 53).
- The one-sentence rewrite of the closing script line drops the clause "explains the gap rather than confirms importance". Keep that clause: it pre-empts the circularity question (#34).

## Five hardest, rehearse these first
1. #34 "Isn't the corrected score circular?" Say yes partly, exact when the gate is fixed, diagnostic only.
2. #27 "Why is the weight below the guess?" Text wins 95%, the weight misses the rare audio cases.
3. #18 "Average first or absolute value first?" Average first; mean-of-abs gives the same pattern.
4. #29 "Does balance guarantee it?" No, say the numbers.
5. #8 "How does it compare?" Within half a point below bcLSTM; not controlled; not state of the art.
