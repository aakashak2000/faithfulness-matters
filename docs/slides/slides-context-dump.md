# Full context for building the 2 midterm slides (give this whole file to the slide tool)

## 0. How to use this
Build exactly 2 slides. Follow section 8 (slide content) and section 9 (design rules). Sections 1 to 7 are background so wording, labels and captions stay accurate. Do not add claims or numbers that are not in this file.

## 1. The talk in one paragraph
Papers that fuse text and audio with a learned weight often read that weight as how much each modality mattered. We ask whether that reading is true. We built a text + audio emotion model on MELD that is at least on par with the published baseline: a balanced version at 58.8 weighted F1 (within half a point of 59.25) and a stronger, text-heavy version at 63.6. We ran intervention tests: swap one modality for another utterance's, rerun the model, and compare how much the prediction moves with what the weight says. On the balanced model (text mattered more in 54% of utterances) the weight agrees with the swap test 75% of the time against 54% for the best trivial guess. On the text-heavy model (95%) the weight is no better than always guessing text. So the weight is informative only while text and audio are comparably strong. A corrected signal (weight times how distinctive each model's vote is) tracks the size of the swap effects much better. Next: a conflict test and the full report.

## 2. Logistics
- Course CSCI 544, Fall 2026, Team 47. Midterm presentation slot: group 24, Oct 22. Slides due Oct 21.
- 2 slides, 2 minutes of speaking (hard cut-off, timer), then 1 minute of questions. Professor's advice: do not put too many things on the slides, control the time.
- Audience: professor, TAs, classmates (peers grade on a Google form). Presenter speaks the detail; slides carry only the headline items.
- Rubric lines the slides should visibly serve: clear problem and question, the task/input/output stated, method, an actual result with a takeaway, and next steps.

## 3. Project in plain language
- Title: Faithfulness MATTERS. MATTERS = Modality Attribution Testing Through Emotion Recognition Signals.
- Question: does the modality weight of a text + audio emotion model match how the prediction actually reacts when we remove the text or the audio? This is an evaluation of explanations, not an accuracy project.
- Dataset: MELD (Poria et al. 2019), dialogue from the TV show Friends. Official split; test set has 2,610 utterances.
- Task: input is one utterance (text transcript and audio clip); output is one of 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise. Metric: weighted F1.

## 4. The model
- A text predictor and an audio predictor each give 7 emotion probabilities.
- Fusion: final probability = w x p_text + (1 - w) x p_audio, with w between 0 and 1 per utterance.
- Two ways to get w: (a) a learned gate, a small neural network trained to decide it; (b) a confidence rule, w = text confidence / (text confidence + audio confidence).
- Reading under test: w above 0.5 means "text mattered more", below 0.5 "audio mattered more".
- Balanced model (each utterance judged alone; small text encoder, strong all-layer audio encoder; text mattered more in 54% of utterances): 58.8 weighted F1 with a class-prior correction chosen on dev data (55.7 without it). Source: results/harness_models.csv, row "M1 stronger audio". (An earlier first model, 51.7, is not shown on the slides.)
- Text-heavy model (our strongest): much stronger encoders (RoBERTa-large for text, WavLM-large for audio), plus the model's outputs for neighbouring utterances in the same dialogue (conversation context), plus a class-prior correction chosen on dev data (never on test): 63.6 weighted F1.
- Source: results/f1_push_v2.csv (stage S2, gate rule, dev alpha 0.5).

## 5. Comparison with published work (MELD, official test set, weighted F1 %)
Each number was read in the paper itself. NOT a controlled comparison: our encoders are newer, we fitted a class-prior correction on dev data, and the other systems differ in inputs and context.

| System | Year | Inputs | MELD |
|---|---|---|---|
| bcLSTM (MELD baseline) | 2019 | text + audio | 59.25 (MELD paper Table 11) |
| DialogueRNN | 2019 | text + audio | 60.25 (same table) |
| Ours, balanced model | 2026 | text + audio | 58.8 |
| Ours, text-heavy model | 2026 | text + audio | 63.6 |
| UniMSE | 2022 | text + audio + video | 65.51 |
| SDT | 2023 | text + audio + video | 66.60 |
| M2FNet | 2022 | text + audio + video | 66.71 |
| MultiEMO | 2023 | text + audio + video | 66.74 |
| TelME | 2024 | text + audio + video | 67.37 (text + audio only variant: 67.19) |
| MERC-PLTAF | 2025 | text + audio | 67.73 |
| MiSTER-E | 2026 | speech + text | 69.5 (fine-tuned 8B text model and 7B speech model) |
Other datasets (different tasks, not comparable to MELD scores): IEMOCAP 70 to 75 (MERC-PLTAF 74.80, MiSTER-E 70.9, TelME 70.48), CMU-MOSI sentiment 87.9 (MiSTER-E).
Why this matters for the story: the model we test is at or above the published baselines, so findings are not an artifact of a weak model. We do not claim state of the art (best recent systems are 4 to 6 points higher). None of these papers tests whether their fusion weights match removal effects.

## 6. The intervention test
- Swap test: replace one modality of an utterance (its text or its audio) with that of another real utterance from a pool of other utterances, rerun the whole model including the weight, and average over all swaps.
- Effect = change in the probability of the originally predicted emotion. The modality whose swap changes it more "mattered more".
- Compare with the weight: if w above 0.5 and text swap moved it more, the call is a match.
- With conversation context: only the explained utterance's text or audio is swapped; its neighbouring utterances stay as they are.
- Why swap and not delete: blank input is far from real data and could cause artifacts; swapping stays on realistic inputs. Blank and mean replacement are kept as checks.
- Same swap-and-average idea as Hessel and Lee 2020 (EMAP), who use it for a different question.
- Protocol was fixed before results were seen (header of scripts/37_swap_with_context.py).

## 7. Results (results/harness_models.csv for the balanced model; results/swap_with_context.csv for the text-heavy model; bootstrap over dialogues)
| Measure | Balanced model (58.8) | Text-heavy model (63.6) |
|---|---|---|
| Text mattered more (swap test) | 54.2% | 95.2% |
| Weight's call matches the swap test | 75.4% | 93.6% [92.6, 94.5] |
| Best trivial guess (usual winner) | 54.2% | 95.2% |
| Weight's gain over the trivial guess | +21.1 points [18.8, 23.6] | -1.6 points [-2.6, -0.6] |
| Rank correlation, weight vs size of swap effects | 0.72 | 0.43 |
| Corrected signal matches the swap test | 93.9% | 95.8% |
| Corrected signal rank correlation | 0.97 | 0.96 |
Takeaway: the weight is informative only while text and audio are comparably strong. When text dominates, the weight says "text" almost always and is right almost always, which adds nothing beyond the guess. The weight reflects trust, not influence; weight times how distinctive each model's vote is (distance from a typical vote) tracks the size of swap effects much better.
Extra fact (text-heavy model): when audio did win (4.8%), the weight said audio 53.9% of the time, and when it said audio it was right 38.3%.
Caveats: the balanced model has no conversation context and its 58.8 uses a class-prior correction fitted on dev; the text-heavy model uses neighbouring utterances plus that correction; the corrected signal is the closed form of the same swap, so it explains the mismatch rather than independently confirming it; the balance finding comes from models we built, so we do not claim it for every system. Weak-audio models are not shown on the slides.

## 8. Slide content

### Slide 1: the problem, our models, the test
- Title: Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals. Subtitle: CSCI 544, Team 47.
- Problem (one line): Models fuse text and audio with a learned weight, and papers read it as how much each modality mattered. Is that reading true? Small citations: Liu et al. 2020; He and Xiao 2025.
- Our solution (diagram + two numbers): Text -> text predictor; Audio -> audio predictor; both -> weight (learned gate or confidence) -> 1 of 7 emotions. Balanced model: 58.8 weighted F1. Text-heavy model (stronger text + neighbouring utterances): 63.6.
- Comparison table: bcLSTM 59.25, DialogueRNN 60.25, Ours balanced 58.8, Ours text-heavy 63.6, TelME 67.37 (text+audio+video), MERC-PLTAF 67.73, MiSTER-E 69.5. Columns: System, Inputs, MELD weighted F1. Highlight the two "Ours" rows. Footnote: not a controlled comparison; our encoders are newer and a class-prior correction is fitted on dev data, never on test.
- The test: Swap one modality for another utterance's, rerun the model, measure how much the prediction changes; compare with the weight. Small citation: Hessel and Lee 2020.

### Slide 2: what we found
- Title: The weight is informative only when text and audio are balanced.
- Chart (grouped bars, y = % of utterances where the call matches the swap test; two groups):
  - Balanced model (text mattered more in 54%): always guess usual winner 54.2 (grey), gate weight 75.4, corrected signal 93.9.
  - Text-heavy model (text mattered more in 95%): always guess text 95.2 (grey), gate weight 93.6, corrected signal 95.8.
  Caption: each bar is how often the call (text or audio) agrees with the swap test; the grey bar is just guessing the usual winner.
- Key lines: Balanced: the weight beats the guess by 21 points. Text-heavy: the weight is no better than always guessing text. So improving accuracy by strengthening one modality removes what the weight can tell us.
- Takeaway box: The weight shows trust, not influence. Weight times how distinctive each model's vote is tracks the swap effects far better: rank correlation 0.97 vs 0.72 on the balanced model.
- Next: Conflict test (pairs where text and audio disagree), then the full report.
- Footnote: results/harness_models.csv, results/swap_with_context.csv; bootstrap over dialogues.

## 9. Design rules
- Few words per slide; no paragraphs. Title plus at most three blocks per slide. Minimum body font about 18 pt, table about 16 pt.
- Slide 1 reads top to bottom: problem, solution, test. Slide 2: chart left, key lines and takeaway right.
- Chart: labelled bars, full 0 to 100 axis, grey for the trivial guess, one accent colour for the gate, a second for the corrected signal. Direct-label bar values. No dual axes.
- No em-dashes; plain language; define jargon in a few words. Use only numbers from this file. Citations in small text, not spoken.

## 10. Spoken script for reference (v11, 246 words, about 98 s at 150 wpm)
Slide 1: "Our project is Faithfulness MATTERS. Many models fuse text and audio with a learned weight, and papers read that weight as how much each modality mattered. Is that reading true? To test it, we first built a text and audio emotion model on MELD, at least on par with the published baseline. It gives a text prediction and an audio prediction for each utterance, and a learned gate or a confidence weight mixes them into one of seven emotions. With text and audio balanced, it scores 58.8 weighted F1, within half a point of the 59.25 bcLSTM baseline. A stronger, text-heavy version reaches 63.6. Recent systems reach 67 to 70. Then, intervention tests: swap one modality for another utterance's, rerun the model, and see how much the prediction changes."
Slide 2: "We read the weight simply: above 0.5, text mattered more. On the balanced model, text mattered more in 54% of utterances. The weight matches the swap test 75% of the time, against 54% for always guessing the usual winner. On the text-heavy model, text mattered in 95%, and the weight, 94%, is no better than always guessing text, 95%. So the weight is informative only while text and audio are comparably strong. It shows trust, not influence. Weight times how distinctive each model's vote is matches the swap test 94% on the balanced model, and tracks its size far better: correlation 0.97, against 0.72. Next: the conflict test, where text and audio disagree, and the full report."

## 11. Likely questions
Task, input, output, labels (MELD annotations, 7 emotions); what the weight is; what the swap test does and why not delete; comparison with published work (balanced 58.8 is within half a point of 59.25, text-heavy 63.6 is above it, recent systems 67 to 70, not controlled); is the balanced F1 fair (no context, class-prior correction chosen on dev, 55.7 without it); why two models (balance decides whether the weight is informative); is the corrected signal independent confirmation (no, closed form of the same swap). Full list in docs/slides/qa.md.
