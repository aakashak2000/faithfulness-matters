# Speaking script (midterm presentation)
Budget: about 260 words for both slides (about 2 minutes at 150 words per minute), leaving about 1 minute for Q&A. Slide 1 target: about 120 words. Status: slide 1 opening FINAL (tightened version, 113 words, agreed 2026-10-20); rest DRAFT. Not yet rehearsed.

## Slide 1, opening (problem statement, citations, method line)

Version B (recommended, 66 words, about 26 seconds):
> Many multimodal models fuse text and audio with a learned weight, and papers often read that weight as how much each modality mattered. Liu and colleagues call their gate weights the contribution of each modality; He and Xiao say theirs assigns importance. We ask: is that reading actually true? We test two weights: a learned gate that mixes the two models' predictions, and a confidence-based weight.

Version A (longer, 78 words, about 31 seconds): same, with the second sentence as "For example, Liu and colleagues in 2022 describe their gate weights as the contribution of each modality, and He and Xiao in 2025 say their gate assigns importance to each modality."

Why B: the slide already shows the years and the full citations, so the spoken version can drop them and leave room for the inputs and outputs, the intervention tests and the gap (about 54 words left for slide 1).

## Basis for each sentence
| Spoken sentence | Basis |
|---|---|
| Many multimodal models fuse text and audio with a learned weight, and papers often read that weight as how much each modality mattered | `docs/slides/slide-01-sources.md` (8 papers; "many" and "often" are deliberate) |
| Liu and colleagues call their gate weights the contribution of each modality | Liu, Li & Meng 2020: "obtains interpretable contribution weights of each modality in emotion recognition" |
| He and Xiao say theirs assigns importance | He & Xiao 2025 (LPGNet): "The multimodal gate then adaptively assigns importance to each modality during the fusion process" |
| We ask: is that reading actually true? | Our research question (`docs/PROJECT_REQUIREMENTS.md` section 2) |
| A learned gate that mixes the two models' predictions | `LearnedGate`, `fuse_gate` in `scripts/common.py` |
| A confidence-based weight | `fuse_confidence` in `scripts/common.py`: c_text / (c_text + c_audio) |

## Delivery notes
- Say "gate weights" and "importance" in the authors' words; do not say they "claim it is faithful", because they do not use that word.
- Pause after "is that reading actually true?" and move to the next slide block (inputs and outputs).
- If asked how many papers: "eight systems in a targeted search, four of them text plus audio emotion recognition; we do not claim a full survey."

## Still to write
Slide 1 remainder (inputs and outputs, intervention tests, prior-work gap), slide 2, timed rehearsals, Q&A one-liners.


---

## Slide 1, opening: presenter's draft (2026-10-20) and suggested edit
Status: DRAFT, not final. Opening only; the inputs and outputs, the intervention tests and the gap are still to write.

**Presenter's draft (148 words, about 59 seconds):**
> Good afternoon Professor, TAs, and our fellow classmates. Our project is called Faithfulness Matters. Matters stands for Modality Attribution Testing through Emotional Recognition Signals. I'll just set the premise for the project. There are many multimodal systems that fuse two modalities. They fuse text or audio, they fuse text or image, and they fuse image and audio. We are specifically looking at text and audio. There are papers that fuse them using either a learned weight using a simple feed-forward network, or they just compute a number. That's a weight for one modality, and one minus that number is the weight for the other modality. Now, these papers claim that this weight is actually the importance that contributes towards the prediction. The question we are trying to ask is: these weights that are being computed by the model, are these actually the importance of these modalities or not?

**Edits suggested, with the reason for each:**
1. "Emotional Recognition Signals" should be "Emotion Recognition Signals" (README).
2. "Text or audio, text or image, image and audio" should read as pairs ("text with audio, text with image, audio with image"), or be cut to save time.
3. Add the two agreed citations: Liu and colleagues, and He and Xiao.
4. "These papers claim that the weight is the importance that contributes towards the prediction" should be "papers often read the weight as how much each modality mattered". Their wording is "contribution" (Liu et al.) and "importance" (He & Xiao); "claim" is stronger than what they wrote and a TA could challenge it.
5. "Or they just compute a number" is supported by the entropy gate of Wu et al. 2025 (a weight set by how certain each modality's features are, though combined with learned logits) and by our own confidence weight; say "compute it from how confident each model is" so it sets up the two weights we test.
6. "Good afternoon" assumes the slot time; confirm it.
7. Time: this opening alone is 148 words. The whole script is about 260 words for both slides, with slide 1 near 120. Trim to about 100 and keep the inputs and outputs, intervention tests and the gap short.

**Suggested tightened version (113 words, about 45 seconds), keeping your structure and voice:**
> Good afternoon Professor, TAs and classmates. Our project is Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals.
> 
> Many multimodal systems fuse two modalities; we look at text and audio. The fusion often uses one weight: a number for one modality, and one minus that number for the other. Some papers learn that weight with a small neural network; others compute it from how confident each model is. And papers often read that weight as how much each modality mattered: Liu and colleagues call it the contribution of each modality, and He and Xiao say it assigns importance. So our question is: are these weights really the importance of each modality, or not?

**Decision 2026-10-20: the tightened version above (113 words, about 45 seconds) is the slide 1 opening, FINAL.**

## Slide 1, remainder (DRAFT, 80 words, about 32 seconds)
> We test this with intervention tests: we replace the text or the audio with real examples from other utterances, rerun the model, and see which removal changes the prediction more. Jacovi and Goldberg ask when an explanation is faithful, and Hessel and Lee use a similar replacement idea for a different question; in the papers we read, none checks a weight this way. We also add a trivial "always text" baseline, so a weight only counts if it beats it.

Inputs and outputs (an utterance's text and audio; an emotion and a text weight) are left to the slide visual, to save about 15 spoken words.

| Spoken sentence | Basis |
|---|---|
| We replace the text or the audio with real examples from other utterances, rerun the model | `scripts/08_resampling_full.py` (donors are the 7,990 classifier-split utterances) |
| Jacovi and Goldberg ask when an explanation is faithful | Jacovi & Goldberg 2020 abstract (to be checked against the paper) |
| Hessel and Lee use a similar replacement idea for a different question | `slide-01-sources.md`, Hessel & Lee section (read in full text) |
| In the papers we read, none checks a weight this way | Keyword search of 8 full texts; keep "in the papers we read" |
| Trivial "always text" baseline | `scripts/05_faithfulness.py`, `scripts/06_analysis.py` |

Running total for slide 1: 193 words (about 77 seconds). Slide 2 then has about 67 words if the whole script stays near 260.

---

## Slide 1 script, consolidated as of 2026-10-20 (185 words, about 74 seconds at 150 words per minute)
Status: **FINAL for slide 1 as written (opening and third paragraph), locked by the presenter 2026-10-20.** Inputs and outputs are left to the slide visual. Not yet rehearsed. Any change needs the presenter's approval.

Good afternoon Professor, TAs and classmates. Our project is Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals.

Many multimodal systems fuse two modalities; we look at text and audio. The fusion often uses one weight: a number for one modality, and one minus that number for the other. Some papers learn that weight with a small neural network; others compute it from how confident each model is. And papers often read that weight as how much each modality mattered: Liu and colleagues call it the contribution of each modality, and He and Xiao say it assigns importance. So our question is: are these weights really the importance of each modality, or not?

We test this with intervention tests: we swap an utterance's text, or its audio, for real examples from thousands of other utterances, rerun the model, and average. Whichever swap changes the prediction more marks the modality that mattered more; we check whether the weight favoured that same one. Hessel and Lee used the same swap-and-average idea for a different question; none of the papers we read uses it to check a weight.

Open items for this script: confirm the slot time ("Good afternoon"); decide whether the baseline ("always text") is mentioned here or at the start of slide 2; rehearse against a timer. ("Thousands of" stays; it was not cut.)

---

## Slide 1 script v2 (2026-10-20): locked text plus ONE pending insertion
Locked text = opening (113 words) and the swap-test paragraph (72 words). The bracketed paragraph is PENDING the presenter's approval; it merges "we test both kinds" with "we built our own model", and replaces the 50 to 60 word versions discussed earlier. Total with the insertion: 225 words, about 90 seconds.

Good afternoon Professor, TAs and classmates. Our project is Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals.

Many multimodal systems fuse two modalities; we look at text and audio. The fusion often uses one weight: a number for one modality, and one minus that number for the other. Some papers learn that weight with a small neural network; others compute it from how confident each model is. And papers often read that weight as how much each modality mattered: Liu and colleagues call it the contribution of each modality, and He and Xiao say it assigns importance. So our question is: are these weights really the importance of each modality, or not?

[PENDING] To test both kinds, we built our own model. The MELD baseline, bcLSTM, has no modality weight; ours has a learned gate and a confidence weight. At 51.7 weighted F1 it is below bcLSTM's 59.25, which reads the whole conversation.

We test this with intervention tests: we swap an utterance's text, or its audio, for real examples from thousands of other utterances, rerun the model, and average. Whichever swap changes the prediction more marks the modality that mattered more; we check whether the weight favoured that same one. Hessel and Lee used the same swap-and-average idea for a different question; none of the papers we read uses it to check a weight.

Basis for the pending paragraph: bcLSTM description in Poria et al. 2017 (features concatenated, no modality weight) and the MELD paper (Table 11: bcLSTM text + audio weighted-F1 59.25, whole-conversation context); our gate weighted-F1 0.5175 (3 seeds, `results/baselines_summary.csv`) = 51.7. Gap to bcLSTM is about 7.5 points. "Our own model" is confirmed true: the presenter built the legacy embeddings and bundle (`data/legacy/fusion_bundle_final.pt`) (confirmed 2026-10-20); the heads, gate and fusion rules were trained in this repo (`scripts/02_train.py`).

---

## Slide 1 script v3 (2026-10-20): all paragraphs locked (225 words, about 90 seconds)
Locked by the presenter. Change = new decision. Not yet rehearsed. Inputs and outputs are left to the slide visual.

Good afternoon Professor, TAs and classmates. Our project is Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals.

Many multimodal systems fuse two modalities; we look at text and audio. The fusion often uses one weight: a number for one modality, and one minus that number for the other. Some papers learn that weight with a small neural network; others compute it from how confident each model is. And papers often read that weight as how much each modality mattered: Liu and colleagues call it the contribution of each modality, and He and Xiao say it assigns importance. So our question is: are these weights really the importance of each modality, or not?

We built our own models: a text model and an audio model, combined two ways, with a learned gate or a confidence weight. The best scores 51.7 weighted F1, below the MELD baseline bcLSTM's 59.25, which reads the whole conversation.

We test this with intervention tests: we swap an utterance's text, or its audio, for real examples from thousands of other utterances, rerun the model, and average. Whichever swap changes the prediction more marks the modality that mattered more; we check whether the weight favoured that same one. Hessel and Lee used the same swap-and-average idea for a different question; none of the papers we read uses it to check a weight.

Notes
- "The best" = the learned gate (weighted-F1 0.5175, 51.7; confidence rule 0.5145, 51.5; `results/baselines_summary.csv`). bcLSTM 59.25 is the text + audio weighted-F1 in Table 11 of the MELD paper; it models the whole conversation.
- The clause "bcLSTM has no modality weight" was dropped from the spoken text to save words. It is NOT currently on the slide either; keep it for Q&A or add it to the slide's "what we add" strip.
- Open: slot time ("Good afternoon"); where the "always text" baseline is first mentioned (proposed: start of slide 2); timed rehearsal.

---

## FULL SCRIPT v1, slides 1 and 2 (2026-10-20): 367 words, about 147 seconds at 150 wpm
Purpose: a complete first pass so we can see where the time goes and edit on the fly. Slide 1 paragraphs are locked (v3); slide 2 is a first draft. The rubric suggests about 260 words (about 2 minutes), so this is about 107 words over.

**Slide 1: title**
> Good afternoon Professor, TAs and classmates. Our project is Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals.

**Slide 1: premise and question**
> Many multimodal systems fuse two modalities; we look at text and audio. The fusion often uses one weight: a number for one modality, and one minus that number for the other. Some papers learn that weight with a small neural network; others compute it from how confident each model is. And papers often read that weight as how much each modality mattered: Liu and colleagues call it the contribution of each modality, and He and Xiao say it assigns importance. So our question is: are these weights really the importance of each modality, or not?

**Slide 1: our models**
> We built our own models: a text model and an audio model, combined two ways, with a learned gate or a confidence weight. The best scores 51.7 weighted F1, below the MELD baseline bcLSTM's 59.25, which reads the whole conversation.

**Slide 1: intervention tests**
> We test this with intervention tests: we swap an utterance's text, or its audio, for real examples from thousands of other utterances, rerun the model, and average. Whichever swap changes the prediction more marks the modality that mattered more; we check whether the weight favoured that same one. Hessel and Lee used the same swap-and-average idea for a different question; none of the papers we read uses it to check a weight.

**Slide 2: findings**
> So what did we find? To judge a weight, we compare it with a trivial rule: always say "text". Across 2,610 test utterances, the learned gate's weight agrees with the swap test 79% of the time, but always saying "text" agrees 77%. That gain of about two points is small and could be zero. Yet the weight does track how much each modality matters: the correlation is 0.70.

**Slide 2: calibration**
> The problem is calibration. Text and audio have equal influence at a text weight of about 0.37, not 0.5. Move the cutoff there, fitted on development data and then frozen, and the gate beats always-text on the test set by about five points.

**Slide 2: what remains**
> What remains: a conflict test with deliberately mismatched text and audio, using about ten random pairings; the full report with our robustness checks; and, if time allows, a stronger audio model.

| Block | Words | Seconds at 150 wpm |
|---|---|---|
| Slide 1: title | 18 | 7 |
| Slide 1: premise and question | 95 | 38 |
| Slide 1: our models | 40 | 16 |
| Slide 1: intervention tests | 72 | 29 |
| Slide 2: findings | 68 | 27 |
| Slide 2: calibration | 43 | 17 |
| Slide 2: what remains | 31 | 12 |
| **Total** | **367** | **147** |

Slide 2 basis: agreement 0.792 vs always-text 0.774, gain +0.019 [-0.003, +0.041], Spearman 0.705 (`results/resampling_full_faithfulness.csv`); equal-influence weight about 0.37 (`results/gate_calibration_primary.csv`, dev cutoffs 0.36 to 0.38 in `results/dev_calibration_thresholds.csv`); dev-fitted cutoff gain +0.049 [+0.035, +0.062] (`results/dev_calibration_test.csv`); remaining work from `results/conflict_h2_diff.csv` and `docs/plans/step-03-strong-audio-head.md`.

---

## FULL SCRIPT v2, slides 1 and 2 (2026-10-20): 346 words, about 138 seconds at 150 wpm
Changes from v1: slide 2 findings and calibration rewritten around the three parts (the weight, our reading of it with a cutoff, the swap test as the answer key), and "correlation" dropped. Slide 1 unchanged (locked). Slide 2 is still a draft pending the presenter's approval. The rubric suggests about 260 words.

**Slide 1: title**
> Good afternoon Professor, TAs and classmates. Our project is Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals.

**Slide 1: premise and question**
> Many multimodal systems fuse two modalities; we look at text and audio. The fusion often uses one weight: a number for one modality, and one minus that number for the other. Some papers learn that weight with a small neural network; others compute it from how confident each model is. And papers often read that weight as how much each modality mattered: Liu and colleagues call it the contribution of each modality, and He and Xiao say it assigns importance. So our question is: are these weights really the importance of each modality, or not?

**Slide 1: our models**
> We built our own models: a text model and an audio model, combined two ways, with a learned gate or a confidence weight. The best scores 51.7 weighted F1, below the MELD baseline bcLSTM's 59.25, which reads the whole conversation.

**Slide 1: intervention tests**
> We test this with intervention tests: we swap an utterance's text, or its audio, for real examples from thousands of other utterances, rerun the model, and average. Whichever swap changes the prediction more marks the modality that mattered more; we check whether the weight favoured that same one. Hessel and Lee used the same swap-and-average idea for a different question; none of the papers we read uses it to check a weight.

**Slide 2: findings**
> Here is what we found. The model gives each utterance a weight for text and audio. We read it simply: above 0.5, text mattered more; below, audio mattered more. Then we compare that reading with the swap test, which tells us what actually mattered. They match 79% of the time, but always guessing "text" matches 77%.

**Slide 2: calibration**
> The weight is useful, just miscalibrated. The real tipping point is 0.37, not 0.5. Fit that cutoff on development data, freeze it, and on the test set we beat always-text by about five points.

**Slide 2: what remains**
> What remains: a conflict test with deliberately mismatched text and audio, using about ten random pairings; the full report with our robustness checks; and, if time allows, a stronger audio model.

| Block | Words | Seconds at 150 wpm |
|---|---|---|
| Slide 1: title | 18 | 7 |
| Slide 1: premise and question | 95 | 38 |
| Slide 1: our models | 40 | 16 |
| Slide 1: intervention tests | 72 | 29 |
| Slide 2: findings | 56 | 22 |
| Slide 2: calibration | 34 | 14 |
| Slide 2: what remains | 31 | 12 |
| **Total** | **346** | **138** |

---

## Slide 1 script v4 (2026-10-20): three edits to paragraphs 3 and 4, requested by the presenter
1. Removed the dangling "this": "We test this with intervention tests" came after the models paragraph, so "this" no longer pointed at the question. Paragraph 3 now opens "To study it" (right after the question) and paragraph 4 opens "On these models, we run intervention tests".
2. Named the "different question": Hessel and Lee used the swap-and-average idea "to ask whether a model really needs both modalities together" (Hessel & Lee 2020, EMAP).
3. Named the weight: "the modality weights they report" (the text-audio weights the papers read as importance).

Replacement for paragraph 3 (43 words):
> To study it, we built our own models: a text model and an audio model, combined two ways, with a learned gate or a confidence weight. The best scores 51.7 weighted F1, below the MELD baseline bcLSTM's 59.25, which reads the whole conversation.

Replacement for paragraph 4 (82 words):
> On these models, we run intervention tests: we swap an utterance's text, or its audio, for real examples from thousands of other utterances, rerun the model, and average. Whichever swap changes the prediction more marks the modality that mattered more; we check whether the weight favoured that same one. Hessel and Lee used the same swap-and-average idea to ask whether a model really needs both modalities together; none of the papers we read uses it to check the modality weights they report.

Paragraphs 1 and 2 are unchanged. Net change: +13 words (slide 1 now 238 words).

---

## Presenter's copy with the v4 fixes applied (candidate v5, 2026-10-20): 346 words, about 138 seconds
Source: the presenter's pasted two-slide copy. Differences from the pasted copy: the acronym expansion restored in the opening; "To study it" and "On these models, we run" replace the dangling "this"; the "different question" and "weight" are named; "always guessing 'text mattered more'" in slide 2. Not locked: presenter to approve.

**Slide 1**

Our project is Faithfulness MATTERS.

Many multimodal systems fuse two modalities; we look at text and audio. The fusion often uses one weight: a number for one modality, and one minus that number for the other.

Some papers learn that weight with a small neural network; others compute it from how confident each model is. And papers often read that weight as how much each modality mattered: Liu and colleagues call it the contribution of each modality, and He and Xiao say it assigns importance.

So our question is: Are these weights really the importance of each modality, or not?

To study it, we built our own models: a text model and an audio model, combined two ways, with a learned gate or a confidence weight. The best scores 51.7 weighted F1, below the MELD baseline bcLSTM's 59.25, which reads the whole conversation.

On these models, we run intervention tests: we swap an utterance's text, or its audio, for real examples from thousands of other utterances, rerun the model, and average.

Whichever swap changes the prediction more marks the modality that mattered more; we check whether the weight favoured that same one. Hessel and Lee used the same swap-and-average idea to ask whether a model really needs both modalities together; none of the papers we read uses it to check the modality weights they report.

**Slide 2**

Here is what we found.

The model gives each utterance a weight for text and audio. We read it simply: above 0.5, text mattered more; below, audio mattered more. Then we compare that reading with the swap test, which tells us what actually mattered.

They match 79% of the time, but always guessing "text mattered more" matches 77%.

The weight is useful, just miscalibrated. On development data, the best cutoff is 0.37, not 0.5. We freeze it, and on the test set we beat always-text by about five points.

What remains: A conflict test with deliberately mismatched text and audio, using about ten random pairings. The full report with our robustness checks. If time allows, a stronger audio model.

Update 2026-10-20: the acronym is NOT spoken; the title on the slide must show "Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals" (saves 8 spoken words). Candidate v5 is now 346 words, about 138 seconds.

Update 2026-10-20: slide 2 wording changed from "the real tipping point is 0.37" to "on development data, the best cutoff is 0.37" (0.37 is a fitted decision cutoff, not a property of the weight). Liu et al. is Interspeech 2020; the script says "Liu and colleagues" with no year, so no spoken change.

---

## Script v6 (candidate, 2026-10-20), cut for the 2-minute hard limit: 236 words, about 94 s at 150 wpm, 109 s at 130 wpm
Reason: the professor cut teams off at 2 minutes (see `lecture-observations-2026-10-20.md`). Changes from v5: opens with a concrete example ("I'm fine" in an angry voice), the premise and "built our own models" paragraphs are shortened, remaining work reduced to two items (the optional audio model dropped). NOT locked; presenter to approve.

**Slide 1**

Our project is Faithfulness MATTERS.

Imagine someone says "I'm fine" in an angry voice. A model fuses the text and the audio with one weight, say 0.6 for text and 0.4 for audio, and papers often read that weight as how much each modality mattered: Liu and colleagues call it a contribution, He and Xiao say it assigns importance. Is that reading true?

We built text and audio models, combined by a learned gate or a confidence weight. They score 51.7 weighted F1, below the 59.25 bcLSTM baseline, which reads whole conversations.

We test the weight with intervention tests: swap an utterance's text, or its audio, for real examples from other utterances, rerun, and average. The swap that changes the prediction more marks the modality that mattered more, and we check whether the weight agrees. Hessel and Lee used this idea for a different question: whether a model needs both modalities together. None of the papers we read uses it to check a weight.

**Slide 2**

Here is what we found.

We read the weight simply: above 0.5, text mattered more. That matches the swap test 79% of the time, but always guessing "text mattered more" matches 77%.

So the weight is informative but miscalibrated. On development data the best cutoff is 0.37; frozen, it beats always-text on the test set by about five points.

Next: a conflict test with about ten random pairings, and the full report.

---

## Script v7 (candidate, 2026-10-20): the quality-ladder result replaces the calibrated-cutoff takeaway: 252 words, about 101 s at 150 wpm, 116 s at 130 wpm
Reason: the team decided to use the stronger-audio result. Changes from v6: slide 1 mentions 54.8 with a stronger audio model; slide 2 is rebuilt around the ladder (`figures/fig5_quality_ladder.png`). The dev-fitted cutoff result moves to Q&A (it holds for the headline model: cutoff 0.37, gain +0.049; for the stronger models the best cutoff is 0.52 to 0.63). NOT locked; the hard cut-off is 2 minutes. Slide 1 174 words, slide 2 78 words.

**Slide 1**

Our project is Faithfulness MATTERS.

Imagine someone says "I'm fine" in an angry voice. A model fuses the text and the audio with one weight, say 0.6 for text and 0.4 for audio, and papers often read that weight as how much each modality mattered: Liu and colleagues call it a contribution, He and Xiao say it assigns importance. Is that reading true?

To study it, we built text and audio models, combined by a learned gate or a confidence weight. They score 51.7 weighted F1, or 54.8 with a stronger audio model, below the 59.25 bcLSTM baseline, which reads whole conversations.

We test the weight with intervention tests: swap an utterance's text, or its audio, for real examples from other utterances, rerun, and average. The swap that changes the prediction more marks the modality that mattered more, and we check whether the weight agrees. Hessel and Lee used this idea for a different question: whether a model needs both modalities together. None of the papers we read uses it to check a weight.

**Slide 2**

Here is what we found.

We read the weight simply: above 0.5, text mattered more. It matches the swap test 79% of the time. With our first model, always guessing "text mattered more" matches 77%. But with a stronger audio model, the weight still matches about 79%, while the best trivial guess falls to 54%. So how informative a weight looks depends on the models.

Next: a conflict test with about ten random pairings, and the full report.

---

## Script v8 (candidate, 2026-10-20 night): rebuilt around the model-quality findings: 270 words, about 108 s at 150 wpm, 125 s at 130 wpm
Changes from v7: slide 1's model line gives the first model (51.7) and the strongest variant (63.6) against bcLSTM's 59.25, with the comparison flagged as uncontrolled; slide 2 is rebuilt around the new figure `figures/fig6_agreement_by_balance.png` (the weight beats the best trivial guess only when text and audio are comparably strong; the corrected signal matches 94 to 99% in every model). Numbers: `results/corrected_signal.csv` (first model, stronger audio, stronger text + audio, stronger text), `results/f1_push_v2.csv`, `results/harness_models.csv`. NOT locked; the hard cut-off is 2 minutes. Slide 1 173 words, slide 2 97 words.

**Slide 1**

Our project is Faithfulness MATTERS.

Imagine someone says "I'm fine" in an angry voice. A model fuses the text and the audio with one weight, say 0.6 for text and 0.4 for audio, and papers often read that weight as how much each modality mattered: Liu and colleagues call it a contribution, He and Xiao say it assigns importance. Is that reading true?

To study it, we built text and audio models, combined by a learned gate or a confidence weight. Our first scores 51.7 weighted F1; our strongest, 63.6, above the 59.25 bcLSTM baseline, though that comparison is not controlled.

We test the weight with intervention tests: swap an utterance's text, or its audio, for real examples from other utterances, rerun, and average. The swap that changes the prediction more marks the modality that mattered more, and we check whether the weight agrees. Hessel and Lee used this idea for a different question: whether a model needs both modalities together. None of the papers we read uses it to check a weight.

**Slide 2**

Here is what we found.

We read the weight simply: above 0.5, text mattered more. With our first model it matches the swap test 79% of the time, but always guessing the usual winner matches 77%. With a stronger audio model, where text and audio each decide about half the time, the weight matches 75% against 54%. When one modality dominates, the weight adds nothing.

The weight reflects trust, not influence. A corrected signal, weight times how distinctive each model's vote is, matches 94 to 99% in every model.

Next: the conflict test, and the full report.


## Script v9 (candidate, 2026-10-20 night): story order problem, solution, intervention tests, results, next steps: 181 words, about 72 s at 150 wpm
Comparison table (docs/slides/comparison-table.md) sits on slide 1 under 'our solution'.

> Our project is Faithfulness MATTERS. Many models fuse text and audio with a learned weight, and papers read that weight as how much each modality mattered. Is that reading true?
> To test it, we first built a text and audio emotion model on MELD, at least on par with the published baseline. Our first scores 51.7 weighted F1; our strongest, 63.6, passes the 59.25 bcLSTM baseline, though newer encoders help us. Recent systems reach 67 to 70.
> Then we ran intervention tests: swap one modality for another utterance's, rerun the model, and see how much the prediction changes.
> Here is what we found. We read the weight simply: above 0.5, text mattered more. It matches the swap 79% of the time, but always guessing the usual winner already matches 77%. Where text and audio each decide about half the time, the weight matches 75% against 54%. When one modality dominates, it adds nothing.
> The weight reflects trust, not influence. Weight times how distinctive each model's vote is matches the swap 94 to 99%.
> Next: the conflict test, and the full report.
## Script v10 (candidate, 2026-10-20 night): every result comes from the 63.6 model (results/swap_with_context.csv): 230 words, about 92 s at 150 wpm, 106 s at 130 wpm
Slide 1 carries the comparison table (docs/slides/comparison-table.md). Replaces v9's results, which came from other models.

**Slide 1**

> Our project is Faithfulness MATTERS. Many models fuse text and audio with a learned weight, and papers read that weight as how much each modality mattered. Is that reading true?
> To test it, we first built a text and audio emotion model on MELD, at least on par with the published baseline. It gives a text prediction and an audio prediction for each utterance, and a learned gate or a confidence weight mixes them into one of seven emotions. Judging each utterance on its own it scores 51.7 weighted F1. Adding the neighbouring utterances and stronger encoders, 63.6, above the 59.25 bcLSTM baseline. Recent systems reach 67 to 70.
> Then, intervention tests: swap one modality for another utterance's, rerun the model, and see how much the prediction changes.
**Slide 2**

> Everything here is on our 63.6 model. Text decided the prediction 95% of the time, audio 5%. We read the weight simply: above 0.5, text mattered more. That matches the swap test 94% of the time, but always guessing text matches 95%, so the weight adds nothing. It caught only half the cases where audio won, and was right on 38% of its audio calls.
> The weight shows trust, not influence. Weight times how distinctive each model's vote is ranks the swap effects far better: correlation 0.96, against 0.43.
> Next: the conflict test, where text and audio disagree, and the full report.
## Script v11 (candidate, 2026-10-21): two-model story, balanced model (58.8) vs text-heavy model (63.6): 246 words, about 98 s at 150 wpm, 113 s at 130 wpm
Sources: results/harness_models.csv (M1 balanced, row 'M1 stronger audio'), results/swap_with_context.csv (63.6 model), results/f1_push_v2.csv (63.6), docs/slides/comparison-table.md.

**Slide 1**

> Our project is Faithfulness MATTERS. Many models fuse text and audio with a learned weight, and papers read that weight as how much each modality mattered. Is that reading true?
> To test it, we first built a text and audio emotion model on MELD, at least on par with the published baseline. It gives a text prediction and an audio prediction for each utterance, and a learned gate or a confidence weight mixes them into one of seven emotions. With text and audio balanced, it scores 58.8 weighted F1, within half a point of the 59.25 bcLSTM baseline. A stronger, text-heavy version reaches 63.6. Recent systems reach 67 to 70.
> Then, intervention tests: swap one modality for another utterance's, rerun the model, and see how much the prediction changes.
**Slide 2**

> We read the weight simply: above 0.5, text mattered more. On the balanced model, text mattered more in 54% of utterances. The weight matches the swap test 75% of the time, against 54% for always guessing the usual winner. On the text-heavy model, text mattered in 95%, and the weight, 94%, is no better than always guessing text, 95%. So the weight is informative only while text and audio are comparably strong.
> It shows trust, not influence. Weight times how distinctive each model's vote is matches the swap test 94% on the balanced model, and tracks its size far better: correlation 0.97, against 0.72.
> Next: the conflict test, where text and audio disagree, and the full report.
## FINAL script v12 (2026-10-21), matches the submitted slides

**Slide 1**

> Our project is Faithfulness MATTERS. Many models mix text and audio with a learned weight, and papers read that weight as how much each modality mattered. Is that reading true?
> To test it, we built text and audio emotion models on MELD: one utterance in, one of seven emotions out. A learned gate or a confidence weight mixes the text prediction and the audio prediction. With text and audio balanced, it scores 58.8 weighted F1, within half a point of the 59.25 bcLSTM baseline. A stronger, text-heavy version reaches 63.6. Recent systems reach 67 to 70.
> Then, the swap test: replace the text or the audio with other utterances', rerun the model, and see how much the prediction changes.
**Slide 2**

> We read the weight simply: above 0.5, text mattered more. On the balanced model, text mattered more in 54% of utterances. The weight matches the swap test 75% of the time, against 54% for always guessing the usual winner. On the text-heavy model, text mattered in 95%, and the weight, 94%, is no better than always guessing text, 95%. So in our models, the weight is informative only while text and audio are balanced.
> It shows trust, not influence. Weight times how distinctive each model's vote is tracks the swap effects far better: correlation 0.97, against 0.72.
> Next: a conflict test, where text and audio disagree, on modality-heavy and balanced models, then the final report.