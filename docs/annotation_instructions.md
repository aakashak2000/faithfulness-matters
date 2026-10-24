# Annotation instructions, Stage A (emotion-conflict experiment)
Version 1, 2026-10-23. For three human reviewers: annotator 1, annotator 2 and annotator 3 (adjudicator). Read this file fully before opening any sheet.

## What this is for
We build pairs of "text of one utterance" and "audio of another utterance" to test whether a model's modality weight stays trustworthy when the two inputs disagree in emotion. MELD only gives one emotion label per utterance, so a pair whose labels differ may or may not sound or read as conflicting. **You decide, using only what you read and hear.** You will not be told any MELD label, and you must not open `KEY_do_not_open_during_annotation.csv`.

## Files you receive (in `data/annotation_package_stageA/`)
`items_text.csv` (text items), `items_audio.csv` and `audio/*.wav` (audio items), `items_pairs.csv` (which text goes with which audio), your three sheets `annotator_N_task1_text.csv`, `..._task2_audio.csv`, `..._task3_pairs.csv`, and `ALLOWED_VALUES.txt`. Work alone, do not discuss items with the other reviewers until the adjudication step.

## Task order (do them in this order, finish each before the next)
1. **Task 1, text only.** For each `text_item_id`, read the text and choose the emotion it expresses: anger, disgust, fear, joy, neutral, sadness, surprise, or cannot_tell. Give a confidence from 1 (guess) to 5 (certain). Judge the words only, as if reading a script.
2. **Task 2, audio tone only.** For each `audio_item_id`, listen (as many times as you like) and choose the emotion expressed by the **voice**: pitch, loudness, speed, voice quality, laughter, sighs. **Ignore what the words say** as far as you can; if you cannot separate tone from words, still pick the best emotion and lower your confidence. Same label set and confidence scale.
3. **Task 3, pairs.** For each `pair_id`, read the text and listen to the audio together and choose a relation: **conflict** (the text and the voice clearly express emotions of opposite polarity, one positive and one negative, for example joyful words with an angry voice), **compatible** (they express the same emotion or emotions that fit each other), or **unclear** (neutral, mixed, hard to tell, or the audio is unusable). Neutral against an emotion is not a conflict. Put a note if the audio is noisy, cut off, music, overlapping speakers or you hear a different speaker than the words suggest.
Use the notes column freely; never edit ids.

## Rules
- No external help, no looking up the show, no guessing from the speaker's identity.
- Do not skip rows; use cannot_tell or unclear when unsure.
- Time: about 2 hours per person for 100 pairs (50 texts, 100 audio clips).
- If a clip is unplayable, write `unplayable` in notes and `unclear` / `cannot_tell`.

## Adjudication process (after all three sheets are returned, before anyone opens the key)
1. The study lead computes Cohen's kappa between annotators 1 and 2 for task 1, task 2 (on the 3 polarity groups positive / negative / other, and on the full 7-way scale) and task 3, and the percentage of pair relations that agree.
2. A **verified label** for an item is the majority of the three reviewers. Items where all three differ are `unresolved` and excluded.
3. Where any two of the three differ on a pair relation, the three reviewers meet (or exchange messages) and discuss only those pairs, with the annotator 3 vote as the tie-breaker; the original independent labels stay in the record, and the post-discussion label is stored in a separate column.
4. Polarity map for verified labels: joy = positive; anger, disgust, fear, sadness = negative; neutral, surprise, cannot_tell = other.
5. A pair is **verified conflict** if its verified relation is conflict AND the verified text polarity and verified tone polarity are opposite (positive vs negative). A pair is **verified control** if its relation is compatible AND the text and tone polarities are equal (both positive or both negative). Everything else is excluded and counted.
6. Only then is the key opened, and the condition each pair was built for (conflict / control) is compared with its verified status. The precision of "labels differ" as a predictor of a verified conflict is reported.

## What happens next
The decision rules for Stage A (what precision or agreement is needed to go on) are in `docs/conflict_experiment_protocol.md`, section 10.
