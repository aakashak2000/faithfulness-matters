# Annotation instructions, Stage A (emotion-conflict experiment)
Version 2, 2026-10-24 (version 1 of 2026-10-23 was never distributed; changes are listed in Amendment 1 of `docs/conflict_experiment_protocol.md`). For three human reviewers. Read this file and `READ_ME_FIRST_quickstart.md` fully before opening any sheet.

## What this is for
We build pairs of "the written text of one utterance" and "the audio recording of another utterance" and want to know whether a model's text/audio weighting stays trustworthy when the two disagree in emotion. Our data only gives one emotion label per original utterance, so whether a given pair really reads and sounds conflicting must be decided by people. **You decide, using only what you read and hear.** You are not told any label, how the pairs were chosen, or how many of them are conflicts. Do not assume any particular share. Judge every row on its own; the same text or the same audio clip can appear in more than one row, and you must not try to remember or reuse earlier answers.

## Two different things you will rate
- **Text emotion:** the emotion conveyed by the **written words** only, as if you read them in a script. Do not imagine how it was spoken.
- **Tone emotion (voice):** the emotion conveyed by **how the voice sounds**: pitch, loudness, speed, voice quality, laughter, crying, sighing, shouting. The recording also contains spoken words. **Do not use the meaning of the words to decide the tone**; use them only if the voice is impossible to read otherwise, and then lower your confidence. If you cannot separate tone from words at all, choose `cannot_tell`.

## Files you open (all inside your own folder)
`annotator_N_task1_text.csv`, `annotator_N_task2_audio.csv`, `annotator_N_task3_pairs.csv` (your sheets, N is your number); `items_text.csv` (text for each text id), `items_audio.csv` plus the folder `audio/` (recordings), `items_pairs.csv` (which text goes with which recording); `ALLOWED_VALUES.txt`. Nothing else.

## Task order (finish each task before starting the next; do not go back and change earlier tasks)
1. **Task 1, text only.** For each `text_item_id`, read the text in `items_text.csv`. Fill `text_emotion` (anger, disgust, fear, joy, neutral, sadness, surprise, or `cannot_tell`) and `text_confidence_1to5` (1 guess, 5 certain). Use `cannot_tell` whenever the words do not clearly carry an emotion or carry a mixed one.
2. **Task 2, tone only.** For each `audio_item_id`, play `audio/<id>.wav` (as often as you like) and fill `tone_emotion` and `tone_confidence_1to5` under the rules above. Use `cannot_tell` for flat, mixed, unclear or unusable recordings.
3. **Task 3, pairs.** For each `pair_id`, look up its text id and audio id in `items_pairs.csv`, read that text and play that recording together, and fill `relation` and `confidence_1to5`:
   - **conflict:** the emotion of the written text and the emotion of the voice are clearly of opposite polarity (one positive, one negative; joy is the positive emotion, anger, disgust, fear and sadness are negative). Example (made up): a text that says "I'm so thrilled!" with a voice that sounds angry.
   - **compatible:** the text and the voice express the same emotion or emotions of the same polarity.
   - **unclear:** neutral against an emotion, mixed or ambiguous emotion, surprise, unusable audio, or you cannot decide.
   Judge the text's emotion against the voice's tone. **Do not judge whether the two sentences make sense together or whether the story fits**; the recording will usually say something different from the text, that is expected.
Use `notes` freely (for example `unplayable`, `noisy`, `music`, `overlapping speakers`). Never change ids or column names.

## Rules
- Work alone. Do not discuss any item, your labels or your impressions with anyone, including the other reviewers, until the study lead announces adjudication.
- Do not look at, ask for, or try to infer anything about how the pairs were built, the show's script, the speakers' identities or any label from the original dataset. Do not use AI tools or any automated emotion tool.
- Do not skip rows. Missing answers are treated as missing data and you will be asked to finish them.
- Time: about 2 hours per reviewer (50 texts, 100 recordings, 100 pairs).

## Adjudication (done later by the study lead, not by you)
1. After all sheets are returned and validated, agreement statistics are computed on the independent labels: Cohen's kappa between reviewers 1 and 2, Fleiss' kappa over all three, for task 3 (conflict / compatible / unclear), for text polarity and for tone polarity (positive = joy; negative = anger, disgust, fear, sadness; other = neutral, surprise, cannot_tell), plus percent agreement. `cannot_tell` and `unclear` count as their own category; missing answers are excluded for that item only.
2. The verified label of an item is the majority of the three independent labels. If all three differ, the item is `unresolved` and excluded.
3. Where any two reviewers differ on a pair relation, the three reviewers may discuss only those pairs; the independent labels stay in the record and the post-discussion label is stored in a separate column; reviewer 3 breaks ties.
4. A pair is a **verified conflict** if its verified relation is conflict AND the verified text polarity and verified tone polarity are positive and negative (in either order). A pair is a **verified control** if its relation is compatible AND the two polarities are equal and not "other". Everything else is excluded and counted.
5. Only then is the key opened and each pair compared with the condition it was built for.
