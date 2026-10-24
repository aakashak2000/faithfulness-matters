# Reviewer quickstart (read this first, about 3 minutes)

**Your job.** You will label emotions for a set of short text snippets and short audio clips from a TV show, and then say whether a given text and a given recording, played together, clearly express *opposite* emotions. You work alone, on your own computer, for about 2 hours.

## 1. What you received
A folder `annotator_N/` (N is your number: 1, 2 or 3). Open **only** these files:
| File | What it is |
|---|---|
| `annotator_N_task1_text.csv` | Task 1 sheet: rate the emotion of 50 texts (you fill it in) |
| `annotator_N_task2_audio.csv` | Task 2 sheet: rate the tone of voice of 100 recordings (you fill it in) |
| `annotator_N_task3_pairs.csv` | Task 3 sheet: rate 100 text+recording pairs (you fill it in) |
| `items_text.csv` | the text for each text id (read only) |
| `audio/` and `items_audio.csv` | the recordings, named by audio id (read only) |
| `items_pairs.csv` | which text id goes with which audio id for each pair id (read only) |
| `ALLOWED_VALUES.txt` | the answers you may type |
| `annotation_instructions.md` | the full rules and definitions |
Open the CSV sheets in a spreadsheet program (Excel, Numbers, Google Sheets) or a plain text editor. Do not change ids, column names or the order of the columns.

## 2. What to do, in this order (finish each task before starting the next)
1. **Task 1 (about 15 min).** Read each text. Type one emotion for what the **written words** express: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`, or `cannot_tell`. Type a confidence from 1 (guess) to 5 (certain).
2. **Task 2 (about 50 min).** Play each recording. Type the emotion of the **voice** (how it sounds, not what is said), plus a confidence 1 to 5. Use `cannot_tell` if the tone is flat, mixed or unreadable.
3. **Task 3 (about 50 min).** For each pair id, find its text id and audio id in `items_pairs.csv`, read that text and play that recording, then type `conflict` (opposite polarity: one positive, one negative), `compatible` (same or fitting emotion), or `unclear`, plus a confidence 1 to 5. Do not judge whether the two sentences fit together; judge the emotion of the text against the emotion of the voice.

## 3. How to record uncertainty
Always answer. If you cannot tell, use `cannot_tell` (tasks 1 and 2) or `unclear` (task 3) and a low confidence. Use the `notes` column for anything odd (`unplayable`, `noisy`, `music`, `overlapping speakers`). Empty cells are not allowed.

## 4. What you must not do
- Do not open any file that is not in your folder, and do not look for or ask about any "key", label, expected answer or how the pairs were made.
- Do not talk about the items, your answers or your impressions with anyone (including the other two reviewers) until you are told adjudication has started.
- Do not use AI tools, emotion detectors or the internet to help; do not try to look up the show.
- Do not go back and change Task 1 or 2 answers after starting a later task.

## 5. Check and return your sheets
1. Save each sheet as CSV (keep the file name), encoding UTF-8.
2. Optional self-check, run by the study lead or you if you have Python: `python scripts/47_validate_annotation_sheets.py --folder <your folder> --annotator N` (it only checks completeness and allowed values; it never uses any key).
3. Send back the **three CSV sheets only** (not the audio folder), by the channel the study lead names, and tell no one else your answers.

**Questions?** Ask the study lead about *how to use the files*, not about the content of any item.
