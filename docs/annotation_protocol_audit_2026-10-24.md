# Audit of the frozen annotation protocol before annotation (2026-10-24)
Constraints kept: the hidden key was not opened (only its file hash is recorded in `docs/annotation_package_manifest.md`), no model was run on constructed pairs, no annotation was filled in by anyone or anything. Amendments: `docs/conflict_experiment_protocol.md`, Amendment 1, made before annotation. Checks were run on the distribution folders only.

| # | Component | Result | Evidence and action |
|---|---|---|---|
| 1a | Same underlying examples for all reviewers | PASS | the three folders hold identical `items_*.csv` and the same id sets in every sheet (checked); only the row order of each sheet differs |
| 1b | Stable ids for matching responses | PASS | 8-hex ids per text, audio and pair, unique (asserted in the builder); the validator checks alignment across reviewers |
| 1c | Condition not revealed to reviewers | **FAIL in version 1, FIXED in version 2** | version 1 listed each text's conflict pair first in `items_pairs.csv` and all conflict audios first in `items_audio.csv`, ids were hashes of the public seed and dev row index (recoverable by a determined reviewer), and the key and `package_stats.json` sat in the package folder. Version 2: master lists shuffled (adjacent rows sharing a text id 1% vs 50% in version 1), ids from a private salt, per-reviewer folders with no key, statistics or other reviewers' sheets. Version 1 was never distributed |
| 1d | Reviewers cannot see model predictions, labels, each other's work | PASS | none in the folders; the instructions forbid discussion and tools |
| 1e | Repeated texts visible | WARNING | each text appears in two pairs and each audio once; reviewers can see repeats and could anchor on earlier answers. Disclosed; they are told to judge each row independently |
| 2a | Text emotion vs vocal tone distinguished | PASS (after A3) | the instructions define both, state that tone is not to be inferred from word meaning, and that Task 3 compares emotions, not sentence coherence |
| 2b | Audio contains words | WARNING (inherent) | reviewers may not be able to separate tone from words; `cannot_tell` is allowed and confidence is recorded; the limitation is stated in the protocol (section 6) and the pairs necessarily differ in wording as well as emotion |
| 2c | Ambiguous option | PASS | `cannot_tell` (tasks 1, 2) and `unclear` (task 3), plus confidence 1 to 5 for all three tasks (task 3 confidence added in version 2) |
| 2d | Different MELD labels are not assumed to imply conflict | PASS | construction only proposes pairs; verification needs opposite polarity by majority of three independent reviewers on text, tone and pair relation |
| 2e | Conflict criterion operational and reproducible | PASS | fixed polarity map and decision rule in instructions section "Adjudication" and in A2/A3 |
| 3a | Agreement statistic appropriate | WARNING, amended | the committed Cohen's kappa uses reviewers 1 and 2 only, although all three rate every item. Amendment A2 keeps it (threshold 0.40, unchanged) and adds Fleiss' kappa over three with the same threshold (stricter); AC1 and percent agreement reported as context |
| 3b | Treatment of ambiguous, missing, disagreeing | PASS (after A2) | `cannot_tell`/`unclear` are categories; missing excluded per item; sheets with more than 5% blank are returned; majority rule, all-different = unresolved; discussion only after independent kappa |
| 3c | Kappa threshold 0.40 | PASS as a gate, WARNING as evidence | threshold kept; the 95% interval at n = 100 is about plus or minus 0.18, so passing is weak evidence of reliability |
| 4a | Power statement | **FAIL as committed, corrected (A4)** | the committed statement (SE 4 to 6 points, 10 to 15 points detectable) was optimistic for M1; corrected table below |
| 4b | Independent unit | PASS after correction | 100 pairs are 50 texts; paired comparison and cluster bootstrap by dialogue; at most 2 texts per dialogue in Stage B (A5b) |
| 4c | Sample size when verification is low | FAIL as committed ("150 and 150 candidates"), amended (A5a) | with precisions 0.5 and 0.8, 150 texts give about 60 complete pairs; the rule N_B = ceil(100 / (precision_conflict x precision_control)), cap 400, is now fixed in advance |
| 5a | Stage A uses dev only | PASS | `scripts/46_build_annotation_package.py` reads only the dev csv and dev wavs; no model |
| 5b | Decisions before Stage B specified | PASS after amendment | sample-size rule, caps, exclusion list freezing, key handling, M1/M3 identity (A5) |
| 5c | Test-stage outcomes cannot influence selection | PASS after A5c | no model quantity on Stage B pairs until verification is complete and the exclusion list is hashed; only an embedding-similarity near-duplicate check against the training pool is allowed earlier |
| 5d | M1 and M3 procedures identical | PASS after A5e | same pairs, gates (10 each), rule, alpha 0.25 for both, donor pool and code path |
| 5e | Holm and endpoints | PASS after A6 | family of two, 0.025 then 0.05, p from the cluster bootstrap, minimum 50 complete pairs per test |
| 6 | Package ready for three independent reviewers | **PASS (WARNINGs 1e, 2b, 3c noted)** | see file list below |

## Detectable differences (approximate; Holm-adjusted two-sided 0.025, 80% power; weight agreement, in points)
| Verified items per condition | M1 unpaired (p near 0.75) | M1 paired, rho 0.3 | M3 unpaired (p near 0.93) | M3 paired, rho 0.3 |
|---|---|---|---|---|
| 25 | 38 (41 with design effect 1.2) | 32 | 22 | 19 |
| 50 | 27 (29) | 22 | 16 | 13 |
| 100 | 19 (21) | 16 | 11 | 9 |
| 150 | 15 (17) | 13 | 9 | 8 |
| 200 | 13 (15) | n/a | 8 | n/a |
Expected complete pairs from N candidate texts = N x precision_conflict x precision_control (control precision assumed 0.8): precision 0.33 / 0.5 / 0.7 gives 40 / 60 / 84 pairs from N = 150 and 100 pairs only from N = 379 / 250 / 179. Stage A (50 conflict candidates) estimates the precision with a 95% Wilson interval of 0.22 to 0.48 for an observed 17 of 50, 0.37 to 0.63 for 25 of 50, and 0.46 to 0.72 for 30 of 50.
**Consequence.** If only 25 to 50 conflict candidates survive verification, the confirmatory test can detect only very large effects (25 to 40 points for M1; 13 to 22 points for M3). A non-significant result is then inconclusive. The experiment can show that the hypothesis is wrong only for effects of that size or larger, and it can show an effect only if large. The earlier natural-pair comparisons moved by a few points, which this design cannot resolve.

## Files to distribute (and nothing else)
Three zip files from `data/annotation_package_stageA/distribution/`: `annotator_1.zip`, `annotator_2.zip`, `annotator_3.zip`, one to each reviewer (each contains its own sheets, the shared item files, the audio, `annotation_instructions.md`, `READ_ME_FIRST_quickstart.md`, `ALLOWED_VALUES.txt`). Do NOT send `shared/` or `KEY_private/`, the manifest, or the protocol documents. Reviewers return only their three completed csv sheets. Keep the key unopened. Validate returned sheets with `python scripts/47_validate_annotation_sheets.py --folder <returned folder> --annotator N --compare <other returned folders>`.
Note: a rebuild draws a new private salt and changes every id and hash; do not rebuild after the zips are sent.
