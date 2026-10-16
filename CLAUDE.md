# Faithfulness MATTERS (CSCI 544, Fall 2026, Team 47)

Research project: do a text+audio emotion model's modality weights (learned gate, confidence rule) match how its prediction actually reacts when we remove the text or the audio? Dataset: MELD. This is an evaluation-of-explanations project, not an accuracy project.

Read before any planning: `docs/PROJECT_REQUIREMENTS.md` (the ask, constraints, definition of done) and `docs/RUBRIC.md` (how we are graded). The proposal is `docs/proposal.pdf`.

## Commands

All scripts run from the repo root with the venv active (`source .venv/bin/activate`):

```bash
python scripts/01_build_dataset.py                          # ~1 min, needs data/legacy + data/meld_csv
python scripts/02_train.py                                  # ~1 min, 3 seeds -> checkpoints/
python scripts/03_silence_embeddings.py --wav-dir data/wavfiles_test   # ~5 min on MPS, WavLM download first time
python scripts/04_interventions.py                          # seconds, must rerun after 02 or 03
python scripts/05_faithfulness.py                           # seconds
python scripts/06_analysis.py                               # ~30 s, bootstrap + figures/
```

There is no test suite. A step "passes" when its script runs clean and its printed numbers match the files in `results/`. Reference numbers to sanity-check against: test weighted-F1 text 0.491, audio 0.368, learned gate 0.517; gate gain over always-text (mean replacement) +0.014 [-0.007, 0.036].

## Rules

- The intervention protocol in the header of `scripts/04_interventions.py` was fixed before results were seen. Do not change a definition, threshold (near-zero = 0.02) or metric to get a nicer number. Proposing a change is fine; it must be written down before rerunning.
- Never tune anything on the test set. Thresholds, recalibration, or selection rules are fit on dev, frozen, then applied to test.
- Every number written into a doc, slide or report must come from a file in `results/` or a script's printed output. Name the file.
- `data/` and `checkpoints/` are local only (gitignored, large). Never commit them. `results/interventions_log.csv` is committed on purpose (teammates need it).
- `MIDTERM_STATUS.md` and `READING_GUIDE.md` are local-only, gitignored on purpose. Read them; do not add them to git.
- Scripts import `scripts/common.py`; keep shared models, fusion rules and metrics there rather than duplicating them.
- Writing for Aakash: no em-dashes; use commas or periods. Writing for teammates: plain language, no jargon without a one-line explanation, never comment on the reader's background or ability.

## Gotchas

- Legacy text embeddings were computed on the raw 2018 MELD CSV text (apostrophes stored as `\x92`). This is intentional for reproducibility; `01_build_dataset.py` aligns with a `raw_meld_c1` decoding.
- 42 test rows are flagged `audio_shared`: 39 share byte-identical audio with another test row (17 groups, MELD's own files) and 3 share it only with the classifier train split. Breakdown in `results/audio_duplicates.json`. Report metrics with and without them.
- Training runs on CPU on purpose (determinism). Only step 3 uses MPS.
- The confidence rule's high agreement is partly structural (its weight is built from the same probabilities it mixes). Never present it as evidence of a good explanation without that caveat.

## How we work in this repo

Work happens in three phases. Do not skip ahead.

1. **Understand**: map the codebase and requirements, then give a step-by-step walkthrough. Output: `docs/STATE.md`.
2. **Review each step** with `/step-review <n>`: discuss, then save the agreed plan to `docs/plans/`. One step per session; `/clear` between steps.
3. **Build** with `/next-phase`: implement the first unchecked phase of `PLAN.md`, run its acceptance check, get it reviewed, tick it.

Subagents: `codebase-auditor` (read-only mapping), `pipeline-runner` (runs scripts, returns only key numbers and errors), `rubric-grader` (fresh-eyes grading and mock Q&A against `docs/RUBRIC.md`), `reviewer` (checks a diff against the current `PLAN.md` phase).
