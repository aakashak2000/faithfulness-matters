# Working on this repo with Claude Code

How the Claude Code setup in this repo works, and the prompts to drive it.

## What's in the harness

| File | What it does |
| --- | --- |
| `CLAUDE.md` | Loaded into every session automatically: commands, rules, gotchas, the 3-phase workflow. Kept short on purpose |
| `docs/PROJECT_REQUIREMENTS.md` | The ask: deliverables, research question, constraints, decision rule for new work, definition of done |
| `docs/RUBRIC.md` | How each deliverable is graded, and what full marks means for us |
| `docs/proposal.pdf` | Our proposal (add it yourself) |
| `docs/STATE.md` | Created in phase 1: the map of what exists |
| `docs/plans/` | One agreed plan per reviewed step, plus `INDEX.md` |
| `PLAN.md` | Created in phase 3: the build plan with checkboxes |
| `.claude/agents/` | Subagents: `codebase-auditor`, `pipeline-runner`, `rubric-grader`, `reviewer` |
| `.claude/skills/step-review/` | `/step-review <n>`: the 7-part review and discussion for one step |
| `.claude/skills/next-phase/` | `/next-phase`: build, check, review and tick one phase |

Why subagents only for these jobs: each one starts with a clean context, which is good for mapping code, absorbing long script output, and grading with fresh eyes. Planning, discussing and building all need the shared conversation, so they stay in the main session.

## One-time setup

```bash
cd ~/Downloads/faithfulness-matters
# unzip the harness here, so CLAUDE.md sits next to README.md
cp ~/Downloads/CS544_Project_Proposal.pdf docs/proposal.pdf
```

1. Fill in the blanks in `docs/PROJECT_REQUIREMENTS.md` section 1 (presentation slot, report deadlines). The deadline buckets depend on them.
2. Commit the harness: `git add CLAUDE.md .claude docs && git commit -m "Add Claude Code harness, rubric and requirements"`.
3. Skip `/init`: `CLAUDE.md` already exists and is tuned.

## Phase 1: understand (one session)

Start Claude Code in the repo (`claude`), then paste:

```text
We're picking up an existing research project. Before changing anything, I want you to understand it fully.

1. Read CLAUDE.md, docs/PROJECT_REQUIREMENTS.md, docs/RUBRIC.md, docs/proposal.pdf, README.md, MIDTERM_STATUS.md and READING_GUIDE.md.
2. Use codebase-auditor subagents in parallel to map: (a) scripts/00_*, 01_* and diag_*; (b) 02_train.py and common.py; (c) 03_* and 04_* including the protocol header; (d) 05_* and 06_*; (e) results/ and figures/, checking that every number in MIDTERM_STATUS.md matches the CSVs.
3. Write docs/STATE.md with: the research question in 3 sentences; the project as the 8 numbered steps from the step-review skill, each with purpose, files, outputs, key numbers (with the results file each comes from) and anything unfinished or inconsistent; a gap check of what the requirements and rubric ask for versus what exists; open questions for me.
4. Then walk me through the project one step at a time in chat, short and concrete, and wait for me to say "next" before moving on. No improvement ideas yet; those come in /step-review.

Do not edit any file except docs/STATE.md. If the presentation slot in docs/PROJECT_REQUIREMENTS.md is still blank, ask me for it first.
```

## Phase 2: review each step (one session per step)

For each step: `/clear`, then `/step-review <n>`. Claude presents the 7 parts (what is done, how good, what else, options, improvements with effort and deadline bucket, recommended plan), you discuss, and once you agree it saves `docs/plans/step-NN-*.md`.

**Suggested order, given the slide deadline:** 1, then 7 (lock the midterm slides early), then 2, 3, 4, 5, 6, then 8. If time is very short before the slides are due, do 1 and 7 only and come back for the rest.

Good things to push on during a review:
- "Which rubric line does this serve?" If none, it's nice-to-have.
- "What's the effort if it goes badly?"
- "Ask the rubric-grader to grade this as the TA would."

## Phase 3: one consolidated plan (one session)

```text
Read docs/plans/INDEX.md and every step plan in docs/plans/. Build PLAN.md: phases in deadline order (before midterm slides, before midterm report, before final, nice to have). Each phase must be independently shippable and have: goal, the step plan it comes from, files touched, a runnable acceptance check, effort, and a checkbox. Flag any conflict between step plans and any phase that doesn't fit its deadline. Have the rubric-grader check the plan against docs/RUBRIC.md and the deadlines, then show me the plan before writing it.
```

## Phase 4: build (one session per phase)

`/clear`, then `/next-phase`. Claude says what it will change, waits for your OK, builds, runs the acceptance check, gets the reviewer's verdict, lists any doc numbers that changed, ticks the phase and proposes a commit message.

Slides: Claude Code can't edit the shared class deck. For step 7 work, have it write slide text, the script and Q&A answers to `docs/midterm/`, then paste them into the deck yourself. Before the deadline, run the rubric-grader on the final slide text.

## Habits that keep this working

- `/clear` between steps and between phases. The plans on disk carry everything forward.
- If you've corrected Claude twice on the same thing, `/clear` and restate the request more specifically.
- Keep `CLAUDE.md` short. Add a line only when Claude makes a mistake that the line would prevent.
- If a script changes a headline number, update `MIDTERM_STATUS.md`, the project story doc and the slides in the same phase.
