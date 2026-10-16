---
name: next-phase
description: Build the first unchecked phase of PLAN.md, verify it with its acceptance check, get it reviewed, and tick it off. Run as /next-phase.
disable-model-invocation: true
---
1. Read PLAN.md and find the first unchecked phase. Read the step plan it links to in docs/plans/ and the files it names. If the phase is ambiguous, ask before writing code.
2. State in 3 lines what you will change and how it will be checked. Wait for the user to confirm.
3. Implement it in this session. Keep changes inside the phase's scope. Shared logic goes in scripts/common.py.
4. Run the phase's acceptance check. Use the pipeline-runner subagent for any script that prints long output. If it fails, fix and rerun; if you've failed the same way twice, stop and explain.
5. Run the reviewer subagent. Fix every Critical and Should-fix item, then rerun the acceptance check.
6. If the phase changed any number that appears in a doc (README.md, MIDTERM_STATUS.md, docs/*, slides), list each doc and the old vs new number. Do not silently leave stale numbers.
7. Tick the phase in PLAN.md with today's date and a one-line result. Propose a commit message; commit only when the user says so. Never add data/ or checkpoints/.
8. Tell the user to /clear before the next phase.
