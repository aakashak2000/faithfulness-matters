---
name: step-review
description: Structured review of one project step (what is done, how good, alternatives, options, improvements, plan), discussed with the user and saved to docs/plans/. Run as /step-review <step number>.
argument-hint: <step number, e.g. 3>
disable-model-invocation: true
---
Review step $ARGUMENTS of this project. This is a discussion, not a build: do not edit any code, script or result during this skill. The only files you may write are the step plan and docs/plans/INDEX.md, and only after the user approves.

## Steps of the project

Use the numbering in docs/STATE.md if it exists. Default numbering:

1. Problem framing: proposal, research question, fit with docs/RUBRIC.md and docs/PROJECT_REQUIREMENTS.md
2. Data and audit: scripts/01_build_dataset.py, scripts/diag_*.py, results/audit_step1.json
3. Models and baselines: scripts/02_train.py, scripts/common.py, results/baselines_*.csv, per_class_f1.csv, legacy_reproduction.csv
4. Intervention design: scripts/03_silence_embeddings.py, the protocol header of scripts/04_interventions.py
5. Faithfulness metrics: scripts/05_faithfulness.py, results/faithfulness_*.csv, method_robustness.csv
6. Uncertainty, calibration and figures: scripts/06_analysis.py, results/bootstrap_ci.csv, gate_calibration.csv, agreement_counts.csv, figures/
7. Midterm presentation: 2 slides, ~2 min script, Q&A answers
8. Plan to the final: H2 conflict set, gate diagnosis, robustness checks, report

## Procedure

1. **Load context.** Read docs/plans/INDEX.md and any earlier step plans (decisions there are binding unless the user reopens them). Read the step's section of docs/STATE.md, its code and its results. Use the codebase-auditor subagent if the step spans several files.

2. **Present, in this order, each section short and concrete:**
   1. **What is being done.** Purpose of the step in the research story, files, outputs, key numbers (name the results file for every number).
   2. **How good is it.** Grade it against docs/PROJECT_REQUIREMENTS.md and the relevant rubric lines in docs/RUBRIC.md. Strengths, then weaknesses, each with evidence. For deliverables (step 7, report sections), get an independent grade from the rubric-grader subagent and include it.
   3. **What else could have been done.** Alternatives a TA or reviewer might expect, and why they were or weren't used.
   4. **Options.** The real decision points for this step, stated as choices.
   5. **Ways to make it better.** A table:

      | # | Improvement | What it fixes | Serves (rubric line or requirement) | Effort (h) | Risk | Bucket |

      Buckets are defined in docs/PROJECT_REQUIREMENTS.md section 6 (before midterm slides / before midterm report / before final / nice to have). Be honest about effort; include "do nothing" when it is a reasonable choice.
   6. **Recommended plan.** The subset you'd actually do, in order, and why. It must be realistic for one implementer, fit its deadline bucket, and serve a rubric line or requirement.

3. **Stop and discuss.** Ask the user only the 1 to 3 decisions that change the plan. Answer their questions, revise, and repeat until they say the plan is agreed. Do not write anything before that.

4. **Save.** Write docs/plans/step-NN-<short-slug>.md using this template, then add or update the step's row in docs/plans/INDEX.md:

   ```markdown
   # Step NN: <title>
   Agreed: <date>

   ## What exists
   ## Assessment
   ## Decisions (with alternatives rejected and why)
   ## Actions
   | # | Action | Serves | Bucket | Effort (h) | Acceptance check |
   ## Open questions
   ## Effects on other steps
   ```

   Every action needs an acceptance check that can be run or verified (a command, a number to match, a file that must exist, a rubric line a draft must meet).

5. **Hand off.** Summarise the agreed plan in 3 lines and tell the user to run /clear, then /step-review <next step>.

## Rules

- Never tune on the test set; any threshold or selection rule is fit on dev first.
- Never change the fixed protocol to get a better number.
- Prefer communicating existing results over new experiments when the deadline is the midterm.
- Plain language when anything is meant for teammates. No em-dashes in anything written for the user.
