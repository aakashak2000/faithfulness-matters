# Grading rubrics

What we are graded on, and what full marks requires from *our* project. The rubric text is quoted from the course staff; the "for us" columns are our interpretation.

## Midterm presentation (5 points)

Source: Piazza @211, Sayan Ghosh, Fall 2026 (corrected version: Progress = 2 points, Q&A = 1 point).

**Format**
- 2 slides, added to the shared class deck (link posted on Piazza and Brightspace once slots are finalized).
- **Slides due 24 hours before our scheduled slot.**
- **3 minutes total, including Q&A.** One team member may present for the whole team.
- Slide 1: an overview of the problem and our method, and how it differs from prior work.
- Slide 2: the main takeaway we have reached so far, and an overview of what remains before the final report.
- Graded by the instructors **and by peers** (Google Form on the day). Peers are classmates, so the slides must make sense to someone who has never heard of our project.

| Criterion | Points | Full marks (verbatim) | Half marks | Zero | What full marks means for us |
| --- | --- | --- | --- | --- | --- |
| Clarity | 1 | "The problem is explicit and well-motivated. The team explains inputs/outputs of the central task." | Topic clear, problem statement too vague or high-level | No identifiable problem statement | State the problem in one sentence. Name the inputs (an utterance's text + its audio clip) and outputs (an emotion label + a text-vs-audio weight; for our evaluation: whether that weight matches the measured effect of removing each input). |
| Novelty | 1 | "The team names specific prior work and articulates how their project goes beyond." | Prior work referenced vaguely, not named or cited | No prior work mentioned | Name papers on the slide, not "prior work shows". Candidates: Jacovi & Goldberg 2020 (plausible vs faithful), Jain & Wallace 2019 / Wiegreffe & Pinter 2019 (attention as explanation), Hessel & Lee 2020 (closest multimodal work), He & Xiao 2025 (gate weights read as importance). Then one line on what we add: an always-one-modality baseline, ranking vs "which mattered more", two removal methods, and (final) modality conflict. |
| Progress so far | 2 | "The team shows concrete completed work (however small) and can clearly articulate what exactly needs to be done before the final report." | Vague progress and/or unclear, unscoped remaining plan | Little or no progress and/or no plan | Show a real number or figure (the gate's gain over always-text with its interval, or the calibration curve). Remaining work as 2 to 4 concrete, scoped items, each one line. |
| Q&A | 1 | "Answers are direct, concise, and clear." | Answers given but not relevant to the question | Cannot answer | With 3 minutes total, Q&A is short. Prepared one-sentence answers for the likely questions (see `MIDTERM_STATUS.md` section 11 and the project story doc). Answer the question asked, first sentence first. |

**Time budget (suggested):** about 2 minutes speaking (roughly 260 words of script), about 1 minute left for questions. Rehearse with a timer.

## Proposal (already graded: 16.5 / 20, overall Level 4)

No written comments were given; the "likely reason" column is our inference.

| Criterion | Score | Likely reason | Carry-forward lesson |
| --- | --- | --- | --- |
| Project idea and objectives | 4 / 4 | | Keep the framing |
| Related work | 3 / 3 | | Keep citing specific papers |
| Novelty and impact | **3 / 5** | Said what we'd test, not what new thing we'd learn or who would act on it | Lead every deliverable with the finding and its implication, and position against Hessel & Lee 2020 explicitly |
| Risk assessment | **1 / 2** | Named risks without warning signs or pivots | Report the risks we actually hit and the decision each one changed; write fallbacks in advance for the final |
| Resource assessment | 1.5 / 2 | "Feasible on one GPU" was vague | Give real hardware and runtimes |
| Milestones | 2 / 2 | | |
| Clarity and presentation | 2 / 2 | | |

## Midterm report (rubric not yet published)

Required content per the course page (4 pages, single-spaced, 10 pt). Use these as a checklist until a rubric appears:

- [ ] What are we trying to do? Objectives in plain language, no jargon
- [ ] Who cares? What difference does success make?
- [ ] Related work, in detail
- [ ] Proposed method and what is new about it
- [ ] How experiments are run, which hypotheses they test, how results are evaluated
- [ ] Preliminary results, discussed
- [ ] Progress against the proposal's milestones, and updated milestones for the rest of the semester

## Final presentation and report

Final presentation: 6 minutes, 5 slides, plus 2 minutes of Q&A; 2 to 3 presenters; graded on clarity, completeness and presentation quality. Final report: 8 pages main text plus up to 15 pages of appendix and references. Full requirements in `docs/PROJECT_REQUIREMENTS.md`. Detailed rubrics not yet published; add them here when they are.
