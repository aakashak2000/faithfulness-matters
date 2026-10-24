# Project requirements: Faithfulness MATTERS

The single source of truth for what this project must deliver, under what constraints, and when it counts as done. Grading criteria are in `docs/RUBRIC.md`; the original plan is `docs/proposal.pdf`.

## 1. Course requirements

CSCI 544 (Applied NLP), Fall 2026. Team project, six members, **55% of the final grade**. Research-focused (not application-focused): we answer a question rather than deploy a system.

| Deliverable | Format | Status |
| --- | --- | --- |
| Proposal | 1 page, single-spaced, 10 pt | Submitted, 16.5 / 20 |
| Midterm presentation | 2 slides, **3 min including Q&A**, 1 presenter, slides due **24 h before our slot** on the shared deck | Slot: **group 24, Oct 22** (time of day not recorded). Slides due 24 h before, i.e. **Oct 21** at the slot time |
| Midterm report | 4 pages, single-spaced, 10 pt | Due: _fill in_ |
| Final presentation | 5 slides, 6 min + 2 min Q&A, 2 to 3 presenters | Due: _fill in_ |
| Final report | 8 pages main text, up to 15 with appendix and references, single-spaced, 10 pt; must incorporate midterm feedback | Due: _fill in_ |

The final report must answer: what we tried to do (no jargon); who cares; how it's done today and its limits; what's new and why we expected it to work; how we ran experiments, which hypotheses they tested and how we evaluated them; results and what we discovered; whether hypotheses were supported; main takeaways; what we'd do next and what new questions arose.

## 2. The research question

**Problem.** Multimodal emotion models mix text and audio with a per-utterance weight. That weight is easy to read as "how much each modality mattered." We test whether that reading is true.

**Inputs and outputs of the central task.**
- Model input: one MELD utterance, as its transcript (text) and its audio clip.
- Model output: one of 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise), plus a text weight `w_t` in [0, 1] (the importance signal).
- Our evaluation's output, per utterance: does the modality the signal favours match the modality whose removal changes the prediction more? Aggregated as agreement, gain over an always-text baseline, and Spearman correlation.

**Hypotheses.**
- H1 (midterm): on natural utterances, importance signals agree with intervention effects beyond a trivial baseline. *Status: not distinguishable from the always-text baseline for the learned gate. Under the primary intervention (full-pool resampling) its gain is +0.019, 95% CI −0.003 to 0.041, so at most a small edge (point estimates across non-blank methods +0.014 to +0.029; `results/resampling_full_faithfulness.csv`). Blank replacement gives −0.032, which we treat as an artifact of the stand-in being far from real data, not as a finding. The confidence rule beats the baseline (+0.225) but part of that is structural, and it cannot be told apart from the gate on the gate's own model (`results/paired_gate_vs_confidence.csv`).*
- H2 (final, from the proposal): agreement drops when text and audio carry conflicting emotion. *Status: not yet tested.*

**Importance signals tested.** Learned gate weight (primary), confidence weight `c_t / (c_t + c_a)` (primary for its own model, secondary on others). Optional third: gradient-based (Integrated Gradients).

**Interventions ("intervention tests").** Replace one modality's embedding and rerun the full model. Primary: full-pool resampling (average the output over every classifier-split utterance as donor; donors from the same dialogue and shared-audio clips excluded). Sensitivity checks: (a) the mean training embedding, (b) the embedding of empty text / length-matched silence, (c) two 50-donor resampling draws.

## 3. Proposal commitments

| Proposal milestone | Status |
| --- | --- |
| Midterm: unimodal and fused baselines on official MELD splits, macro- and weighted-F1 | Done, 3 seeds |
| Midterm: show how removing text or audio affects predictions | Done, both methods, full test set, with faithfulness metrics (ahead of plan) |
| Final: full faithfulness comparison across importance signals | Partly done (gate, confidence); gradient signal optional |
| Final: text-audio conflict examples | Not started; this is H2 |

Also promised: comparison against the MELD paper's bcLSTM baseline (59.25 weighted-F1, text+audio). Our model is lower by design (frozen encoders, no dialogue context); this must be stated, not hidden.

## 4. Current state (as of the midterm snapshot, tag `midterm`)

Done: data audit; baselines (3 seeds); silence and empty-text embeddings; both interventions on all 2,610 test utterances; agreement, always-text baseline, Spearman; dialogue-level bootstrap CIs; gate calibration analysis; 4 figures. Details in `MIDTERM_STATUS.md` (local) and `docs/STATE.md` (generated in phase 1).

## 5. Constraints

- **Time.** Midterm presentation this week. Anything for the midterm slides must be finished, checked and rehearsed before the slide deadline (24 h before the slot). Bias hard toward communicating existing results over producing new ones.
- **People.** One person (Aakash) implements and runs experiments. Five teammates read, review results, draft report sections and prepare Q&A. Several are new to Python and ML, so anything handed to them must be explained plainly.
- **Compute.** MacBook (Apple M4, MPS) and Google Colab Pro. Frozen MiniLM and WavLM-large encoders; only small heads are trained. After the one-time embedding step, every experiment runs in under a minute on CPU. New WavLM passes (e.g. dev-set silence, conflict audio) take minutes on MPS.
- **Data.** MELD official splits only. Raw media and caches live in `data/` locally; never in git.
- **Method integrity.** The intervention protocol is fixed (see `scripts/04_interventions.py`). Thresholds are chosen on dev, never on test.

## 6. Decision rule for new work

Every proposed piece of work must state: which rubric criterion or requirement it serves, effort in hours, risk, and its deadline bucket:

| Bucket | Allowed when |
| --- | --- |
| **Before midterm slides** | It changes what goes on the 2 slides or the Q&A answers, takes under about 2 hours, and cannot break existing results |
| **Before midterm report** | It strengthens a required report section (especially novelty and risk, where the proposal lost points) |
| **Before final** | It tests H2, explains the gate finding, or checks robustness of a headline claim |
| **Nice to have** | Anything else. Done only if all of the above are finished |

## 7. Definition of done

**Midterm presentation**
- 2 slides in the shared deck before the deadline, meeting every full-marks line in `docs/RUBRIC.md`.
- Every number on the slides traceable to a file in `results/`.
- A ~2-minute script, rehearsed against a timer, and one-sentence answers to the likely questions.

**Midterm report**
- All required sections (`docs/RUBRIC.md` checklist), 4 pages, related work naming and differentiating specific papers, a risk section that reports risks hit and decisions made, real resource numbers.

**Final**
- H2 tested with matched same-label controls and judged by gain over the always-text baseline, with bootstrap CIs.
- Gate diagnosis done (dev-fitted threshold, frozen, applied to test).
- Headline claims checked under at least one alternative effect metric.
- Code reproducible from the README; final presentation and report delivered.

## 8. Out of scope

- Improving emotion-classification accuracy, fine-tuning the encoders, dialogue-context models.
- New datasets or modalities (video).
- Claims about gates in general; conclusions apply to our model on MELD.
