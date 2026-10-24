# Final-experiment plan and timeline (revised 2026-10-23; dates are internal, confirm LMS due dates)
Priorities were reordered after the audit. Classification-F1 improvements are optional and only done if a specific research question needs a stronger model.

## 0. Provenance of hypotheses (to disclose in the reports)
| Item | Origin | Status |
|---|---|---|
| H2: agreement between importance signals and intervention effects drops when text and audio conflict (`docs/proposal.pdf`) | prospective | protocol frozen in `docs/conflict_experiment_protocol.md`; no model run on a constructed pair yet |
| H1: the gate's gain over a constant rule falls as one modality dominates | exploratory (seen on test outputs) | needs the controlled series below |
| H3: gate adaptation changes the winner on a measurable share of utterances | exploratory, now measured: 4.2 to 8.0% of utterances, `results/gate_adaptation_*.csv` | reported; causal role in the weight's failure on the text-heavy model NOT supported (frozen gain is only +1.0 to +1.3, `docs/gate_adaptation_analysis.md`) |
| H4: a gate that sees the representation that drives the text head tracks swap effects better | new, from the audit | untested; descriptive diagnostic only |
| The corrected score (weight x distinctiveness) | derived after seeing test results | diagnostic; algebra of the swap |

## 1. Priority order of experiments
| # | Experiment | Why | Controls | Primary outcome | Cost | Stop / fallback |
|---|---|---|---|---|---|---|
| 1 | **Conflict versus matched controls (H2)**, Stage A then Stage B | the proposal's central hypothesis, still untested | natural / matched same-emotion / conflict conditions per text item; human-verified pairs; clf-pool donors; M1 and M3 | paired difference in weight agreement, Holm over 2 models | human annotation about 2 h per reviewer per 100 pairs; model runs minutes | Stage A precision below 50% or kappa below 0.40: amend construction; fewer than 100 verified items: report underpowered |
| 2 | **Controlled gate-input ablation (H4)** on M2 and M3 | tests the MiniLM-vs-RoBERTa mismatch separately from dominance (heads fixed, so text strength and swap-win share barely change) | same heads, split, donors, alpha; gate variants: current MiniLM input, RoBERTa-PCA input, both; 10 gates each | agreement and gain over dev constant per gate variant, paired by dialogue | about 2 h per model | if no change, the mismatch is not the cause |
| 3 | **Controlled modality-strength series (H1)** | the uncontrolled M-series cannot support a dominance claim | one architecture (M1), degrade the TEXT head (and separately the audio head) in a pre-listed series (training-set fraction 100, 50, 25, 10 percent; label noise 0, 10, 20 percent), 3 to 5 head seeds per level | gain vs swap-win share, CI over head seeds | 1 to 2 days | abort a level if weighted F1 collapses below 0.30 |
| 4 | **Gate-adaptation decomposition, extended** | mechanism for the 2 to 4 point drop and the 4 to 8% reversals | existing arrays (done at the summary level, `docs/gate_adaptation_analysis.md`); extend by per-donor decomposition and by the gate-input ablation | share of the frozen-to-adaptive gap explained by the gate term, per condition | hours | none |
| 5 | **More head seeds and uncertainty** | the "10 seeds" are 10 gates on one head | retrain heads with seeds 0 to 4 (M1, M2, M3), new gates | variance of gain across heads vs across gates | 3 to 4 h | replaces the "10 seeds" wording |
| 6 | **Alternative intervention robustness** | the swap donor distribution is a design choice | blank/mean replacement, matched-donor swaps (same speaker or same emotion), mean-of-abs, flips, on M1 to M5 | sign and size of the gain under each | about 1 day | report all, including contradictions |
| opt | audio-dominant series; second dataset (IEMOCAP, license permitting); equal-footing attribution baselines (occlusion, gradient); F1 improvements | publication stretch | | | days to weeks | only after 1 to 6 are stable |

## 2. Milestone reconciliation (proposal vs actual, as of 2026-10-23)
| Original commitment | Done? | Evidence | Deviation and why | Remaining | Target |
|---|---|---|---|---|---|
| Unimodal and fused baselines, macro and weighted F1 | yes | `results/audit_f1_recompute.csv`, `harness_models.csv` | dialogue-level re-split of training data; correction fitted on dev | none | midterm |
| Effects of removing text or audio | yes, changed | `interventions_log.csv`, swap results | primary intervention became real-utterance swaps; silence and mean replacement kept as checks on the earlier model | experiment 6 | Nov 24 |
| Gate weight vs confidence signals | partly | `harness_models.csv` | confidence rule's agreement is partly structural | none | midterm |
| Gradient or SHAP (optional) | no | | | optional | if time |
| Spearman and ranking agreement | yes | `audit_bootstrap.csv`, `corrected_rule_swap_*.csv` | | | midterm |
| Silence vs mean comparison and instability report | earlier model only | `docs/robustness_results.md` | | experiment 6 | Nov 24 |
| Conflict test | pilot on the earlier model only | `conflict_h2_*.csv` | now a frozen protocol with human verification | experiment 1 | Nov 10 (Stage A done by team), Nov 17 (Stage B results) |
| Full comparison across signals incl. conflict | not yet | | | all above | Dec |

## 3. Reading of M2, M3 and M5 so far (no new experiment)
- Text swap-win share: M2 97.1%, M3 92.8%, M5 95.2%; gain over the majority winner: -0.4, -0.1, -1.6 points. Gate weighted F1 at alpha 0 / dev alpha: M2 59.69 / 63.48, M3 60.87 / 62.70, M5 57.09 / 63.57.
- M5 differs from M3 in context, stage-2 refit and gate count (3 vs 10) at once, so its lower gain cannot be attributed to context. Required control: a context-free model with the same stage structure and 3 gates. Context is not shown to help F1.

## 4. Calendar
| Window | Work | Exit gate |
|---|---|---|
| Oct 23 to 27 | commit all of the above; read the four key papers beyond what is recorded; hand the Stage A package to three reviewers | reviewers have the package and instructions |
| Oct 28 to Nov 2 | midterm report submission (4 pages, 10 pt, rendered and inspected); Stage A annotation and adjudication; head-seed runs (experiment 5) | kappa and precision per protocol section 10; decision to proceed or amend |
| Nov 3 to 10 | Stage B pair construction with the frozen rule; gate-input ablation (experiment 2) | frozen Stage B list and hash before annotation |
| Nov 11 to 17 | Stage B annotation and adjudication; run conflict analysis; start experiment 3 | H2 result with uncertainty, whatever it is |
| Nov 18 to 24 | experiments 3, 4 (extended) and 6 | tables for the report |
| Nov 25 to Dec 1 | replication or fixes only; final freeze and one reproduce command | all numbers traceable |
| Dec 2 onward | 5-slide deck (630 to 660 words at about 118 wpm), Q&A sheet, 8-page report with appendix | deadlines met |
