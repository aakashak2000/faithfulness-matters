# Final-experiment plan and timeline (draft 2026-10-23; dates are internal, confirm LMS due dates)

## 0. Provenance of hypotheses (to disclose in the reports)
| Item | Origin | Status |
|---|---|---|
| Proposal hypothesis: agreement between modality-importance signals and intervention effects drops when text and audio carry conflicting emotional evidence (`docs/proposal.pdf`) | prospective, written before any result | untested; keep unchanged as H2 |
| Observation that the gate weight beats a constant rule on the balanced model (M1) but not on text-dominant models | exploratory, seen on test outputs | preliminary; becomes H1 below |
| Corrected score (weight x distinctiveness) and its Spearman values | derived AFTER seeing test results; algebra of the swap | diagnostic only (H3) |
| Original proposal also promised: macro- and weighted-F1 baselines, silence and mean-embedding interventions, confidence-based signals next to the gate, optional gradient/SHAP signals, Spearman agreement | prospective | status in the milestone table below |

## 1. Pre-specified hypotheses (frozen before the runs below)
- **H1 (dominance).** With architecture, split, donor pool, gate form and context held fixed, the gate's gain over the dev-selected constant rule decreases as one modality's swap-win share rises. Primary comparison: gain at lowest vs highest dominance level within the controlled family. Cross-model trends from the uncontrolled M-series are not evidence for H1.
- **H2 (conflict).** Agreement and gain over the dev-selected constant are lower on verified text-audio conflict pairs than on matched same-emotion controls. Null and opposite outcomes are possible and will be reported.
- **H3 (gate adaptation).** After accounting for the exact fixed-gate arithmetic, gate re-weighting changes the winner on a measurable share of utterances (observed so far: 6.1% for M1, 4.2% for M5) concentrated in near-ties. The fixed-gate identity is a theorem of the architecture, not a hypothesis.
- **H4 (gate input mismatch; new from the audit).** In models whose text head reads RoBERTa, a gate that also sees the RoBERTa-based representation tracks swap effects better than the current MiniLM-input gate.

## 2. Experiments, priority order
| P | Experiment | Controls | Primary outcome | Cost | Acceptance / fallback |
|---|---|---|---|---|---|
| P0 | Corrected-rule swap, M1 then M5 (protocol `docs/corrected_rule_swap_protocol.md`) | rule is the only change | gain over dev constant with CI, paired with raw rule | M1 about 1 h CPU, M5 about 1 h | validations pass; report whichever result |
| P0 | Cheap checks (F1 recompute, dev constant, duplicate sensitivity), step 42 | none | all conclusions unchanged or listed as changed | done or running | `results/audit_*.csv` |
| P1 | **Gate-input mismatch test (H4)**: for M2 and M3, train gates on [RoBERTa mean-pooled representation reduced by PCA to 384-d fitted on clf only] + audio, keep heads, donors, split, alpha fixed; also a gate that sees only MiniLM (current). Same swap protocol. | heads unchanged, so text strength and its swap-win share barely move; the only change is what the gate sees | agreement and gain over dev constant, per input variant | about 2 h per model | If no change, mismatch is not the cause of the lost edge. Run on M2 and M3 only; M5 needs a stage-2 equivalent and is deferred |
| P1 | **Dominance series (H1)**: fix M1's architecture and degrade the TEXT head (and separately the audio head) in a pre-listed series (e.g. training-set fraction 100, 50, 25, 10 percent, or label noise 0, 10, 20 percent), several head seeds each; swap and constant baselines per level | one backbone, one gate form, one donor pool | gain vs swap-win share across levels, CI over head seeds | 1 to 2 days | needs at least 3 head seeds per level (uncertainty currently ignores head variation); abort if F1 collapses |
| P1 | **Conflict study (H2)**: candidate pairs from MELD labels (text of utterance i with audio of an utterance of a different emotion), compared with (a) same-emotion donors, (b) random donors, (c) matched donors where possible; human audit of 50 to 100 pairs for audible emotion; protocol frozen on dev | MELD label mismatch does not guarantee audible conflict, so audit sample is mandatory | agreement, gain over dev constant, decision margin; no accuracy on synthetic pairs | 3 to 4 days incl. human audit | report null results; record attrition |
| P1 | Context vs strength: compare M3 (no context) and M5 (context) at matched alpha settings and matched gate counts; M5 swap with neighbours fixed vs variants | uses existing models | whether context changes swap-win share or gain | hours | Without a matched context-free control, do not claim a context effect (see section 4) |
| P2 | Audio-dominant series, second dataset (IEMOCAP), second fusion mechanism, equal-footing attribution baselines (occlusion, gradient) | | | days to weeks | only if P1 is stable; never claim SHAP or IG without implementing them |
| P2 | Head-seed variance for M1, M2, M3 (retrain heads with seeds 0 to 4, new gates) | | variance of gain across heads vs across gates | 3 to 4 h | replaces "10 independent seeds" wording |

## 3. Milestone reconciliation (proposal vs actual, as of 2026-10-23)
| Original commitment | Done by Oct 22? | Evidence | Deviation and why | Remaining | New target |
|---|---|---|---|---|---|
| Unimodal and fused baselines reproduced on official MELD splits, macro and weighted F1 | Yes, with caveats | `results/baselines_*.csv` (earlier pipeline), `results/harness_models.csv`, `results/audit_f1_recompute.csv` (step 42) | later models use a dialogue-level re-split of the training data and a dev-fitted prior correction; both weighted and macro F1 are in the CSVs | report both with and without correction; keep the official dev/test | midterm report |
| Tests of removing text or audio and the effect on predictions | Yes, extended | silence and mean replacement on the original headline model (`results/interventions_log.csv`, `robustness_results.md`); swap test on later models | primary intervention changed from silence/mean to swapping real utterances (silence and mean kept as checks on the headline model only); the change was made after seeing that removal creates unrealistic inputs | blank/mean checks for M1 to M5 | Nov 2 |
| Compare gate weight with confidence-based signals | Partly | `results/harness_models.csv` (`confW_*`, `gateCONF_*`) | confidence rule analysed; its agreement is partly structural | none beyond reporting the caveat | midterm |
| Gradient/SHAP importance (optional) | No | | optional in the proposal; not started | P2 | final, if time |
| Spearman plus ranking agreement | Yes | `results/harness_models.csv`, `results/audit_bootstrap.csv` | none | corrected-rule version | midterm follow-up |
| Silence vs mean-embedding comparison and reporting any instability | Yes on the original model | `results/robustness_results.md` | not repeated on M1 to M5 | P1 robustness | Nov 24 |
| Conflict test with labelled mismatched pairs (H2) | Pilot only (earlier headline model, `results/conflict_h2_*.csv`) | | no controlled or audited conflict set yet | P1 above | Nov 10 protocol freeze, Nov 17 results |
| Final: full comparison across importance signals incl. conflict | not yet | | | all of section 2 | Dec |

## 4. Reading of M2, M3 and M5 so far (no new experiment)
- Swap-win share of text: M2 97.1% (strong text, no context), M3 92.8% (strong text and audio, no context), M5 95.2% (adds neighbouring utterances). Gains over the test-majority constant: M2 -0.4, M3 -0.1, M5 -1.6 points.
- Weighted F1 (gate): M2 59.69 / 63.48, M3 60.87 / 62.70, M5 57.09 / 63.57 (alpha 0 / dev alpha). So at alpha 0 the context model is the LOWEST of the three, and with the correction it is within 0.1 points of M2. These data do not show that context improves F1, and they do not separate context from the prior correction.
- M5 differs from M3 in three ways at once (context, a stage-2 refit, gate count 3 vs 10), so its lower gain cannot be attributed to context. Required control: a context-free model with 3 gates and the same stage structure.

## 5. Calendar
| Window | Work | Exit gate |
|---|---|---|
| Oct 23 to 27 | cheap checks, corrected-rule M1 and M5, architecture/result registries, related-work reading in full | every slide number tagged VERIFIED / NOT VERIFIED; contradictions investigated |
| Oct 28 to Nov 2 | midterm report (4 pages, 10 pt) and visual inspection; head-seed variance | no unsupported number; milestone table complete |
| Nov 3 to 10 | gate-input test (H4) set-up; conflict-pair construction, human audit, protocol freeze | frozen protocols, counts before outcomes |
| Nov 11 to 17 | conflict study and dominance series, replicates | H1 and H2 results with uncertainty, nulls kept |
| Nov 18 to 24 | decomposition per donor, robustness across effect definitions, per-emotion and error audit | paper-ready tables |
| Nov 25 to Dec 1 | audio-heavy or second dataset only if the above is stable; otherwise fix weaknesses | final experiment freeze, one reproduce command |
| Dec 2 onward | 5-slide deck, rehearsals at about 118 words per minute (630 to 660 words for 6 minutes), Q&A sheet, 8-page final report with appendix | traceable numbers, deadlines met |
