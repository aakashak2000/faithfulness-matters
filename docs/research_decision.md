# Research decision document (2026-10-23)
Scope: our own models on MELD (M1 balanced, M5 text-heavy and the M0b to M3 series). Sources: `docs/research_audit_2026-10-23.md`, `docs/result_registry.md`, `docs/gate_adaptation_analysis.md`.

## 1. What is established
1. **Reproducibility.** The reported F1 values, the swap results (54.2 / 75.4 / 95.2 / 93.6 and the gains) and the frozen-gate identity were recomputed from row-level data. Duplicate-audio exclusion, the dev-selected constant and the corrected decision rule leave the conclusions unchanged (`results/audit_*.csv`, `results/corrected_rule_swap_*.csv`).
2. **A balanced pair of weak-text and strong-audio models gives an informative weight; text-dominated models do not.** M1: the weight's agreement with the swap winner exceeds always-text by +21 points under both decision rules ([18.5, 23.6]). M5, M2, M3: gain between -1.6 and -0.1, never significantly positive.
3. **The sign of the M5 result depends on the procedure.** With the gate frozen the gain is +1.0 [0.1, 2.0]; with the gate recomputed -1.6 [-2.6, -0.6]. Adaptation lowers the gain by 2 to 4 points in all settings and reverses 4 to 8% of winners.
4. **The corrected score is an exact identity under a frozen gate** (agreement 1.0, error 5e-7). Its high agreement and 0.97 rank correlation follow from the arithmetic.
5. **A measured fact about our F1 claim.** The swap-tested rule scored 55.7 (M1) and 57.1 (M5) before correction; the quoted 58.8 and 63.6 include a dev-fitted class-prior correction. We reran the swaps on the corrected rule; the conclusions hold.

## 2. What remains uncertain
- Whether the balanced-versus-dominant contrast is due to modality dominance itself, to the specific encoders, or to the gate not seeing the text head's input (the gate reads MiniLM in every model). No controlled test exists yet.
- Whether the finding extends beyond two model families, other gates, other datasets, or feature-level gates (Liu et al.; He and Xiao).
- Variation across independently trained heads (heads are shared across the 10 gates; M5 has 3 gates).
- Whether the swap-effect reference is a good reference: random donors create unnatural pairs (Hessel and Lee note the same for EMAP); matched-donor and blank/mean checks exist only for an earlier model.
- Whether context helps (no matched control).
- What happens under text-audio conflict (H2): untested. Conflicting pairs may or may not behave like natural ones, and verified conflict needs human audit.
- Whether any later journal version of He and Xiao exists and what Liu et al.'s weights mean at utterance level (we cite only what the papers say).

## 3. What could falsify the current interpretation
| Current interpretation | Experiment | Result that would contradict it |
|---|---|---|
| The weight is informative only when modalities are comparably strong | controlled strength series (3): degrade the text head of M1 stepwise | gain stays near +21 while text swap-win share rises above 90%, or gain falls with no change in dominance |
| The gate's losing its edge is about dominance, not input mismatch | gate-input ablation (2) on M2 and M3 | a RoBERTa-aware gate recovers a large positive gain at unchanged swap-win share |
| Gate adaptation is a secondary contributor, not the cause | extended decomposition (4) | with the gate-input variant, the frozen-versus-adaptive gap is large and explains most of the M5 shortfall |
| The balanced result is not a property of M1 alone | additional head seeds (5) and other balanced pairs | gain varies by more than its interval across head seeds, or vanishes for another weak-text pair |
| The result is not an artifact of random-donor swaps | robustness (6): matched donors, blank/mean | sign or size changes materially under matched donors |
| The weight tracks sensitivity less when modalities conflict | conflict study (1) | agreement is equal or higher on verified conflict pairs; or construction fails the Stage A precision rule |

## 4. What a defensible publication would need
- A sharp question beyond known tools: swapping (Hessel and Lee) and removal saliency (Jin et al.) already exist, so the contribution must be the audit of gate weights and the conditions of failure, supported by controlled manipulations, not by the uncontrolled model series.
- Controlled dominance and conflict experiments with human-verified conflict, matched controls, pre-registered protocols and nulls reported.
- At least one more architecture or dataset and one feature-level gate, or an explanation (not only arithmetic) of why the scalar late-fusion result transfers or not.
- Honest uncertainty: head and split variation, not only test-set resampling.
- Reproducible artifacts, license-aware (MELD redistribution) and a short human-verification study.
- Clear authorship and policy agreements. As stands, the work is a solid course project, not yet a publishable paper: the corrected score is structural and the balanced finding rests on two models we built.
