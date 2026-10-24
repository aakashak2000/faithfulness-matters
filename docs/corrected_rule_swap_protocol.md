# Corrected-rule swap protocol (written 2026-10-23, before any corrected-rule number on test was computed)

**Status of this document.** This is a NEW follow-up specified after the audit in `docs/research_audit_2026-10-23.md`. It does not replace or edit the original experiment (steps 33 to 41, `results/harness_*.csv`, `results/swap_with_context.csv`, `results/corrected_signal.csv`, `results/frozen_gate_audit.csv`, `results/audit_bootstrap.csv`, `results/mean_of_abs_check.csv`). Those results stay as they are and are called "the original (raw-rule) experiment". Every number from this protocol goes in new files and carries the label "corrected rule".

## 1. Why
The F1 values 58.8 (M1) and 63.6 (M5) are scored after a dev-fitted class-prior correction; the original swap experiment measured effects on the fused probability BEFORE that correction (55.7 and 57.1 weighted F1 without it). So the model whose predictions score 58.8 / 63.6 was not the object whose weights were tested. This protocol repeats the swap test on the decision rule that produces the quoted F1.

## 2. Models and frozen parameters
| Item | M1 ("balanced") | M5 ("text-heavy", 63.6) |
|---|---|---|
| Heads, gate inputs, split | as in step 33 / 38 (`scripts/audit_common.py::build_harness`) | as in step 37 / 39 (`scripts/audit_common.py::build_m5`) |
| Gates | the 10 gates trained in step 42 and cached (`data/audit_cache/gates_M1.pt`), seeds 0 to 9 | 3 gates, seeds 0 to 2, cached in `data/audit_cache/gates_M5.pt` |
| Class prior pi | class frequencies of the clf split (training rows), computed from labels | same |
| alpha | **0.25**, selected on dev in step 33 (`results/harness_models.csv`, column `alpha_dev`) | **0.5**, selected on dev in step 30 (`data/v2_pipeline_v2.pt`, key `alpha`, "S2 + context, bidirectional") |
| Test set used to choose alpha, pi, donors or any threshold | no | no |
If the dev re-selection in step 42 (own F1 code) disagrees with these values, the stored values above are used (they produced the reported F1) and the disagreement is reported.

## 3. Decision rule
For one gate, an utterance and a modality pair (text t, audio a): p = w t + (1 - w) a, with w the gate output for that input pair. The corrected rule is
q = softmax(log p + alpha log pi) (so q is a normalised probability vector, q_k proportional to p_k pi_k^alpha), prediction = argmax q.
The correction is applied to the fused probability after the gate. The gate itself, the heads and pi are not changed.

## 4. Original predicted class and swaps (corrected rule)
- Original class c' = argmax q0, with q0 computed from the unswapped input and its own gate weight w0.
- Text swap: replace the utterance's text input by that of each valid donor d; the gate is recomputed on (text_d, audio_0), giving w_d; p_d = w_d t_d + (1 - w_d) a_0; q_d = softmax(log p_d + alpha log pi).
  Audio swap: symmetric. For M5 only the explained utterance's stage-1 logits and gate embedding are replaced; its neighbours stay (as in step 37).
- Donors: the same pool and masks as the original experiment (clf split; a donor is excluded for an utterance when its replaced-modality headline embedding is byte-identical to the utterance's own). Test utterances are never donors.
- c' is held fixed for all donors, including donors for which argmax q_d differs from c'.

## 5. Primary effect and metrics
- Effect of modality k: Delta_k = | mean over valid donors of q_d[c'] - q0[c'] | (average first, then absolute value), exactly as in the original primary measure but on q instead of p.
- Winner = text if Delta_text > Delta_audio, else audio. Exact ties (none observed in the original experiment) are dropped from agreement and counted.
- Weight call = text if w0 > 0.5 (the gate output on the unswapped input).
- **Primary metric:** agreement between the weight call and the winner, per gate, averaged over gates; **primary comparison:** gain over the dev-selected constant rule = agreement of the weight minus agreement of "always say the modality that won on more than half of the DEV utterances" (selected per gate from dev swaps under the same rule, `data/audit_dev_arrays_{M1,M5}.npz`).
- Uncertainty: dialogue-cluster bootstrap, 1000 draws, `numpy.random.default_rng(0)`, resampling whole dialogues of the test set, gates averaged inside each draw, 95% percentile interval. Paired differences (corrected minus raw) use the same draws.
- Ranking: Spearman correlation between 2 w0 - 1 and the signed effect difference Delta_text - Delta_audio, per gate, averaged, with the same bootstrap.

## 6. Secondary analyses (reported whatever the primary result)
1. Descriptive test-majority baseline: agreement of "always the modality that won most often on TEST", labelled descriptive; gain over it.
2. Mean-of-absolute-deviations effect: mean over donors of |q_d[c'] - q0[c']| for each modality.
3. Flip-share effect: the share of donors for which argmax q_d != c', per modality; winner = larger share.
4. Frozen-gate effect: the swap with the gate weight held at w0 for every donor.
5. Differences between the raw-rule and corrected-rule conclusions: p(text won), agreement, gains, Spearman, and the paired difference with its interval.
6. Rows where the original class changes between rules (c_raw != c'): share, and agreement under each rule on those rows and on the others.
Not computed under the corrected rule, because the identity does not carry over: the closed-form "corrected signal" of step 34 (q is a nonlinear function of p). It stays labelled as a raw-rule diagnostic.

## 7. Handling a changed predicted class without mixing definitions
Two separate analyses, each internally consistent, are reported side by side and never combined:
- Raw rule: class c_raw = argmax p0; probabilities p (the original experiment, reproduced here as a check).
- Corrected rule: class c' = argmax q0; probabilities q.
A "hybrid" (effect on q of the class chosen by p, or the reverse) is not computed. The rows with c_raw != c' are counted and examined in 6.6 but are not removed.

## 8. Required sanity tests (run before every execution; the script stops on failure)
| # | Test | Where |
|---|---|---|
| 1 | alpha = 0 makes the two rules identical (effects and class) | `selftest` in step 43 |
| 2 | Manual numpy reference for the corrected-rule text effect, gate recomputed | selftest |
| 3 | A donor equal to the utterance's own modality (constant gate) gives zero effect under both rules | selftest |
| 4 | Constant gate: real equals frozen effect | selftest |
| 5 | Permuting the donor order leaves the effects unchanged | selftest |
| 6 | Raw-rule frozen-gate closed form holds to 1e-5 | selftest and every real run |
| 7 | Masked donors do not contribute | selftest |
| 8 | Raw-rule effects on the real test run equal the saved step 38/39 arrays (max abs diff < 1e-4) | step 43 `VALIDATION` line |
| 9 | Classes tracked under each rule reproduce the F1 recomputed in step 42 (to 1e-6) | step 43 `VALIDATION` line |
| 10 | q rows sum to 1 and argmax q equals argmax(log p + alpha log pi) | step 42 |
| 11 | Dev alpha re-selected in step 42 equals the value frozen in section 2 | step 42 log |

## 9. Order and decision rules
1. Run `--selftest`, then M1 on test. If any validation fails, stop and fix the code without looking at corrected-rule results.
2. Only if M1 validated and finished, run M5 (3 gates).
3. No parameter (alpha, pi, donors, tie rule, effect definition, bootstrap settings) is changed after seeing a corrected-rule result. If something is found to be wrong, the change is written here as a dated amendment, and both versions are kept.
4. Reading rule, fixed now: if the corrected-rule M1 gain over the dev-selected constant has an interval that excludes 0 and is positive, the statement "the weight is informative against a constant baseline when text and audio are balanced" is supported for this rule; if its interval includes 0 or is negative, the report says the balanced-model edge does not survive the corrected decision rule, and the contradiction is investigated (class-change rows first). For M5 the same rule applies with the original result (negative gain) as the reference. Whatever is found is reported.

## 10. Outputs
`results/corrected_rule_swap_M1.csv`, `results/corrected_rule_swap_M5.csv`; local arrays `data/corrected_rule_arrays_{M1,M5}.npz`; the run log under `logs/`. Source: `scripts/audit_common.py`, `scripts/42_audit_cheap_checks.py`, `scripts/43_corrected_rule_swap.py`.
