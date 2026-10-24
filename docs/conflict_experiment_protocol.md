# Emotion-conflict experiment protocol (written 2026-10-23, before any model is run on a constructed pair)
Status: FROZEN at the commit that adds this file. No model output on constructed pairs exists at this time. Amendments, if any, are appended as dated sections with the old text kept. This follows the proposal's prospective hypothesis (`docs/proposal.pdf`) and is the confirmatory test of H2 in `docs/final_experiment_plan.md`.

## 1. Hypotheses and outcome
- **H2 (from the proposal, direction unchanged).** The agreement between the modality weight and the swap-effect winner is lower on text-audio pairs that genuinely conflict in emotion than on matched control pairs that do not. The opposite direction is possible and will be reported; the test is two-sided.
- **Primary outcome.** For each text item i, the paired difference D_i = agree_conflict(i) - agree_control(i), where agree(.) in {0, 1} is whether the weight's call (w0 > 0.5 means text) equals the swap winner, under the corrected decision rule (the rule behind the reported F1; alpha frozen below), gate recomputed. The estimand is the mean of D_i over verified items, with a dialogue-cluster bootstrap CI (section 9). Per model, so two primary tests.
- **Secondary (exploratory, no multiplicity claim):** (a) control minus natural (T_i, A_i) agreement, which isolates the cost of a different recording from the cost of emotional conflict; (b) gain over the dev-selected constant rule in each condition; (c) swap-win share of text per condition; (d) Spearman between 2w - 1 and the signed effect gap; (e) frozen-gate version; (f) raw-rule version; (g) decision margin (q0[c'] minus the second-largest q0). Accuracy on constructed pairs is NOT an outcome: a conflict pair has no single gold label.

## 2. Models and frozen settings
Primary models (both without conversation context, because a constructed pair has no coherent neighbours): **M1** (balanced, 10 cached gates) and **M3** (strong text + strong audio, 10 cached gates). M5 is excluded from the confirmatory test for that reason; M2 is optional exploratory. Gates, heads, alpha (M1 0.25, M3 0.25, dev-selected in step 33, `results/harness_models.csv`), class prior pi and donor pools are exactly those of `scripts/audit_common.py` and `docs/corrected_rule_swap_protocol.md` sections 2 to 5. Nothing is retrained.

## 3. Two stages and where the data come from
- **Stage A (dev split only; annotation package built by `scripts/46_build_annotation_package.py`, seed 20261023).** 50 text items from dev, each with one conflict-candidate audio and one matched control audio = 100 pairs. Purpose: measure how often "MELD labels differ" yields a perceived conflict, check annotator agreement, and decide whether the construction rule may be used on the test split. No model is run on Stage A pairs.
- **Stage B (test split, confirmatory).** Explained text items i from the test split; audio j (conflict candidate) and k (control) from the test split; the swap donors come from the clf training pool, as in all earlier experiments. Stage B pairs are built with the rule frozen after Stage A (section 4), annotated, verified, and only then run through the model. Target: at least 150 conflict and 150 control candidate pairs annotated, to obtain at least 100 verified items per condition (section 10).
- The test split is not used before Stage B: no Stage A selection looks at test items. Stage A was built before the flag and near-duplicate rules of section 5 were applied to dev (no dev flags were computed), so Stage A pairs are used only to estimate construction precision and annotator agreement, not for any model result.

## 4. Construction rule (frozen; any change after Stage A is an amendment)
For a text item i (utterance with at least 3 words, at least 1 s of audio, label in {joy, anger, sadness}; neutral, surprise, fear, disgust text labels are excluded because their polarity is ambiguous):
- **Conflict-candidate audio j:** an utterance from a different dialogue whose MELD label has opposite polarity to the label of i (joy vs {anger, sadness, fear, disgust}), at least 1 s long.
- **Matched control audio k:** a different utterance from a different dialogue with the SAME label as i, j != k, k != i.
- **Matching of j and k** (among 150 random eligible candidates per side, seeded): minimise |ln duration_j - ln duration_k| + |RMS dB_j - RMS dB_k| / 6 + 0.5 [speaker_j != speaker_k]. Speaker identity of j and k is matched to each other where possible; no audio clip is used more than once in the package. **Recording quality** is proxied by loudness (RMS dB) and duration only; no other recording-quality measure is available, and this is a limitation. **Dialogue context:** i, j, k are always from three different dialogues; episodes are not matched.
- **Balance check, reported:** standardised mean differences of duration and RMS dB between j and k below 0.1 (Stage A: duration -0.005, RMS -0.003, `docs/annotation_package_manifest.md`); the share of pairs with audio speaker equal to text speaker is only 10% in both conditions, so speaker mismatch between text and audio is the norm and is a limitation common to both conditions.
- MELD labels are used to propose pairs only; the human audit decides which pairs enter the analysis.

## 5. Leakage and near-duplicate rules
1. Explained text i, audio j and audio k are all from the same split (dev for Stage A, test for Stage B). Swap donors are clf-pool training rows only. No test utterance is a donor.
2. Rows flagged `audio_shared` (42 test rows, `results/audio_duplicates.json`) are excluded as i, j and k in Stage B. For every j and k in Stage B, a donor is skipped where the donor's replaced-modality embedding is byte-identical (as in all earlier experiments).
3. Near-duplicate text: an item i is excluded if its text string (lower-cased, punctuation stripped) also occurs in the clf pool with a different label; and if two items in the package share a text string only one is kept.
4. Near-duplicate audio: any clip j or k whose headline audio embedding has cosine similarity above 0.995 to any clf-pool clip is excluded (checked by embedding, no labels or model outputs involved).
5. The annotation key is unopened until adjudication is complete; its sha256 is recorded in the manifest before the sheets are sent out.

## 6. Independent verification of emotion
By three human reviewers following `docs/annotation_instructions.md`: text-only emotion, audio tone-only emotion, and a pair relation, blind to MELD labels, with random item order per reviewer, then adjudication. MELD supplies utterance-level labels, not independent text and audio labels, so these human labels are the verification. Limitation: audio carries words as well as tone; reviewers are told to judge tone but may not be able to ignore words. The audit does not establish that the model "hears" the conflict.

## 7. How the model, gate, correction and effects are recomputed on a pair (T_i, A_j)
Features: the text features of utterance i (all inputs the text head and the gate read) and the audio features of utterance j. For each cached gate: w0 = gate(MiniLM embedding of i, audio embedding of j); p0 = w0 p_text(T_i) + (1 - w0) p_audio(A_j); q0 = softmax(log p0 + alpha log pi); c' = argmax q0. Text swap: replace T_i by each valid donor text in the clf pool (A_j fixed), recompute the gate on (donor text, A_j), fuse, correct, record q_d[c']; audio swap likewise with T_i fixed. Effect = |mean_d q_d[c'] - q0[c']|; winner = larger effect; weight call = text if w0 > 0.5. This is `scripts/audit_common.py::core2` with `pt`, `pa`, `w0` computed on the constructed pair (no code change except feature assembly). The same code on natural pairs (T_i, A_i) must reproduce the already-published corrected-rule results (sanity test 1).

## 8. Conditions per text item
natural (T_i, A_i), control (T_i, A_k), conflict (T_i, A_j). Only items whose conflict and control pairs are both human-verified (section 6, verified conflict and verified control) enter the primary analysis; the natural condition uses the original utterance.

## 9. Statistics
- Unit: text item i; cluster: its dialogue. Paired difference D_i as in section 1; mean over verified items; 95% percentile CI from a dialogue-cluster bootstrap (2,000 draws, `numpy.random.default_rng(0)`), gates averaged inside each draw.
- Two primary tests (M1, M3): two-sided bootstrap-based p-value for mean D = 0 (twice the smaller tail share), Holm correction over the two tests, family-wise alpha 0.05. All other analyses are exploratory and are labelled as such, with raw intervals only.
- **Power statement (fixed now):** with about 100 verified items, from 80 to 100 dialogue clusters, the standard error of a paired difference in agreement is expected to be around 4 to 6 percentage points; only differences of roughly 10 to 15 points can be detected. A non-significant result is reported as "no evidence at this sample size", not as evidence of no effect.
- Clustering: any item analysis reports the number of clusters. If fewer than 60 clusters remain, intervals are marked unreliable.

## 10. Stage A decision rules and failure conditions (fixed now)
- Annotator agreement: Cohen's kappa (annotators 1 and 2) on the pair relation must be at least 0.40 and on text polarity and tone polarity at least 0.40; otherwise the instructions are revised and a fresh Stage A sample is annotated.
- Precision of the construction: among conflict-condition pairs, the share verified as conflict must be at least 50%, and among control-condition pairs the share verified as control at least 70%. If not, the construction is not used for Stage B; the rule must be revised (new amendment, new Stage A sample) before any test pair is made.
- Verified numbers: Stage B proceeds with the candidates annotated; if fewer than 100 verified items per condition (conflict and control for the same text) result, the experiment is reported as underpowered and the primary test is not run as confirmatory.
- Exclusions (counted and reported): not verified, flagged audio, unplayable audio, near-duplicates, missing features, annotation unresolved.

## 11. Sanity tests (the run stops on failure)
1. Natural (T_i, A_i) reproduces the main corrected-rule results for the same items to 1e-5.
2. Pair-builder tests: i, j, k from three different dialogues; no clip reused; i != j != k.
3. Matching balance: |SMD| below 0.1 for duration and RMS dB in the verified set; if above 0.25, stop and amend.
4. The 7 synthetic tests of `scripts/43_corrected_rule_swap.py --selftest`.
5. Each condition's distribution of original predicted classes and of w0 is reported before outcomes are interpreted (to catch degenerate cases).
6. The key hash recorded before annotation equals the hash at opening.

## 12. Order of work
Build Stage A package (done) -> annotate (human) -> adjudicate -> open key, compute precision and kappa -> decision under section 10 -> amend construction if needed -> build Stage B pairs with the frozen rule, annotate and adjudicate -> run section 7 on verified pairs -> section 9 analysis. Nothing after the first step is run by the agent without the human annotations.
