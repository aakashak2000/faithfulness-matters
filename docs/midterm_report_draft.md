---
title: "Faithfulness MATTERS: Do Modality Weights in Text+Audio Emotion Models Match What Swapping a Modality Does?"
author: "Team 47: T. Beede, A. Inamdar, J. Nagpal, A. Samandi, R. Sanchez, A. Kumar"
date: "CSCI 544 midterm report, draft of 2026-10-23"
geometry: margin=0.8in
fontsize: 10pt
---

**Status of this draft.** Every number is traceable to a file in `results/` (named in brackets) and was recomputed or re-checked in the audit of 2026-10-23 (`docs/research_audit_2026-10-23.md`). Items we have not verified are marked *unverified*. Results are exploratory, on models we built, and scoped to them.

## 1. What we are trying to do, and why it matters
When an AI model reads what someone says and also hears how they say it, it can report that one input mattered more, for example "the model leaned on the words". We ask whether such a report matches what the model actually does when we change one input. If a developer debugging a wrong prediction is told "the audio was ignored" and that is untrue of the model's behaviour, they will look in the wrong place.

**Task.** Emotion recognition on MELD (Poria et al. 2019), dialogue from the TV show *Friends*: the input is one utterance (its transcript and its audio clip), the output is one of seven emotions (anger, disgust, fear, joy, neutral, sadness, surprise). We use the official train/dev/test split (9,989 / 1,109 / 2,610 utterances) and report weighted and macro F1. **The quantity under test** is a per-utterance *modality weight* w between 0 and 1, produced by a learned gate or a confidence rule, that mixes a text prediction and an audio prediction into the final one: p = w p_text + (1 - w) p_audio. Papers often read w above 0.5 as "text mattered more". **"Influence" in this report is operational:** how much the model's predicted-class probability moves when that modality is replaced by that of other real utterances. We do not claim to identify the true cause of a prediction.

**Research questions.** (RQ1) How often does the weight's call (text or audio) match which modality's replacement moves the prediction more, compared with a trivial rule that always names the usually-winning modality? (RQ2) How does the answer change as one modality dominates? (RQ3, from our proposal) Does agreement fall when text and audio carry conflicting emotional evidence?

## 2. Related work and the gap
*Gated fusion makes weights tempting to interpret.* Liu, Li and Meng (2020) fuse speech and text with a "group gated fusion" layer on IEMOCAP, and He and Xiao (2025, LPGNet, preprint) use "dual-gated fusion" to combine modalities; both report accuracy. Their gates are feature-level and are not the scalar probability-level weight we study, so we do not claim to replicate them. We have read their abstracts in full and have not yet confirmed in the full text whether the gate values are interpreted as importance.
*Intervention-based analysis of modalities.* Hessel and Lee (2020) propose EMAP, which removes cross-modal interactions to test whether a model uses them, on seven image+text tasks, and recommend reporting unimodal baselines. Jin et al. (2021, MSD) define per-modality saliency scores to weight distillation losses. Both are precedents for intervening on modalities; neither tests whether a fusion weight matches intervention effects. We use the same family of idea (substitution from other examples) for a different question.
*What counts as faithful.* Jain and Wallace (2019) found attention weights frequently uncorrelated with gradient-based importance, Wiegreffe and Pinter (2019) argued the answer depends on the definition and proposed baselines and calibration, and Jacovi and Goldberg (2020) argued for graded rather than binary faithfulness. Attention is not a late-fusion gate, but these papers motivate scoped claims and trivial baselines.
*Gap and contribution.* We do not claim no prior study exists (our search was targeted, not exhaustive). Our contribution is a focused empirical audit: for utterance-level text+audio emotion models, does the original weight predict which modality's swap moves the output more, beyond a trivial constant rule, and when does that advantage vanish. It is not a new theory of causality or a new swap method.

## 3. Method
**Models (all built by us; encoders frozen, only heads and the gate trained).** Text: MiniLM sentence embedding (384-d) or RoBERTa-large (all 25 layers, 25,600 features); audio: WavLM-large (last layer 1,024-d, or all 25 layers with mean and std, 51,200 features); linear heads on standardised features. The gate is an MLP (Linear 1,408 to 128, ReLU, Linear to 1, sigmoid; 180,481 parameters) on the MiniLM text embedding and the 1,024-d audio embedding **in every model**. *Balanced* (M1): MiniLM text head and all-layer audio head, each utterance judged alone. *Text-heavy* (M5): RoBERTa text, all-layer audio, plus a second stage that uses neighbouring utterances' first-stage outputs within the same dialogue (the audio stage also uses following utterances) [`docs/model_architecture_registry.md`]. A class-prior correction q = softmax(log p + alpha log pi), alpha chosen on dev, is applied to the fused probability for the reported F1 (never tuned on test).

**Swap test (primary).** For an utterance, let c be the emotion with the highest fused probability on the unswapped input. For each valid donor utterance from the classifier-training split (test utterances are never donors; byte-identical embeddings are excluded), replace one modality of the explained utterance with the donor's, recompute the gate and the fused probability, and record the probability of c. Average over donors, subtract the original probability of c, take the absolute value: that is the modality's effect. The modality with the larger effect "mattered more". The **weight's call** is text if w > 0.5. **Agreement** is the share of utterances where call and winner match. **Baselines:** always the modality that wins most often; we select it on dev (it is text for every model) and also report the test-majority rule, which is identical here and descriptive only. **Uncertainty:** dialogue-level bootstrap (1,000 draws), gates averaged inside each draw; this covers test-set sampling only. The 10 gates share one head per model (3 gates for M5), so variation across independently trained heads is *not* measured.

**Hypotheses (provenance).** H2 (conflict lowers agreement) is from our proposal and untested. The balanced-versus-text-heavy contrast (H1) and the corrected score below were found after seeing test outputs and are labelled exploratory.

## 4. Preliminary results
Table 1 lists the tested models; the first swap experiment used the fused probability **before** the class-prior correction, so the model it tested scores lower than the quoted F1. We therefore ran a second experiment, specified and committed before its test run (`docs/corrected_rule_swap_protocol.md`, commit 4f4c662), on the corrected rule that produces the quoted F1.

| | Weighted F1 (gate), no correction / with dev correction | Text swap larger | Weight agrees | Gain over always-text [95% CI] |
|---|---|---|---|---|
| M1 balanced, original rule | 55.7 / **58.8** | 54.2% | 75.4% | **+21.1** [18.8, 23.6] |
| M1 balanced, corrected rule | | 52.8% | 73.6% | **+20.8** [18.5, 23.2] |
| M5 text-heavy, original rule | 57.1 / **63.6** | 95.2% | 93.6% | **-1.6** [-2.6, -0.6] |
| M5 text-heavy, corrected rule | | 92.9% | 91.9% | **-0.9** [-1.9, -0.0] |
*Table 1.* Sources: `audit_f1_recompute.csv`, `audit_bootstrap.csv`, `corrected_rule_swap_M1.csv`, `corrected_rule_swap_M5.csv` (all in `results/`). The published bcLSTM baseline is 59.25 and DialogueRNN 60.25 (Poria et al. 2019; not a controlled comparison, our encoders are newer): M1 with the correction is 0.45 below bcLSTM, M5 with it is 4.3 above, and without the correction both are below it.

**Finding 1.** On the balanced model the weight agrees with the swap test far more often than guessing the usual winner (+21 points), and this survives the corrected rule. **Finding 2.** On the text-heavy models, text moves the prediction more in about 93 to 95% of utterances and the weight does no better than always guessing text (gain -1.6, and -0.9 on the corrected rule, interval touching 0). When audio did win (4.8% of utterances) the weight said so 53.9% of the time and was right 38.3% of the time it said so. A stronger-text model without context (M2) gives the same picture (text wins 97.1%, gain -0.4 [-0.7, -0.2]). **We do not conclude** that balanced modalities guarantee informative weights: models with a flattened text share of 72% or 42% gained +9.9 and +2.3 points, and weak-text models lost about 12.7 [`results/balance_experiments.csv`].

**Diagnostic: why a weight can mislead.** With the gate frozen at its original value, the swap effect of a modality is exactly the weight times the absolute difference between its predicted-class probability and the donors' average probability. A high weight therefore gives little effect if the modality's prediction looks like a typical replacement's. This "corrected score" matches the frozen-gate swap winner in 100% of cases (a theorem of the architecture, to 5e-7), and the real recomputed-gate winner in 93.9% (M1) and 95.8% (M5); its rank correlation with swap-effect differences is 0.97 against 0.72 for the weight on M1. Because it is the swap's own algebra, **this is an explanation of the gap, not independent validation.** Letting the gate re-weight per donor changes the winner for 6.1% (M1) and 4.2% (M5) of utterances, mostly near-ties. Under a frozen gate, M5's gain is +1.0 instead of -1.6, so the gate's own adaptation matters to the sign for M5 [`results/frozen_gate_audit.csv`, `results/audit_duplicate_sensitivity.csv`].

**Checks.** Excluding the 42 test utterances flagged for duplicate audio (39 within test, 3 shared with train) changes no gain by more than 0.2 points [`results/audit_duplicate_sensitivity.csv`]. Averaging absolute per-donor changes instead gives the same signs for M1, M2, M3, M5; the earlier headline-recipe model's small edge (+1.8 [-0.4, +4.0]) turns to -1.5 under that definition.

## 5. Threats and limitations
(1) Two models carry the main claims; one pair, built by us, so no claim about other systems. (2) Heads are shared across gates, so intervals miss head-training variance. (3) The gate in text-strong models does not see the RoBERTa representation that drives their text predictions (an untested mismatch hypothesis). (4) Swapping in random donors creates unnatural input pairs; blank and mean replacements were checked on an earlier model only. (5) The always-text baseline is also near-optimal when text dominates, which limits what any weight can show. (6) The context model's F1 gain over a no-context model (M3) is 0.9 points with the correction and negative without it, and it also differs in gate count, so we make no claim that context helps. (7) Duplicate audio is flagged only when byte-identical.

## 6. Proposal versus actual (as of 2026-10-23)
| Proposal commitment | Status | Note |
|---|---|---|
| Baselines on official splits, macro and weighted F1 | done | later models use a dialogue-level re-split of training data; both F1 reported |
| Effects of removing text or audio | done, changed | primary intervention became real-utterance swaps; silence and mean replacement kept as checks, run on the earlier model only |
| Gate weight versus confidence signals | partly | confidence rule analysed; its agreement is partly structural |
| Gradient or SHAP signals (optional) | not started | |
| Spearman and ranking agreement | done | |
| Conflict test (H2) | pilot only | controlled, audited conflict set is the next step |

## 7. Next steps (dated; targets, not promises)
Nov 10: freeze the conflict-pair protocol with matched same-emotion and random controls and a human audit of 50 to 100 pairs. Nov 17: run the conflict study and a controlled dominance series (one architecture, degraded text head, several head seeds). Nov 24: gate-input mismatch test on M2 and M3, per-donor decomposition, effect-definition and replacement robustness on the new models. Dec: final report and presentation.

## References
Poria, S., et al. (2019). MELD: A multimodal multi-party dataset for emotion recognition in conversations. ACL.

Liu, P., Li, K., Meng, H. (2020). Group gated fusion on attention-based bidirectional alignment for multimodal emotion recognition. Interspeech (arXiv 2201.06309).

He, Z., Xiao, Y. (2025). LPGNet. arXiv 2508.08925.

Hessel, J., Lee, L. (2020). Does my multimodal model learn cross-modal interactions? It's harder to tell than you might think! EMNLP.

Jin, W., et al. (2021). MSD: Saliency-aware knowledge distillation for multimodal understanding. Findings of EMNLP.

Jain, S., Wallace, B. (2019). Attention is not explanation. NAACL.

Wiegreffe, S., Pinter, Y. (2019). Attention is not not explanation. EMNLP-IJCNLP.

Jacovi, A., Goldberg, Y. (2020). Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? ACL.
