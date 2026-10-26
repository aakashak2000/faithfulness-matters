---
title: "When a Model Says \"the Words Mattered Most\": Testing Text--Audio Fusion Weights Against What Swapping an Input Does"
author: "Team 47: T. Beede, A. Inamdar, J. Nagpal, A. Samandi, R. Sanchez, A. Kumar"
date: "CSCI 544 midterm report"
geometry: margin=0.72in
fontsize: 10pt
header-includes: |
  ```{=latex}
  \usepackage{caption}
  \captionsetup{font=small,skip=3pt}
  \usepackage{titlesec}
  \titlespacing*{\section}{0pt}{7pt}{3pt}
  \titleformat*{\section}{\large\bfseries}
  \setlength{\parskip}{3pt}
  \setlength{\parindent}{0pt}
  \renewcommand{\arraystretch}{1.05}
  \usepackage{etoolbox}
  \usepackage{needspace}
  \AtBeginEnvironment{longtable}{\footnotesize}
  ```
---

## 1. Why this matters, and what we found
A voice assistant reads "Great, thanks." as joy. The developer opens the debugging panel and sees *text weight 0.9, audio weight 0.1*, and concludes that the tone of voice was ignored. Is that conclusion safe? Many text+audio emotion models combine a text prediction and an audio prediction with a learned weight, and papers read that weight as how much each modality mattered. If the weight does not match what the model actually does when an input changes, a developer will look in the wrong place.

**Task.** Emotion recognition on MELD (Poria et al. 2019), dialogue from the TV show *Friends*: one utterance, its transcript and its audio clip, goes in; one of seven emotions (anger, disgust, fear, joy, neutral, sadness, surprise) comes out. We use the official split (9,989 / 1,109 / 2,610 utterances) and report weighted and macro F1. **The quantity under test** is a per-utterance weight *w* in (0, 1) that mixes the two predictions, p = *w*·p~text~ + (1 − *w*)·p~audio~. Reading it as "*w* above 0.5 means text mattered more" is the claim we check. "Influence" here means something measurable: how much the model's output moves when that modality is replaced by another utterance's.

**What we found.** In a balanced model, the weight's call agrees with which modality's replacement moves the prediction more far beyond the best trivial guess, **+20.8 points** over always answering "text". In a text-dominated model it adds nothing, **−0.9 points** (interval touching zero). Both models were built by us on MELD. We explain why a weight can mislead (it ignores how unusual each modality's prediction is) and show this follows exactly from the fusion arithmetic. We do not attribute the contrast between the two models to any single cause; that is the subject of the experiments planned in Section 5.

**Research questions.** RQ1: how often does the weight's call match the swap-test winner, compared with an always-text rule? RQ2: how does this change as one modality dominates? RQ3 (from our proposal): does agreement fall when text and audio express conflicting emotions? RQ1 and RQ2 are answered preliminarily here; RQ3 is not yet tested. **Hypotheses.** H1 (exploratory): the weight's gain over always-text shrinks as one modality dominates. H2 (from the proposal, prospective): agreement is lower on verified conflicting text--audio pairs than on matched same-emotion controls.

## 2. Related work: what is known, and what we add
| Work | What is gated or intervened on | Gate weights checked against interventions? | How our audit differs |
|:----------|:------------------------|:-------------|:-----------------------------|
| Liu, Li, Meng 2020 | gates on feature vectors; claims "interpretable contribution weights" (Sec. 3.6) | no | one scalar weight on class probabilities, tested utterance by utterance |
| He, Xiao 2026 (LPGNet) | feature gates and softmax weights over modalities; "adaptively assigns importance to each modality" | no (whole-module ablation) | same |
| Hessel, Lee 2020 | EMAP: output averaged over other examples' values of one input (image+text) | n/a: no gates; the intervention is the tool | same averaging idea, read at one utterance's predicted class and compared with the model's own weight |
| Jin et al. 2021 | per-instance modality saliency by masking, used to weight distillation losses | n/a: masking saliency used for training; no gate weights compared | real-utterance substitution; audits a weight instead of training with scores |

Gate values are already read as importance (Liu et al.; He and Xiao), and substitution and masking are established tools (Hessel and Lee; Jin et al.). We do not claim to have invented either. **Our contribution is the audit that joins them:** for utterance-level text+audio emotion models with a scalar late-fusion weight, we measure how often the weight identifies the modality whose replacement matters more, against a trivial constant rule, and we show when it stops adding anything. To our knowledge, among the work we read, no paper tests a fusion weight this way. Our gate is simpler than the published feature-level gates, so our results do not transfer to them without separate tests. Jain and Wallace (2019), Wiegreffe and Pinter (2019) and Jacovi and Goldberg (2020) motivate the framing: define faithfulness carefully, use trivial baselines, and make graded claims.

## 3. Method in plain terms
**Models** (all built by us; pretrained encoders frozen, only small linear heads and the gate trained). Each head outputs seven emotion probabilities; the gate, a small network (one hidden layer of 128 units, sigmoid output) reading a 384-d text embedding and the audio embedding, outputs *w*. Table 1 lists the three models. M3 is M5 without the context stage; we introduce it because it has M5's encoders and strength but judges each utterance alone, which the conflict experiment needs (Section 5).

| Model | Text side | Audio side | Context | Gates | W-F1 / M-F1 |
|:--------------|:--------------------|:----------------|:------------------------|:----|:-------------|
| M1 balanced | MiniLM embedding, linear head | WavLM-large, all layers | none | 10 | 58.8 / 41.1 |
| M3 strong, no context | RoBERTa-large, all layers | as M1 | none | 10 | 62.7 / 46.4 |
| M5 text-heavy | as M3 | as M3 | neighbouring utterances' first-stage outputs (audio stage also uses following ones) | 3 | 63.6 / 46.5 |

Table: Models. W-F1 and M-F1 are weighted and macro F1 (%) of the learned-gate model with the dev-fitted class-prior correction described below. The published bcLSTM baseline is 59.25 weighted F1 and DialogueRNN 60.25 (Poria et al. 2019); the comparison is descriptive, not controlled, since our encoders are newer.

**Class-prior correction (plain English).** MELD is dominated by neutral. Our heads are trained with class weights, which pushes predictions toward rare emotions; the correction partly undoes that by multiplying each emotion's fused probability by (its training frequency)^α^ and renormalising, with α chosen on the dev set (never on test; 0.25 for M1, 0.5 for M5). *Illustrative example (made-up numbers, three emotions instead of seven):* fused probabilities neutral 0.38, joy 0.42, anger 0.20 and training frequencies 0.50, 0.30, 0.20 (summing to one); with α = 0.25 the scores are 0.320, 0.311, 0.134, which renormalise to neutral 0.418, joy 0.407, anger 0.175, so the prediction moves from joy to neutral. All results below that carry an F1 use this corrected rule.

**Swap test (plain English).** For one utterance we freeze the emotion the model first predicted. To measure the text's influence, we put in the text of another utterance, rerun the whole model (including the gate), and record the probability of that same emotion. We do this with every utterance in a pool of 8,004 training utterances (all of them, no sampling; on average 8,003 valid text donors and 8,004 audio donors per test utterance, excluding clips whose embedding is identical to the explained utterance's), average the answers, and measure how far the average sits from the original probability. Doing the same for audio gives two numbers; the modality with the larger one "mattered more". *Illustrative example (made-up numbers):* the model says anger with probability 0.68; averaging over text donors gives 0.64 (effect 0.04) and over audio donors 0.50 (effect 0.18). Audio mattered more, even if the weight said 0.8 for text, so the weight and the swap test disagree on this utterance. **Agreement** is the share of test utterances where the weight's call (text if *w* > 0.5) equals the swap winner. **Baseline:** always answering the modality that wins most often, which here is "text" in every model (chosen on dev swaps, so no test outcome selects it). **Gain** is agreement minus that baseline. Intervals are 95% from a bootstrap that resamples whole test dialogues (1,000 draws).

```{=latex}
\needspace{16\baselineskip}
```
## 4. Results
| Model | Text swap larger | Weight agrees | Gain over always-text [95% CI] |
|:--------------------|:--:|:--:|:------------------|
| M1 balanced | 52.8% | 73.6% | **+20.8** [18.5, 23.2] |
| M5 text-heavy | 92.9% | 91.9% | **−0.9** [−1.9, −0.0] |
| *Sensitivity: swap on the uncorrected fused probability* | | | |
| M1 | 54.2% | 75.4% | +21.1 [18.8, 23.6] |
| M3 (strong, no context) | 92.8% | 92.7% | −0.1 [−0.6, +0.4] |
| M5 | 95.2% | 93.6% | −1.6 [−2.6, −0.6] |

Table: Swap-test results, primary rule (first two rows) and sensitivity check. Intervals: dialogue-bootstrap.

![Corrected decision rule, primary measure. (a) How often a call matches the swap-test winner. (b) The weight's gain over always answering "text", with 95% dialogue-bootstrap intervals.](../figures/fig7_gain_corrected_rule.pdf){width=100%}

**Finding 1: the weight is informative in the balanced model.** It matches the swap winner in 73.6% of utterances, against 52.8% for always-text, and its ranking of how much more text than audio mattered correlates at 0.70 with the measured gap (Spearman, 95% interval 0.68 to 0.72). **Finding 2: it adds nothing in the text-heavy model.** Text wins about 93% of the time and the weight does no better than always answering text (−0.9 points, interval touching zero: no evidence of an advantage, and at most a small disadvantage). When audio did win (4.8% of utterances, uncorrected probability), the weight said so in 53.9% of those cases, and was right in 38.3% of the audio calls it made. M2, a strong-text model without context, shows the same pattern (text wins 97.1%; −0.4 [−0.7, −0.1], uncorrected probability).

**Why a weight can mislead.** Replacing a modality with an average replacement changes the output by roughly *weight × how different its prediction is from the average replacement's prediction*. A high weight therefore means little if the modality's prediction already looks like a typical replacement. With the gate held fixed this identity is exact (it matches the swap winner in 100% of utterances), and the weight times this "distinctiveness" has rank correlation 0.97 with the measured gap, against 0.72 for the weight alone (M1, uncorrected probability). **Because it is the swap's own arithmetic, this explains the gap; it is not independent evidence of importance.** Letting the gate re-weight per replacement changes the winner for 4 to 8% of utterances and lowers the gain by 2 to 4 points relative to a frozen gate, but even with the gate frozen M5's gain is only +1.0 to +1.3, so we do not call re-weighting the cause of the text-heavy result.

**Robustness.** Dropping the 42 test utterances with shared audio changes no gain by more than 0.2 points. Averaging absolute per-replacement changes instead gives the same signs for M1, M2, M3 and M5. The two models differ in encoders, strength, context and the number of trained gates (10 for M1 and M3, 3 for M5), so the contrast between them is a finding to explain, not an established cause.

## 5. Status against the proposal, and what we test next
**Conflict experiment (RQ3).** The proposal's hypothesis, that agreement falls when text and audio express different emotions, is untested. We have written and frozen the protocol before running any model on a constructed pair, and prepared a 100-pair package that three reviewers will annotate blind to the dataset labels (differing MELD labels do not guarantee that the emotions sound or read as conflicting). We will compare, per text, a verified conflicting pairing with a matched same-emotion control, with weight-versus-swap agreement as the outcome and Holm correction over two models. With about 100 verified items only large effects (roughly 10 to 20 points) are detectable, so a null result will be reported as inconclusive. **M3 replaces M5** here because a constructed pair has no coherent neighbouring utterances for M5's context stage; M3 has the same encoders and a similar text dominance (text wins 92.8%, uncorrected probability), so M1 (balanced) and M3 (strong text) are run under one identical procedure.

| Proposal commitment | Status |
|:-------------------------------|:------------------------------------------|
| Baselines on official splits, weighted and macro F1 | done (Table 1) |
| Effects of removing or changing a modality | done, extended: swap test; silence and mean replacement run on an earlier model only |
| Gate weight versus confidence-based signal | partly: confidence rule analysed, its agreement is partly structural |
| Spearman and ranking agreement | done |
| Gradient / SHAP signals (optional) | not started |
| Conflict test | protocol frozen, annotation pending, no results |

Table: Proposal milestones and their status.

**Next.** Oct 28 to Nov 2: annotation and adjudication of the pilot package (proceed only if reviewer agreement is acceptable). Nov 3 to 17: build, verify and run the test-split conflict study on M1 and M3; ablate what the gate sees (it reads MiniLM text embeddings even where the text head reads RoBERTa). Nov 18 to 24: a controlled series that degrades one modality in a single architecture with several trained heads, and robustness to matched replacements. Dec: final report.

**Limitations.** Two model families we built on one dataset; heads are shared across gates, so intervals reflect test-set sampling, not head-training variance; random replacements create unnatural pairs (Hessel and Lee note the same for EMAP); the gate's design differs from published feature-level gates. Code, protocols and result tables are in the project repository (corrected-rule protocol committed as 4f4c662 before its run; conflict protocol frozen in 5f878c7 and amended before annotation in ff55b48).

## References
\footnotesize
Poria et al. (2019). MELD: A multimodal multi-party dataset for emotion recognition in conversations. *ACL*.

Liu, P., Li, K., Meng, H. (2020). Group gated fusion on attention-based bidirectional alignment for multimodal emotion recognition. *Interspeech 2020* (arXiv:2201.06309).

He, Z., Xiao, Y. (2026). LPGNet: A lightweight network with parallel attention and gated fusion for multimodal emotion recognition. *Circuits, Systems, and Signal Processing* 45(9):7634--7652. doi:10.1007/s00034-026-03547-w (arXiv:2508.08925; quotations here are from the arXiv v1 text).

Hessel, J., Lee, L. (2020). Does my multimodal model learn cross-modal interactions? It's harder to tell than you might think! *EMNLP* (arXiv:2010.06572).

Jin, W., Sanjabi, M., Nie, S., Tan, L., Ren, X., Firooz, H. (2021). MSD: Saliency-aware knowledge distillation for multimodal understanding. *Findings of EMNLP 2021* (arXiv:2101.01881).

Jain, S., Wallace, B. C. (2019). Attention is not explanation. *NAACL*. Wiegreffe, S., Pinter, Y. (2019). Attention is not not explanation. *EMNLP-IJCNLP*. Jacovi, A., Goldberg, Y. (2020). Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? *ACL*.
