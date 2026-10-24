# Related-work matrix (full-text reading, 2026-10-23)
Method: I read these papers in full from their arXiv PDFs on 2026-10-23: Liu et al. (arXiv 2201.06309v1, 5 pages), Hessel and Lee (arXiv 2010.06572v1, pages 1 to 6 read closely, the rest skimmed for the results discussion), Jin et al. (arXiv 2101.01881, text extracted and Sections 1 to 5 read; later sections and appendix not read in detail), He and Xiao (arXiv 2508.08925v1, 19 pages, sections 3.4, 5 and the ablation read closely; the rest skimmed). Page numbers are arXiv PDF page numbers, not the venue's pagination, and must be re-checked against the published versions before final citation. Section, equation and figure numbers are the papers' own. Quotes are short and verbatim.
**Not found:** a 2026 journal version of He and Xiao. Only the arXiv preprint (submitted 12 Aug 2025, "under peer review" at the time of the fetch) was found, by search and on arXiv; the report should cite the preprint unless the team has the journal reference.
Search scope: targeted, not exhaustive. No claim is made that nobody has studied this.

## 1. Per-paper record
### Liu, Li, Meng (2020), "Group Gated Fusion on Attention-based Bidirectional Alignment for Multimodal Emotion Recognition", Interspeech 2020 (arXiv 2201.06309)
- **Question and evaluation.** Does attention-based bidirectional alignment of speech and text, plus a gated fusion layer, improve utterance-level emotion recognition? 5-fold cross-validation on IEMOCAP (4 classes), weighted and unweighted accuracy (Sec. 3, 3.2; Tables 1 to 3, p. 3 to 4).
- **What the mechanism computes.** The group gated fusion (GGF) layer takes four representations (aligned speech a_s, aligned text a_t, last hidden states h_s, h_t) in two groups. Within a group a sigmoid gate z (Eq. 17, 20) is computed from the concatenated pair, and the group output is z ⊙ p_s + (1 - z) ⊙ p_t (Eq. 15 to 21, Sec. 2.3, p. 3). The gate acts on feature vectors, element by element, not on class probabilities, and not as one scalar per utterance. Final representation h = Group 1 + Group 2, then a fully connected layer and softmax (Eq. 22, 23).
- **Gate values as importance.** Yes. Contribution (2), p. 1 to 2: GGF "obtains interpretable contribution weights of each modality in emotion recognition"; Sec. 3.6 "Weights Analysis" (p. 4, Fig. 2) averages the gate weights of each representation over test folds and observes that "both a_t and h_t have consistently higher weights ... indicating that the text modality contribute more", and says this "illustrates the interpretable benefit of the GGF layer".
- **What it establishes.** Representation-level accuracy of h_t and a_t exceeds that of speech (Table 1), consistent with the text-heavy weights; GGF gives the best accuracy (Table 2, 3).
- **What it does not establish.** No intervention or removal test of the gate weights; weights are averaged over data rather than examined per utterance; weights are element-wise vectors averaged for display; no baseline for what the weights would be if they were uninformative.
- **How ours differs.** Scalar, utterance-level, probability-level weight on separately trained text and audio predictions; the weight's call is tested against a swap-effect winner with a constant-rule baseline. Their gate is feature-level; we do not replicate GGF.

### He and Xiao (2025 preprint), "LPGNet: A Lightweight Network with Parallel Attention and Gated Fusion for Multimodal Emotion Recognition" (arXiv 2508.08925v1)
- **Question and evaluation.** Can parallel attention (LPIA) plus dual-gated fusion give a lighter speaker-independent ERC model? IEMOCAP, "over 87% accuracy and F1-score in 4-class" (abstract); ablations (Table 3, p. 13).
- **What the mechanism computes.** Sec. 3.4 (p. 8): unimodal gating Z = sigma(W_g H), H_gated = Z ⊙ H (Eq. 9) applied to intra- and cross-modal branches; then a multimodal gate: alpha_T = Softmax(W T_fused), alpha_A = Softmax(W A_fused) (Eq. 13), F_final = alpha_T · T_fused + alpha_A · A_fused (Eq. 14). Per time step and feature vector, softmax-normalised attention-style weights.
- **Gate values as importance.** The prose does: "The multimodal gate then adaptively assigns importance to each modality during the fusion process, allowing the model to balance their contributions based on context" (Sec. 5 ablation text, p. 12). No gate values are reported or analysed in what I read.
- **What it establishes.** Removing the whole dual-gated module lowers F1 from 87.96% to 86.59% (p. 12 to 13, Table 3).
- **What it does not establish.** The ablation removes the module, not individual weights; nothing tests whether alpha_T versus alpha_A matches the effect of changing a modality.
- **How ours differs.** As above. We also do not claim to evaluate LPGNet itself.

### Hessel and Lee (2020), "Does my multimodal model learn cross-modal interactions? It's harder to tell than you might think!", EMNLP 2020 (arXiv 2010.06572)
- **Question and evaluation.** Does a model use cross-modal feature interactions, or only additive unimodal structure? EMAP is computed for seven image+text tasks, a synthetic task and balanced VQA tasks, comparing performance of the model to its projection onto the multimodally-additive functions (Sec. 3 to 5, p. 3 to 6).
- **What the mechanism computes.** For a model f(t, v), EMAP f~(t, v) = E_v f(t, v) + E_t f(t, v) - E_{t,v} f(t, v) (p. 3 to 4), estimated by evaluating f on all pairs (t_i, v_j) of the evaluation set and averaging over j for each i (Algorithm 1, p. 4; Eq. 3; footnote 4: the expectation is over the marginal, not the conditional). They prove it is the optimal multimodally-additive approximation in squared error (Appendix A, "Claim", p. 4). The recommendation is to compute it on the evaluation set (p. 4; footnote 5).
- **Gate values as importance.** Not applicable; the paper has no gates. It does not interpret internal weights.
- **What it establishes.** On balanced VQA tasks EMAP hurts accuracy sharply (e.g. LXMERT VQA2 70.3 to 40.5, Table 2, p. 5); on the seven unbalanced tasks the additive projection often performs close to the full model (Sec. 5.1, p. 6 onward; Table 4).
- **What it does not establish.** Nothing about per-example attribution to one modality, and nothing about a model's reported weights. It also notes the projection evaluates mismatched pairs that "do not occur in the observed data" (p. 4), the same limitation our swap test has.
- **How ours differs and what we reuse.** Our swap effect for one modality is the same building block as EMAP's partial dependence: the model's output averaged over other examples' values of the other input. We do not invent it. Differences: (a) we read the partial dependence at the originally predicted class for one utterance and take the change from the unswapped output, as a per-utterance sensitivity; (b) we compare that sensitivity with a model-internal weight, which EMAP does not; (c) donors come from a separate pool, not the evaluation set; (d) text+audio emotion models, not image+text.

### Jin et al. (2021), "MSD: Saliency-aware Knowledge Distillation for Multimodal Understanding", Findings of EMNLP 2021 (arXiv 2101.01881)
- **Question and evaluation.** How much does each modality matter in knowledge distillation, and can modality-specific distillation losses weighted by saliency help a small student? Four datasets (Hateful Memes, MM-IMDB, SNLI-VE, VQA2) (Sec. 3 to 5, Table 1).
- **What the mechanism computes.** A per-instance modality saliency score S(m) = delta(t(x), t(x_-m)), where x_-m is the input after masking out modality m and delta is a KL divergence between the teacher's predictions (Eq. 5, Sec. 4.2, p. 4); a loss-based variant; and a weight-learner MLP over teacher predictions (Sec. 4.3). The scores set the weights of auxiliary distillation losses.
- **Gate values as importance.** They interpret saliency scores, "proxy scores to modality importance" (Sec. 1), and discuss them, e.g. text saliency is higher on MM-IMDB, and text saliency is correlated with labels in SNLI-VE (Sec. 5, Q1, Figs. 2 to 3, p. 5). These are computed by removal from the teacher's outputs, not read from a gate.
- **What it establishes.** Saliency-weighted losses outperform plain KD on several datasets (Table 1).
- **What it does not establish.** In the parts read, saliency scores are not compared with any internal fusion weight of the model, and they are not validated as explanations; they are used as training weights.
- **How ours differs.** Removal-based per-instance modality saliency is established in this paper; we do not claim it. We use real-utterance substitution with averaging and measure the change in the predicted-class probability, and we audit a model's own scalar weight against it, on emotion recognition.

### Attention-explanation papers (abstract level only; full texts not read)
Jain and Wallace (NAACL 2019), Wiegreffe and Pinter (EMNLP-IJCNLP 2019), Jacovi and Goldberg (ACL 2020): see abstracts in the earlier version of this file (checked 2026-10-23). They are cited for the need to define faithfulness, use baselines and report graded claims, not for any method we use.

### Poria et al. (2019), MELD (ACL)
Dataset and baselines (bcLSTM 59.25, DialogueRNN 60.25 weighted F1, Table 11 as noted earlier in `docs/slides/comparison-table.md`); not re-read today. Other comparison systems are in that file.

## 2. Group/feature-level gates versus our scalar late-fusion gate
| Property | Liu et al. GGF | He and Xiao dual-gated | Ours |
|---|---|---|---|
| What is gated | feature vectors (representations) | feature vectors per time step | class-probability vectors of two separately trained predictors |
| Gate output | element-wise vector in (0,1) | softmax over features, per modality | one scalar w in (0,1) per utterance |
| Trained with | end to end with the classifier | end to end | after the heads, on a held-out split, from MiniLM text embedding and audio embedding |
| Claim in the paper about the gate | "interpretable contribution weights" (Sec. 3.6) | "adaptively assigns importance to each modality" (p. 12) | the reading we test: w above 0.5 means text mattered more |
| Tested against interventions? | no | no (module ablation only) | yes, by substitution |
Our scalar late-fusion weight is a deliberately simple relative of these gates. Results about it do not transfer to feature-level gates without separate tests.

## 3. Novelty statement (conservative; replaces earlier wording)
Substitution of a modality from other examples and averaging (Hessel and Lee) and removal-based per-instance modality saliency (Jin et al.) are existing tools, and gate values are already read as modality importance (Liu et al.; He and Xiao). We make no claim to have invented swapping or intervention-based modality saliency. What we add is an audit: for utterance-level text+audio emotion models with a scalar late-fusion weight, we measure how often the weight's call agrees with which modality's substitution moves the prediction more, against a trivial constant rule, and how this changes as one modality dominates. The audit covers models we built on MELD only, and it is not a test of the published gated architectures.

## 4. Still to do
Read in full: Hessel and Lee Sections 5.2 to 7 (results discussion and recommendations); Jin et al. Sections 5 Q4 and Q5 and appendix; the venue versions of all four (page numbers). Read the three attention-explanation papers' full texts if they are cited for more than motivation. Confirm whether a journal version of He and Xiao exists.
