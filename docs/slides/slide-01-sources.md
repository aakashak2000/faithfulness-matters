# Slide 1 sources: systems that fuse modalities with a learned weight and read the weight as importance

Compiled 2026-10-20. Every quote below was extracted from the paper's full text (PDF downloaded and searched locally), not from a search summary. This is a targeted search, not a systematic survey: say "many" and "often" on the slide, never "most" or "we found that systems typically...".

## What the search found, in one paragraph
Eight papers fuse text with audio (or with other modalities) through a learned gate or weight, and each one reads that weight as the importance, contribution or reliability of a modality. In the passages searched, none of them checks the reading by removing a modality and re-running the model. One paper (ReH-FUSE, 2026) says in its own limitations that such tests are still needed.

## Group A: text + audio systems for emotion recognition (closest to our setting)

| # | Paper | Task and data | Learned weight | How the weight is read |
|---|---|---|---|---|
| A1 | Liu, Li & Meng, "Group Gated Fusion on Attention-based Bidirectional Alignment for Multimodal Emotion Recognition", Interspeech 2020; arXiv:2201.06309, posted Jan 2022 | Speech + text, IEMOCAP | Group Gated Fusion (sigmoid weights over representations) | Contribution of each modality |
| A2 | He & Xiao, "LPGNet: A Lightweight Network with Parallel Attention and Gated Fusion for Multimodal Emotion Recognition", arXiv:2508.08925 (Aug 2025) | Text + audio, IEMOCAP | Multimodal gate (softmax weights alpha_T, alpha_A) | Importance assigned to each modality |
| A3 | Li et al., "GatedxLSTM: A Multimodal Affective Computing Approach for Emotion Recognition in Conversations", arXiv:2503.20919 (Mar 2025) | Text + audio (CLAP), IEMOCAP | Gate over utterances and features | Importance of each input feature |
| A4 | Wen, Tseng & Chen, "ReH-FUSE: Reliability-Aware Hierarchical Fusion of Experts for Multimodal Emotion Recognition in Conversation", arXiv:2609.13857 (Sep 2026) | Text, audio, cross-modal experts; IEMOCAP and MELD | Decision-level routing weights | Reliability or trust, with an explicit caveat |

## Group B: text + audio + visual systems (sentiment and emotion)

| # | Paper | Task and data | Learned weight | How the weight is read |
|---|---|---|---|---|
| B1 | Kumar & Vepa, "Gated Mechanism for Attention Based Multimodal Sentiment Analysis", arXiv:2003.01043 (Feb 2020) | Text, audio, visual sentiment | Conditional gate | Relative importance of modalities |
| B2 | Wu, Sun, Yang & Wong, "Beyond Simple Fusion: Adaptive Gated Fusion for Robust Multimodal Sentiment Analysis", arXiv:2510.01677 (Oct 2025) | Text, audio, video; CMU-MOSI and CMU-MOSEI | Modality importance gate | Modality importance |
| B3 | Zadeh, Liang, Poria, Cambria & Morency, "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph", ACL 2018 (P18-1208) | Language, vision, acoustic; CMU-MOSEI sentiment and emotion | Efficacies (sigmoid weights on fusion edges) | Which modality combinations the model relies on |

## Group C: the original gated unit

| # | Paper | Task and data | Learned weight | How the weight is read |
|---|---|---|---|---|
| C1 | Arevalo, Solorio, Montes-y-Gomez & Gonzalez, "Gated Multimodal Units for Information Fusion", ICLR 2017 workshop track, arXiv:1702.01992 | Plot text + poster image, movie genre (not audio) | Gate z in [0, 1] | Which modality influences the model more |

## Exact lines

**A1, Liu, Li & Meng 2020** (contributions list, then the analysis of the GGF weights, Figure 2)
> "The proposed GGF method allows integration of multiple representations effectively, and obtains interpretable contribution weights of each modality in emotion recognition"

> "We observe that both a_t and h_t have consistently higher weights within their respective groups (i.e., Group-1 and Group-2) on all the five folds, indicating that the text modality contribute more in emotion classification. [...] Besides, this analysis also illustrates the interpretable benefit of the proposed GGF layer, which allows to present the contribution of a particular modality in multimodal emotion recognition."

Check used: compared with which representations are more discriminative on their own (their Section 3.3). No experiment removes a modality and reruns the model in the text I searched.

**A2, He & Xiao 2025** (Section 4, "Effect of Dual-Gated Fusion", ablation discussion)
> "The multimodal gate then adaptively assigns importance to each modality during the fusion process, allowing the model to balance their contributions based on context."

Related-work framing: "...others lack dynamic weighting, treating all modalities equally, even when one modality carries more emotional information than others." (the paper uses an em dash there). Gate definition (Section 3.4, "Multimodal Gated Fusion"): "a multimodal gating mechanism that adaptively assigns weights to each modality", with alpha_T = Softmax(W T_fused), alpha_A = Softmax(W A_fused). The paper does not analyse the learned weights; its check is removing the whole gate module (F1 87.96% to 86.59%), which is a component ablation, not a modality removal. This is very likely the "He & Xiao 2025" named in `docs/RUBRIC.md`; the rubric does not give a title, so confirm with the owner before relying on that.

**A3, Li et al. 2025** (results discussion, Figure 2)
> "Additionally, one of the key contributions of our model is its ability to explain the importance of each input feature. Using the 48-seed model, the weights are visualised in Figure 2. This visualisation provides interpretability, demonstrating that audio has a greater influence on the ERC tasks, which aligns with our intuition."

Check used: agreement with intuition. No modality removal found.

**A4, Wen, Tseng & Chen 2026** (Section 4 analysis and Section 5 limitations). This paper reads the weights as reliability, then warns about it, and it is the closest statement of our motivation:
> "Class-conditional routing is descriptive rather than causal: it shows variation across emotions, but not that larger weight identifies the objectively cleaner modality. Controlled degradation is needed to test whether operational reliability tracks signal quality. Together, these analyses qualify the reliability claim. Gate weights are learned proxies for trust, not calibrated measurements of modality quality, and may degrade under unobserved corruption or distribution shift."

> "Class-conditional gate averages do not establish causal correspondence to signal quality. [...] tests omit controlled missing-modality, acoustic-noise, and transcript-noise conditions."

**B1, Kumar & Vepa 2020** (abstract)
> "Proposed gating mechanism selectively learns the relative importance of different modalities based on the linguistic information, tone of the speaker and facial expressions of an utterance."

**B2, Wu et al. 2025** (introduction)
> "AGFN employs a dual-gated fusion mechanism that explicitly models information reliability (via information entropy gate) and modality importance (via modality importance gate)"

Check used: Table 3 removes the gate module itself ("the effect of removing the Modality Importance Gate"), not a modality.

**B3, Zadeh et al. 2018** (introduction and Section 5)
> "DFG is easily interpretable through what is called efficacies in graph connections. [...] Once the model is trained end to end, we analyze the efficacies in the DFG to study the fusion mechanism learned for modalities in multimodal language."

> "Trace of Multimodal Fusion: We trace the dominant path that every modality undergoes during fusion"

**C1, Arevalo et al. 2017** (results, Figure 8)
> "In order to evaluate which modality influences more the model when assigning a particular label, we averaged the activations of a subset of z gates of the test samples to which the model assigned them such label. We counted the number of samples that pays more attention to the textual modality (z <= 0.5) or to the visual modality (z > 0.5)."

> "As expected, the model is generally more influenced by the textual modality."

Check used: consistency with unimodal performance (their Table 3). Not a removal test.

## Limits of this list
- Eight papers from a targeted search; "many" is supported, "most" is not.
- Only A1 to A4 are text + audio emotion systems; B and C are other modality sets and tasks, so cite them as "multimodal models" not "emotion models".
- "Does not test by removing a modality" is based on the full text of each paper searched for removal, masking, missing-modality and corruption terms, plus the passages above. It could miss a test described in other words.
- A1 (Liu et al.) was published at Interspeech 2020 (arXiv record: "Published in INTERSPEECH-2020"; arXiv upload January 2022), so cite it as 2020. B1 and B2 are arXiv versions only; confirm their venues before the report.
- A3's gate acts over utterance and feature inputs, so "weight per modality" is a looser fit than A1 and A2.

## Suggested citations for the problem sentence (needs your approval)
The current sentence cites (Arevalo et al. 2017; Wu et al. 2025). Better, because they are text + audio emotion systems that state the reading in words: (Liu et al. 2020; He & Xiao 2025), with Arevalo et al. 2017 as the origin of the gate. ReH-FUSE (Wen et al. 2026) belongs in the prior-work or Q&A line: it says the same test is still needed.

---

# Second pass: what each paper built, what it concluded, and whether the conclusion stands

Compiled 2026-10-20 from the full text of each paper (abstract, method and analysis sections). Quotes are exact; descriptions are my paraphrase.

## What each team built and concluded

- **A1. Liu, Li & Meng (2020)** built a **Gated Bidirectional Alignment Network (GBAN)** that aligns speech and text with bidirectional attention over LSTM hidden states, then fuses four representations (aligned and last-state, for speech and for text) in a **group gated fusion (GGF)** layer with two sigmoid gates, evaluated by 5-fold cross validation on IEMOCAP, and they conclude that the gate weights are "interpretable contribution weights of each modality": text weights were higher on all five folds, "indicating that the text modality contribute more in emotion classification".
- **A2. He & Xiao (2025)** built **LPGNet**, a lightweight network that models within- and between-modality relations with parallel dot-product attention over utterance-level text and audio features, drops speaker embeddings, and fuses with a two-stage gate whose last stage gives softmax weights alpha_T and alpha_A to the text and audio streams, evaluated on IEMOCAP (4 classes, over 87% accuracy and F1), and they conclude that "the multimodal gate then adaptively assigns importance to each modality during the fusion process, allowing the model to balance their contributions based on context".
- **A3. Li et al. (2025)** built **GatedxLSTM**, a speech-text conversation model that embeds audio and text with CLAP into a shared space, passes features from the current and preceding utterances (both speakers) through a gate that sets "their contribution to the final emotion recognition task", and decodes with an xLSTM and a Dialogical Emotion Decoder, evaluated on IEMOCAP (4 classes), and they conclude from the average absolute gate weights that "audio has a greater influence on the ERC tasks, which aligns with our intuition".
- **A4. Wen, Tseng & Chen (2026)** built **ReH-FUSE**, which has dialogue-aware text, audio and cross-modal experts and a decision-level router that first sets the text-versus-audio preference and then balances that mixture against the cross-modal expert (the routed output is a weighted sum of expert logits, with weights summing to 1), evaluated on IEMOCAP (74.34 weighted F1) and MELD (68.03 weighted F1), and they conclude that the weights are "learned routing preference, not calibrated signal quality" (audio outweighs text for angry and happy utterances, text outweighs audio for excited ones) and that "controlled degradation is needed to test whether operational reliability tracks signal quality".
- **B1. Kumar & Vepa (2020)** built a **gated attention network** for text, audio and video sentiment that forms contextual representations with self attention, cross attention between modality pairs, and a conditional gate on each cross-fused vector, evaluated on CMU-MOSI and CMU-MOSEI (83.9% and 81.1% accuracy), and they conclude that the gate "selectively learns the relative importance of different modalities".
- **B2. Wu, Sun, Yang & Wong (2025)** built the **Adaptive Gated Fusion Network (AGFN)**, which adds an information-entropy gate (reliability) and a modality importance gate to text, audio and video features after cross-modal attention, evaluated on CMU-MOSI and CMU-MOSEI, and they conclude that the importance gate weights modalities by importance, helping "the model focus on more reliable modalities", while their own ablation calls it "not a dominant contributor".
- **B3. Zadeh et al. (2018)** built the **Dynamic Fusion Graph (DFG)**, used inside Graph-MFN, in which every combination of language, vision and acoustic is a vertex and the links between them are scaled by learned sigmoid "efficacies", trained on CMU-MOSEI (which they introduce), and they conclude that the efficacies show which modality interactions the model relies on, for example that visual efficacies drop when the visual modality is "either uninformative or contradictory".
- **C1. Arevalo et al. (2017)** built the **Gated Multimodal Unit (GMU)**, which combines a text vector and an image vector with multiplicative gates z, evaluated on MM-IMDb (movie plot text and poster image, genre prediction; no audio), and they conclude, by averaging z over test samples per predicted genre, that "the model is generally more influenced by the textual modality" but the visual modality affects it more for Animation and Family.

## How each team checked its reading, and whether the check is enough

| Paper | What they did to support the reading | What that does and does not show |
|---|---|---|
| A1 GBAN | Compared gate weights with which representations are more discriminative on their own (their Section 3.3) | Agreement with unimodal accuracy. A modality can be more accurate alone and still not change this model's prediction when removed. No removal test found. |
| A2 LPGNet | Removed the whole gate module (F1 87.96% to 86.59%) | Shows the gate helps accuracy, which is a different question from whether its weights say which modality mattered. The weights themselves are not analysed. |
| A3 GatedxLSTM | Plotted average absolute weights per output neuron (Figure 2) and noted they match intuition | Parameter magnitudes depend on the scale of the input features (my inference), and "matches intuition" is not a test. No removal test found. |
| A4 ReH-FUSE | Class-level averages of routing weights as a "behavioral check", plus accuracy ablations of routing versus uniform averaging | Descriptive only, and the authors say so. They list missing-modality and corruption tests as not done. |
| B1 Kumar & Vepa | Gated versus ungated accuracy | Shows the gate helps accuracy. The gate values are not analysed in the text I read. |
| B2 AGFN | Removed each gate (small accuracy drops) and showed a qualitative case study of a text-versus-visual conflict | Utility of a component, not faithfulness of its weights. Their own result says the importance gate is minor. |
| B3 DFG | Visualised efficacies on hand-picked cases where a modality is informative, uninformative or contradictory | The closest to a real check, because it asks whether the weights respond to input quality. But the cases are hand-picked, the weights sit on fusion edges, and nothing is removed and rerun. |
| C1 GMU | Averaged the gate per predicted genre and compared with unimodal performance per genre (their Table 3) | Consistency with unimodal accuracy. The gate is a vector per hidden unit, averaged over selected units, so "modality weight" is looser than in a late-fusion mixture. |

## Does any of it make the claim stand?
No. Three kinds of evidence recur, and none answers "if I remove this modality, does the prediction change in the way the weight says?":
1. **Accuracy ablations** of the gate module (A2, B1, B2): they show the gate is useful, not that its weights are faithful.
2. **Agreement with unimodal performance or with intuition** (A1, A3, C1): a modality that is accurate alone is not necessarily the one this model relies on.
3. **Hand-picked or class-level descriptions** (B3, A4): descriptive, and A4 says so.
The nearest to a real check is B3 (weights respond to whether a modality is informative). The most candid is A4, which states the test that is missing.

## Is our test worth doing?
Yes, for three reasons.
- **The assumption is widely used and untested.** Eight papers read their weights as importance, and in the text searched none tests the reading by removing a modality.
- **One group asks for it.** A4 says controlled degradation and missing-modality tests "would help determine whether routing responds to degraded evidence".
- **The reading is not simply right or wrong, which is a finding.** In our model the gate's weights rank effects well (Spearman 0.705) but at the 0.5 cutoff they do not beat always saying "text" (gain +0.019 [-0.003, +0.041]), and a cutoff fitted on dev does (+0.049 [+0.035, +0.062]). The confidence weight scores a gain of +0.225 even with shuffled labels, so a high match does not by itself show a good explanation. Files: `results/resampling_full_faithfulness.csv`, `results/dev_calibration_test.csv`, `results/shuffled_control.csv`.

## What makes our setting different (and what we cannot claim)
- **Closest in design to A4.** Both mix at decision level (ours: probabilities; theirs: logits), so the weight has a clean meaning, the share of the output given to each modality. That makes our test cleaner than for feature-level gates (A1, A2, B1, B2, C1).
- **What we add that none of them has:** an intervention on the inputs with real replacement examples, an always-text baseline, dialogue-level confidence intervals, and a label-shuffled control.
- **What we cannot claim:** our result is about our model (frozen encoders, single utterances, MELD). It does not show that the models above are unfaithful. It shows the reading can fail in a measurable way, so it needs a test.

---

# Hessel & Lee 2020, read in full text (2026-10-20)

Jack Hessel and Lillian Lee, "Does my multimodal model learn cross-modal interactions? It's harder to tell than you might think!", EMNLP 2020, arXiv:2010.06572.

**What they did.** They propose EMAP (empirical multimodally-additive function projection), a diagnostic for a trained image+text classifier. For every text t_i and image v_j, including mismatched pairs, they run the model f(t_i, v_j). They then replace each prediction with f_T(t_i) + f_V(v_i) - mean, where f_T(t_i) is the average of f(t_i, v_j) over all images j and f_V(v_i) is the average of f(t_j, v_i) over all texts j. The result is the prediction of the closest possible ensemble of a text-only and an image-only classifier, so cross-modal interactions are removed.

**What they concluded.** On seven image+text tasks, "removing cross-modal interactions results in little to no performance degradation" in many cases, even when more expressive models outperform simpler ones, so "performance improvements, even when present, often cannot be attributed to consideration of cross-modal feature interactions". They recommend reporting unimodal baselines and EMAP for the best model. They state that model comparison "provide[s] rather indirect evidence", which is the same complaint we make about weight-reading.

**How this relates to us.**
- *Same building block.* Our "remove one modality" step (average the model's output over real donor examples of that modality, `scripts/08_resampling_full.py`) is the same marginalisation as their f_T and f_V terms. This supports our choice of resampling as the primary intervention, because it uses real embeddings and averages over them, and not an artificial blank. State it as "the same replacement idea", since they average logits for an accuracy question and we average probabilities and let the gate recompute its weight for every donor.
- *Different question.* They ask whether a whole model needs cross-modal interactions to be accurate. We ask whether a model's own per-utterance weight matches how much its prediction changes when each modality is removed. They do not evaluate gate or attention weights.
- *Different data.* Image+text classification tasks, not text+audio emotion.
- *Support for the baseline.* Their recommendation to report unimodal baselines matches our always-text baseline in spirit; ours is a baseline for a weight's decision, not for a model's accuracy.
