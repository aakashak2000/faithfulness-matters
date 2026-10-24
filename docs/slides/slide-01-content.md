# Slide 1 content (midterm presentation)

Title on the slide (FINAL, 2026-10-20): **Faithfulness MATTERS: Modality Attribution Testing Through Emotion Recognition Signals**. The acronym is not spoken, so the full expansion must be visible in the title (source: `README.md`).
Status 2026-10-20: FINAL = sections 1 (problem statement and citations), 2 (method line), 3 (inputs and outputs), 4 (intervention tests). DRAFT = sections 5 (prior work), 5b (why test, not for the slide), 6 (what we add).

## 1. Problem statement and citations (FINAL)
> Many multimodal models fuse text and audio with a learned weight, and papers often read that weight as how much each modality mattered (Liu et al. 2020; He & Xiao 2025). We test whether that reading is true.

Systems behind the sentence (full list, exact quotes and checks in `slide-01-sources.md`):
- Text + audio emotion recognition: Liu, Li & Meng 2020 (speech + text, IEMOCAP); He & Xiao 2025 (LPGNet, text + audio, IEMOCAP); Li et al. 2025 (GatedxLSTM, IEMOCAP); Wen, Tseng & Chen 2026 (ReH-FUSE, IEMOCAP and MELD, which also warns the reading is unproven).
- Text + audio + visual sentiment: Kumar & Vepa 2020; Wu et al. 2025 (AGFN); Zadeh et al. 2018 (Dynamic Fusion Graph).
- Origin of the gate: Arevalo et al. 2017 (text + image).
- In the text searched, none of them checks the reading by removing a modality.

Notes
- "Many" and "often" are deliberate: the sources show the practice, they are not a survey. Do not change to "we found" or "most".
- "Emotion" is left out of the claim on purpose; the next line names the task (emotion recognition on MELD).
- Citations FINAL (option C, agreed 2026-10-20): Liu et al. 2020 and He & Xiao 2025, the two text + audio emotion systems that state the reading in words. Arevalo et al. 2017 stays in the prior-work block as the origin of gated fusion, not in this sentence. Rejected: keeping Arevalo and Wu (neither is text + audio emotion), adding a third name, no inline citations.

## 2. Method line (FINAL, updated 2026-10-20: now names "intervention tests"; the explanation sits in section 4)
> We test a learned gate that mixes the two models' predictions, and a confidence-based weight, using **intervention tests**.

Notes
- Replaces "in the style of Arevalo et al. 2017": Arevalo's gates act on hidden feature vectors, ours is one scalar weight over the two heads' output probabilities (`LearnedGate`, `fuse_gate` in `scripts/common.py`).
- The confidence weight is c_text / (c_text + c_audio), each c the head's top-class probability (`fuse_confidence`).

## 3. Inputs and outputs: slide text (FINAL, approved 2026-10-20)
Heading: **The task**
- **Input:** one utterance (MELD): its text + its audio clip
- **Model output:** one of 7 emotions + a text weight w (audio weight = 1 − w)
- **Our test:** does w match which modality's removal changes the prediction more?

Basis: `docs/PROJECT_REQUIREMENTS.md` section 2 (7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise); `scripts/common.py` (`fuse_gate`: p = w * p_text + (1 - w) * p_audio).

## 4. Intervention tests: slide text (FINAL, approved 2026-10-20)
Heading: **Intervention tests**
1. **Swap:** replace the text (or the audio) with real examples from thousands of other utterances
2. **Rerun** the model on each swap and average
3. **Compare:** did the weight favour the modality whose swap changed the prediction more?

Basis: `scripts/08_resampling_full.py` (donors: 7,990 classifier-split utterances; output averaged over donors; gate recomputed per donor); metric definitions in `scripts/04_interventions.py` header. Wording matches the locked script ("swap", "rerun", "average", "favoured that same one").

## 5. Prior work (draft, built from `slide-01-sources.md`)

**Gates and weights read as importance**
- Liu, Li & Meng 2020 (GBAN, speech + text): gate weights give "the contribution of a particular modality".
- He & Xiao 2025 (LPGNet, text + audio): the gate "assigns importance to each modality".
- Li et al. 2025 (GatedxLSTM, speech + text): weights show "audio has a greater influence".
- Wen et al. 2026 (ReH-FUSE, MELD and IEMOCAP): routing weights, but "controlled degradation is needed" to test what they mean.
- Same practice in other settings: Arevalo et al. 2017; Zadeh et al. 2018; Kumar & Vepa 2020; Wu et al. 2025.

**When is an explanation faithful?**
- Jacovi & Goldberg 2020: a faithful explanation reflects what the model really did, not just what sounds plausible.
- Jain & Wallace 2019; Wiegreffe & Pinter 2019: whether attention weights are explanations is disputed, and depends on the test.
- Hessel & Lee 2020: to tell whether a multimodal model uses both modalities, average its output over mismatched examples (EMAP). The same replacement idea is our "remove one modality" step; we use it to test a weight, not the model's accuracy.

**The gap (one line):** in the papers we read, the weights are supported by accuracy ablations or by "matches intuition", and none tests the reading by removing a modality and rerunning the model.

Hessel & Lee 2020 was read in full text on 2026-10-20 (see `slide-01-sources.md`); the presenter should still read its abstract, Figure 1 and Algorithm 1 (about 5 minutes) before the talk.
Open: He & Xiao 2025 is very likely LPGNet (arXiv:2508.08925); the rubric file gives no title, so the owner should confirm.
Wording choice: "in the papers we read" is deliberate. It is what the search found and is safe if someone knows a counterexample. Do not shorten to "none do".

## 5b. Why it makes sense for us to test faithfulness (draft)
1. **The reading is common and untested.** Eight systems across emotion and sentiment tasks treat a learned weight as modality importance, and their support is accuracy ablations or intuition, not interventions.
2. **A weight can look right and be wrong.** The attention debate shows weights can mislead, and ReH-FUSE itself calls gate weights "learned proxies for trust, not calibrated measurements of modality quality".
3. **Our setting makes a clean test possible.** Late fusion gives the weight an exact meaning (the share of the output given to each modality), so we can remove a modality, rerun the model, and check whether the prediction moves the way the weight says.
4. **Early evidence says the answer is not simply yes or no** (for the slide notes, not the slide): the gate ranks effects well but at the 0.5 cutoff does not beat "always text"; the confidence weight scores well even with shuffled labels (`results/resampling_full_faithfulness.csv`, `results/shuffled_control.csv`). The result applies to our model only.

## 6. What we add (draft)
- An always-text baseline, so only what beats a trivial rule counts.
- A ranking measure next to "which mattered more", and several ways of removing a modality.
- For the final: a test with deliberately conflicting text and audio.
