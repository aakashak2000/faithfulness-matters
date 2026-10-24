# On-screen content for the two slides (DRAFT v1, 2026-10-20)
Sections 1 to 4 of slide 1 are FINAL (see `slide-01-content.md`); the prior-work strip and all of slide 2 are drafts. Every number names its file in `claim-ledger.md`.

## SLIDE 1

**Title:** Faithfulness MATTERS
**Subtitle:** Modality Attribution Testing Through Emotion Recognition Signals

**Problem (one sentence, full width):**
Many multimodal models fuse text and audio with a learned weight, and papers often read that weight as how much each modality mattered (Liu et al. 2020; He & Xiao 2025). We test whether that reading is true.

**The task** (left, with a pipeline diagram: text + audio -> two models -> weight w -> prediction)
- Input: one utterance (MELD): its text + its audio clip
- Model output: one of 7 emotions + a text weight w (audio weight = 1 - w)
- Our test: does w match which modality's removal changes the prediction more?

**What we test** (middle)
We test a learned gate that mixes the two models' predictions, and a confidence-based weight, using intervention tests.
Our own models: 51.7 weighted F1 (MELD baseline bcLSTM: 59.25, uses the whole conversation)

**Intervention tests** (right)
1. Swap: replace the text (or the audio) with real examples from thousands of other utterances
2. Rerun the model on each swap and average
3. Compare: did the weight favour the modality whose swap changed the prediction more?

**Prior work** (bottom strip, names only)
Weights read as importance: Liu et al. 2020 · He & Xiao 2025 · Li et al. 2025 · Wen et al. 2026
Faithfulness: Jacovi & Goldberg 2020 · Jain & Wallace 2019 · Wiegreffe & Pinter 2019
Same swap-and-average idea, different question: Hessel & Lee 2020
**Gap:** in the papers we read, none checks a modality weight by removing a modality.

## SLIDE 2

**Title:** What we found so far

**How to read the results** (small, left)
- The weight: the share of the vote given to text (w) and audio (1 - w)
- Our reading: w above a cutoff means "text mattered more"
- The swap test: what actually mattered

**Result** (left, three lines, big numbers)
Agreement with the swap test, 2,610 test utterances:
- Weight, cutoff 0.5: **79%**
- Always guess "text mattered more": **77%**
- Weight, cutoff 0.37 (fitted on dev, then frozen): **82%** (+5 points, 95% CI 3.5 to 6.2)

**Takeaway** (headline above the figure)
The weight is informative but miscalibrated.

**Figure** (right): how often text mattered more, by the gate's text weight; it crosses 50% at about 0.37, not 0.5. (Currently `figures/fig3b_gate_calibration_primary.png`, drawn from test data; a dev-data version is pending.)

**What remains** (bottom)
- Conflict test: mismatched text and audio, about ten random pairings
- Full report with robustness checks
- If time allows: a stronger audio model

## Open points on the slides
1. Figure data: the current figure uses test data; build a dev-data version (the cutoff comes from dev) or keep the test curve and say so.
2. The slide 1 strip is dense; if it overflows, drop "Li et al. 2025 · Wen et al. 2026" and keep them for Q&A.
3. The long title needs two lines; check legibility when the deck is built.
