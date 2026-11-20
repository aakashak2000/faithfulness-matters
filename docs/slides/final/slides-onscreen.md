# Final presentation: on-screen content (DRAFT v4, 2026-11-19; fixes from the rubric-grader pass)
5 slides, 5 minutes, hard stop. Every number is in `claim-ledger.md`. Fixed vocabulary everywhere (slides, script, Q&A): **words-listener**, **voice-listener**, **trust dial (gate weight w)**, **swap test**, **shuffled dial**, **ranking score (50 = coin flip)**, **95% range**.
Every results slide has the same shape: one question, one chart comparing two things, one number, one sentence of meaning.

## Slide 1: The mystery
**Title:** Can a model tell us whether it listened to the words or the voice?
**Diagram:** clip → words-listener (text model) + voice-listener (audio model) → each guesses an emotion → **trust dial (gate weight w)** mixes them.
**Real clip** (MELD test; the first of 127 clips meeting a rule written before looking; one illustration, not a typical case):
> Rachel: *"Ross, didn't you say that there was an elevator in here?"*
> Words-listener: surprise, 28% · Voice-listener: neutral, 69% · **Trust dial: 61% words**
> Swap in other clips' words: score moves 0.02 · Swap in other voices: score moves **0.20 (9×)**
> The dial says "mostly words". The voice decided.
(The swap test measures how much the predicted emotion's score moves, not whether the label changes.)
**Prior work, one line each:** dials are read as "how much each part mattered" (Liu et al. 2020; He & Xiao 2025) · trust weights matter for accuracy only when they pick the right model (Moon et al., Interspeech 2026) · gate weights can disagree with removal tests (Deshmukh et al. 2026). **Ours, beyond these:** a shuffled-dial control, a check of how the answer depends on the swap-in clips, and retraining under deliberately broken audio to see what the dial actually learns.

## Slide 2: How we investigated
**Roadmap (six steps, icons):** built 2 text+audio emotion models → tested the dial against swaps → challenged our own test with a shuffled dial → changed what we swap in → retrained under deliberately broken audio → checked what the dial's inputs could know.
**Setup:** MELD, 2,610 test clips from *Friends*, 7 emotions · two models: balanced, words-heavy · swap test with 8,004 swap-in clips · baselines: shuffled dial, a single fixed dial setting.
**Since the midterm (one line):** 6 exploratory experiments · 33 dials retrained · rules written before each run · 4 of our own earlier claims withdrawn.
**Change from the midterm plan (pick the true line):** (H2 not done) "Conflict test: [package built, annotation not run]. We paused it to first check whether the agreement it relies on means anything." / (H2 done) "[H2 result in one line]."

## Slide 3: Finding 1: agreement is not evidence
**Question:** The dial agrees with the swap test. Is that because it learned something?
**Chart:** `figures/final/fig1_shuffled_dial.png`: learned 78% vs shuffled 79% (balanced); 94% vs 94% (words-heavy).
**Meaning:** a shuffled dial agrees just as often, so agreement alone doesn't show the dial learned which part mattered for each clip.
Footnote: a fair shuffle needs the dial held fixed during swaps (raw mixing rule). Our midterm numbers (79% on the slides for an earlier model; 74% / 92% in the report, dial recomputed, corrected rule) used other versions of the test.

## Slide 4: Finding 2: "what mattered" depends on the comparison
**Question:** Does the swap test give one stable answer?
**Chart:** `figures/final/fig2_reference.png`: swap in similar clips: **17%** of answers flip (random-swap noise 5%); words-heavy 8% (noise 4%).
**Meaning:** "did the voice matter?" means something different against a random clip than against a similar one. Overall agreement barely moves (73% to 74% across swap-in sets), so the headline hides this.

## Slide 5: Finding 3 and takeaways
**Question:** So what does the dial actually learn?
**Chart:** `figures/final/fig3_what_dial_learns.png`.
- **Overall:** break the voice-listener on purpose (overconfident, sharper, flatter, 30% wrong audio) and the dial turns the voice down every time: 59% → 67 to 77% words (sharper and flatter compared with their own 58% reference). A single fixed setting moves about as far in 4 of 5 comparisons (post hoc; our stricter all-cases rule was not met).
- **Per clip:** which listener is right is only weakly knowable from the dial's inputs (a detector trained for it: 64), and the dial gets most of it (59).
**Takeaway band:** The trust dial tells you how the model mixes its two listeners, not which one mattered for a given clip. · In our balanced model, confidence in a dial reading is better judged from its break-even point than from 50/50 (post hoc); a learned confidence score did not beat that rule. · **Limits:** exploratory analyses of our own models, one dataset, no human labels yet. **Next:** MUStARD++ human labels of whether sarcasm lives in the words or the voice.

## Backup slides (not presented; for Q&A)
B1 fusion equation and the corrected rule · B2 why agreement is structural (influence = weight × distance from usual guess; break-even point) and the shuffled-dial construction · B3 swap-in reference sets, noise floor over 4 draws, holds with the dial fixed (14% / 6%) · B4 phase 2 conditions (C0, C1, C2, S0 to S2, P), manipulation check, constant-gate comparison · B5 probes and the planted-audio result (51 to 52) · B6 selective attribution: learned score vs break-even rule (honest null) · B7 claims withdrawn and why.
