# Rule for the slide example clip (written 2026-11-19, before any clip was looked at)
Model M1, cached gate 0 (first gate in `data/corrected_rule_arrays_M1.npz`). Scan MELD test rows in file order and take the FIRST row that meets all of:
1. not `audio_shared`;
2. the text head and the audio head predict different emotions;
3. gate text weight w >= 0.6 (the dial says "mostly words");
4. the primary swap test (corrected rule, gate recomputed, full pool) says audio mattered more (e_cor_real < 0);
5. the utterance has 3 to 15 words.
Reported as-is: speaker, utterance, gold emotion, each head's top emotion and probability, w, fused prediction, text and audio swap effects. Selected by `scripts/63_slide_material.py`; no other clip is considered.
