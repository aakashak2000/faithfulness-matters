# Final spoken script v2 (221 words, about 112 s at 118 wpm; built from the user's own 263-word draft)

## Slide 1 (98 words)
Our project is Faithfulness MATTERS. Multimodal emotion models combine text and audio using learned weights, as in Liu and in He and Xiao, and these weights are often read as modality importance. Does a higher weight really mean greater influence? We built text and audio classifiers on MELD, from Poria et al. Our balanced model scores 58.8 weighted F1, just below the 59.25 bcLSTM baseline. Our text-heavy model reaches 63.6. Inspired by Hessel and Lee, our swap test replaces text or audio with other utterances', averages the predictions, and measures how far the originally predicted emotion's probability moves.

## Slide 2 (123 words)
We check whether the weight agrees with which modality has the larger swap effect. In our balanced model, text has the larger effect in 54% of cases, and the weight agrees 75% of the time, versus 54% for always guessing text. In our text-heavy model, text wins 95% of cases, and the weight is no better than always guessing text. So why might a higher weight not mean greater influence? Even with a high weight, replacing text may barely change the prediction. A corrected score, weight times how different the original prediction is from a typical replacement's, raises rank correlation with the swap effects from 0.72 to 0.97, largely because of how weighted fusion works. Next: conflicting text-audio pairs and other modality-dominant models.

## Notes
- Previous version (206 words, with "explains the gap rather than confirms importance") is in git history of this file's first version only; the circularity answer lives in qa-v4.md row 34.
- If you run short on time, cut "Inspired by Hessel and Lee," last, never the "versus 54%" clause.
