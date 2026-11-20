# Q&A sheet (1 minute; answer in one or two sentences, then stop)

**Isn't this just math? The fusion equation tells you all this.**
The equation tells you how a given dial setting turns into influence. It doesn't tell you what a trained dial learns, that it captures only part of the per-clip information available, or that the answer changes with the swap-in clips. Those we had to measure.

**Why does the shuffled-dial number (78%) differ from the midterm's 74%?**
Two things changed: a fair shuffle needs the dial held fixed during swaps (the midterm report recomputed it), and the shuffle uses the raw mixing rule rather than the class-corrected one (raw with the dial recomputed gives 75%). We ran the shuffle only in that version, and there the learned dial adds nothing over a shuffled one.

**So is the trust dial useless?**
Not for prediction: in the balanced model, accuracy is 53.5% [51.6, 55.5] with the learned dial and 50.8% [49.0, 52.6] shuffled (suggestive; we did not test the paired difference), and the words-heavy model barely differs (52.9% vs 52.2%). It's just not an explanation of what mattered.

**Doesn't 17% flipping just mean noise?**
Random swap-in sets give 5%; similar clips give 17%. It also holds when the dial is held fixed (14%).

**Why couldn't the dial spot the wrong-audio clips?**
Nothing could from its inputs: a detector trained only for that scored 51, a coin flip. So that's not the dial's failure.

**How is this different from Moon et al. (2026)?**
They show reliability weights matter for accuracy only when they predict the right model. We test the weights as explanations, with a shuffled-dial control, and show what the dial learns.

**Did anything improve the explanation?**
A simple rule did: judge confidence from the dial's break-even point, not 50/50 (in the balanced system, wrong claims among the half we keep drop from 14.6% to 11.1%). A learned score did not beat that rule.

**Would this hold for bigger models or other fusion designs?**
Untested; our models, one dataset, one fusion style. That's our main limit.

**Why MELD / why sarcasm next?**
MELD is the standard text-and-audio emotion set. MUStARD++ has human labels saying whether sarcasm lives in the words or the voice, which gives ground truth the dial never saw.

**What happened to the conflict test from the midterm?** (only if H2 not done)
We paused it to first check whether the agreement it relies on means anything. It is in the report's next steps.

**Were all these experiments pre-registered?**
Every analysis had its rules written down before it ran, and committed. Some follow-up checks were motivated by earlier results; those are labelled post hoc in the protocols.

**Why not show the shuffled dial with the midterm's 74%?**
That version recomputes the dial after each swap, so a shuffled dial isn't well defined there. The fair shuffle holds the dial fixed, which gives 78%.

**At the midterm you said 79%, "informative but miscalibrated". Now agreement means nothing. Which is it?**
Both observations hold; we now know why. The midterm's better cutoff (0.37 instead of 0.5) was an average break-even point: how much a listener moves the answer depends on how far its guess is from its usual guess, so the right cutoff differs per clip. And a shuffled dial matches the learned one, so the agreement never showed per-clip learning. The midterm "ranks influence well" claim is one of the four we withdrew.

**If finding 2 shows the swap test depends on the swap-in clips, why use it as the yardstick in finding 1?**
We use it as the model's own behaviour that the dial claims to describe, not as ground truth for importance. Finding 1 holds for the standard swap set; finding 2 says any such claim must name its comparison.

**How does the dial compare with simply saying "words mattered" every time?**
Balanced model: dial 74%, always-words 53%. Words-heavy model: dial 92%, always-words 93%, so there the dial is no better (midterm report numbers, dial recomputed).

**Isn't the example clip only interesting because the words-listener was unsure (28%)?**
Yes, and that is the point: the dial reflects general trust, while the swap measures how strongly each listener pushed on this clip. When one listener is unsure, swapping it matters little. 127 clips met the selection rule; we showed the first.
