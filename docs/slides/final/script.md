# Final presentation: spoken script (DRAFT v4, 2026-11-19)
About 480 spoken words (excluding the 30-second version), about 4 minutes at 120 words per minute. Split at slide boundaries. Same vocabulary as the slides.

## The 30-second version (every presenter should be able to say this unprompted)
Emotion models mix what's said and how it's said with a trust dial, and people read that dial as an explanation. We tested it by swapping the words or the voice and seeing what changes the answer. The dial agrees with that test, but a shuffled dial agrees just as often. And when we broke the voice on purpose, the dial learned to trust the voice less overall, but only weakly which clips to trust. So the dial tells you how the model mixes its listeners, not which one mattered for a given clip.

## Slide 1, the mystery (about 55 s)
Our system guesses emotion in TV dialogue with two listeners: one reads the words, one hears the voice. A trust dial mixes their guesses, and papers read that dial as the explanation: "this clip was decided mostly by the words."
Here's a real clip, the first of 127 that met a rule we wrote before looking. Rachel says, "Ross, didn't you say there was an elevator in here?" The words-listener weakly guesses surprise. The voice-listener confidently says neutral. The dial says 61% words. But swap in other clips' voices and the score for its answer moves nine times more than when we swap the words. The voice decided.
Others have shown these weights matter for accuracy, or can disagree with removal tests. We add a shuffled-dial control and controlled retraining to ask what the dial actually explains.

## Slide 2, the investigation (about 40 s)
We built two models on 2,610 test clips from Friends, then tested the dial from several angles: against swaps, against a shuffled dial, with different swap-in clips, and by retraining it with deliberately broken audio. Since the midterm that meant six exploratory experiments and 33 retrained dials, with each test's rules written before we ran it. We also checked four of our own earlier claims, and withdrew them.

## Slide 3, finding 1 (about 55 s)
The dial agrees with the swap test 78% of the time in the balanced model. So we shuffled the dial: same settings, handed to the wrong clips. It agrees 79% of the time. In the words-heavy model, 94 and 94.
So agreement alone doesn't show the dial learned which part mattered for each clip. It comes from the dial's general habits, not from judging each clip.

## Slide 4, finding 2 (about 55 s)
Next we questioned the swap test itself. "Did the voice matter?" depends on compared with what. Compare against a random clip, or against a very similar clip, and those are different questions.
When we swapped in similar clips, 17% of answers about which listener mattered flipped, against 5% from random noise; 8% in the words-heavy model. Overall agreement barely moved, 73 to 74 percent, so the headline number hides this.

## Slide 5, finding 3 and takeaways (about 75 s)
So what does the dial learn? We broke the voice-listener on purpose: overconfident, sharper, flatter, or 30% of clips with the wrong audio. Every time, the dial turned the voice down, from 59% words to between 67 and 77. In four of five comparisons, a single fixed setting moves about as far.
Clip by clip, which listener is right is only weakly knowable from what the dial sees. A detector trained just for that scores 64, where 50 is a coin flip, and the dial gets 59, most of it.
Our takeaway: the trust dial tells you how the model mixes its two listeners, not which one mattered for a given clip. In our balanced model, confidence is better judged from the dial's break-even point than from fifty-fifty, and a learned confidence score didn't beat that simple rule. Limits: exploratory work on our own models, one dataset. Next: human labels of whether sarcasm lives in the words or the voice. Thank you.
