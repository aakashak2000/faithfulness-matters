"""Diagnostic: show which rows share byte-identical audio embeddings.

Identical audio and text embeddings: possibly a duplicated input clip, a preprocessing
  artifact, or an embedding collision.
Identical audio but different text embeddings: suspicious collision; the audio for at
  least one row may not belong to its utterance. See diag_source_files.py to trace it.

Usage: python scripts/diag_duplicates.py
"""
from collections import defaultdict

import pandas as pd
import torch

D = torch.load("data/dataset.pt", map_location="cpu", weights_only=False)
E = D["emotions"]
S = D["splits"]
meta = {s: pd.DataFrame(D["meta"][s]) for s in ["dev", "test"]}

groups = defaultdict(list)
for split in ["clf", "gate", "dev", "test"]:
    for i, row in enumerate(S[split]["audio"]):
        groups[row.numpy().tobytes()].append((split, i))

dups = [g for g in groups.values() if len(g) > 1]
print(f"{len(dups)} groups of identical audio, {sum(len(g) for g in dups)} rows total\n")

def describe(split, i):
    label = E[int(S[split]["y"][i])]
    if split in meta:
        m = meta[split].iloc[i]
        return (f"{split:<5} row {i:<5} dia{m.dialogue_id}_utt{m.utterance_id:<3} "
                f"{m.duration_s:5.2f}s {label:<9} {m.speaker:<10} {str(m.utterance)[:45]!r}")
    return f"{split:<5} row {i:<5} {'(no metadata for train rows)':<32} {label}"

same_text, diff_text, involves_train_only = 0, 0, 0
for g in sorted(dups, key=lambda g: (-len(g), g[0])):
    texts = set()
    for split, i in g:
        texts.add(S[split]["text"][i].numpy().tobytes())
    verdict = "SAME text" if len(texts) == 1 else "DIFFERENT text"
    same_text += len(texts) == 1
    diff_text += len(texts) > 1
    print(f"--- {len(g)} rows, {verdict}")
    for split, i in g:
        print("   ", describe(split, i))

print(f"\nSummary: {same_text} groups with identical text, {diff_text} groups with different text")
