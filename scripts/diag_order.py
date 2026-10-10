"""Diagnostic: for each bundle row, find its best-matching MELD CSV row.
Shows what order the legacy pipeline stored utterances in.

Usage: python scripts/diag_order.py dev
"""
import sys

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
B = torch.load("data/legacy/fusion_bundle_final.pt", map_location="cpu", weights_only=False)
bt = B[f"X_{split}_text"].float()
bt = bt / bt.norm(dim=1, keepdim=True)

df = pd.read_csv(f"data/meld_csv/{split}_sent_emo.csv")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
ct = model.encode(df["Utterance"].astype(str).tolist(), convert_to_tensor=True,
                  normalize_embeddings=True).float().cpu()

sim = bt @ ct.T
best_cos, best_idx = sim.max(dim=1)

print(f"{'row':>4} {'csv':>5} {'cos':>6}  dia_utt      text")
for i in range(30):
    j = int(best_idx[i])
    print(f"{i:>4} {j:>5} {float(best_cos[i]):6.3f}  "
          f"dia{df.Dialogue_ID[j]}_utt{df.Utterance_ID[j]:<4} {df.Utterance[j][:50]!r}")

print("\nbest-match cosine distribution over all rows:")
for t in [0.999, 0.99, 0.95, 0.9]:
    print(f"  >= {t}: {(best_cos >= t).sum().item()} / {len(best_cos)}")
low = (best_cos < 0.99).nonzero().flatten()[:5].tolist()
print("first rows with best cos < 0.99:", low)
for i in low:
    j = int(best_idx[i])
    print(f"  row {i}: cos={float(best_cos[i]):.3f} nearest csv text {df.Utterance[j]!r}")
