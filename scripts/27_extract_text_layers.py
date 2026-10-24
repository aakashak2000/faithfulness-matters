"""Step 27: RoBERTa-large features from EVERY layer for all MELD utterances (text side of the model-quality work).

Why: the headline text head reads one 384-number MiniLM sentence vector. The audio side improved by about
10 weighted-F1 points once we used several layers, standardised features and a properly trained head; this
extracts the same kind of multi-layer features for text so the same recipe can be tried (step 28).

Definitions, fixed before running:
  - Encoder: FacebookAI/roberta-large (general-purpose, not trained on MELD or on emotion labels).
  - Text: the MELD CSV utterance (utf-8) with typographic quotes and dashes mapped to ASCII; nothing else.
  - Features: for each of the 25 hidden states (embedding output + 24 layers) the attention-mask-aware mean over
    tokens (max 128 tokens), stored as float16 (N, 25, 1024).
  - Row order = the order of data/wavlm_layers_<split>.pt ("ids"), so text and audio rows align exactly.
  - Check: the number of rows per split equals the audio file (train 9,988, dev 1,108, test 2,610) and no text
    is empty.

Usage: python scripts/27_extract_text_layers.py     Outputs: data/roberta_layers_{train,dev,test}.pt (local)
"""
import time

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from common import DATA

MAP = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}


def clean(t):
    for a, b in MAP.items():
        t = t.replace(a, b)
    return t.strip()


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def embed(texts, tok, enc, dev, batch=64):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.zeros(len(texts), 25, 1024, dtype=torch.float16)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        x = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
        hs = enc(**x, output_hidden_states=True).hidden_states
        m = x["attention_mask"].unsqueeze(-1).float()
        pooled = torch.stack([(h.float() * m).sum(1) / m.sum(1).clamp_min(1) for h in hs], 1).cpu()
        out[idx] = pooled.half()
    return out


def main():
    dev = device()
    print("device", dev)
    tok = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
    enc = AutoModel.from_pretrained("FacebookAI/roberta-large").to(dev).eval()
    t0 = time.time()
    for split in ("test", "dev", "train"):
        ids = torch.load(DATA / f"wavlm_layers_{split}.pt", map_location="cpu", weights_only=False)["ids"]
        csv = pd.read_csv(DATA / "meld_csv" / f"{split}_sent_emo.csv")
        lookup = {(a, b): t for a, b, t in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Utterance)}
        texts = [clean(str(lookup[tuple(i)])) for i in ids]
        assert all(len(t) > 0 for t in texts), "empty text"
        feats = embed(texts, tok, enc, dev)
        torch.save({"ids": ids, "mean": feats, "texts": texts}, DATA / f"roberta_layers_{split}.pt")
        print(f"{split}: {len(texts)} utterances, {time.time() - t0:.0f}s; example: {texts[0][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
