"""Step 9: WavLM-large features from EVERY layer, for train, dev and test clips.

Why: the cached audio embedding is the LAST layer, mean-pooled. Emotion information in
speech models sits mostly in middle layers, so the audio head is weak. This script
stores, per clip and per layer (25 = CNN output + 24 transformer layers):
  mean : attention-mask-aware mean over time        (N, 25, 1024) float16
  std  : attention-mask-aware std over time         (N, 25, 1024) float16
  last_mean : layer 24 mean, float32, for matching clips to cached rows
Extraction settings are the same as the legacy cache and 03_silence_embeddings.py:
librosa 16 kHz mono, first 20 s, WavLM-large feature extractor with padding + mask.
Clips are sorted by length before batching to limit padding.

Checks (stop the run if they fail):
  - layer 24 mean reproduces the cached test/dev audio embedding (cosine >= 0.99)

Usage:
  python scripts/09_extract_layers.py --split test
  python scripts/09_extract_layers.py --split dev
  python scripts/09_extract_layers.py --split train
  add --limit 64 for a timing run (writes nothing)
Output: data/wavlm_layers_<split>.pt
"""
import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel

from common import DATA, load_dataset

SR = 16000
MAX_SECONDS = 20
WAV_ROOT = DATA / "all_data"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def clip_list(split):
    """(dialogue_id, utterance_id) in the order the rest of the project uses."""
    if split in ("dev", "test"):
        D = load_dataset()
        m = D["meta"][split]
        return list(zip(m.dialogue_id.tolist(), m.utterance_id.tolist()))
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    return [(a, b) for a, b in zip(csv.Dialogue_ID, csv.Utterance_ID)
            if (WAV_ROOT / f"wavfiles_train/dia{a}_utt{b}.wav").exists()]


@torch.no_grad()
def pooled_layers(waves, extractor, encoder, device):
    inputs = extractor(waves, return_tensors="pt", sampling_rate=SR,
                       padding=True, return_attention_mask=True).to(device)
    out = encoder(**inputs, output_hidden_states=True)
    # hidden_states[24] is taken BEFORE WavLM's final layer norm; last_hidden_state is after it
    # (and is what the cache uses). Entries 0..23 are as returned, entry 24 is last_hidden_state.
    hs = list(out.hidden_states[:24]) + [out.last_hidden_state]  # 25 x (B, T, 1024)
    fmask = encoder._get_feature_vector_attention_mask(hs[0].shape[1], inputs.attention_mask)
    mask = fmask.unsqueeze(-1).float()
    n = mask.sum(1).clamp_min(1.0)
    means, stds = [], []
    for h in hs:
        h = h.float()
        mu = (h * mask).sum(1) / n
        var = ((h - mu.unsqueeze(1)) ** 2 * mask).sum(1) / n
        means.append(mu)
        stds.append(var.sqrt())
    return torch.stack(means, 1).cpu(), torch.stack(stds, 1).cpu()  # (B, 25, 1024)


def main(args):
    device = pick_device()
    print(f"device: {device}")
    ids = clip_list(args.split)
    if args.limit:
        ids = ids[:args.limit]
    wav_dir = WAV_ROOT / f"wavfiles_{args.split}"
    paths = [wav_dir / f"dia{a}_utt{b}.wav" for a, b in ids]
    print(f"{args.split}: {len(paths)} clips")

    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    encoder = AutoModel.from_pretrained("microsoft/wavlm-large").to(device).eval()

    lengths = np.array([min(librosa.get_duration(path=str(p)), MAX_SECONDS) for p in paths])
    order = np.argsort(lengths)
    N = len(paths)
    mean = torch.zeros(N, 25, 1024, dtype=torch.float16)
    std = torch.zeros(N, 25, 1024, dtype=torch.float16)
    last = torch.zeros(N, 1024)

    t0 = time.time()
    for s in range(0, N, args.batch):
        idx = order[s:s + args.batch]
        waves = [librosa.load(str(paths[i]), sr=SR, mono=True)[0][:SR * MAX_SECONDS] for i in idx]
        m, sd = pooled_layers(waves, extractor, encoder, device)
        mean[idx] = m.half()
        std[idx] = sd.half()
        last[idx] = m[:, 24]
        if (s // args.batch) % 50 == 0:
            done = s + len(idx)
            el = time.time() - t0
            print(f"  {done}/{N}  {el:.0f}s elapsed, ~{el / done * (N - done):.0f}s left", flush=True)
    print(f"extraction time {time.time() - t0:.0f}s")

    if args.limit:
        print("timing run, nothing saved")
        return

    if args.split in ("dev", "test"):
        cached = load_dataset()["splits"][args.split]["audio"]
        cos = torch.nn.functional.cosine_similarity(last, cached, dim=1)
        print(f"reproduction vs cache: cosine min {cos.min():.5f} mean {cos.mean():.5f} "
              f"share>=0.99 {(cos >= 0.99).float().mean():.4f}")
        if cos.mean() < 0.99:
            raise SystemExit("layer-24 mean does not reproduce the cache; stop and inspect")

    torch.save({"ids": ids, "mean": mean, "std": std, "last_mean": last,
                "clip_length_s": torch.tensor(lengths, dtype=torch.float32)},
               DATA / f"wavlm_layers_{args.split}.pt")
    print(f"saved {DATA / f'wavlm_layers_{args.split}.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    main(ap.parse_args())
