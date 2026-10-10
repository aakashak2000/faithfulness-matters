"""Step 3: embeddings for the 'silence' and 'empty text' interventions.

Audio: WavLM-large on an all-zero waveform with the same length as each test clip,
       extracted exactly like the legacy cache (librosa 16 kHz, max 20 s,
       last_hidden_state, attention-mask-aware mean pooling).
Text:  MiniLM on the empty string "".

Before computing silence, it re-extracts a few REAL test clips and compares them
with the cached embeddings, to prove the extraction settings match. If that check
fails, the silence vectors would not be comparable, so the script stops.

Clip lengths are bucketed to 0.1 s (one WavLM pass per bucket) to keep it fast.

Usage:
  python scripts/03_silence_embeddings.py --wav-dir <folder with test diaX_uttY.wav>
Output:
  data/intervention_embeddings.pt
"""
import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoFeatureExtractor, AutoModel

from common import DATA, load_dataset

SR = 16000
MAX_SECONDS = 20
BUCKET_S = 0.1


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def wavlm_embed(waveforms, extractor, encoder, device, batch_size=8):
    """Identical to the legacy get_wavlm_embeddings."""
    out = []
    for i in range(0, len(waveforms), batch_size):
        batch = waveforms[i:i + batch_size]
        inputs = extractor(batch, return_tensors="pt", sampling_rate=SR,
                           padding=True, return_attention_mask=True).to(device)
        hidden = encoder(**inputs).last_hidden_state
        fmask = encoder._get_feature_vector_attention_mask(hidden.shape[1], inputs.attention_mask)
        mask = fmask.unsqueeze(-1).to(hidden.dtype)
        out.append(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)).float().cpu())
    return torch.cat(out)


def main(args):
    wav_dir = Path(args.wav_dir).expanduser()
    device = pick_device()
    print(f"device: {device}")
    D = load_dataset()
    meta = D["meta"]["test"]
    cached = D["splits"]["test"]["audio"]

    # --- real clip lengths, exactly as the legacy loader saw them
    paths, lengths = [], []
    for m in meta.itertuples():
        p = wav_dir / f"dia{m.dialogue_id}_utt{m.utterance_id}.wav"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; check --wav-dir")
        paths.append(p)
        lengths.append(min(librosa.get_duration(path=str(p)), MAX_SECONDS))
    lengths = np.array(lengths)
    print(f"test clips: {len(paths)}, length min {lengths.min():.2f}s "
          f"median {np.median(lengths):.2f}s max {lengths.max():.2f}s")

    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    encoder = AutoModel.from_pretrained("microsoft/wavlm-large").to(device).eval()

    # --- 1. verify the extraction settings reproduce the cache
    random.seed(0)
    check = random.sample(range(len(paths)), args.verify)
    waves = [librosa.load(str(paths[i]), sr=SR, mono=True)[0][:SR * MAX_SECONDS] for i in check]
    re_emb = wavlm_embed(waves, extractor, encoder, device, batch_size=1)
    cos = torch.nn.functional.cosine_similarity(re_emb, cached[check], dim=1)
    print(f"\nreproduction check on {len(check)} real clips: cosine min {cos.min():.5f} "
          f"mean {cos.mean():.5f}")
    if cos.min() < 0.99:
        raise SystemExit("Re-extracted embeddings do not match the cache (cos < 0.99). "
                         "Paste this output before going further.")

    # --- 2. silence, one pass per length bucket
    buckets = np.maximum(np.round(lengths / BUCKET_S) * BUCKET_S, BUCKET_S)
    unique = np.unique(buckets)
    print(f"\nsilence: {len(unique)} length buckets of {BUCKET_S}s")
    silent = [np.zeros(int(round(b * SR)), dtype=np.float32) for b in unique]
    sil_emb = wavlm_embed(silent, extractor, encoder, device, batch_size=1)
    lookup = {round(float(b), 3): sil_emb[k] for k, b in enumerate(unique)}
    audio_silence = torch.stack([lookup[round(float(b), 3)] for b in buckets])

    # how different are silence vectors across lengths? (informative for the report)
    s = torch.nn.functional.normalize(sil_emb, dim=1)
    pair = s @ s.T
    print(f"silence vectors across lengths: pairwise cosine min {pair.min():.4f} "
          f"mean {pair.mean():.4f}")

    # --- 3. empty text
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    text_empty = torch.tensor(text_model.encode([""]))[0].float()

    # --- how far are the intervention vectors from real data?
    real_a = torch.nn.functional.normalize(cached, dim=1)
    real_t = torch.nn.functional.normalize(D["splits"]["test"]["text"], dim=1)
    mean_a = torch.nn.functional.normalize(D["splits"]["clf"]["audio"].mean(0, keepdim=True), dim=1)
    mean_t = torch.nn.functional.normalize(D["splits"]["clf"]["text"].mean(0, keepdim=True), dim=1)
    sil_n = torch.nn.functional.normalize(audio_silence, dim=1)
    emp_n = torch.nn.functional.normalize(text_empty[None], dim=1)
    print("\naverage cosine to the real test embeddings:")
    print(f"  silence audio      {float((sil_n * real_a).sum(1).mean()):.4f}")
    print(f"  mean-train audio   {float((mean_a @ real_a.T).mean()):.4f}")
    print(f"  empty text         {float((emp_n @ real_t.T).mean()):.4f}")
    print(f"  mean-train text    {float((mean_t @ real_t.T).mean()):.4f}")

    out = {"test": {"audio_silence": audio_silence, "text_empty": text_empty,
                    "clip_length_s": torch.tensor(lengths, dtype=torch.float32),
                    "bucket_s": BUCKET_S},
           "reproduction_check": {"rows": check, "cosine": cos.tolist()}}
    torch.save(out, DATA / "intervention_embeddings.pt")
    print(f"\nSaved {DATA / 'intervention_embeddings.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--verify", type=int, default=20, help="real clips to re-extract as a check")
    main(ap.parse_args())
