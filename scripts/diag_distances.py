"""Item 8: save the embedding-distance numbers that were only printed by 03_silence_embeddings.py.

Average cosine similarity between each replacement vector and the real test embeddings
(same four numbers as 03_silence_embeddings.py lines 112-123), computed from the saved
embeddings so no WavLM pass is needed. Also reports cosine DISTANCE (1 - similarity) and,
for the silence vectors, the spread across clips.

Usage: python scripts/diag_distances.py
Output: results/embedding_distances.json
"""
import json

import torch
import torch.nn.functional as F

from common import DATA, RESULTS, load_dataset


def main():
    D = load_dataset()
    S = D["splits"]
    E = torch.load(DATA / "intervention_embeddings.pt", map_location="cpu", weights_only=False)["test"]
    real_a = F.normalize(S["test"]["audio"], dim=1)
    real_t = F.normalize(S["test"]["text"], dim=1)
    mean_a = F.normalize(S["clf"]["audio"].mean(0, keepdim=True), dim=1)
    mean_t = F.normalize(S["clf"]["text"].mean(0, keepdim=True), dim=1)
    sil = F.normalize(E["audio_silence"], dim=1)
    emp = F.normalize(E["text_empty"][None], dim=1)

    sims = {
        "silent_audio": (sil * real_a).sum(1),
        "mean_train_audio": (mean_a @ real_a.T).squeeze(0),
        "empty_text": (emp @ real_t.T).squeeze(0),
        "mean_train_text": (mean_t @ real_t.T).squeeze(0),
    }
    out = {"note": "average cosine similarity to the 2,610 real test embeddings; "
                   "distance = 1 - similarity. Computed from data/intervention_embeddings.pt "
                   "and the clf-split mean.",
           "replacement": {k: {"mean_cosine_similarity": round(float(v.mean()), 4),
                               "mean_cosine_distance": round(float(1 - v.mean()), 4),
                               "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
                           for k, v in sims.items()}}
    (RESULTS / "embedding_distances.json").write_text(json.dumps(out, indent=2))
    for k, v in out["replacement"].items():
        print(f"{k:18s} similarity {v['mean_cosine_similarity']:.4f}  (range {v['min']:.3f} to {v['max']:.3f})")
    print("saved results/embedding_distances.json")


if __name__ == "__main__":
    main()
