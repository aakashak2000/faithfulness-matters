"""Diagnostic: export the extraction reproduction check made in step 3.

03_silence_embeddings.py re-embeds 20 real test clips and compares them with the cached
embeddings (cosine). The result is stored in data/intervention_embeddings.pt only; this
copies it to results/extraction_check.json so it is citable.

Usage: python scripts/diag_extraction_check.py
"""
import json

import torch

chk = torch.load("data/intervention_embeddings.pt", map_location="cpu", weights_only=False)["reproduction_check"]
cos = [float(c) for c in chk["cosine"]]
out = {"n_clips": len(cos), "rows": [int(r) for r in chk["rows"]],
       "min_cosine": min(cos), "mean_cosine": sum(cos) / len(cos)}
with open("results/extraction_check.json", "w") as f:
    json.dump(out, f, indent=2)
print({k: v for k, v in out.items() if k != "rows"})
