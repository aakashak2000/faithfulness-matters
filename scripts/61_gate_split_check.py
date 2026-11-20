"""Step 61 (post hoc, added after step 60's results; see docs/controlled_series_protocol.md "Results"): audio-head quality on the
GATE split, where the gate is trained, for C0 and the three C1 heads of step 59 (same refits). Explains whether the C1/C2 gate shift
tracks the head's likelihood on the gate's own training data. Exploratory.

Usage:  python scripts/61_gate_split_check.py
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402

s59 = importlib.import_module("59_controlled_series")

B = ac.build_harness()
n_clf = len(B["y"]["clf"])
heads = {"C0": s59.fit_audio_head(B)}
for s in (0, 1, 2):
    rows = np.sort(np.random.default_rng(s).choice(n_clf, int(round(0.25 * n_clf)), replace=False))
    heads[f"C1_seed{s}"] = s59.fit_audio_head(B, torch.tensor(rows))
for name, h in heads.items():
    for split in ("gate", "dev", "test"):
        y = B["y"][split].numpy()
        p = s59.probs(h, B["EA"][split])
        print(f"GATESPLIT {name} {split}: accuracy {(p.argmax(1) == y).mean():.4f}, NLL {s59.nll(p, y).mean():.4f}, "
              f"mean max prob {p.max(1).mean():.4f}", flush=True)
