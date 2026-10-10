"""Step 0: inspect the legacy assets from the MELD challenge build.

Prints the structure of every .pt file in a folder (keys, tensor shapes,
dtypes, label values, list lengths) so we know exactly what we can reuse.

Usage:
    python scripts/00_inspect_assets.py data/legacy
"""
import sys
from pathlib import Path

import torch


def describe(obj, name="root", depth=0, max_depth=4, max_keys=40):
    pad = "  " * depth

    if isinstance(obj, torch.Tensor):
        info = f"Tensor shape={tuple(obj.shape)} dtype={obj.dtype}"
        if obj.numel() > 0 and obj.dim() == 1 and not obj.is_floating_point():
            uniq = torch.unique(obj)
            shown = uniq.tolist()[:10]
            info += f" unique={shown}{' ...' if len(uniq) > 10 else ''}"
        elif obj.numel() > 1 and obj.is_floating_point():
            f = obj.float()
            info += f" mean={f.mean():.4f} std={f.std():.4f}"
            if obj.dim() == 2 and obj.shape[1] <= 10:
                info += f" row_sums~{f.sum(dim=1).mean():.3f}"  # ~1.0 means probabilities
        print(f"{pad}{name}: {info}")
        return

    if isinstance(obj, dict):
        print(f"{pad}{name}: dict ({len(obj)} keys)")
        if depth < max_depth:
            for i, (k, v) in enumerate(obj.items()):
                if i >= max_keys:
                    print(f"{pad}  ... {len(obj) - i} more keys")
                    break
                describe(v, str(k), depth + 1)
        return

    if isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        if obj and depth < max_depth:
            describe(obj[0], "[0]", depth + 1)
        return

    if hasattr(obj, "columns") and hasattr(obj, "shape"):  # pandas DataFrame
        print(f"{pad}{name}: DataFrame shape={obj.shape} columns={list(obj.columns)}")
        return

    if hasattr(obj, "classes_"):  # sklearn LabelEncoder
        print(f"{pad}{name}: {type(obj).__name__} classes={list(obj.classes_)}")
        return

    text = repr(obj)
    print(f"{pad}{name}: {type(obj).__name__} = {text[:150]}")


def main(folder):
    files = sorted(Path(folder).glob("*.pt"))
    if not files:
        print(f"No .pt files found in {folder}")
        return
    for path in files:
        size_mb = path.stat().st_size / 1e6
        print("=" * 70)
        print(f"{path.name}  ({size_mb:.1f} MB)")
        print("=" * 70)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            describe(obj)
        except Exception as e:
            print(f"  could not load: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/legacy")
