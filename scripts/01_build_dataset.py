"""Step 1: build one canonical dataset file from the legacy bundle, and audit it.

What it does
  1. Loads fusion_bundle_final.pt (the post-leakage-fix bundle).
  2. Re-embeds the MELD CSV text with MiniLM and aligns CSV rows to bundle rows
     for dev and test. This recovers utterance IDs, speaker and duration, and
     proves that text, audio and labels line up.
  3. Runs audit checks: label mapping, finite values, embedding norms,
     clf/gate stratification, duplicate audio rows within/across splits,
     and how the old standalone *_tensors.pt files compare.
  4. Saves data/dataset.pt and results/audit_step1.json.

Usage
    python scripts/01_build_dataset.py \
        --bundle data/legacy/fusion_bundle_final.pt \
        --csv-dir data/meld_csv \
        --legacy-dir data/legacy
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]  # sklearn LabelEncoder order
EXACT_THR = 0.999   # same text, same encoder
FUZZY_THR = 0.90    # same utterance, text cleaned slightly differently


# ----------------------------------------------------------------------------- helpers

def to_seconds(ts):
    """'00:16:16,059' -> 976.059"""
    hms, ms = ts.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def load_csv(csv_dir, split):
    df = pd.read_csv(Path(csv_dir) / f"{split}_sent_emo.csv")
    df["duration_s"] = df["EndTime"].map(to_seconds) - df["StartTime"].map(to_seconds)
    return df.reset_index(drop=True)


def mojibake(s, enc):
    """UTF-8 text mis-read with a single-byte encoding: 'You’re' -> 'Youâ€™re'."""
    raw = s.encode("utf-8")
    try:
        return raw.decode(enc)
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def raw_meld_c1(s):
    """How the original 2018 MELD CSVs store text: each Windows-1252 character
    kept as its byte value, so ’ becomes the control character '\\x92' and … becomes '\\x85'."""
    out = []
    for ch in s:
        try:
            out.append(ch.encode("cp1252").decode("latin-1"))
        except UnicodeEncodeError:
            out.append(ch)
    return "".join(out)


def ascii_fold(s):
    return (s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
             .replace("…", "...").replace("—", "-").replace("–", "-"))


# Ways the legacy pipeline may have decoded the CSV text before embedding it.
TEXT_VARIANTS = {
    "raw_meld_c1": raw_meld_c1,
    "utf8": lambda s: s,
    "cp1252_mojibake": lambda s: mojibake(s, "cp1252"),
    "latin1_mojibake": lambda s: mojibake(s, "latin-1"),
    "ascii_folded": ascii_fold,
}


def embed_text(sentences, batch_size=128):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(list(sentences), batch_size=batch_size, convert_to_tensor=True,
                       show_progress_bar=False)
    return emb.float().cpu()


def unit(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sequential_align(bundle_text, csv_variant_embs):
    """Walk both lists in order. The bundle is the CSV minus a few broken clips,
    so each bundle row matches the next CSV row, or we skip a CSV row.
    Similarity = best cosine over all text-decoding variants.

    Returns csv index per bundle row, cosine and winning variant of each match,
    match type, and the CSV rows that were skipped."""
    b = unit(bundle_text)
    names = list(csv_variant_embs)
    C = torch.stack([unit(csv_variant_embs[n]) for n in names])  # (V, nc, d)
    nb, nc = b.shape[0], C.shape[1]
    idx, cos, var, kind, skipped = [], [], [], [], []
    i = j = 0

    def sim(bi, cj):
        if bi >= nb or cj >= nc:
            return -1.0, None
        sims = C[:, cj] @ b[bi]
        k = int(sims.argmax())
        return float(sims[k]), names[k]

    while i < nb:
        if j >= nc:
            raise RuntimeError(f"Ran out of CSV rows at bundle row {i}: order is not preserved.")
        s0, v0 = sim(i, j)
        if s0 >= EXACT_THR:
            idx.append(j); cos.append(s0); var.append(v0); kind.append("exact"); i += 1; j += 1
            continue
        # look ahead a few CSV rows: were one or more clips dropped from the bundle?
        jump = None
        for k in range(1, 6):
            sk, _ = sim(i, j + k)
            if sk >= EXACT_THR or (sk >= FUZZY_THR and sk > s0 + 0.05):
                jump = k
                break
        if jump is not None:
            skipped.extend(range(j, j + jump)); j += jump
        elif s0 >= FUZZY_THR:
            idx.append(j); cos.append(s0); var.append(v0); kind.append("fuzzy"); i += 1; j += 1
        else:
            raise RuntimeError(
                f"Could not align bundle row {i} with CSV row {j} (best cos={s0:.3f}). "
                "Paste this error.")
    skipped.extend(range(j, nc))
    return np.array(idx), np.array(cos), var, kind, skipped


def dup_audio_rows(a, b=None):
    """Count rows of `a` that exactly equal some row of `b` (or another row of `a`)."""
    keys_a = [r.numpy().tobytes() for r in a]
    if b is None:
        counts = Counter(keys_a)
        return sum(v - 1 for v in counts.values() if v > 1)
    keys_b = set(r.numpy().tobytes() for r in b)
    return sum(k in keys_b for k in keys_a)


# ----------------------------------------------------------------------------- main

def main(args):
    report = {}
    B = torch.load(args.bundle, map_location="cpu", weights_only=False)
    splits = ["clf", "gate", "dev", "test"]
    data = {s: {"text": B[f"X_{s}_text"].float(),
                "audio": B[f"X_{s}_audio"].float(),
                "y": B[f"y_{s}"].long()} for s in splits}

    # --- basic checks
    basic = {}
    for s, d in data.items():
        n = d["y"].shape[0]
        assert d["text"].shape == (n, 384) and d["audio"].shape == (n, 1024), f"shape mismatch in {s}"
        tnorm = d["text"].norm(dim=1)
        basic[s] = {
            "n": n,
            "finite": bool(torch.isfinite(d["text"]).all() and torch.isfinite(d["audio"]).all()),
            "text_norm_mean": round(float(tnorm.mean()), 4),
            "text_norm_min": round(float(tnorm.min()), 4),
            "class_counts": {EMOTIONS[k]: int(v) for k, v in sorted(Counter(d["y"].tolist()).items())},
        }
    report["basic"] = basic

    # --- clf/gate stratification: class proportions should be nearly identical
    p = lambda s: np.array([basic[s]["class_counts"].get(e, 0) for e in EMOTIONS]) / basic[s]["n"]
    report["clf_vs_gate_max_proportion_gap"] = round(float(np.abs(p("clf") - p("gate")).max()), 4)

    # --- duplicate audio rows (exact equality means a clip appears twice: leakage)
    report["duplicate_audio"] = {
        "within_test": dup_audio_rows(data["test"]["audio"]),
        "test_in_train": dup_audio_rows(data["test"]["audio"], torch.cat([data["clf"]["audio"], data["gate"]["audio"]])),
        "dev_in_train": dup_audio_rows(data["dev"]["audio"], torch.cat([data["clf"]["audio"], data["gate"]["audio"]])),
        "clf_in_gate": dup_audio_rows(data["clf"]["audio"], data["gate"]["audio"]),
    }

    # --- align dev/test with the MELD CSVs to recover metadata and verify labels
    meta = {}
    align_report = {}
    for s in ["dev", "test"]:
        df = load_csv(args.csv_dir, s)
        texts = df["Utterance"].astype(str).tolist()
        print(f"\nEmbedding {s} CSV text ({len(df)} rows x {len(TEXT_VARIANTS)} decodings)...")
        variant_embs = {name: embed_text([f(t) for t in texts]) for name, f in TEXT_VARIANTS.items()}
        idx, cos, var, kind, skipped = sequential_align(data[s]["text"], variant_embs)
        aligned = df.iloc[idx].reset_index(drop=True)
        label_from_csv = aligned["Emotion"].map(EMOTIONS.index).values
        label_agree = float((label_from_csv == data[s]["y"].numpy()).mean())
        non_ascii = aligned["Utterance"].astype(str).str.contains(r"[^\x00-\x7f]").values
        align_report[s] = {
            "bundle_rows": int(data[s]["y"].shape[0]),
            "csv_rows": len(df),
            "skipped_csv_rows": [f"dia{df.Dialogue_ID[k]}_utt{df.Utterance_ID[k]}" for k in skipped],
            "exact_matches": kind.count("exact"),
            "fuzzy_matches": kind.count("fuzzy"),
            "min_cosine": round(float(cos.min()), 4),
            "label_agreement_with_csv": round(label_agree, 4),
            "rows_with_non_ascii_text": int(non_ascii.sum()),
            "winning_decoding_on_non_ascii_rows": dict(Counter(np.array(var)[non_ascii].tolist())),
        }
        meta[s] = pd.DataFrame({
            "dialogue_id": aligned["Dialogue_ID"].values,
            "utterance_id": aligned["Utterance_ID"].values,
            "speaker": aligned["Speaker"].values,
            "utterance": aligned["Utterance"].values,
            "emotion": aligned["Emotion"].values,
            "duration_s": aligned["duration_s"].values,
        })
    report["csv_alignment"] = align_report

    # --- old standalone tensors vs bundle (documents why we don't use them)
    legacy = {}
    for s in ["dev", "test"]:
        path = Path(args.legacy_dir) / f"{s}_tensors.pt"
        if not path.exists():
            continue
        L = torch.load(path, map_location="cpu", weights_only=False)
        same_n = L["X"].shape[0] == data[s]["audio"].shape[0]
        entry = {"rows": int(L["X"].shape[0]), "same_row_count": same_n}
        if same_n:
            rowcos = (unit(L["X"].float()) * unit(data[s]["audio"])).sum(1)
            entry["rowwise_cosine_mean"] = round(float(rowcos.mean()), 4)
            entry["identical"] = bool(torch.allclose(L["X"].float(), data[s]["audio"]))
            entry["string_labels_agree"] = round(float(np.mean(
                [EMOTIONS.index(lbl) == int(y) for lbl, y in zip(L["Y"], data[s]["y"])])), 4)
        legacy[s] = entry
    report["legacy_standalone_vs_bundle"] = legacy

    # --- save
    out = {
        "emotions": EMOTIONS,
        "splits": data,
        "meta": {s: m.to_dict(orient="list") for s, m in meta.items()},
        "legacy_probs": {s: {"text": B[f"text_probs_{s}"], "audio": B[f"audio_probs_{s}"]}
                         for s in ["gate", "dev", "test"]},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))

    print("\n" + json.dumps(report, indent=2, default=str))
    print(f"\nSaved {args.out} and {args.report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/legacy/fusion_bundle_final.pt")
    ap.add_argument("--csv-dir", default="data/meld_csv")
    ap.add_argument("--legacy-dir", default="data/legacy")
    ap.add_argument("--out", default="data/dataset.pt")
    ap.add_argument("--report", default="results/audit_step1.json")
    main(ap.parse_args())
