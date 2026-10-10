"""Diagnostic: are the duplicate-audio rows duplicates in the SOURCE files?

Hashes the original audio/video file for every dev/test row whose cached audio
embedding is shared with another row, then groups them.
  identical file hashes   -> MELD itself ships copied clips (dataset issue)
  different file hashes   -> files differ, so the collision came from our extraction

Usage:
  python scripts/diag_source_files.py <folder with diaX_uttY.wav or .mp4 files> [split]
  e.g. python scripts/diag_source_files.py ~/MELD.Raw/output_repeated_splits_test test
"""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from common import load_dataset


def file_hash(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_file(folder, dia, utt):
    for ext in (".wav", ".mp4"):
        p = folder / f"dia{dia}_utt{utt}{ext}"
        if p.exists():
            return p
    return None


def main(folder, split):
    folder = Path(folder).expanduser()
    D = load_dataset()
    meta = D["meta"][split]
    audio = D["splits"][split]["audio"]

    by_vector = defaultdict(list)
    for i in meta.index[meta["audio_shared"]]:
        by_vector[audio[i].numpy().tobytes()].append(i)

    identical, differ, missing = 0, 0, 0
    for rows in by_vector.values():
        if len(rows) < 2:      # shared only with a train row; nothing to compare here
            continue
        hashes = {}
        for i in rows:
            m = meta.loc[i]
            p = find_file(folder, m.dialogue_id, m.utterance_id)
            hashes[i] = (file_hash(p)[:12], p.stat().st_size) if p else None
        if any(v is None for v in hashes.values()):
            missing += 1
            verdict = "FILE MISSING"
        elif len({v[0] for v in hashes.values()}) == 1:
            identical += 1
            verdict = "SOURCE FILES IDENTICAL"
        else:
            differ += 1
            verdict = "SOURCE FILES DIFFER"
        print(f"--- {verdict}")
        for i in rows:
            m = meta.loc[i]
            print(f"    row {i:<5} dia{m.dialogue_id}_utt{m.utterance_id:<3} csv {m.duration_s:5.2f}s  "
                  f"file {hashes[i]}  {str(m.utterance)[:40]!r}")

    print(f"\nSummary ({split}): {identical} groups with identical source files, "
          f"{differ} groups with different source files, {missing} with missing files")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "test")
