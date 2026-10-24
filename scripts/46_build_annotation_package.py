"""Step 46: build the Stage A annotation package for the emotion-conflict experiment (protocol: docs/conflict_experiment_protocol.md).

Written and run BEFORE any model output exists for the constructed pairs; no model, embedding or prediction is read here. Only MELD labels, speaker
names, durations and audio loudness are used. The package is a local build under data/annotation_package_stageA/ (MELD text and audio are not
redistributed through git); the committed pieces are this script, the instructions and the manifest of file hashes (docs/annotation_package_manifest.md).

Selection (fixed, seed 20261023; the DEV split only, so the confirmatory test split is not inspected):
  text items   : dev utterances, 3 or more words, emotion in {joy, anger, sadness} (coarse polarity: joy positive; anger, sadness negative),
                 stratified 17 / 17 / 16 text items. Neutral, surprise, fear, disgust text items are excluded from Stage A.
  conflict audio j : a dev utterance of OPPOSITE polarity to the text item's label (text positive -> audio emotion in {anger, sadness, fear, disgust};
                 text negative -> audio emotion joy), different dialogue from i, 1 or more seconds long.
  control audio k  : a dev utterance with the SAME emotion label as the text item, different dialogue, j != k, k != i.
  matching     : among 150 randomly drawn eligible candidates per side, choose the (j, k) minimising
                 |ln dur_j - ln dur_k| + |RMS_dB_j - RMS_dB_k| / 6 + 0.5 [speaker_j != speaker_k]  (speaker identity and loudness are matched between j and k;
                 audio items are used at most once in the whole package).
  output       : 50 conflict pairs and 50 matched control pairs = 100 pairs; 50 text items; 100 distinct audio items.
Labels are hidden from annotators; the key (labels, condition, matching statistics) is written to a separate file and must not be opened by annotators.

Version 2 (2026-10-24, protocol Amendment 1, made BEFORE any annotation): version 1 leaked the experimental condition through the ORDER of the
master item files (items_pairs.csv listed each text's conflict pair first, items_audio.csv listed all conflict audios first) and derived item ids
from the public seed. Version 2 shuffles every master file, draws ids from a private random salt kept only in KEY_private/, and writes one
self-contained folder per reviewer under distribution/ (no key, no statistics, no other reviewer's sheets). The selection of text items and audio
clips (seed 20261023) is unchanged from version 1.
Usage: python scripts/46_build_annotation_package.py [--salt HEX]     (without --salt a fresh random salt is drawn and stored in KEY_private/salt.txt)
"""
import argparse
import hashlib
import json
import secrets
import shutil
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "annotation_package_stageA"
SEED = 20261023
POS, NEG = {"joy"}, {"anger", "sadness", "fear", "disgust"}


def audio_stats(path):
    with wave.open(str(path), "rb") as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if sw == 1:
        x = x - 128.0
    x = x / float(2 ** (8 * sw - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    return n / sr, 20 * np.log10(rms + 1e-12)


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA / "meld_csv" / "dev_sent_emo.csv")
    df["wav"] = [DATA / "all_data" / "wavfiles_dev" / f"dia{d}_utt{u}.wav" for d, u in zip(df.Dialogue_ID, df.Utterance_ID)]
    df = df[df.wav.map(Path.exists)].reset_index(drop=True)
    st = [audio_stats(p) for p in df.wav]
    df["dur"], df["rms_db"] = [s[0] for s in st], [s[1] for s in st]
    df["nwords"] = df.Utterance.str.split().str.len()
    df["speaker"] = df.Speaker.str.strip().str.lower()
    cand_text = df[(df.nwords >= 3) & (df.Emotion.isin(["joy", "anger", "sadness"])) & (df.dur >= 1.0)]
    quota = {"joy": 17, "anger": 17, "sadness": 16}
    texts = pd.concat([cand_text[cand_text.Emotion == e].sample(n=q, random_state=int(rng.integers(1 << 31))) for e, q in quota.items()])
    texts = texts.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    used_audio = set(texts.index)          # a text item's own audio is never used as a donor audio
    pairs = []
    for i, r in texts.iterrows():
        opp = NEG if r.Emotion in POS else POS
        eligible = df[(df.Dialogue_ID != r.Dialogue_ID) & (df.dur >= 1.0) & (~df.index.isin(used_audio))]
        J = eligible[eligible.Emotion.isin(opp)]
        K = eligible[eligible.Emotion == r.Emotion]
        Jc = J.sample(n=min(150, len(J)), random_state=int(rng.integers(1 << 31)))
        Kc = K.sample(n=min(150, len(K)), random_state=int(rng.integers(1 << 31)))
        best = None
        for j, rj in Jc.iterrows():
            for k, rk in Kc.iterrows():
                if j == k:
                    continue
                cost = abs(np.log(rj.dur) - np.log(rk.dur)) + abs(rj.rms_db - rk.rms_db) / 6 + 0.5 * (rj.speaker != rk.speaker)
                if best is None or cost < best[0]:
                    best = (cost, j, k)
        _, j, k = best
        used_audio.update([j, k])
        pairs.append({"text_row": i, "conflict_audio_row": j, "control_audio_row": k, "match_cost": best[0]})
    P = pd.DataFrame(pairs)

    # ---- anonymise (ids from a private salt, never from the public seed or the condition)
    ap = argparse.ArgumentParser()
    ap.add_argument("--salt", default=None)
    args = ap.parse_args()
    salt = args.salt or secrets.token_hex(8)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "shared" / "audio").mkdir(parents=True)
    (OUT / "KEY_private").mkdir()
    hexid = lambda tag, n: [hashlib.sha1(f"{salt}-{tag}-{x}".encode()).hexdigest()[:8] for x in n]
    text_ids = dict(zip(P.text_row, hexid("T", P.text_row)))
    audio_rows = list(P.conflict_audio_row) + list(P.control_audio_row)
    audio_ids = dict(zip(audio_rows, hexid("A", audio_rows)))
    assert len(set(audio_ids.values())) == len(audio_rows) and len(set(text_ids.values())) == len(text_ids)
    shuf = lambda d, tag: d.sample(frac=1, random_state=int(hashlib.sha1(f"{salt}-shuffle-{tag}".encode()).hexdigest()[:8], 16)).reset_index(drop=True)
    items_text = shuf(pd.DataFrame({"text_item_id": [text_ids[i] for i in P.text_row], "text": [df.loc[i, "Utterance"] for i in P.text_row]}), "text")
    items_audio = shuf(pd.DataFrame({"audio_item_id": [audio_ids[a] for a in audio_rows], "file": [f"audio/{audio_ids[a]}.wav" for a in audio_rows]}), "audio")
    for a in audio_rows:
        shutil.copy(df.loc[a, "wav"], OUT / "shared" / "audio" / f"{audio_ids[a]}.wav")
    pr = []
    for _, r in P.iterrows():
        for cond, col in (("conflict", "conflict_audio_row"), ("control", "control_audio_row")):
            pr.append({"pair_id": hashlib.sha1(f"{salt}-P-{r.text_row}-{cond}".encode()).hexdigest()[:8], "text_item_id": text_ids[r.text_row], "audio_item_id": audio_ids[r[col]],
                       "_cond": cond, "_text_label": df.loc[r.text_row, "Emotion"], "_audio_label": df.loc[r[col], "Emotion"], "_text_dur": df.loc[r.text_row, "dur"],
                       "_audio_dur": df.loc[r[col], "dur"], "_audio_rms_db": df.loc[r[col], "rms_db"], "_audio_speaker": df.loc[r[col], "speaker"],
                       "_text_speaker": df.loc[r.text_row, "speaker"], "_text_dialogue": int(df.loc[r.text_row, "Dialogue_ID"]), "_text_utt": int(df.loc[r.text_row, "Utterance_ID"]),
                       "_audio_dialogue": int(df.loc[r[col], "Dialogue_ID"]), "_audio_utt": int(df.loc[r[col], "Utterance_ID"])})
    PR = pd.DataFrame(pr)
    assert PR.pair_id.is_unique
    K = OUT / "KEY_private"
    PR[[c for c in PR.columns if c.startswith("_") or c in ("pair_id", "text_item_id", "audio_item_id")]].to_csv(K / "KEY_do_not_open_during_annotation.csv", index=False)
    (K / "salt.txt").write_text(salt + "\n")
    items_pairs = shuf(PR[["pair_id", "text_item_id", "audio_item_id"]], "pairs")
    shared = OUT / "shared"
    items_text.to_csv(shared / "items_text.csv", index=False)
    items_audio.to_csv(shared / "items_audio.csv", index=False)
    items_pairs.to_csv(shared / "items_pairs.csv", index=False)
    labels = "anger|disgust|fear|joy|neutral|sadness|surprise|cannot_tell"
    (shared / "ALLOWED_VALUES.txt").write_text(f"emotion fields (text_emotion, tone_emotion): {labels}\nrelation: conflict | compatible | unclear\nconfidence: integer 1 (guess) to 5 (certain)\n")
    # one self-contained folder per reviewer; each gets her own random order, identical underlying items
    for n, ann in ((1, "annotator_1"), (2, "annotator_2"), (3, "annotator_3")):
        arng = np.random.default_rng(int(hashlib.sha1(f"{salt}-{ann}".encode()).hexdigest()[:8], 16))
        d = OUT / "distribution" / ann
        shutil.copytree(shared, d)
        shutil.copy(ROOT / "docs" / "annotation_instructions.md", d / "annotation_instructions.md")
        if (ROOT / "docs" / "annotation_reviewer_quickstart.md").exists():
            shutil.copy(ROOT / "docs" / "annotation_reviewer_quickstart.md", d / "READ_ME_FIRST_quickstart.md")
        items_text.sample(frac=1, random_state=int(arng.integers(1 << 31)))[["text_item_id"]].assign(text_emotion="", text_confidence_1to5="", notes="").to_csv(d / f"{ann}_task1_text.csv", index=False)
        items_audio.sample(frac=1, random_state=int(arng.integers(1 << 31)))[["audio_item_id"]].assign(tone_emotion="", tone_confidence_1to5="", notes="").to_csv(d / f"{ann}_task2_audio.csv", index=False)
        items_pairs.sample(frac=1, random_state=int(arng.integers(1 << 31)))[["pair_id"]].assign(relation="", confidence_1to5="", notes="").to_csv(d / f"{ann}_task3_pairs.csv", index=False)
        shutil.make_archive(str(OUT / "distribution" / ann), "zip", d)
    s = {}
    for c in ("conflict", "control"):
        x = PR[PR._cond == c]
        s[c] = {"n": int(len(x)), "audio_dur_mean": float(x._audio_dur.mean()), "audio_rms_db_mean": float(x._audio_rms_db.mean()),
                "share_audio_speaker_equals_text_speaker": float((x._audio_speaker == x._text_speaker).mean())}
    s["smd_duration"] = float((s["conflict"]["audio_dur_mean"] - s["control"]["audio_dur_mean"]) / PR._audio_dur.std())
    s["smd_rms_db"] = float((s["conflict"]["audio_rms_db_mean"] - s["control"]["audio_rms_db_mean"]) / PR._audio_rms_db.std())
    s["seed"] = SEED
    (K / "package_stats.json").write_text(json.dumps(s, indent=1))
    h = {f"shared/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(shared.glob("*.csv"))}
    h.update({f"shared/audio/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((shared / "audio").glob("*.wav"))})
    zips = {f"distribution/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((OUT / "distribution").glob("*.zip"))}
    keyhash = hashlib.sha256((K / "KEY_do_not_open_during_annotation.csv").read_bytes()).hexdigest()[:16]
    lines = ["# Stage A annotation package manifest (version 2, 2026-10-24)",
             f"Built by `scripts/46_build_annotation_package.py` (selection seed {SEED}; private id salt NOT recorded here). Local build: `data/annotation_package_stageA/` (not committed; MELD text and audio are not redistributed).",
             "Version 1 (2026-10-23) was never distributed; see Amendment 1 in `docs/conflict_experiment_protocol.md`.", "",
             "Distribute ONLY `distribution/annotator_N.zip` (one per reviewer). Never distribute `KEY_private/` or `shared/` outside the zips.", "",
             f"**Key file hash recorded before annotation (protocol section 5.5):** `KEY_private/KEY_do_not_open_during_annotation.csv` sha256[:16] = `{keyhash}`", "",
             "## Per-reviewer zips", "| file | sha256[:16] |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in zips.items()]
    lines += ["", "## Shared items (identical in all three zips)", "| file | sha256[:16] |", "|---|---|"] + [f"| {k} | {v} |" for k, v in h.items()]
    lines += ["", "Matching statistics and the key are in `KEY_private/` (not distributed, not listed here)."]
    (ROOT / "docs" / "annotation_package_manifest.md").write_text("\n".join(lines) + "\n")
    print(len(items_text), "text items;", len(items_audio), "audio items;", len(PR), "pairs; zips:", list(zips))


if __name__ == "__main__":
    main()
