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
Usage: python scripts/46_build_annotation_package.py
"""
import hashlib
import json
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

    # ---- anonymise
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "audio").mkdir(parents=True)
    hexid = lambda tag, n: [hashlib.sha1(f"{SEED}-{tag}-{x}".encode()).hexdigest()[:8] for x in n]
    text_ids = dict(zip(P.text_row, hexid("T", P.text_row)))
    audio_rows = list(P.conflict_audio_row) + list(P.control_audio_row)
    audio_ids = dict(zip(audio_rows, hexid("A", audio_rows)))
    items_text = pd.DataFrame({"text_item_id": [text_ids[i] for i in P.text_row], "text": [df.loc[i, "Utterance"] for i in P.text_row]})
    items_audio = pd.DataFrame({"audio_item_id": [audio_ids[a] for a in audio_rows], "file": [f"audio/{audio_ids[a]}.wav" for a in audio_rows]})
    for a in audio_rows:
        shutil.copy(df.loc[a, "wav"], OUT / "audio" / f"{audio_ids[a]}.wav")
    pr = []
    for _, r in P.iterrows():
        for cond, col in (("conflict", "conflict_audio_row"), ("control", "control_audio_row")):
            pr.append({"pair_id": hashlib.sha1(f"{SEED}-P-{r.text_row}-{cond}".encode()).hexdigest()[:8], "text_item_id": text_ids[r.text_row], "audio_item_id": audio_ids[r[col]],
                       "_cond": cond, "_text_label": df.loc[r.text_row, "Emotion"], "_audio_label": df.loc[r[col], "Emotion"], "_text_dur": df.loc[r.text_row, "dur"],
                       "_audio_dur": df.loc[r[col], "dur"], "_audio_rms_db": df.loc[r[col], "rms_db"], "_audio_speaker": df.loc[r[col], "speaker"],
                       "_text_speaker": df.loc[r.text_row, "speaker"], "_text_dialogue": int(df.loc[r.text_row, "Dialogue_ID"]), "_text_utt": int(df.loc[r.text_row, "Utterance_ID"]),
                       "_audio_dialogue": int(df.loc[r[col], "Dialogue_ID"]), "_audio_utt": int(df.loc[r[col], "Utterance_ID"])})
    PR = pd.DataFrame(pr)
    key = PR[[c for c in PR.columns if c.startswith("_") or c in ("pair_id", "text_item_id", "audio_item_id")]]
    key.to_csv(OUT / "KEY_do_not_open_during_annotation.csv", index=False)
    items_text.to_csv(OUT / "items_text.csv", index=False)
    items_audio.to_csv(OUT / "items_audio.csv", index=False)
    PR[["pair_id", "text_item_id", "audio_item_id"]].to_csv(OUT / "items_pairs.csv", index=False)
    # annotator sheets: each annotator gets her own random order, three tasks in this order: T (text only), A (audio tone only), P (pair)
    labels = "anger|disgust|fear|joy|neutral|sadness|surprise|cannot_tell"
    for ann in ("annotator_1", "annotator_2", "annotator_3_adjudicator"):
        arng = np.random.default_rng(int(hashlib.sha1(f"{SEED}-{ann}".encode()).hexdigest()[:8], 16))
        t = items_text.sample(frac=1, random_state=int(arng.integers(1 << 31)))[["text_item_id"]].assign(text_emotion="", text_confidence_1to5="", notes="")
        a = items_audio.sample(frac=1, random_state=int(arng.integers(1 << 31)))[["audio_item_id"]].assign(tone_emotion="", tone_confidence_1to5="", notes="")
        p = PR[["pair_id"]].sample(frac=1, random_state=int(arng.integers(1 << 31))).assign(relation="", notes="")
        t.to_csv(OUT / f"{ann}_task1_text.csv", index=False)
        a.to_csv(OUT / f"{ann}_task2_audio.csv", index=False)
        p.to_csv(OUT / f"{ann}_task3_pairs.csv", index=False)
    (OUT / "ALLOWED_VALUES.txt").write_text(f"emotion fields: {labels}\nrelation: conflict | compatible | unclear\nconfidence: 1 (guess) to 5 (certain)\n")
    # package statistics for the matched design (conflict vs control)
    s = {}
    for c in ("conflict", "control"):
        x = PR[PR._cond == c]
        s[c] = {"n": int(len(x)), "audio_dur_mean": float(x._audio_dur.mean()), "audio_rms_db_mean": float(x._audio_rms_db.mean()),
                "share_audio_speaker_equals_text_speaker": float((x._audio_speaker == x._text_speaker).mean())}
    s["smd_duration"] = float((s["conflict"]["audio_dur_mean"] - s["control"]["audio_dur_mean"]) / PR._audio_dur.std())
    s["smd_rms_db"] = float((s["conflict"]["audio_rms_db_mean"] - s["control"]["audio_rms_db_mean"]) / PR._audio_rms_db.std())
    s["seed"] = SEED
    (OUT / "package_stats.json").write_text(json.dumps(s, indent=1))
    h = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.json"))}
    h.update({f"audio/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((OUT / "audio").glob("*.wav"))})
    lines = ["# Stage A annotation package manifest", f"Built by `scripts/46_build_annotation_package.py`, seed {SEED}. Local build: `data/annotation_package_stageA/` (not committed; MELD text and audio are not redistributed).",
             "", "## Matching statistics (conflict vs control audio)", "```", json.dumps(s, indent=1), "```", "", "## File hashes (sha256, first 16 hex)", "| file | hash |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in h.items()]
    (ROOT / "docs" / "annotation_package_manifest.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(s, indent=1))
    print(len(items_text), "text items;", len(items_audio), "audio items;", len(PR), "pairs")


if __name__ == "__main__":
    main()
