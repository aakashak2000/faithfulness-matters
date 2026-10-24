"""Step 47: validate a reviewer's completed annotation sheets. Checks only completeness, allowed values and id alignment.

It never opens the hidden key (any path containing 'KEY' is refused), never reads model outputs, and does not compute agreement or adjudicate.
Checks per sheet: expected columns (exact), id set equals the id set of the shared item files, duplicate ids, missing or blank answers, allowed values,
confidence integer 1 to 5, referential integrity of items_pairs.csv, existence of every audio file. Optional --compare folders (other reviewers'
returned folders) checks that all reviewers cover exactly the same ids.
Usage: python scripts/47_validate_annotation_sheets.py --folder PATH --annotator N [--compare PATH2 PATH3] ;  python scripts/47_validate_annotation_sheets.py --selftest
Exit code 0 only when every check passes.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

EMO = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise", "cannot_tell"}
REL = {"conflict", "compatible", "unclear"}
SHEETS = {
    "task1_text": (["text_item_id", "text_emotion", "text_confidence_1to5", "notes"], "text_item_id", "text_emotion", EMO, "text_confidence_1to5", "items_text.csv"),
    "task2_audio": (["audio_item_id", "tone_emotion", "tone_confidence_1to5", "notes"], "audio_item_id", "tone_emotion", EMO, "tone_confidence_1to5", "items_audio.csv"),
    "task3_pairs": (["pair_id", "relation", "confidence_1to5", "notes"], "pair_id", "relation", REL, "confidence_1to5", "items_pairs.csv"),
}


def read(path):
    if "KEY" in str(path).upper().replace("MONKEY", ""):
        raise SystemExit(f"refusing to open a path containing KEY: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def check_sheet(folder, ann, name, problems):
    cols, idcol, ans, allowed, conf, itemfile = SHEETS[name]
    f = folder / f"annotator_{ann}_{name}.csv"
    if not f.exists():
        problems.append(f"{name}: sheet file missing ({f.name})")
        return None
    d = read(f)
    if list(d.columns) != cols:
        problems.append(f"{name}: columns are {list(d.columns)}, expected {cols}")
        return d
    items = read(folder / itemfile)
    expected = set(items[idcol])
    ids = d[idcol].str.strip()
    dup = ids[ids.duplicated()].unique().tolist()
    if dup:
        problems.append(f"{name}: duplicate ids {dup[:5]}{' ...' if len(dup) > 5 else ''}")
    miss, extra = sorted(expected - set(ids)), sorted(set(ids) - expected)
    if miss:
        problems.append(f"{name}: {len(miss)} item ids missing from the sheet, e.g. {miss[:3]}")
    if extra:
        problems.append(f"{name}: {len(extra)} ids not in the item list, e.g. {extra[:3]}")
    blank = d[d[ans].str.strip() == ""]
    if len(blank):
        problems.append(f"{name}: {len(blank)} rows with no answer in '{ans}'")
    bad = d[(d[ans].str.strip() != "") & (~d[ans].str.strip().isin(allowed))]
    if len(bad):
        problems.append(f"{name}: {len(bad)} invalid values in '{ans}', e.g. {sorted(set(bad[ans]))[:4]} (allowed: {sorted(allowed)})")
    cv = d[conf].str.strip()
    nb = d[cv == ""]
    if len(nb):
        problems.append(f"{name}: {len(nb)} rows with no confidence")
    bad_c = d[(cv != "") & (~cv.isin(["1", "2", "3", "4", "5"]))]
    if len(bad_c):
        problems.append(f"{name}: {len(bad_c)} confidence values not an integer 1 to 5, e.g. {sorted(set(bad_c[conf]))[:4]}")
    return d


def check_items(folder, problems):
    t, a, p = (read(folder / x) for x in ("items_text.csv", "items_audio.csv", "items_pairs.csv"))
    for nm, df, col in (("items_text", t, "text_item_id"), ("items_audio", a, "audio_item_id"), ("items_pairs", p, "pair_id")):
        if df[col].duplicated().any():
            problems.append(f"{nm}: duplicate ids")
    if not set(p.text_item_id) <= set(t.text_item_id):
        problems.append("items_pairs: some text ids are not in items_text")
    if not set(p.audio_item_id) <= set(a.audio_item_id):
        problems.append("items_pairs: some audio ids are not in items_audio")
    for f in a.file:
        if not (folder / f).exists():
            problems.append(f"audio file missing: {f}")
            break


def run(folder, ann, compare=()):
    folder = Path(folder)
    problems = []
    check_items(folder, problems)
    ids = {}
    for name in SHEETS:
        d = check_sheet(folder, ann, name, problems)
        if d is not None and SHEETS[name][1] in d.columns:
            ids[name] = set(d[SHEETS[name][1]].str.strip())
    for other in compare:
        other = Path(other)
        for k in sorted(other.glob("annotator_*_task*.csv")):
            name = next((n for n in SHEETS if n in k.name), None)
            if name is None:
                continue
            o = set(read(k)[SHEETS[name][1]].str.strip())
            if name in ids and o != ids[name]:
                problems.append(f"alignment: {k.parent.name}/{k.name} covers a different id set ({len(o ^ ids[name])} differing ids)")
    print(f"annotator_{ann}: {'ALL CHECKS PASSED' if not problems else str(len(problems)) + ' problem(s)'}")
    for x in problems:
        print("  -", x)
    return not problems


def selftest():
    import shutil
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "audio").mkdir()
        pd.DataFrame({"text_item_id": ["t1", "t2"], "text": ["a", "b"]}).to_csv(td / "items_text.csv", index=False)
        pd.DataFrame({"audio_item_id": ["a1", "a2", "a3"], "file": [f"audio/a{i}.wav" for i in (1, 2, 3)]}).to_csv(td / "items_audio.csv", index=False)
        for i in (1, 2, 3):
            (td / "audio" / f"a{i}.wav").write_bytes(b"x")
        pd.DataFrame({"pair_id": ["p1", "p2"], "text_item_id": ["t1", "t2"], "audio_item_id": ["a1", "a2"]}).to_csv(td / "items_pairs.csv", index=False)
        good = {"task1_text": pd.DataFrame({"text_item_id": ["t1", "t2"], "text_emotion": ["joy", "cannot_tell"], "text_confidence_1to5": ["4", "1"], "notes": ["", ""]}),
                "task2_audio": pd.DataFrame({"audio_item_id": ["a1", "a2", "a3"], "tone_emotion": ["anger", "neutral", "sadness"], "tone_confidence_1to5": ["5", "3", "2"], "notes": ["", "", ""]}),
                "task3_pairs": pd.DataFrame({"pair_id": ["p1", "p2"], "relation": ["conflict", "unclear"], "confidence_1to5": ["3", "2"], "notes": ["", ""]})}
        for k, v in good.items():
            v.to_csv(td / f"annotator_1_{k}.csv", index=False)
        assert run(td, 1), "valid sheets should pass"
        cases = {"missing answer": ("task1_text", lambda d: d.assign(text_emotion=["joy", ""])),
                 "invalid value": ("task2_audio", lambda d: d.assign(tone_emotion=["anger", "angry", "sadness"])),
                 "duplicate id": ("task3_pairs", lambda d: d.assign(pair_id=["p1", "p1"])),
                 "bad confidence": ("task1_text", lambda d: d.assign(text_confidence_1to5=["4", "7"])),
                 "missing id": ("task2_audio", lambda d: d.iloc[:2]),
                 "extra id": ("task1_text", lambda d: pd.concat([d, pd.DataFrame({"text_item_id": ["t9"], "text_emotion": ["joy"], "text_confidence_1to5": ["1"], "notes": [""]})])),
                 "wrong columns": ("task3_pairs", lambda d: d.rename(columns={"relation": "label"}))}
        for nm, (sheet, fn) in cases.items():
            for k, v in good.items():
                v.to_csv(td / f"annotator_1_{k}.csv", index=False)
            fn(good[sheet]).to_csv(td / f"annotator_1_{sheet}.csv", index=False)
            print(f"[selftest case: {nm}]")
            assert not run(td, 1), f"case '{nm}' should fail"
        # alignment between reviewers
        other = td / "other"
        shutil.copytree(td, other)
        for k, v in good.items():
            v.to_csv(td / f"annotator_1_{k}.csv", index=False)
        good["task2_audio"].iloc[:2].to_csv(other / "annotator_2_task2_audio.csv", index=False)
        print("[selftest case: reviewer alignment]")
        assert not run(td, 1, compare=[other])
        try:
            read(td / "KEY_do_not_open.csv")
            raise AssertionError("key path must be refused")
        except SystemExit:
            pass
    print("selftest passed: 1 valid case, 7 failing cases, alignment and key refusal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder")
    ap.add_argument("--annotator", type=int)
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        sys.exit(0 if run(a.folder, a.annotator, a.compare) else 1)
