"""Diagnostic (item 4): what do the heads say about the MEAN embedding, and does that
confound the mean-replacement effects?

Question: mean replacement swaps a modality for the clf-split mean embedding. If a head's
output on that mean embedding already favours class m, then removing that modality barely
moves the fused probability for utterances predicted as m, regardless of how much the
modality mattered. This script checks it. It changes no protocol definition.

Part A  per seed: the text head's and audio head's probabilities on the mean embedding
        -> results/mean_embedding_head_outputs.csv
Part B  learned-gate model, mean method, grouped by the model's predicted class:
        n, mean |dp_text|, mean |dp_audio|, agreement, always-text baseline, gain, and
        whether the predicted class equals the text / audio head's mean-embedding class
        -> results/mean_embedding_by_pred_class.csv

Usage: python scripts/diag_mean_embedding.py   (needs checkpoints/ and results/interventions_log.csv)
"""
import importlib
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_dataset  # noqa: E402

load_models = importlib.import_module("04_interventions").load_models
SEEDS = [0, 1, 2]


def main():
    D = load_dataset()
    emotions = D["emotions"]
    clf = D["splits"]["clf"]
    mean_t = clf["text"].mean(0, keepdim=True)
    mean_a = clf["audio"].mean(0, keepdim=True)

    rows, mean_class = [], {}
    for seed in SEEDS:
        text_head, audio_head, _ = load_models(seed)
        with torch.no_grad():
            pt = torch.softmax(text_head(mean_t), 1)[0]
            pa = torch.softmax(audio_head(mean_a), 1)[0]
        mean_class[seed] = (int(pt.argmax()), int(pa.argmax()))
        for head, p in (("text", pt), ("audio", pa)):
            rows.append({"seed": seed, "head": head, "argmax_class": emotions[int(p.argmax())],
                         "max_prob": float(p.max()), **{e: float(p[i]) for i, e in enumerate(emotions)}})
    heads = pd.DataFrame(rows)
    heads.to_csv(RESULTS / "mean_embedding_head_outputs.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    print("A. Head output on the mean embedding")
    print(heads)

    log = pd.read_csv(RESULTS / "interventions_log.csv")
    d = log[(log.model == "fusion_learned_gate") & (log.method == "mean")].copy()
    d["abs_text"], d["abs_audio"] = d.dp_text.abs(), d.dp_audio.abs()
    d["e"], d["s"] = d.abs_text - d.abs_audio, 2 * d.gate_w_t - 1
    d["agree"] = (d.s > 0) == (d.e > 0)
    d["text_wins"] = d.e > 0
    d["pred_is_text_mean_class"] = [p == mean_class[s][0] for s, p in zip(d.seed, d.pred)]
    d["pred_is_audio_mean_class"] = [p == mean_class[s][1] for s, p in zip(d.seed, d.pred)]
    d["pred_name"] = [emotions[p] for p in d.pred]
    g = d.groupby("pred_name").agg(
        n=("row", "size"), mean_abs_dp_text=("abs_text", "mean"), mean_abs_dp_audio=("abs_audio", "mean"),
        agreement=("agree", "mean"), always_text_baseline=("text_wins", "mean"),
        share_pred_is_text_mean_class=("pred_is_text_mean_class", "mean"),
        share_pred_is_audio_mean_class=("pred_is_audio_mean_class", "mean"))
    g["gain"] = g.agreement - g.always_text_baseline
    g = g.sort_values("n", ascending=False)
    g.to_csv(RESULTS / "mean_embedding_by_pred_class.csv")
    print("\nB. Gate model, mean replacement, by predicted class (pooled over 3 seeds)")
    print(g)
    # direct test: effect when the prediction equals the head's mean-embedding class vs not
    for head in ("text", "audio"):
        col = f"pred_is_{head}_mean_class"
        print(f"\nmean |dp_{head}| when pred == {head} head's mean-embedding class vs not:")
        print(d.groupby(col)[f"abs_{head}"].agg(["size", "mean"]))


if __name__ == "__main__":
    main()
