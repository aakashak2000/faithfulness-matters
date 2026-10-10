"""Step 2: train the unimodal heads and the fusion models, 3 seeds.

For each seed:
  - text head and audio head on the clf split (class-weighted CE, early stop on dev)
  - learned gate on the held-out gate split, using the heads' out-of-sample probs
  - evaluate text-only, audio-only, average, confidence and learned-gate fusion on test
Also scores the legacy cached probabilities, to check we reproduce the old numbers.

Outputs
  checkpoints/seed{N}.pt           head + gate weights
  results/baselines_per_seed.csv   every model x seed
  results/baselines_summary.csv    mean +- std over seeds
  results/per_class_f1.csv         per-emotion F1, mean over seeds
  results/legacy_reproduction.csv  old cached probs scored with the same metrics

Usage: python scripts/02_train.py
"""
import pandas as pd
import torch

from common import (CKPT, RESULTS, class_weights, fuse_average, fuse_confidence, fuse_gate,
                    load_dataset, metrics, per_class_f1, set_determinism, train_gate, train_head)

SEEDS = [0, 1, 2]


def main():
    D = load_dataset()
    E, S = D["emotions"], D["splits"]
    clf, gate_s, dev, test = S["clf"], S["gate"], S["dev"], S["test"]
    w_clf = class_weights(clf["y"])
    w_gate = class_weights(gate_s["y"])
    CKPT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    rows, class_rows = [], []
    for seed in SEEDS:
        set_determinism(seed)
        text_head = train_head(clf["text"], clf["y"], dev["text"], dev["y"], w_clf, seed)
        audio_head = train_head(clf["audio"], clf["y"], dev["audio"], dev["y"], w_clf, seed)

        with torch.no_grad():
            probs = {s: (torch.softmax(text_head(S[s]["text"]), 1),
                         torch.softmax(audio_head(S[s]["audio"]), 1))
                     for s in ["gate", "dev", "test"]}

        gate = train_gate(gate_s["text"], gate_s["audio"], *probs["gate"], gate_s["y"],
                          (dev["text"], dev["audio"], *probs["dev"], dev["y"]), w_gate, seed)

        with torch.no_grad():
            p_t, p_a = probs["test"]
            outputs = {
                "text_only": p_t,
                "audio_only": p_a,
                "fusion_average": fuse_average(p_t, p_a)[0],
                "fusion_confidence": fuse_confidence(p_t, p_a)[0],
            }
            p_gate, w_gate_test = fuse_gate(gate, test["text"], test["audio"], p_t, p_a)
            outputs["fusion_learned_gate"] = p_gate

        for name, p in outputs.items():
            rows.append({"model": name, "seed": seed, **metrics(test["y"], p)})
            class_rows.append({"model": name, "seed": seed, **per_class_f1(test["y"], p, E)})

        torch.save({"text_head": text_head.state_dict(), "audio_head": audio_head.state_dict(),
                    "gate": gate.state_dict(), "seed": seed}, CKPT / f"seed{seed}.pt")
        print(f"seed {seed}: gate text-weight on test mean={w_gate_test.mean():.3f} "
              f"std={w_gate_test.std():.3f} min={w_gate_test.min():.3f} max={w_gate_test.max():.3f}")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS / "baselines_per_seed.csv", index=False)

    order = ["text_only", "audio_only", "fusion_average", "fusion_confidence", "fusion_learned_gate"]
    summary = per_seed.groupby("model")[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    summary = summary.loc[order]
    summary.to_csv(RESULTS / "baselines_summary.csv")

    per_class = pd.DataFrame(class_rows).groupby("model")[E].mean().loc[order]
    per_class.to_csv(RESULTS / "per_class_f1.csv")

    # legacy cached probabilities, scored with the same metrics
    L = D["legacy_probs"]["test"]
    legacy = {"text_only": L["text"], "audio_only": L["audio"],
              "fusion_average": fuse_average(L["text"], L["audio"])[0],
              "fusion_confidence": fuse_confidence(L["text"], L["audio"])[0]}
    legacy_df = pd.DataFrame([{"model": k, **metrics(test["y"], p)} for k, p in legacy.items()])
    legacy_df.to_csv(RESULTS / "legacy_reproduction.csv", index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.precision", 3)
    print("\n=== Test set, mean (std) over seeds", SEEDS, "===")
    print(summary)
    print("\n=== Per-class F1 (mean over seeds) ===")
    print(per_class)
    print("\n=== Legacy cached probabilities (reproduction check) ===")
    print(legacy_df.set_index("model"))
    print(f"\nSaved checkpoints to {CKPT} and tables to {RESULTS}")


if __name__ == "__main__":
    main()
