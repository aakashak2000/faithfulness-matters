"""Step 4: run the modality interventions and log everything per utterance.

INTERVENTION PROTOCOL (fixed before looking at any results)
-----------------------------------------------------------
Models tested (each recomputed with its OWN fusion rule after every intervention):
  fusion_learned_gate   w_t = gate(e_t, e_a)
  fusion_confidence     w_t = c_t / (c_t + c_a),  c = max class prob of each head
  fusion_average        w_t = 0.5 (no importance signal; reference only)

Importance signals (the model's "claim"), always computed on the ORIGINAL input:
  gate_w_t   learned gate's text weight            -> primary signal for the gate model
  conf_w_t   unimodal-confidence text weight       -> primary signal for the confidence model,
                                                      secondary (external) signal for the others
  importance_text = w_t, importance_audio = 1 - w_t, signal difference = 2 * w_t - 1

Interventions (one modality replaced, the other untouched, everything downstream recomputed):
  method "mean"  text  -> mean MiniLM embedding of the clf split
                 audio -> mean WavLM embedding of the clf split
  method "blank" text  -> MiniLM embedding of the empty string ""
                 audio -> WavLM embedding of an all-zero waveform of the clip's own length

Effects, relative to the ORIGINALLY predicted class c of that model:
  dp_text   = p_fused(text removed)[c]  - p_fused(original)[c]      (signed)
  dp_audio  = p_fused(audio removed)[c] - p_fused(original)[c]      (signed)
  effect    = |dp|                                                  (primary)
  flip      = predicted class changes
  tvd       = 0.5 * sum |p_removed - p_original|   (whole distribution, secondary)
  effect difference = |dp_text| - |dp_audio|

Metrics (computed in 05_faithfulness.py):
  ranking agreement: sign(2 w_t - 1) == sign(|dp_text| - |dp_audio|)
  near-zero rule: also report agreement excluding utterances with
                  | |dp_text| - |dp_audio| | < 0.02   (threshold fixed now)
  Spearman across utterances: (2 w_t - 1) vs (|dp_text| - |dp_audio|)
  every metric per seed, per method, full test set and without audio_shared rows

Output: results/interventions_log.csv  (one row per seed x utterance x model x method)
        data/intervention_probs.pt     full 7-class probabilities for every condition,
                                       keyed "seed|condition|model"
Usage:  python scripts/04_interventions.py
"""
import pandas as pd
import torch

from common import (CKPT, DATA, RESULTS, Head, LearnedGate, fuse_average, fuse_confidence,
                    fuse_gate, load_dataset)

SEEDS = [0, 1, 2]


def load_models(seed):
    ck = torch.load(CKPT / f"seed{seed}.pt", map_location="cpu", weights_only=False)
    text_head, audio_head, gate = Head(384), Head(1024), LearnedGate()
    text_head.load_state_dict(ck["text_head"])
    audio_head.load_state_dict(ck["audio_head"])
    gate.load_state_dict(ck["gate"])
    for m in (text_head, audio_head, gate):
        m.eval()
    return text_head, audio_head, gate


@torch.no_grad()
def run_all_models(e_t, e_a, text_head, audio_head, gate):
    """Full forward pass: returns {model: (fused probs, w_t)} plus the unimodal probs."""
    p_t = torch.softmax(text_head(e_t), 1)
    p_a = torch.softmax(audio_head(e_a), 1)
    return {
        "fusion_learned_gate": fuse_gate(gate, e_t, e_a, p_t, p_a),
        "fusion_confidence": fuse_confidence(p_t, p_a),
        "fusion_average": fuse_average(p_t, p_a),
    }, p_t, p_a


def effect(p_new, p_orig, c):
    idx = torch.arange(len(c))
    dp = p_new[idx, c] - p_orig[idx, c]
    flip = p_new.argmax(1) != c
    tvd = 0.5 * (p_new - p_orig).abs().sum(1)
    return dp, flip, tvd


def main():
    D = load_dataset()
    test, meta = D["splits"]["test"], D["meta"]["test"]
    clf = D["splits"]["clf"]
    iv = torch.load(DATA / "intervention_embeddings.pt", map_location="cpu", weights_only=False)["test"]
    n = test["y"].shape[0]
    assert iv["audio_silence"].shape[0] == n, "intervention embeddings do not match the test set"

    e_t, e_a = test["text"], test["audio"]
    replacements = {
        "mean": {"text": clf["text"].mean(0, keepdim=True).expand(n, -1),
                 "audio": clf["audio"].mean(0, keepdim=True).expand(n, -1)},
        "blank": {"text": iv["text_empty"][None].expand(n, -1),
                 "audio": iv["audio_silence"]},
    }

    frames, prob_store = [], {}
    for seed in SEEDS:
        heads = load_models(seed)
        orig, p_t, p_a = run_all_models(e_t, e_a, *heads)
        prob_store[(seed, "original", "unimodal")] = (p_t, p_a)
        for model, (p, w) in orig.items():
            prob_store[(seed, "original", model)] = (p, w)
        gate_w_t = orig["fusion_learned_gate"][1].squeeze(1)
        conf_w_t = orig["fusion_confidence"][1].squeeze(1)
        text_pred, audio_pred = p_t.argmax(1), p_a.argmax(1)

        for method, rep in replacements.items():
            no_text, pt_nt, pa_nt = run_all_models(rep["text"], e_a, *heads)
            no_audio, pt_na, pa_na = run_all_models(e_t, rep["audio"], *heads)
            prob_store[(seed, f"{method}_no_text", "unimodal")] = (pt_nt, pa_nt)
            prob_store[(seed, f"{method}_no_audio", "unimodal")] = (pt_na, pa_na)
            for model in orig:
                prob_store[(seed, f"{method}_no_text", model)] = no_text[model]
                prob_store[(seed, f"{method}_no_audio", model)] = no_audio[model]

            for model in orig:
                p0, _ = orig[model]
                c = p0.argmax(1)
                dpt, flt, tvt = effect(no_text[model][0], p0, c)
                dpa, fla, tva = effect(no_audio[model][0], p0, c)
                frames.append(pd.DataFrame({
                    "seed": seed, "row": range(n), "model": model, "method": method,
                    "label": test["y"].numpy(), "pred": c.numpy(),
                    "correct": (c == test["y"]).numpy(),
                    "p_pred": p0.max(1).values.numpy(),
                    "gate_w_t": gate_w_t.numpy(), "conf_w_t": conf_w_t.numpy(),
                    "text_pred": text_pred.numpy(), "audio_pred": audio_pred.numpy(),
                    "dp_text": dpt.numpy(), "dp_audio": dpa.numpy(),
                    "flip_text": flt.numpy(), "flip_audio": fla.numpy(),
                    "tvd_text": tvt.numpy(), "tvd_audio": tva.numpy(),
                    "audio_shared": meta["audio_shared"].values,
                    "audio_shared_with_train": meta["audio_shared_with_train"].values,
                    "dialogue_id": meta["dialogue_id"].values,
                    "utterance_id": meta["utterance_id"].values,
                }))
        print(f"seed {seed} done")

    log = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    log.to_csv(RESULTS / "interventions_log.csv", index=False)
    # full 7-class probability vectors (and fusion weights) for every condition
    torch.save({"|".join(map(str, k)): tuple(t.clone() for t in v) for k, v in prob_store.items()},
               DATA / "intervention_probs.pt")

    # quick sanity summary
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    summary = (log.assign(abs_text=log.dp_text.abs(), abs_audio=log.dp_audio.abs())
               .groupby(["model", "method"])[["abs_text", "abs_audio", "flip_text", "flip_audio",
                                              "tvd_text", "tvd_audio"]].mean())
    print("\nMean effect of removing each modality (all seeds):")
    print(summary)
    print(f"\nSaved {len(log)} rows to {RESULTS / 'interventions_log.csv'}")


if __name__ == "__main__":
    main()
