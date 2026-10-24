"""Step 39: the step 38 frozen-gate audit applied to M5, the 63.6 model (strong text + strong audio + context).
Same protocol as 38 (written there). For M5 the donor's vote depends on the utterance's own context, as in 37:
a swap replaces only the explained utterance's stage-1 logits and headline gate embedding; neighbours stay.
Output: appends one row to results/frozen_gate_audit.csv (3 gates, as step 37).
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, class_weights, load_dataset, set_determinism, train_gate  # noqa: E402

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
rs07 = importlib.import_module("07_resampling")
f38 = importlib.import_module("38_frozen_gate")


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    P2 = torch.load(DATA / "v2_pipeline_v2.pt", weights_only=False)
    rows, Z1, ids, dl, ut = P["rows"], P["Z1"], P["ids"], P["dlg"], P["utt"]
    cfg = P2["cfg"]
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    emo = {e: i for i, e in enumerate(D["emotions"])}
    lab = {(a, b): emo[e] for a, b, e in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Emotion)}
    ytr = torch.tensor([lab[tuple(i)] for i in ids["train"]])
    y = {"clf": ytr[rows["clf"]], "gate": ytr[rows["gate"]], "dev": S["dev"]["y"], "test": S["test"]["y"]}
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    txt_head = torch.zeros(len(ids["train"]), 384)
    aud_head = torch.zeros(len(ids["train"]), 1024)
    for sp in ("clf", "gate"):
        m = mp[mp.split == sp].sort_values("row")
        txt_head[torch.tensor(m.file_index.values)] = S[sp]["text"]
        aud_head[torch.tensor(m.file_index.values)] = S[sp]["audio"]
    G = {"clf": (txt_head[rows["clf"]], aud_head[rows["clf"]]), "gate": (txt_head[rows["gate"]], aud_head[rows["gate"]]),
         "dev": (S["dev"]["text"], S["dev"]["audio"]), "test": (S["test"]["text"], S["test"]["audio"])}
    own, ctxpart, Z2 = {}, {}, {}
    for kind in ("text", "audio"):
        c = cfg[kind]
        C = {s: m29.ctx_features(Z1[kind][s], dl[s], ut[s], c["k"], c["mode"]) for s in ("clf", "gate", "dev", "test")}
        W, b = b22.fit(C["clf"], y["clf"], w_clf, float(c["wd"]))
        own[kind] = W[:7]
        ctxpart[kind] = (C["test"][:, 7:] @ W[7:] + b)
        Z2[kind] = {s: C[s] @ W + b for s in ("gate", "dev", "test")}
    donor = {k: Z1[k]["clf"] @ own[k] for k in ("text", "audio")}
    pt, pa = torch.softmax(Z2["text"]["test"], 1), torch.softmax(Z2["audio"]["test"], 1)
    ok_t, ok_a = rs07.valid_masks(G["test"][0], G["clf"][0]), rs07.valid_masks(G["test"][1], G["clf"][1])
    # unmasked pool mean vote of the explained utterance (as in step 34, but with its own context)
    vall = []
    for kind in ("text", "audio"):
        acc = torch.zeros(len(pt), 7)
        for lo in range(0, donor[kind].shape[0], 200):
            acc += torch.softmax(donor[kind][lo:lo + 200][None] + ctxpart[kind][:, None], 2).sum(1)
        vall.append(acc / donor[kind].shape[0])
    per = []
    for seed in range(3):
        set_determinism(seed)
        Pg = {s: (torch.softmax(Z2["text"][s], 1), torch.softmax(Z2["audio"][s], 1)) for s in ("gate", "dev")}
        g = train_gate(G["gate"][0], G["gate"][1], *Pg["gate"], y["gate"], (G["dev"][0], G["dev"][1], *Pg["dev"], y["dev"]), w_gate, seed)
        g.eval()
        lin1, lin2 = g.net[0], g.net[2]
        Wt, Wa, b1 = lin1.weight[:, :384], lin1.weight[:, 384:], lin1.bias
        with torch.no_grad():
            w0 = g(*G["test"]).squeeze(1)
            At_pool, Aa_pool = G["clf"][0] @ Wt.T, G["clf"][1] @ Wa.T
            At_i, Aa_i = G["test"][0] @ Wt.T, G["test"][1] @ Wa.T
            pool_fn = lambda kind, sl: torch.softmax(donor[kind][sl][None] + ctxpart[kind][:, None], 2)
            pre_fn = lambda kind, sl: ((At_pool[sl][None] + Aa_i[:, None]) if kind == "text" else (At_i[:, None] + Aa_pool[sl][None])) + b1
            out = f38.core(pt, pa, w0, pool_fn, pre_fn, lin2.weight[0], lin2.bias[0], ok_t, ok_a)
            d = f38.per_gate(pt, pa, w0, out, vall[0], vall[1], None)
        per.append(d)
        print(f"  gate {seed} done ({time.time() - t0:.0f}s)", flush=True)
    row = f38.summarise("M5 best model: strong text + strong audio + context (63.6)", per)
    old = pd.read_csv(RESULTS / "frozen_gate_audit.csv")
    old = old[~old.model.str.startswith("M5")]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS / "frozen_gate_audit.csv", index=False)
    print({k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
