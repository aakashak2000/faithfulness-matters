"""Shared code for the audit follow-ups (steps 42 and 43), written 2026-10-23.

Nothing here changes an earlier definition. It rebuilds the same models as steps 33/38 (harness M0b-M3) and 37/39 (M5),
and adds ONE new swap routine, `core2`, that evaluates the original swap effect under two decision rules at once:
  raw : the fused probability p = w p_text + (1 - w) p_audio (the rule tested in steps 33 to 41);
  cor : the dev-prior-corrected probability q = softmax(log p + alpha * log pi), pi = class frequencies of the clf split
        (the rule behind the F1 values 58.8 and 63.6). alpha is frozen from dev, never from test.
For each rule the "original class" is argmax of that rule's probability on the unswapped input and is kept for every donor.
Effects (signed, for modality k): mean over valid donors of prob_d[c] minus prob_0[c] (average first; the primary measure takes
|.| of this afterwards), plus the mean-of-absolute-changes variant and the share of donors that flip the argmax.
The gate is recomputed for each donor ("real") and, as a secondary analysis, held at the original weight ("fro").
Validation: with alpha irrelevant, the "raw real" effect must equal the effect saved by step 38 (data/frozen_gate_arrays_*.npz).
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, LearnedGate, load_dataset, set_determinism, train_gate  # noqa: E402

f38 = importlib.import_module("38_frozen_gate")
m29 = importlib.import_module("29_context_pipeline")
b22 = importlib.import_module("22_better_audio")
rs07 = importlib.import_module("07_resampling")
CHUNK = 100
ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]
RULES = ("raw", "cor")
CACHE = DATA / "audit_cache"


def qrule(F, alpha, lp):
    """Dev-prior-corrected, normalised probabilities. For alpha = 0 this is F itself."""
    return torch.softmax(F.clamp_min(1e-12).log() + alpha * lp, -1)


def core2(pt, pa, w0, pool_fn, pre_fn, w2, b2c, ok_t, ok_a, lp, alpha):
    """Swap effects under both rules. pt, pa (n,7); w0 (n,). Returns dict of numpy arrays (n,)."""
    n = pt.shape[0]
    idx = torch.arange(n)
    p0 = w0[:, None] * pt + (1 - w0[:, None]) * pa
    q0 = qrule(p0, alpha, lp)
    base = {"raw": p0, "cor": q0}
    cls = {"raw": p0.argmax(1), "cor": q0.argmax(1)}
    b0 = {v: base[v][idx, cls[v]] for v in RULES}
    out = {"c_raw": cls["raw"].numpy(), "c_cor": cls["cor"].numpy(), "w0": w0.numpy(),
           "orig_class_changes": (cls["raw"] != cls["cor"]).numpy()}
    for kind, ok in (("text", ok_t), ("audio", ok_a)):
        acc = {(v, mode, s): torch.zeros(n) for v in RULES for mode in ("real", "fro") for s in ("mean", "mabs", "flip")}
        vote = torch.zeros(n)
        for lo in range(0, ok.shape[1], CHUNK):
            sl = slice(lo, lo + CHUNK)
            m_ = ok[:, sl].float()
            Pd = pool_fn(kind, sl).expand(n, -1, -1)
            wd = torch.sigmoid(torch.relu(pre_fn(kind, sl)) @ w2 + b2c)
            other = pa if kind == "text" else pt
            if kind == "text":
                real = wd[:, :, None] * Pd + (1 - wd[:, :, None]) * other[:, None, :]
                fro = w0[:, None, None] * Pd + (1 - w0[:, None, None]) * other[:, None, :]
            else:
                real = wd[:, :, None] * other[:, None, :] + (1 - wd[:, :, None]) * Pd
                fro = w0[:, None, None] * other[:, None, :] + (1 - w0[:, None, None]) * Pd
            gi = cls["raw"][:, None, None].expand(n, Pd.shape[1], 1)
            vote += (m_ * Pd.gather(2, gi).squeeze(2)).sum(1)
            for mode, F in (("real", real), ("fro", fro)):
                for v in RULES:
                    Q = F if v == "raw" else qrule(F, alpha, lp)
                    ci = cls[v][:, None, None].expand(n, Q.shape[1], 1)
                    qc = Q.gather(2, ci).squeeze(2)
                    acc[(v, mode, "mean")] += (m_ * qc).sum(1)
                    acc[(v, mode, "mabs")] += (m_ * (qc - b0[v][:, None]).abs()).sum(1)
                    acc[(v, mode, "flip")] += (m_ * (Q.argmax(2) != cls[v][:, None]).float()).sum(1)
        cnt = ok.sum(1).float()
        for (v, mode, s), a in acc.items():
            val = a / cnt
            out[f"{v}_{mode}_{kind}_{s}"] = (val - b0[v] if s == "mean" else val).numpy()
        out[f"vote_{kind}"] = (vote / cnt).numpy()
        out[f"closed_form_check_{kind}"] = None
    # frozen closed form check (raw rule): fro effect = w * (vote - p_own[c])
    for kind, own, wk in (("text", pt, w0), ("audio", pa, 1 - w0)):
        cf = wk.numpy() * (out[f"vote_{kind}"] - own[idx, cls["raw"]].numpy())
        out[f"closed_form_err_{kind}"] = np.abs(out[f"raw_fro_{kind}_mean"] - cf)
        out.pop(f"closed_form_check_{kind}")
    # effect differences (text minus audio), the quantities compared with the weight
    for v in RULES:
        for mode in ("real", "fro"):
            out[f"e_{v}_{mode}"] = np.abs(out[f"{v}_{mode}_text_mean"]) - np.abs(out[f"{v}_{mode}_audio_mean"])
            out[f"e_{v}_{mode}_mabs"] = out[f"{v}_{mode}_text_mabs"] - out[f"{v}_{mode}_audio_mabs"]
            out[f"e_{v}_{mode}_flip"] = out[f"{v}_{mode}_text_flip"] - out[f"{v}_{mode}_audio_flip"]
    out["s_w"] = (2 * w0 - 1).numpy()
    return out


# ---------------------------------------------------------------- harness models (M0b..M3), as in steps 33 and 38
def build_harness():
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    specs, (ET, EA, H, y, w_gate, dT, dA), (ok_t, ok_a) = f38.harness_models(D, S, mp)
    ok = {"test": (ok_t, ok_a), "dev": (rs07.valid_masks(H["dev"][0], H["clf"][0]), rs07.valid_masks(H["dev"][1], H["clf"][1]))}
    pi = torch.bincount(y["clf"], minlength=7).float()
    lp = (pi / pi.sum()).log()
    return dict(D=D, specs=specs, ET=ET, EA=EA, H=H, y=y, w_gate=w_gate, dT=dT, dA=dA, ok=ok, lp=lp)


def harness_gate(B, th, ah, seed):
    ET, EA, H, y, dT, dA = B["ET"], B["EA"], B["H"], B["y"], B["dT"], B["dA"]
    set_determinism(seed)
    with torch.no_grad():
        pr = {s: (torch.softmax(th(ET[s]), 1), torch.softmax(ah(EA[s]), 1)) for s in ("gate", "dev")}
    g = train_gate(H["gate"][0], H["gate"][1], *pr["gate"], y["gate"], (H["dev"][0], H["dev"][1], *pr["dev"], y["dev"]), B["w_gate"], seed)
    w1 = g.net[0].weight.data
    Gp = LearnedGate(dT + 384, dA + 1024)
    Gp.net[0].weight.data = torch.cat([torch.zeros(w1.shape[0], dT), w1[:, :384], torch.zeros(w1.shape[0], dA), w1[:, 384:]], 1)
    Gp.net[0].bias.data = g.net[0].bias.data.clone()
    Gp.net[2].load_state_dict(g.net[2].state_dict())
    Gp.eval()
    return Gp


def harness_split(B, th, ah, Gp, split, alpha):
    """Original-rule probabilities and the swap effects (both rules) for one split and one gate."""
    ET, EA = B["ET"], B["EA"]
    lin1, lin2 = Gp.net[0], Gp.net[2]
    dT = B["dT"]
    Wtt, Waa, b1 = lin1.weight[:, :dT + 384], lin1.weight[:, dT + 384:], lin1.bias
    with torch.no_grad():
        Pp_t, Pp_a = torch.softmax(th(ET["clf"]), 1), torch.softmax(ah(EA["clf"]), 1)
        pt, pa = torch.softmax(th(ET[split]), 1), torch.softmax(ah(EA[split]), 1)
        w0 = Gp(ET[split], EA[split]).squeeze(1)
        At_pool, Aa_pool = ET["clf"] @ Wtt.T, EA["clf"] @ Waa.T
        At_i, Aa_i = ET[split] @ Wtt.T, EA[split] @ Waa.T
        pool_fn = lambda kind, sl: (Pp_t[sl] if kind == "text" else Pp_a[sl])[None]
        pre_fn = lambda kind, sl: ((At_pool[sl][None] + Aa_i[:, None]) if kind == "text" else (At_i[:, None] + Aa_pool[sl][None])) + b1
        ok_t, ok_a = B["ok"][split]
        out = core2(pt, pa, w0, pool_fn, pre_fn, lin2.weight[0], lin2.bias[0], ok_t, ok_a, B["lp"], alpha)
    return pt, pa, w0, out


# ---------------------------------------------------------------- M5 (63.6 model), as in steps 37 and 39
def build_m5():
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    P2 = torch.load(DATA / "v2_pipeline_v2.pt", weights_only=False)
    rows, Z1, ids, dl, ut = P["rows"], P["Z1"], P["ids"], P["dlg"], P["utt"]
    cfg, alpha_saved = P2["cfg"], P2["alpha"]["S2 + context, bidirectional"]
    from common import class_weights
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    emo = {e: i for i, e in enumerate(D["emotions"])}
    lab = {(a, b): emo[e] for a, b, e in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Emotion)}
    ytr = torch.tensor([lab[tuple(i)] for i in ids["train"]])
    y = {"clf": ytr[rows["clf"]], "gate": ytr[rows["gate"]], "dev": S["dev"]["y"], "test": S["test"]["y"]}
    w_clf, w_gate = class_weights(y["clf"]), class_weights(y["gate"])
    pi = torch.bincount(y["clf"], minlength=7).float()
    lp = (pi / pi.sum()).log()
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
        ctxpart[kind] = {s: C[s][:, 7:] @ W[7:] + b for s in ("dev", "test")}
        Z2[kind] = {s: C[s] @ W + b for s in ("gate", "dev", "test")}
        err = (Z2[kind]["test"] - P2["Z2"][kind]["test"]).abs().max().item()
        assert err < 1e-3, err
    donor = {k: Z1[k]["clf"] @ own[k] for k in ("text", "audio")}
    ok = {sp: (rs07.valid_masks(G[sp][0], G["clf"][0]), rs07.valid_masks(G[sp][1], G["clf"][1])) for sp in ("dev", "test")}
    return dict(D=D, G=G, y=y, w_gate=w_gate, lp=lp, Z2=Z2, ctxpart=ctxpart, donor=donor, ok=ok, alpha_saved=alpha_saved)


def m5_gate(M, seed):
    Z2, G, y = M["Z2"], M["G"], M["y"]
    set_determinism(seed)
    Pg = {s: (torch.softmax(Z2["text"][s], 1), torch.softmax(Z2["audio"][s], 1)) for s in ("gate", "dev")}
    g = train_gate(G["gate"][0], G["gate"][1], *Pg["gate"], y["gate"], (G["dev"][0], G["dev"][1], *Pg["dev"], y["dev"]), M["w_gate"], seed)
    g.eval()
    return g


def m5_split(M, g, split, alpha):
    Z2, G, donor, ctxpart = M["Z2"], M["G"], M["donor"], M["ctxpart"]
    lin1, lin2 = g.net[0], g.net[2]
    Wt, Wa, b1 = lin1.weight[:, :384], lin1.weight[:, 384:], lin1.bias
    with torch.no_grad():
        pt, pa = torch.softmax(Z2["text"][split], 1), torch.softmax(Z2["audio"][split], 1)
        w0 = g(*G[split]).squeeze(1)
        At_pool, Aa_pool = G["clf"][0] @ Wt.T, G["clf"][1] @ Wa.T
        At_i, Aa_i = G[split][0] @ Wt.T, G[split][1] @ Wa.T
        pool_fn = lambda kind, sl: torch.softmax(donor[kind][sl][None] + ctxpart[kind][split][:, None], 2)
        pre_fn = lambda kind, sl: ((At_pool[sl][None] + Aa_i[:, None]) if kind == "text" else (At_i[:, None] + Aa_pool[sl][None])) + b1
        ok_t, ok_a = M["ok"][split]
        out = core2(pt, pa, w0, pool_fn, pre_fn, lin2.weight[0], lin2.bias[0], ok_t, ok_a, M["lp"], alpha)
    return pt, pa, w0, out
