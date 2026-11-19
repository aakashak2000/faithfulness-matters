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


# ---------------------------------------------------------------- loaders for steps 49 to 52 (moved from 49_rescoring and 51_gate_free_tau, unchanged)
def load_rescoring_inputs(tag):
    """Test and dev head probabilities, gate weights (all cached gates) and labels for M1 or M5 (steps 49 to 52)."""
    cache = torch.load(CACHE / f"gates_{tag}.pt")
    out = {}
    if tag == "M1":
        B = build_harness()
        th, ah = B["specs"]["M1 stronger audio"]
        ET, EA = B["ET"], B["EA"]
        with torch.no_grad():
            for sp in ("dev", "test"):
                pt, pa = torch.softmax(th(ET[sp]), 1).numpy(), torch.softmax(ah(EA[sp]), 1).numpy()
                ws = []
                for _, sd in cache.items():
                    g = LearnedGate(B["dT"] + 384, B["dA"] + 1024)
                    g.load_state_dict(sd)
                    g.eval()
                    ws.append(g(ET[sp], EA[sp]).squeeze(1).numpy())
                out[sp] = dict(pt=pt, pa=pa, w=np.stack(ws), y=B["y"][sp].numpy())
        out["meta"] = B["D"]["meta"]["test"]
    else:
        M = build_m5()
        with torch.no_grad():
            for sp in ("dev", "test"):
                pt, pa = torch.softmax(M["Z2"]["text"][sp], 1).numpy(), torch.softmax(M["Z2"]["audio"][sp], 1).numpy()
                ws = []
                for _, sd in cache.items():
                    g = LearnedGate(384, 1024)
                    g.load_state_dict(sd)
                    g.eval()
                    ws.append(g(*M["G"][sp]).squeeze(1).numpy())
                out[sp] = dict(pt=pt, pa=pa, w=np.stack(ws), y=M["y"][sp].numpy())
        out["meta"] = M["D"]["meta"]["test"]
    return out


def donor_pool_means(tag, split="test", ok=None):
    """Per row of `split` (default test): mean donor probability vector for text and audio, over valid donors only (or over the donors in
    `ok`, a (text, audio) mask pair, if given). Returns pt, pa, mt, ma, y, meta."""
    B = build_harness() if tag == "M1" else None
    M = build_m5() if tag == "M5" else None
    with torch.no_grad():
        if tag == "M1":
            th, ah = B["specs"]["M1 stronger audio"]
            pt, pa = torch.softmax(th(B["ET"][split]), 1), torch.softmax(ah(B["EA"][split]), 1)
            Pt, Pa = torch.softmax(th(B["ET"]["clf"]), 1), torch.softmax(ah(B["EA"]["clf"]), 1)
            ok_t, ok_a = (m.float() for m in (B["ok"][split] if ok is None else ok))
            mt = (ok_t @ Pt) / ok_t.sum(1, keepdim=True)
            ma = (ok_a @ Pa) / ok_a.sum(1, keepdim=True)
            return pt.numpy(), pa.numpy(), mt.numpy(), ma.numpy(), B["y"][split].numpy(), B["D"]["meta"][split]
        pt, pa = torch.softmax(M["Z2"]["text"][split], 1), torch.softmax(M["Z2"]["audio"][split], 1)
        out = {}
        for kind in ("text", "audio"):
            okk = (M["ok"][split] if ok is None else ok)[0 if kind == "text" else 1].float()
            ms = []
            for s in range(0, okk.shape[0], 100):
                P = torch.softmax(M["donor"][kind][None] + M["ctxpart"][kind][split][s:s + 100, None], 2)   # (rows, donors, 7)
                o = okk[s:s + 100]
                ms.append((o[:, :, None] * P).sum(1) / o.sum(1, keepdim=True))
            out[kind] = torch.cat(ms)
        return pt.numpy(), pa.numpy(), out["text"].numpy(), out["audio"].numpy(), M["y"][split].numpy(), M["D"]["meta"][split]


# ---------------------------------------------------------------- donor pools and swap-free features (steps 53 to 56)
POOL_K = 50


def donor_speakers():
    """Speaker of each clf-split donor, in donor order (same order for M1 and M5)."""
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    csv = pd.read_csv(DATA / "meld_csv" / "train_sent_emo.csv")
    spk = {(a, b): s for a, b, s in zip(csv.Dialogue_ID, csv.Utterance_ID, csv.Speaker)}
    tr = P["ids"]["train"]
    return np.array([spk[tuple(int(x) for x in tr[i])] for i in np.asarray(P["rows"]["clf"])])


def build_pool_masks(ok, emb, emb_pool, row_spk, don_spk, pools, seed=544, k=POOL_K):
    """Per-utterance donor masks with exactly k valid donors. ok: (valid_text, valid_audio) bool (n, n_donors).
    global50: random; speaker50: random from the same speaker (needs global50 first; rows without k such donors fall back to
    global50 and are flagged not included); nn50: nearest by cosine in the swapped modality's embedding. One generator, rows in order."""
    n = ok[0].shape[0]
    rng = np.random.default_rng(seed)
    masks, included = {}, np.ones(n, bool)
    for pool in pools:
        mt, ma = torch.zeros_like(ok[0]), torch.zeros_like(ok[1])
        if pool == "nn50":
            for m, valid, e, ep in ((mt, ok[0], emb[0], emb_pool[0]), (ma, ok[1], emb[1], emb_pool[1])):
                sim = torch.nn.functional.normalize(e.float(), dim=1) @ torch.nn.functional.normalize(ep.float(), dim=1).T
                sim[~valid] = -float("inf")
                m.scatter_(1, sim.topk(k, dim=1).indices, True)
        else:
            for i in range(n):
                for m, valid in ((mt, ok[0]), (ma, ok[1])):
                    cand = np.flatnonzero(valid[i].numpy())
                    if pool == "speaker50":
                        same = cand[don_spk[cand] == row_spk[i]]
                        if len(same) < k:
                            included[i] = False
                            m[i] = masks["global50"][0 if m is mt else 1][i]
                            continue
                        cand = same
                    m[i, rng.choice(cand, k, replace=False)] = True
        for m, valid in ((mt, ok[0]), (ma, ok[1])):
            assert (m.sum(1) == k).all() and not (m & ~valid).any(), f"{pool}: mask check failed"
        masks[pool] = (mt, ma)
    return masks, included


def pool_runner(tag, alpha, return_model=False):
    """Rebuild a model with its cached gates. Returns meta, gate embeddings, the mutable valid-donor dict, gates, and
    split_fn(gate, split) -> core2 output dict (uses whatever masks are in the dict at call time)."""
    cache = torch.load(CACHE / f"gates_{tag}.pt")
    if tag == "M1":
        B = build_harness()
        th, ah = B["specs"]["M1 stronger audio"]
        gates = []
        for _, sd in cache.items():
            g = LearnedGate(B["dT"] + 384, B["dA"] + 1024)
            g.load_state_dict(sd)
            g.eval()
            gates.append(g)
        fn = lambda g, split: harness_split(B, th, ah, g, split, alpha)[3]
        return (B["D"]["meta"], B["H"], B["ok"], gates, fn) + ((B,) if return_model else ())
    M = build_m5()
    gates = []
    for _, sd in cache.items():
        g = LearnedGate(384, 1024)
        g.load_state_dict(sd)
        g.eval()
        gates.append(g)
    fn = lambda g, split: m5_split(M, g, split, alpha)[3]
    return (M["D"]["meta"], M["G"], M["ok"], gates, fn) + ((M,) if return_model else ())


def swapfree_features(W, pt, pa, mt, ma):
    """Swap-free features per (gate, row), given donor-pool means. W: (gates, n). Returns (gates, n, 16) features, R2 and tau."""
    from common import js_div
    n = pt.shape[0]
    ii = np.arange(n)
    ent = lambda p: -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    c0 = (0.5 * pt + 0.5 * pa).argmax(1)
    d0t, d0a = np.abs(pt[ii, c0] - mt[ii, c0]), np.abs(pa[ii, c0] - ma[ii, c0])
    shared = [pt.max(1), pa.max(1), ent(pt), ent(pa), (pt.argmax(1) == pa.argmax(1)).astype(float), js_div(pt, pa), d0t, d0a]
    F, R2, TAU = [], [], []
    for w in W:
        c = (w[:, None] * pt + (1 - w[:, None]) * pa).argmax(1)
        dt, da = np.abs(pt[ii, c] - mt[ii, c]), np.abs(pa[ii, c] - ma[ii, c])
        tau = da / np.clip(dt + da, 1e-12, None)
        r2 = w * dt / np.clip(w * dt + (1 - w) * da, 1e-12, None)
        F.append(np.column_stack([w, np.abs(w - 0.5), r2, np.abs(r2 - 0.5), tau, np.abs(w - tau), dt, da] + shared))
        R2.append(r2)
        TAU.append(tau)
    return np.stack(F), np.stack(R2), np.stack(TAU)


def risk_curve(score, wrong, cover):
    """Share of wrong claims among the top-scored fraction, for each coverage level."""
    order = np.argsort(-score, kind="stable")
    cw = np.cumsum(wrong[order])
    k = np.maximum(1, np.ceil(cover * len(score)).astype(int))
    return cw[k - 1] / k
