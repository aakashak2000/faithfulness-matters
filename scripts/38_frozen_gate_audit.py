"""Step 38: frozen-gate audit of the corrected signal (methodological audit, questions 1 to 3).

Questions: (1) is the corrected signal s_corr = w |p_t[c] - m_t[c]| - (1 - w) |p_a[c] - m_a[c]| numerically identical to the swap
effect difference when the gate's weight is held at its original value? (2) how do the recomputed-gate swap (the primary test),
the frozen-gate swap and s_corr compare on identical utterances and donors? (3) is the residual disagreement between s_corr and
the real swap explained by the gate re-weighting, and how large is that term?

Definitions, fixed before running (same donors, masks, class and effect as the primary swap of 33/37):
  c = class predicted by the gate model on the original input; w0 = gate weight on the original input; p0 = fused output.
  Donor d is valid for utterance i unless its headline embedding of the swapped modality is byte-identical to i's.
  REAL    : fused_d = w_d P_d + (1 - w_d) p_a (text swap), w_d recomputed by the gate; dp_text = mean_d fused_d[c] - p0[c]; audio likewise.
  FROZEN  : the same loop with w_d := w0 (computed by explicit per-donor forward, not by the closed form).
  CLOSED  : w0 (m_t[c] - p_t[c]), m_t = mean over the VALID donors of P_d (masked mean).
  CORR34  : s_corr of 34 (m over the whole clf2 pool, unmasked), as published in results/corrected_signal.csv.
  e = |dp_text| - |dp_audio| (positive: text mattered more). The re-weighting term is G = REAL - FROZEN (exact, the fused output is
  linear in the weight). Models: M0b, M1, M2, M3 (10 gates each, as 33) and M5 = the 63.6 model (3 gates, as 37). Test set.
  Reported per model (mean over gates): agreement and Spearman of every pair of (REAL, FROZEN, CLOSED, CORR34, plain weight),
  max and mean absolute differences, share of utterances where sign(e) differs between REAL and FROZEN, the margin |e_frozen| of those
  utterances against all utterances, medians of |G|, and a counterfactual: how often adding only one of the two G terms restores the
  REAL sign. Check: REAL must reproduce the pipeline of 08/33 (rs08.exact_average) to 1e-5 for the harness models.
Usage: python scripts/38_frozen_gate_audit.py     Output: results/frozen_gate_audit.csv
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, RESULTS, Head, LearnedGate, class_weights, load_dataset, set_determinism, train_gate,  # noqa: E402
                    train_head)

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
iv04 = importlib.import_module("04_interventions")
CHUNK = 100


def audit(p_t, p_a, w0, pd_fn, wd_fn, ok):
    """Returns dict of per-utterance arrays for REAL, FROZEN, CLOSED, plus donor means."""
    n = p_t.shape[0]
    idx = torch.arange(n)
    p0 = w0[:, None] * p_t + (1 - w0[:, None]) * p_a
    c = p0.argmax(1)
    out = {"c": c, "p0c": p0[idx, c]}
    for kind in ("text", "audio"):
        okm = ok[kind].float()
        n_valid = okm.sum(1)
        S_real, S_fro, V = torch.zeros(n, 7), torch.zeros(n, 7), torch.zeros(n, 7)
        for lo in range(0, okm.shape[1], CHUNK):
            sl = slice(lo, lo + CHUNK)
            m_ = okm[:, sl]
            Pd = pd_fn(kind, sl)                       # (n or 1, chunk, 7)
            wd = wd_fn(kind, sl)                       # (n, chunk)
            other = p_a if kind == "text" else p_t
            wo = w0[:, None, None]
            if kind == "text":
                real = wd[:, :, None] * Pd + (1 - wd[:, :, None]) * other[:, None, :]
                fro = wo * Pd + (1 - wo) * other[:, None, :]
            else:
                real = wd[:, :, None] * other[:, None, :] + (1 - wd[:, :, None]) * Pd
                fro = wo * other[:, None, :] + (1 - wo) * Pd
            S_real += (m_[:, :, None] * real).sum(1)
            S_fro += (m_[:, :, None] * fro).sum(1)
            V += (m_[:, :, None] * Pd.expand(n, -1, -1)).sum(1)
        out[f"dp_real_{kind}"] = S_real[idx, c] / n_valid - out["p0c"]
        out[f"dp_fro_{kind}"] = S_fro[idx, c] / n_valid - out["p0c"]
        out[f"vote_{kind}"] = V / n_valid[:, None]
    return out


def pair_stats(a, b):
    v = (a != 0) & (b != 0)
    return (np.sign(a[v]) == np.sign(b[v])).mean(), np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def summarise(name, runs):
    """runs: list of dicts (one per gate) with arrays e_real, e_fro, e_closed, e_corr34, w (text weight), G terms."""
    acc = {}
    for r in runs:
        sig = {"real": r["e_real"], "frozen": r["e_fro"], "closed": r["e_closed"], "corr34": r["e_corr34"], "weight": 2 * r["w"] - 1}
        for a, b in (("weight", "real"), ("weight", "frozen"), ("real", "frozen"), ("corr34", "real"), ("closed", "real"),
                     ("corr34", "frozen"), ("closed", "frozen"), ("corr34", "closed")):
            ag, rho = pair_stats(sig[a], sig[b])
            acc.setdefault(f"agree_{a}_vs_{b}", []).append(ag)
            acc.setdefault(f"spearman_{a}_vs_{b}", []).append(rho)
        acc.setdefault("max_abs_dp_brute_vs_closed", []).append(r["max_dp_err"])
        acc.setdefault("max_abs_e_corr34_vs_frozen", []).append(np.abs(r["e_corr34"] - r["e_fro"]).max())
        acc.setdefault("mean_abs_e_corr34_vs_frozen", []).append(np.abs(r["e_corr34"] - r["e_fro"]).mean())
        acc.setdefault("mean_abs_e_real_vs_frozen", []).append(np.abs(r["e_real"] - r["e_fro"]).mean())
        v = (r["e_real"] != 0) & (r["e_fro"] != 0)
        dis = v & (np.sign(r["e_real"]) != np.sign(r["e_fro"]))
        acc.setdefault("share_sign_differs_real_vs_frozen", []).append(dis.sum() / v.sum())
        acc.setdefault("median_margin_frozen_where_differs", []).append(np.median(np.abs(r["e_fro"][dis])) if dis.any() else np.nan)
        acc.setdefault("median_margin_frozen_all", []).append(np.median(np.abs(r["e_fro"][v])))
        acc.setdefault("median_abs_G_text", []).append(np.median(np.abs(r["G_t"])))
        acc.setdefault("median_abs_G_audio", []).append(np.median(np.abs(r["G_a"])))
        for lab, (gt, ga) in {"only_G_text": (r["G_t"], 0 * r["G_a"]), "only_G_audio": (0 * r["G_t"], r["G_a"])}.items():
            e_cf = (np.abs(r["dpf_t"] + gt) - np.abs(r["dpf_a"] + ga))
            acc.setdefault(f"restores_real_sign_{lab}", []).append(
                (np.sign(e_cf[dis]) == np.sign(r["e_real"][dis])).mean() if dis.any() else np.nan)
        acc.setdefault("p_text_real", []).append((r["e_real"][r["e_real"] != 0] > 0).mean())
        acc.setdefault("p_text_frozen", []).append((r["e_fro"][r["e_fro"] != 0] > 0).mean())
        acc.setdefault("check_real_vs_pipeline_max_abs", []).append(r.get("check", np.nan))
    return {"model": name, "n_gates": len(runs), **{k: float(np.nanmean(v)) for k, v in acc.items()},
            **{k.replace("max_abs", "worst_max_abs"): float(np.nanmax(v)) for k, v in acc.items() if k.startswith("max_abs")}}


def run_gate(p_t, p_a, w0, pd_fn, wd_fn, ok, pool_votes_unmasked):
    o = audit(p_t, p_a, w0, pd_fn, wd_fn, ok)
    n = p_t.shape[0]
    idx = torch.arange(n)
    c = o["c"]
    closed_t = w0 * (o["vote_text"][idx, c] - p_t[idx, c])
    closed_a = (1 - w0) * (o["vote_audio"][idx, c] - p_a[idx, c])
    mt, ma = pool_votes_unmasked
    corr34 = w0 * (p_t[idx, c] - mt[c]).abs() - (1 - w0) * (p_a[idx, c] - ma[c]).abs()
    r = {"e_real": (o["dp_real_text"].abs() - o["dp_real_audio"].abs()).numpy(),
         "e_fro": (o["dp_fro_text"].abs() - o["dp_fro_audio"].abs()).numpy(),
         "e_closed": (closed_t.abs() - closed_a.abs()).numpy(), "e_corr34": corr34.numpy(), "w": w0.numpy(),
         "G_t": (o["dp_real_text"] - o["dp_fro_text"]).numpy(), "G_a": (o["dp_real_audio"] - o["dp_fro_audio"]).numpy(),
         "dpf_t": o["dp_fro_text"].numpy(), "dpf_a": o["dp_fro_audio"].numpy(),
         "max_dp_err": float(max((o["dp_fro_text"] - closed_t).abs().max(), (o["dp_fro_audio"] - closed_a).abs().max())),
         "dp_real_t": o["dp_real_text"], "dp_real_a": o["dp_real_audio"], "c": c}
    return r


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    rows, ids, Wb = P["rows"], P["ids"], P["Wb"]
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
    H = {"clf": (txt_head[rows["clf"]], aud_head[rows["clf"]]), "gate": (txt_head[rows["gate"]], aud_head[rows["gate"]]),
         "dev": (S["dev"]["text"], S["dev"]["audio"]), "test": (S["test"]["text"], S["test"]["audio"])}
    ok = {"text": rs07.valid_masks(H["test"][0], H["clf"][0]), "audio": rs07.valid_masks(H["test"][1], H["clf"][1])}
    results = []

    # ------------------------------------------------ harness models M0b, M1, M2, M3 (as 34)
    rt = {sp: torch.load(DATA / f"roberta_layers_{sp}.pt", map_location="cpu", weights_only=False)["mean"] for sp in ("train", "dev", "test")}
    wl = {sp: torch.load(DATA / f"wavlm_layers_{sp}.pt", map_location="cpu", weights_only=False) for sp in ("train", "dev", "test")}
    bt = lambda m, idx=None: torch.cat([(m if idx is None else m[idx])[:, l].float() for l in range(25)], 1)
    ba = lambda sp, idx=None: b22.build(wl[sp]["mean"] if idx is None else wl[sp]["mean"][idx],
                                        wl[sp]["std"] if idx is None else wl[sp]["std"][idx], m29.AUDIO_LAYERS, m29.AUDIO_POOL)
    rawT = {"clf": bt(rt["train"], rows["clf"]), "gate": bt(rt["train"], rows["gate"]), "dev": bt(rt["dev"]), "test": bt(rt["test"])}
    rawA = {"clf": ba("train", rows["clf"]), "gate": ba("train", rows["gate"]), "dev": ba("dev"), "test": ba("test")}
    FT = {k: ((v - rawT["clf"].mean(0)) / (rawT["clf"].std(0) + 1e-6)).contiguous() for k, v in rawT.items()}
    FA = {k: ((v - rawA["clf"].mean(0)) / (rawA["clf"].std(0) + 1e-6)).contiguous() for k, v in rawA.items()}
    del rawT, rawA, rt, wl
    dT, dA = FT["clf"].shape[1], FA["clf"].shape[1]
    ET = {k: torch.cat([FT[k], H[k][0]], 1).contiguous() for k in FT}
    EA = {k: torch.cat([FA[k], H[k][1]], 1).contiguous() for k in FA}
    (Wt, bt_), (Wa, ba_) = Wb["text"], Wb["audio"]

    def padded(W_blocks, b, dim_total, offset, n_in):
        h = Head(dim_total)
        w = torch.zeros(7, dim_total)
        w[:, offset:offset + n_in] = W_blocks
        h.lin.weight.data, h.lin.bias.data = w, b.clone()
        h.eval()
        return h

    set_determinism(0)
    th_mini = train_head(H["clf"][0], y["clf"], H["dev"][0], y["dev"], w_clf, 0)
    ah_head = train_head(H["clf"][1], y["clf"], H["dev"][1], y["dev"], w_clf, 0)
    T_MINI = padded(th_mini.lin.weight.data, th_mini.lin.bias.data, dT + 384, dT, 384)
    T_ROB = padded(Wt.T.contiguous(), bt_, dT + 384, 0, dT)
    A_HEAD = padded(ah_head.lin.weight.data, ah_head.lin.bias.data, dA + 1024, dA, 1024)
    A_STR = padded(Wa.T.contiguous(), ba_, dA + 1024, 0, dA)
    specs = {"M0b headline recipe, v2 split": (T_MINI, A_HEAD), "M1 stronger audio (balanced)": (T_MINI, A_STR),
             "M2 stronger text": (T_ROB, A_HEAD), "M3 stronger text + audio": (T_ROB, A_STR)}
    for name, (th, ah) in specs.items():
        with torch.no_grad():
            Pt_pool, Pa_pool = torch.softmax(th(ET["clf"]), 1), torch.softmax(ah(EA["clf"]), 1)
            p_t, p_a = torch.softmax(th(ET["test"]), 1), torch.softmax(ah(EA["test"]), 1)
            unmasked = (Pt_pool.mean(0), Pa_pool.mean(0))
        runs = []
        for seed in range(10):
            set_determinism(seed)
            with torch.no_grad():
                pr = {s: (torch.softmax(th(ET[s]), 1), torch.softmax(ah(EA[s]), 1)) for s in ("gate", "dev")}
            g = train_gate(H["gate"][0], H["gate"][1], *pr["gate"], y["gate"], (H["dev"][0], H["dev"][1], *pr["dev"], y["dev"]), w_gate, seed)
            w1 = g.net[0].weight.data
            Gp = LearnedGate(dT + 384, dA + 1024)
            Gp.net[0].weight.data = torch.cat([torch.zeros(w1.shape[0], dT), w1[:, :384], torch.zeros(w1.shape[0], dA), w1[:, 384:]], 1)
            Gp.net[0].bias.data = g.net[0].bias.data.clone()
            Gp.net[2].load_state_dict(g.net[2].state_dict())
            Gp.eval()
            with torch.no_grad():
                lin1, lin2 = Gp.net[0], Gp.net[2]
                Wt1, Wa1, b1 = lin1.weight[:, :dT + 384], lin1.weight[:, dT + 384:], lin1.bias
                w2, bb2 = lin2.weight[0], lin2.bias[0]
                At_pool, Aa_pool = ET["clf"] @ Wt1.T, EA["clf"] @ Wa1.T
                At_i, Aa_i = ET["test"] @ Wt1.T, EA["test"] @ Wa1.T
                w0 = Gp(ET["test"], EA["test"]).squeeze(1)

                def pd_fn(kind, sl):
                    return (Pt_pool if kind == "text" else Pa_pool)[sl][None]

                def wd_fn(kind, sl):
                    pre = At_pool[sl][None] + Aa_i[:, None] if kind == "text" else At_i[:, None] + Aa_pool[sl][None]
                    return torch.sigmoid(torch.relu(pre + b1) @ w2 + bb2)

                r = run_gate(p_t, p_a, w0, pd_fn, wd_fn, ok, unmasked)
                if seed == 0:       # validation against the pipeline of 08/33
                    old = iv04.load_models
                    iv04.load_models = lambda s: (th, ah, Gp)
                    try:
                        orig, _, _, cls, pb, _, _ = rs08.exact_average(0, ET["test"], EA["test"], ET["clf"], EA["clf"], ok["text"], ok["audio"])
                    finally:
                        iv04.load_models = old
                    m_ = "fusion_learned_gate"
                    dpt, _, _ = iv04.effect(pb[m_][0], orig[m_][0], cls[m_])
                    dpa, _, _ = iv04.effect(pb[m_][1], orig[m_][0], cls[m_])
                    r["check"] = float(max((dpt - r["dp_real_t"]).abs().max(), (dpa - r["dp_real_a"]).abs().max()))
                    assert r["check"] < 1e-5, r["check"]
            runs.append(r)
        results.append(summarise(name, runs))
        print(name, "done", f"({time.time() - t0:.0f}s)", {k: round(v, 4) for k, v in results[-1].items() if k in
              ("agree_real_vs_frozen", "agree_corr34_vs_frozen", "agree_corr34_vs_real", "share_sign_differs_real_vs_frozen")}, flush=True)
    del ET, EA, FT, FA

    # ------------------------------------------------ M5, the 63.6 model (as 37)
    P2 = torch.load(DATA / "v2_pipeline_v2.pt", weights_only=False)
    Z1, dl, ut, cfg = P["Z1"], P["dlg"], P["utt"], P2["cfg"]
    own, ctxp, Z2 = {}, {}, {}
    for kind in ("text", "audio"):
        cc = cfg[kind]
        C = {s: m29.ctx_features(Z1[kind][s], dl[s], ut[s], cc["k"], cc["mode"]) for s in ("clf", "gate", "dev", "test")}
        W, b = b22.fit(C["clf"], y["clf"], w_clf, float(cc["wd"]))
        own[kind] = W[:7]
        ctxp[kind] = C["test"][:, 7:] @ W[7:] + b
        Z2[kind] = {s: C[s] @ W + b for s in ("gate", "dev", "test")}
    donor = {k: Z1[k]["clf"] @ own[k] for k in ("text", "audio")}
    p_t, p_a = torch.softmax(Z2["text"]["test"], 1), torch.softmax(Z2["audio"]["test"], 1)
    votes_unmasked = {k: torch.softmax(donor[k][None] + ctxp[k][:, None], 2).mean(1) for k in ("text", "audio")}   # per-utterance, unmasked
    runs = []
    for seed in range(3):
        set_determinism(seed)
        Pg = {s: (torch.softmax(Z2["text"][s], 1), torch.softmax(Z2["audio"][s], 1)) for s in ("gate", "dev")}
        g = train_gate(H["gate"][0], H["gate"][1], *Pg["gate"], y["gate"], (H["dev"][0], H["dev"][1], *Pg["dev"], y["dev"]), w_gate, seed)
        g.eval()
        with torch.no_grad():
            lin1, lin2 = g.net[0], g.net[2]
            Wt1, Wa1, b1 = lin1.weight[:, :384], lin1.weight[:, 384:], lin1.bias
            w2, bb2 = lin2.weight[0], lin2.bias[0]
            At_pool, Aa_pool = H["clf"][0] @ Wt1.T, H["clf"][1] @ Wa1.T
            At_i, Aa_i = H["test"][0] @ Wt1.T, H["test"][1] @ Wa1.T
            w0 = g(*H["test"]).squeeze(1)

            def pd_fn(kind, sl):
                return torch.softmax(donor[kind][sl][None] + ctxp[kind][:, None], 2)

            def wd_fn(kind, sl):
                pre = At_pool[sl][None] + Aa_i[:, None] if kind == "text" else At_i[:, None] + Aa_pool[sl][None]
                return torch.sigmoid(torch.relu(pre + b1) @ w2 + bb2)

            n = p_t.shape[0]
            # unmasked per-utterance vote means (context makes m depend on the utterance), indexed at the predicted class later
            o = audit(p_t, p_a, w0, pd_fn, wd_fn, ok)
            idx = torch.arange(n)
            c = o["c"]
            mt_c, ma_c = votes_unmasked["text"][idx, c], votes_unmasked["audio"][idx, c]
            closed_t = w0 * (o["vote_text"][idx, c] - p_t[idx, c])
            closed_a = (1 - w0) * (o["vote_audio"][idx, c] - p_a[idx, c])
            corr34 = w0 * (p_t[idx, c] - mt_c).abs() - (1 - w0) * (p_a[idx, c] - ma_c).abs()
            r = {"e_real": (o["dp_real_text"].abs() - o["dp_real_audio"].abs()).numpy(),
                 "e_fro": (o["dp_fro_text"].abs() - o["dp_fro_audio"].abs()).numpy(),
                 "e_closed": (closed_t.abs() - closed_a.abs()).numpy(), "e_corr34": corr34.numpy(), "w": w0.numpy(),
                 "G_t": (o["dp_real_text"] - o["dp_fro_text"]).numpy(), "G_a": (o["dp_real_audio"] - o["dp_fro_audio"]).numpy(),
                 "dpf_t": o["dp_fro_text"].numpy(), "dpf_a": o["dp_fro_audio"].numpy(),
                 "max_dp_err": float(max((o["dp_fro_text"] - closed_t).abs().max(), (o["dp_fro_audio"] - closed_a).abs().max()))}
            runs.append(r)
    results.append(summarise("M5 best model (63.6): text-heavy + context", runs))
    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "frozen_gate_audit.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 4)
    print(res.set_index("model").T.to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
