"""Step 38: frozen-gate audit of the corrected attribution signal (methodological audit, 2026-10-21).

Questions, fixed before running:
  Q1  Is s_corr = w |p_t[c] - m_t[c]| - (1 - w) |p_a[c] - m_a[c]| (step 34) EXACTLY the effect difference of a swap in
      which the gate's weight is frozen at its original value w? Notation: c = class predicted by the original fused
      probability, m = mean donor probability over the valid donors (byte-identical headline embeddings excluded), effect
      of a modality = mean over donors of the fused probability of c, minus the original fused probability of c (the
      primary effect of 06/33: the donor-AVERAGED fused vector, then the change in class c, then absolute value).
  Q2  Three quantities on the SAME utterances and donor sets:
        real    = swap with the gate recomputed for each donor (as in 08/33/37)
        frozen  = swap with the gate weight held at the original w, computed by looping over donors (no formula)
        s_corr  = step 34's formula (m over the whole clf pool, as in step 34) and a masked version (m over valid donors)
      Report: sign agreement, Spearman, absolute errors, and the check frozen = closed form to floating-point precision.
  Q3  Exact decomposition: real dp = frozen dp + gate term, gate term = mean over donors of (w_d - w) (P_d[c] - p_other[c])
      for the swapped modality. Report how often the gate terms reverse which modality has the larger effect, how large
      the terms are against the margin, and what the reversed utterances look like. Nothing is assumed: the reversal
      share is the measured share of sign(real) != sign(frozen).
  Validation: my real effect must equal the effect from rs08.exact_average (the pipeline behind the main results).
  Models: M0b, M1, M2, M3 of step 33 (10 gates each, test set) and M5 of step 37 (the 63.6 model, 3 gates).
  No tuning, no test selection, nothing is fitted.

Usage: python scripts/38_frozen_gate.py
Output: results/frozen_gate_audit.csv
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
from common import (DATA, RESULTS, Head, LearnedGate, class_weights, load_dataset, set_determinism,  # noqa: E402
                    train_gate, train_head)

b22 = importlib.import_module("22_better_audio")
m29 = importlib.import_module("29_context_pipeline")
rs07 = importlib.import_module("07_resampling")
rs08 = importlib.import_module("08_resampling_full")
iv04 = importlib.import_module("04_interventions")
CHUNK = 100


def core(pt, pa, w0, pool_logits_fn, gate_pre_fn, w2, b2c, ok_t, ok_a):
    """Loop over donors. pt, pa: (n,7); w0: (n,); returns dict of tensors.
    pool_logits_fn(kind, sl) -> donor probability (n or 1, k, 7); gate_pre_fn(kind, sl) -> (n, k, hidden) pre-activation + bias."""
    n = pt.shape[0]
    idx = torch.arange(n)
    p0 = w0[:, None] * pt + (1 - w0[:, None]) * pa
    c = p0.argmax(1)
    out = {"c": c, "p0c": p0[idx, c]}
    for kind, ok in (("text", ok_t), ("audio", ok_a)):
        Sreal, Sfro, Vote = torch.zeros(n, 7), torch.zeros(n, 7), torch.zeros(n, 7)
        Areal, Afro = torch.zeros(n), torch.zeros(n)      # per-donor |change| of class c, summed (mean-of-absolute variant)
        G = torch.zeros(n)
        for lo in range(0, ok.shape[1], CHUNK):
            sl = slice(lo, lo + CHUNK)
            m_ = ok[:, sl].float()                                        # (n, k)
            Pd = pool_logits_fn(kind, sl)                                  # (n|1, k, 7)
            Pd = Pd.expand(n, -1, -1)
            wd = torch.sigmoid(torch.relu(gate_pre_fn(kind, sl)) @ w2 + b2c)   # (n, k) gate recomputed for each donor
            other = pa if kind == "text" else pt
            # real: swapped modality's vote from the donor, gate recomputed
            # fused_d = w_t,d * P_text,d + (1 - w_t,d) * p_a  (text swap)  or  w_t,d * p_t + (1 - w_t,d) * P_audio,d (audio swap)
            if kind == "text":
                real = wd[:, :, None] * Pd + (1 - wd[:, :, None]) * other[:, None, :]
                fro = w0[:, None, None] * Pd + (1 - w0[:, None, None]) * other[:, None, :]
            else:
                real = wd[:, :, None] * other[:, None, :] + (1 - wd[:, :, None]) * Pd
                fro = w0[:, None, None] * other[:, None, :] + (1 - w0[:, None, None]) * Pd
            Sreal += (m_[:, :, None] * real).sum(1)
            Sfro += (m_[:, :, None] * fro).sum(1)
            Vote += (m_[:, :, None] * Pd).sum(1)
            gi = c[:, None, None].expand(n, real.shape[1], 1)
            Areal += (m_ * (real.gather(2, gi).squeeze(2) - out['p0c'][:, None]).abs()).sum(1)
            Afro += (m_ * (fro.gather(2, gi).squeeze(2) - out['p0c'][:, None]).abs()).sum(1)
        cnt = ok.sum(1).float()[:, None]
        out[f"real_{kind}"] = (Sreal / cnt)[idx, c] - out["p0c"]
        out[f"fro_{kind}"] = (Sfro / cnt)[idx, c] - out["p0c"]
        out[f"mabs_real_{kind}"] = Areal / cnt.squeeze(1)
        out[f"mabs_fro_{kind}"] = Afro / cnt.squeeze(1)
        out[f"vote_{kind}"] = (Vote / cnt)[idx, c]                        # m[c], masked
        out[f"voteall_{kind}"] = None
    return out


def rank(a):
    return rankdata(a)


def summarise(name, per_seed):
    ag = lambda a, b: np.mean([(np.sign(x[(x != 0) & (y != 0)]) == np.sign(y[(x != 0) & (y != 0)])).mean() for x, y in zip(a, b)])
    sp = lambda a, b: np.mean([np.corrcoef(rank(x), rank(y))[0, 1] for x, y in zip(a, b)])
    R = {k: np.stack([s[k] for s in per_seed]) for k in per_seed[0]}
    np.savez_compressed(DATA / f"frozen_gate_arrays_{name.split()[0]}.npz", **R)      # local, used by 40_audit_bootstrap.py
    e_real, e_fro, e_c34, e_cm, e_w = R["e_real"], R["e_fro"], R["e_c34"], R["e_cmask"], R["s_w"]
    dis = e_real * e_fro < 0                                              # real and frozen disagree on which modality won
    marg_fro = np.abs(e_fro)
    row = {"model": name, "n_gates": len(per_seed),
           "p_text_real": float((e_real[e_real != 0] > 0).mean()), "p_text_frozen": float((e_fro[e_fro != 0] > 0).mean()),
           "agree_real_vs_frozen": ag(e_real, e_fro), "agree_real_vs_corr34": ag(e_real, e_c34),
           "agree_frozen_vs_corr34": ag(e_fro, e_c34), "agree_frozen_vs_corr_masked": ag(e_fro, e_cm),
           "agree_weight_vs_real": ag(e_w, e_real), "agree_weight_vs_frozen": ag(e_w, e_fro),
           "spearman_real_vs_frozen": sp(e_real, e_fro), "spearman_real_vs_corr34": sp(e_real, e_c34),
           "spearman_frozen_vs_corr34": sp(e_fro, e_c34), "spearman_frozen_vs_corr_masked": sp(e_fro, e_cm),
           "max_abs_frozen_vs_closed_form_dp": float(max(R["err_cf_text"].max(), R["err_cf_audio"].max())),
           "mean_abs_e_corr34_vs_frozen": float(np.abs(e_c34 - e_fro).mean()), "max_abs_e_corr34_vs_frozen": float(np.abs(e_c34 - e_fro).max()),
           "mean_abs_e_corrmasked_vs_frozen": float(np.abs(e_cm - e_fro).mean()), "max_abs_e_corrmasked_vs_frozen": float(np.abs(e_cm - e_fro).max()),
           "max_abs_real_vs_pipeline_dp": float(R["val_err"].max()) if "val_err" in R else np.nan,
           "share_reversed_by_gate": float(dis.mean()),
           "median_margin_frozen_all": float(np.median(marg_fro)), "median_margin_frozen_reversed": float(np.median(marg_fro[dis])) if dis.any() else np.nan,
           "median_abs_gate_term_text": float(np.median(np.abs(R["g_text"]))), "median_abs_gate_term_audio": float(np.median(np.abs(R["g_audio"]))),
           "median_abs_frozen_effect_text": float(np.median(np.abs(R["fro_text"]))), "median_abs_frozen_effect_audio": float(np.median(np.abs(R["fro_audio"]))),
           "share_reversed_with_frozen_text_win": float((dis & (e_fro > 0)).sum() / max(dis.sum(), 1)),
           "share_reversed_with_frozen_audio_win": float((dis & (e_fro < 0)).sum() / max(dis.sum(), 1))}
    # counterfactual: reverse only the text gate term, or only the audio gate term
    for nm, tt, aa in (("text_gate_term_only", R["g_text"], 0.0), ("audio_gate_term_only", 0.0, R["g_audio"])):
        e_cf = np.abs(R["fro_text"] + tt) - np.abs(R["fro_audio"] + aa)
        row[f"agree_real_vs_frozen_plus_{nm}"] = ag(e_real, e_cf)
    return row


def per_gate(pt, pa, w0, core_out, voteall_t, voteall_a, w_pipeline):
    c = core_out["c"]
    idx = torch.arange(len(c))
    d = {}
    for kind, p_own in (("text", pt), ("audio", pa)):
        w = w0 if kind == "text" else 1 - w0
        cf = w * (core_out[f"vote_{kind}"] - p_own[idx, c])                  # closed form, masked donors, signed
        d[f"err_cf_{kind}"] = (core_out[f"fro_{kind}"] - cf).abs().numpy()
        d[f"fro_{kind}"] = core_out[f"fro_{kind}"].numpy()
        d[f"g_{kind}"] = (core_out[f"real_{kind}"] - core_out[f"fro_{kind}"]).numpy()
    d["e_real"] = (core_out["real_text"].abs() - core_out["real_audio"].abs()).numpy()
    d["e_fro"] = (core_out["fro_text"].abs() - core_out["fro_audio"].abs()).numpy()
    d["e_cmask"] = (w0 * (core_out["vote_text"] - pt[idx, c]).abs() - (1 - w0) * (core_out["vote_audio"] - pa[idx, c]).abs()).numpy()
    d["e_c34"] = (w0 * (voteall_t[idx, c] - pt[idx, c]).abs() - (1 - w0) * (voteall_a[idx, c] - pa[idx, c]).abs()).numpy()
    d["e_mabs"] = (core_out["mabs_real_text"] - core_out["mabs_real_audio"]).numpy()
    d["e_mabs_fro"] = (core_out["mabs_fro_text"] - core_out["mabs_fro_audio"]).numpy()
    d["s_w"] = (2 * w0 - 1).numpy()
    return d


def harness_models(D, S, mp):
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
    specs = {"M0b headline recipe, v2 split": (T_MINI, A_HEAD), "M1 stronger audio": (T_MINI, A_STR),
             "M2 stronger text": (T_ROB, A_HEAD), "M3 stronger text + audio": (T_ROB, A_STR)}
    ok_t, ok_a = rs07.valid_masks(H["test"][0], H["clf"][0]), rs07.valid_masks(H["test"][1], H["clf"][1])
    return specs, (ET, EA, H, y, w_gate, dT, dA), (ok_t, ok_a)


def main():
    t0 = time.time()
    D = load_dataset()
    S = D["splits"]
    mp = pd.read_csv(RESULTS / "strong_audio_row_map.csv")
    specs, (ET, EA, H, y, w_gate, dT, dA), (ok_t, ok_a) = harness_models(D, S, mp)
    print(f"setup done ({time.time() - t0:.0f}s)", flush=True)
    res = []
    for name, (th, ah) in specs.items():
        with torch.no_grad():
            Pp_t, Pp_a = torch.softmax(th(ET["clf"]), 1), torch.softmax(ah(EA["clf"]), 1)
            vall = (Pp_t.mean(0, keepdim=True).expand(len(ET["test"]), -1), Pp_a.mean(0, keepdim=True).expand(len(ET["test"]), -1))
            pt, pa = torch.softmax(th(ET["test"]), 1), torch.softmax(ah(EA["test"]), 1)
        per = []
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
            lin1, lin2 = Gp.net[0], Gp.net[2]
            Wtt, Waa, b1 = lin1.weight[:, :dT + 384], lin1.weight[:, dT + 384:], lin1.bias
            with torch.no_grad():
                w0 = Gp(ET["test"], EA["test"]).squeeze(1)
                At_pool, Aa_pool = ET["clf"] @ Wtt.T, EA["clf"] @ Waa.T
                At_i, Aa_i = ET["test"] @ Wtt.T, EA["test"] @ Waa.T
                pool_fn = lambda kind, sl: (Pp_t[sl] if kind == "text" else Pp_a[sl])[None]
                pre_fn = lambda kind, sl: ((At_pool[sl][None] + Aa_i[:, None]) if kind == "text" else (At_i[:, None] + Aa_pool[sl][None])) + b1
                out = core(pt, pa, w0, pool_fn, pre_fn, lin2.weight[0], lin2.bias[0], ok_t, ok_a)
                d = per_gate(pt, pa, w0, out, vall[0], vall[1], None)
                if seed == 0:       # validation against the pipeline behind the main results
                    old = iv04.load_models
                    iv04.load_models = lambda s_: (th, ah, Gp)
                    try:
                        orig, _, _, cls, pb, _, _ = rs08.exact_average(0, ET["test"], EA["test"], ET["clf"], EA["clf"], ok_t, ok_a)
                    finally:
                        iv04.load_models = old
                    c = cls["fusion_learned_gate"]
                    idx = torch.arange(len(c))
                    ref_t = pb["fusion_learned_gate"][0][idx, c] - orig["fusion_learned_gate"][0][idx, c]
                    ref_a = pb["fusion_learned_gate"][1][idx, c] - orig["fusion_learned_gate"][0][idx, c]
                    val = max((out["real_text"] - ref_t).abs().max().item(), (out["real_audio"] - ref_a).abs().max().item())
                    print(f"  {name}: my real effect vs rs08.exact_average max |diff| {val:.2e}", flush=True)
                    assert (out["c"] == c).all()
                d["val_err"] = np.array([val if seed == 0 else 0.0])
            per.append(d)
        r = summarise(name, per)
        res.append(r)
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "model"}, f"[{time.time() - t0:.0f}s]", flush=True)
    pd.DataFrame(res).to_csv(RESULTS / "frozen_gate_audit.csv", index=False)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
