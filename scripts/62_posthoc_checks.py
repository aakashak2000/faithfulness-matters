"""Step 62: post-hoc checks on phase 2 (Amendment 1 of docs/controlled_series_protocol.md, written and committed in 7c27d09 before this
script was run). (a) probes with the gate's architecture and inputs: planted vs clean (arm P), which head is correct (C0);
(b) the best constant weight per arm versus the learned gates' mean shift; (c) gate-split audio NLL and audio's marginal value given text.
Exploratory, post hoc.

Usage:  python scripts/62_posthoc_checks.py
Output: results/controlled_series_posthoc.csv
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS, LearnedGate, auroc, dialogue_draws, set_determinism, spearman, train_full_batch  # noqa: E402

s59 = importlib.import_module("59_controlled_series")
N_BOOT = importlib.import_module("06_analysis").N_BOOT
U = torch.linspace(-8, 8, 1601)


def constant_gate(B, th, ah):
    """Best single weight (logit) on the gate split under the gate's training loss, and audio's marginal value given text."""
    pt, pa = torch.softmax(th(B["ET"]["gate"]), 1).detach(), torch.softmax(ah(B["EA"]["gate"]), 1).detach()
    y, wt = B["y"]["gate"], B["w_gate"]
    losses = torch.stack([F.nll_loss(torch.log((torch.sigmoid(u) * pt + (1 - torch.sigmoid(u)) * pa).clamp_min(1e-8)), y, weight=wt) for u in U])
    k = int(losses.argmin())
    text_only = F.nll_loss(torch.log(pt.clamp_min(1e-8)), y, weight=wt).item()
    audio_nll = F.nll_loss(torch.log(pa.clamp_min(1e-8)), y).item()
    return float(U[k]), text_only - losses[k].item(), audio_nll


def probe(B, target, rows, seed):
    """Gate-architecture probe on the gate's inputs. target/rows: dicts per split (bool target, bool row mask)."""
    set_determinism(seed)
    m = LearnedGate(384, 1024)
    t = {s: torch.tensor(target[s][rows[s]], dtype=torch.float32) for s in ("gate", "dev")}
    X = {s: (B["H"][s][0][torch.tensor(rows[s])], B["H"][s][1][torch.tensor(rows[s])]) for s in ("gate", "dev", "test")}
    train_full_batch(m, lambda: F.binary_cross_entropy(m(*X["gate"]).squeeze(1), t["gate"]),
                     lambda: F.binary_cross_entropy(m(*X["dev"]).squeeze(1), t["dev"]))
    with torch.no_grad():
        return m(*X["test"]).squeeze(1).numpy()


def main():
    B = ac.build_harness()
    th, _ = B["specs"]["M1 stronger audio"]
    settings = pd.read_csv(RESULTS / "controlled_series_arm_settings.csv", index_col=0)
    cs = pd.read_csv(RESULTS / "controlled_series.csv")
    learned = cs[(cs.analysis == "retrained_gate") & (cs.metric == "mean_logit_w")].set_index("arm").value
    Z = np.load(DATA / "controlled_series_arrays.npz")
    rows = []
    h0 = s59.fit_audio_head(B)
    heads = {"C0": h0}
    n_clf = len(B["y"]["clf"])
    for s in (0, 1, 2):
        r = np.sort(np.random.default_rng(s).choice(n_clf, int(round(0.25 * n_clf)), replace=False))
        h = s59.fit_audio_head(B, torch.tensor(r))
        heads[f"C1_seed{s}"] = h
        heads[f"C2_seed{s}"] = s59.TempHead(h, settings.loc[f"C2_seed{s}", "T"])
    for a in ("S0", "S1", "S2"):
        heads[a] = s59.TempHead(h0, settings.loc[a, "T"])
    const = {a: constant_gate(B, th, h) for a, h in heads.items()}

    # (a2) C0: which head is correct, on rows where exactly one is
    tgt, rws = {}, {}
    for sp in ("gate", "dev", "test"):
        y = B["y"][sp].numpy()
        t_ok = s59.probs(th, B["ET"][sp]).argmax(1) == y
        a_ok = s59.probs(h0, B["EA"][sp]).argmax(1) == y
        tgt[sp], rws[sp] = t_ok, t_ok != a_ok
    pr_c0 = np.mean([probe(B, tgt, rws, s) for s in (0, 1, 2)], 0)
    lab_c0 = tgt["test"][rws["test"]]
    w_c0 = Z["C0__w"][:, rws["test"]]

    # (P) plant in place, then constant gate and the planted probe
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    planted = s59.plant(B, P["ids"]["train"], P["rows"], B["D"]["meta"])
    assert (planted["test"] == Z["P__planted"].astype(bool)).all(), "plant does not reproduce step 59"
    hp = s59.fit_audio_head(B)
    const["P"] = constant_gate(B, th, hp)
    allrows = {sp: np.ones(len(planted[sp]), bool) for sp in ("gate", "dev", "test")}
    pr_p = np.mean([probe(B, planted, allrows, s) for s in (0, 1, 2)], 0)
    w_p = Z["P__w"]

    # ---- (a) bootstrap: probe vs gate on the same draws
    meta = B["D"]["meta"]["test"]
    draws = dialogue_draws(meta["dialogue_id"].values, N_BOOT)
    idx_c0 = np.flatnonzero(rws["test"])
    pos = {r: i for i, r in enumerate(idx_c0)}

    def stat(d):
        dc = np.array([pos[r] for r in d if r in pos])
        r = {"probe_auroc_which_head_correct_C0": auroc(pr_c0[dc], lab_c0[dc]),
             "gate_auroc_which_head_correct_C0": np.mean([auroc(w[dc], lab_c0[dc]) for w in w_c0]),
             "probe_auroc_planted_P": auroc(pr_p[d], planted["test"][d]),
             "gate_auroc_planted_P": np.mean([auroc(w[d], planted["test"][d]) for w in w_p])}
        r["probe_minus_gate_C0"] = r["probe_auroc_which_head_correct_C0"] - r["gate_auroc_which_head_correct_C0"]
        r["probe_minus_gate_P"] = r["probe_auroc_planted_P"] - r["gate_auroc_planted_P"]
        return r
    p_ = stat(np.arange(len(meta)))
    bs = [stat(d) for d in draws]
    for k, v in p_.items():
        lo, hi = np.percentile([b[k] for b in bs], [2.5, 97.5])
        rows.append({"check": "a_probe", "arm": "", "metric": k, "value": v, "lo": lo, "hi": hi})

    # ---- (b) and (c)
    group = {"C1": ["C1_seed0", "C1_seed1", "C1_seed2"], "C2": ["C2_seed0", "C2_seed1", "C2_seed2"], "S1": ["S1"], "S2": ["S2"], "P": ["P"]}
    ref = {"C1": "C0", "C2": "C0", "P": "C0", "S1": "S0", "S2": "S0"}
    pts = []
    for g, members in group.items():
        c_logit = np.mean([const[m][0] for m in members])
        c_shift = c_logit - const[ref[g]][0]
        l_shift = learned[g] - learned[ref[g]]
        d_nll = np.mean([const[m][2] for m in members]) - const[ref[g]][2]
        d_mv = np.mean([const[m][1] for m in members]) - const[ref[g]][1]
        pts.append((l_shift, d_nll, d_mv))
        for k, v in (("constant_gate_logit", c_logit), ("constant_gate_shift", c_shift), ("learned_mean_logit_shift", l_shift),
                     ("constant_shift_over_learned_shift", c_shift / l_shift), ("gate_split_audio_nll_change", d_nll),
                     ("audio_marginal_value_change", d_mv)):
            rows.append({"check": "b_c", "arm": g, "metric": k, "value": v})
    for a in ("C0", "S0", "C2_seed0", "C2_seed1", "C2_seed2", "C1_seed0", "C1_seed1", "C1_seed2", "P"):
        rows.append({"check": "c", "arm": a, "metric": "gate_split_audio_nll", "value": const[a][2]})
        rows.append({"check": "c", "arm": a, "metric": "audio_marginal_value", "value": const[a][1]})
    pts = np.array(pts)
    rows.append({"check": "c", "arm": "all", "metric": "spearman_shift_vs_gate_split_nll_change", "value": spearman(pts[:, 0], pts[:, 1])})
    rows.append({"check": "c", "arm": "all", "metric": "spearman_shift_vs_marginal_value_change", "value": spearman(pts[:, 0], pts[:, 2])})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "controlled_series_posthoc.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
