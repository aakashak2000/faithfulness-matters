"""Step 59: phase 2 controlled series on M1: what does a likelihood-trained gate adapt to? Protocol: docs/controlled_series_protocol.md
(revision 2, committed in d19b5b0 before this script was written or run). Exploratory. Only the AUDIO head is manipulated.

Arms: C0 (refit of the M1 audio head), C1 (fit on 25% of clf; 3 subset seeds), C2 (C1 + dev temperature restoring mean
distinctiveness), S0/S1/S2 (C0 at T*, T*/2, 2T*; T* = dev-NLL-optimal temperature), P (30% of rows in every split get another
dialogue's audio; head and gates refitted). Gate seeds 0, 1, 2 per head. Influence always on the full donor pool.

Usage:  python scripts/59_controlled_series.py
Output: data/controlled_series_arrays.npz, results/controlled_series.csv, results/controlled_series_manipulation_check.csv
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import DATA, RESULTS, Head, class_weights, dialogue_draws, ece  # noqa: E402

b22 = importlib.import_module("22_better_audio")
N_BOOT = importlib.import_module("06_analysis").N_BOOT
AUDIO_WD = 1.0
ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]
GATE_SEEDS = (0, 1, 2)
SUBSET_SEEDS = (0, 1, 2)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


class TempHead(nn.Module):
    def __init__(self, base, T):
        super().__init__()
        self.base, self.T = base, float(T)

    def forward(self, x):
        return self.base(x) / self.T


def fit_audio_head(B, rows=None):
    dA = B["dA"]
    X = B["EA"]["clf"][:, :dA] if rows is None else B["EA"]["clf"][rows, :dA]
    y = B["y"]["clf"] if rows is None else B["y"]["clf"][rows]
    W, b = b22.fit(X, y, class_weights(y), AUDIO_WD)
    h = Head(dA + 1024)
    w = torch.zeros(7, dA + 1024)
    w[:, :dA] = W.T
    h.lin.weight.data, h.lin.bias.data = w, b.clone()
    h.eval()
    return h


def probs(head, E):
    with torch.no_grad():
        return torch.softmax(head(E), 1).numpy()


def pool_mean(P_clf, ok):
    okf = ok.float().numpy()
    return (okf @ P_clf) / okf.sum(1, keepdims=True)


def mean_d0(B, th, ah, split, kind="audio"):
    """Mean distinctiveness of a head at the gate-free class, full valid pool, on `split`."""
    pt, pa = probs(th, B["ET"][split]), probs(ah, B["EA"][split])
    Pc = probs(ah, B["EA"]["clf"]) if kind == "audio" else probs(th, B["ET"]["clf"])
    ok = B["ok"][split][1 if kind == "audio" else 0]
    m = pool_mean(Pc, ok)
    c0 = (0.5 * pt + 0.5 * pa).argmax(1)
    own = pa if kind == "audio" else pt
    ii = np.arange(len(c0))
    return np.abs(own[ii, c0] - m[ii, c0]).mean()


def nll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def head_check(B, th, ah, arm):
    rows = []
    for split in ("dev", "test"):
        y = B["y"][split].numpy()
        pa = probs(ah, B["EA"][split])
        rows.append({"arm": arm, "split": split, "audio_accuracy": (pa.argmax(1) == y).mean(), "audio_nll": nll(pa, y).mean(),
                     "audio_ece": ece(pa, y, bins=15), "audio_mean_d0": mean_d0(B, th, ah, split)})
    return rows


def run_arm(B, th, ah, gates=None):
    """Train gates (unless given), pick alpha on dev, evaluate on test with the full pool. Returns a dict of arrays per gate."""
    if gates is None:
        gates = [ac.harness_gate(B, th, ah, s) for s in GATE_SEEDS]
    yd = B["y"]["dev"].numpy()
    ptd, pad = probs(th, B["ET"]["dev"]), probs(ah, B["EA"]["dev"])
    lp = B["lp"].numpy()
    f1s = []
    for a in ALPHAS:
        s = []
        for g in gates:
            with torch.no_grad():
                w = g(B["ET"]["dev"], B["EA"]["dev"]).squeeze(1).numpy()
            p = w[:, None] * ptd + (1 - w[:, None]) * pad
            s.append(f1_score(yd, (np.log(np.clip(p, 1e-12, None)) + a * lp).argmax(1), average="weighted", zero_division=0))
        f1s.append(np.mean(s))
    alpha = ALPHAS[int(np.argmax(f1s))]
    outs = [ac.harness_split(B, th, ah, g, "test", alpha) for g in gates]
    pt, pa = outs[0][0].numpy(), outs[0][1].numpy()
    keys = ("e_cor_real", "e_cor_fro", "cor_real_text_mean", "cor_fro_text_mean", "cor_real_audio_mean", "cor_fro_audio_mean", "c_cor")
    R = {k: np.stack([o[3][k] for o in outs]) for k in keys}
    R["w"] = np.stack([o[2].numpy() for o in outs])
    R["pt"], R["pa"], R["alpha"] = pt, pa, alpha
    R["mt"] = pool_mean(probs(th, B["ET"]["clf"]), B["ok"]["test"][0])
    R["ma"] = pool_mean(probs(ah, B["EA"]["clf"]), B["ok"]["test"][1])
    R["dev_audio_nll"] = nll(pad, yd).mean()
    return R, gates


def plant(B, ids_train, rows_v2, meta):
    """In place: 30% of rows in every split get the audio features of a random row from a different dialogue in the same split."""
    rng = np.random.default_rng(2026)
    planted = {}
    dl = {"clf": np.array([ids_train[i][0] for i in np.asarray(rows_v2["clf"])]),
          "gate": np.array([ids_train[i][0] for i in np.asarray(rows_v2["gate"])]),
          "dev": meta["dev"]["dialogue_id"].values, "test": meta["test"]["dialogue_id"].values}
    for split in ("clf", "gate", "dev", "test"):
        n = len(dl[split])
        pick = np.sort(rng.choice(n, int(round(0.3 * n)), replace=False))
        src = np.empty(len(pick), int)
        for k, i in enumerate(pick):
            j = rng.integers(n)
            while dl[split][j] == dl[split][i]:
                j = rng.integers(n)
            src[k] = j
        ea_src, ha_src = B["EA"][split][src].clone(), B["H"][split][1][src].clone()
        B["EA"][split][pick] = ea_src
        ht, ha = B["H"][split]
        ha = ha.clone()
        ha[pick] = ha_src
        B["H"][split] = (ht, ha)
        mask = np.zeros(n, bool)
        mask[pick] = True
        planted[split] = mask
    for split in ("dev", "test"):
        B["ok"][split] = (B["ok"][split][0], ac.rs07.valid_masks(B["H"][split][1], B["H"]["clf"][1]))
    return planted


def main():
    B = ac.build_harness()
    th, ah_orig = B["specs"]["M1 stronger audio"]
    log("harness built")
    P = torch.load(DATA / "v2_pipeline.pt", weights_only=False)
    checks, arms, ref = [], {}, {}

    # ---- C0 and validation
    h0 = fit_audio_head(B)
    d = np.abs(probs(h0, B["EA"]["dev"]) - probs(ah_orig, B["EA"]["dev"])).max()
    log(f"VALIDATION C0 refit vs original M1 audio head, dev probabilities: max |diff| {d:.1e}")
    assert d < 1e-4
    checks += head_check(B, th, h0, "C0")
    arms["C0"], g_c0 = run_arm(B, th, h0)
    log(f"C0 done (alpha {arms['C0']['alpha']})")
    d0_c0_dev = mean_d0(B, th, h0, "dev")

    # ---- C1 and C2 (three subset seeds), with the fixed fallback 25% -> 10%
    n_clf = len(B["y"]["clf"])
    for frac in (0.25, 0.10):
        c1_heads = []
        for s in SUBSET_SEEDS:
            rows = np.sort(np.random.default_rng(s).choice(n_clf, int(round(frac * n_clf)), replace=False))
            c1_heads.append(fit_audio_head(B, torch.tensor(rows)))
        yd = B["y"]["dev"].numpy()
        acc0, nll0 = (probs(h0, B["EA"]["dev"]).argmax(1) == yd).mean(), nll(probs(h0, B["EA"]["dev"]), yd)
        drops = [acc0 - (probs(h, B["EA"]["dev"]).argmax(1) == yd).mean() for h in c1_heads]
        ddev = dialogue_draws(B["D"]["meta"]["dev"]["dialogue_id"].values, N_BOOT)
        nll_diff = [np.mean([(nll(probs(h, B["EA"]["dev"]), yd)[dd] - nll0[dd]).mean() for h in c1_heads]) for dd in ddev]
        nll_lo = np.percentile(nll_diff, 2.5)
        log(f"manipulation check C1 frac {frac}: dev accuracy drops {np.round(drops, 4)}, dev NLL increase lower bound {nll_lo:.4f}")
        if np.mean(drops) >= 0.02 or nll_lo > 0:
            break
        log("C1 manipulation failed the pre-registered check; using the fixed fallback (10%)")
    c1_frac = frac
    for s, h in zip(SUBSET_SEEDS, c1_heads):
        checks += head_check(B, th, h, f"C1_seed{s}")
        arms[f"C1_seed{s}"], _ = run_arm(B, th, h)
        arms[f"C1_seed{s}_C0gates"], _ = run_arm(B, th, h, gates=g_c0)
        lo, hi = 0.05, 20.0                      # C2: temperature with mean dev distinctiveness equal to C0's (bisection, d falls with T)
        for _ in range(40):
            mid = np.sqrt(lo * hi)
            if mean_d0(B, th, TempHead(h, mid), "dev") > d0_c0_dev:
                lo = mid
            else:
                hi = mid
        T2 = np.sqrt(lo * hi)
        hc2 = TempHead(h, T2)
        checks += head_check(B, th, hc2, f"C2_seed{s}")
        arms[f"C2_seed{s}"], _ = run_arm(B, th, hc2)
        arms[f"C2_seed{s}"]["T"] = T2
        arms[f"C2_seed{s}_C0gates"], _ = run_arm(B, th, hc2, gates=g_c0)
        log(f"C1/C2 seed {s} done (C2 temperature {T2:.3f})")

    # ---- S arms around the dev-NLL-optimal temperature of C0
    yd = B["y"]["dev"].numpy()
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 61))
    with torch.no_grad():
        zd = h0(B["EA"]["dev"])
    nlls = [F.cross_entropy(zd / T, B["y"]["dev"]).item() for T in grid]
    Tstar = float(grid[int(np.argmin(nlls))])
    log(f"C0 audio head dev-NLL-optimal temperature T* = {Tstar:.3f}")
    for name, T in (("S0", Tstar), ("S1", Tstar / 2), ("S2", Tstar * 2)):
        hs = TempHead(h0, T)
        checks += head_check(B, th, hs, name)
        arms[name], _ = run_arm(B, th, hs)
        arms[name]["T"] = T
        if name != "S0":
            arms[f"{name}_C0gates"], _ = run_arm(B, th, hs, gates=g_c0)
        log(f"{name} done (T {T:.3f})")

    # ---- P (planted), last: modifies B in place
    planted = plant(B, P["ids"]["train"], P["rows"], B["D"]["meta"])
    hp = fit_audio_head(B)
    checks += head_check(B, th, hp, "P")
    arms["P"], _ = run_arm(B, th, hp)
    arms["P_C0gates"], _ = run_arm(B, th, hp, gates=g_c0)
    arms["P"]["planted"] = planted["test"]
    log("P done")

    pd.DataFrame(checks).to_csv(RESULTS / "controlled_series_manipulation_check.csv", index=False)
    flat = {f"{a}__{k}": v for a, R in arms.items() for k, v in R.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(DATA / "controlled_series_arrays.npz", c1_frac=c1_frac, Tstar=Tstar, **flat)
    meta_extra = {a: {k: v for k, v in R.items() if not isinstance(v, np.ndarray)} for a, R in arms.items()}
    pd.DataFrame(meta_extra).T.to_csv(RESULTS / "controlled_series_arm_settings.csv")
    log("arrays saved; run scripts/60_controlled_series_analysis.py for the statistics")


if __name__ == "__main__":
    main()
