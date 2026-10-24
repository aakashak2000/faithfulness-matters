"""Step 45: descriptive diagnostic of the gate-input mismatch (audit follow-up P1), no training and no swaps. Written 2026-10-23.

The gate reads the MiniLM text embedding (384-d) and the 1,024-d audio embedding in every model (steps 29/33). In M2 and M3 the text HEAD reads
RoBERTa features (25,600-d), so the gate cannot see the representation that drives the text prediction; in M0b and M1 the text head and the gate
read the same MiniLM embedding. For each model (10 cached gates from step 42, test set) we report, descriptively:
  - Spearman correlation of the gate weight w with the confidence-rule weight c_t / (c_t + c_a), with c_t and with c_a;
  - AUC of w for telling apart utterances where only the text head is correct from utterances where only the audio head is correct
    (a reliability-detection measure; 0.5 = no information).
No hypothesis is tested here; the controlled test is H4 in docs/final_experiment_plan.md.
Output: results/gate_input_diagnostic.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402
from common import RESULTS  # noqa: E402


def main():
    B = ac.build_harness()
    y = B["y"]["test"].numpy()
    cache_tag = {"M0b headline recipe, v2 split": "M0b", "M1 stronger audio": "M1", "M2 stronger text": "M2", "M3 stronger text + audio": "M3"}
    rows = []
    for name, (th, ah) in B["specs"].items():
        tag = cache_tag[name]
        gates = torch.load(ac.CACHE / f"gates_{tag}.pt")
        with torch.no_grad():
            pt, pa = torch.softmax(th(B["ET"]["test"]), 1), torch.softmax(ah(B["EA"]["test"]), 1)
        ct, ca = pt.max(1).values.numpy(), pa.max(1).values.numpy()
        t_ok, a_ok = pt.argmax(1).numpy() == y, pa.argmax(1).numpy() == y
        only = t_ok != a_ok
        res = {k: [] for k in ("rho_w_conf", "rho_w_ct", "rho_w_ca", "auc_text_only_vs_audio_only", "mean_w", "share_text_correct", "share_audio_correct", "share_only_one_correct")}
        for seed, sd in gates.items():
            G_ = ac.LearnedGate(B["dT"] + 384, B["dA"] + 1024)
            G_.load_state_dict(sd)
            G_.eval()
            with torch.no_grad():
                w = G_(B["ET"]["test"], B["EA"]["test"]).squeeze(1).numpy()
            res["rho_w_conf"].append(spearmanr(w, ct / (ct + ca))[0])
            res["rho_w_ct"].append(spearmanr(w, ct)[0])
            res["rho_w_ca"].append(spearmanr(w, ca)[0])
            res["auc_text_only_vs_audio_only"].append(roc_auc_score(t_ok[only], w[only]))
            res["mean_w"].append(w.mean())
            res["share_text_correct"].append(t_ok.mean())
            res["share_audio_correct"].append(a_ok.mean())
            res["share_only_one_correct"].append(only.mean())
        r = {"model": tag, "n_gates": len(gates), "text_head_reads": "MiniLM (same as gate)" if tag in ("M0b", "M1") else "RoBERTa all layers (gate sees MiniLM only)"}
        r.update({k: float(np.mean(v)) for k, v in res.items()})
        r["auc_sd_across_gates"] = float(np.std(res["auc_text_only_vs_audio_only"], ddof=1))
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "gate_input_diagnostic.csv", index=False)
    print(d.round(4).to_string())


if __name__ == "__main__":
    main()
