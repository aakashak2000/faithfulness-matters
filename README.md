# Faithfulness MATTERS

**Modality Attribution Testing Through Emotion Recognition Signals**
CSCI 544, Fall 2026, Team 47: Tyler Beede, Amogh Inamdar, Jayant Nagpal, Aria Samandi, Ryan Sanchez, Aakash Kumar

When a multimodal emotion model says "I relied on the tone of voice for this one," is that true?
We take a text + audio emotion classifier on MELD, read off what it *claims* about which modality
mattered (fusion-gate weights, per-modality confidence), and test that claim against what actually
happens to its prediction when we remove the text or the audio.

**Team:** the midterm status report (results, what they mean, what's left) and the reading guide are shared separately in the group chat. Read those first, then come back here for the code.

## Pipeline

```
MELD utterance ──► MiniLM (frozen) ──► text embedding (384) ──► text head  ──► p_text ─┐
            └────► WavLM-large (frozen) ► audio embedding (1024) ► audio head ► p_audio ┤
                                                                                       ▼
                         fused p = w_t · p_text + (1 − w_t) · p_audio     w_t from one of:
                                                                            average     0.5
                                                                            confidence  c_t / (c_t + c_a)
                                                                            learned     gate MLP([e_t; e_a])
```

Interventions replace one modality's embedding (mean training embedding, or empty text / silent audio),
rerun the whole model, and measure how much the prediction changes. Faithfulness = does the modality the
model *weighted* more match the modality whose removal *changed* the prediction more?

## Scripts (run in order)

| Step | Script | What it does | Runtime |
|---|---|---|---|
| 0 | `scripts/00_inspect_assets.py` | Prints the structure of the legacy `.pt` files | seconds |
| 1 | `scripts/01_build_dataset.py` | Builds `data/dataset.pt` from the legacy bundle, aligns it to the MELD CSVs, runs the audit | ~1 min |
| 2 | `scripts/02_train.py` | Trains text/audio heads + learned gate, 3 seeds; baseline F1 tables | ~1 min |
| 3 | `scripts/03_silence_embeddings.py` | WavLM on silent audio, MiniLM on empty text; verifies extraction matches the cache | ~5 min (MPS) |
| 4 | `scripts/04_interventions.py` | Runs every intervention, logs per-utterance effects and full probabilities | seconds |
| 5 | `scripts/05_faithfulness.py` | Agreement, always-text baseline, Spearman, robustness checks | seconds |
| 6 | `scripts/06_analysis.py` | Dialogue-level bootstrap CIs and the four figures | ~30 s |

`scripts/common.py` holds the shared models, fusion rules, training loop and metrics.
`scripts/diag_*.py` are the one-off diagnostics from the data audit (order, duplicates, source-file hashes).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt librosa
```

**Data (not in git).** Ask Aakash for access to the shared Drive folder, then place:

```
data/legacy/fusion_bundle_final.pt     cached MiniLM + WavLM embeddings, leakage-safe splits
data/meld_csv/{train,dev,test}_sent_emo.csv
data/wavfiles_test/                    extracted test WAVs (only needed for step 3)
```

The MELD CSVs can also be fetched from the official repo:

```bash
mkdir -p data/meld_csv && cd data/meld_csv
curl -sSfLO https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv
curl -sSfLO https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv
curl -sSfLO https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv
```

Then run steps 1 to 6. Everything after step 3 runs on CPU in under a minute.

## Outputs

| Path | Contents |
|---|---|
| `results/audit_step1.json` | Data audit: alignment, label mapping, duplicates, split stratification |
| `results/baselines_summary.csv`, `baselines_per_seed.csv`, `per_class_f1.csv` | Classification results |
| `results/legacy_reproduction.csv` | Old cached probabilities scored the same way (reproduction check) |
| `results/faithfulness_summary.csv`, `faithfulness_per_seed.csv` | Faithfulness metrics |
| `results/bootstrap_ci.csv`, `agreement_counts.csv` | 95% intervals and denominators |
| `results/method_robustness.csv` | Mean vs blank intervention agreement |
| `results/gate_calibration.csv` | Share where text mattered more, by gate weight |
| `figures/fig1..fig4_*.png` | Presentation and report figures |
