# Time Series Classification on LSST — Setting 1: Foundation Model Adaptation

**Course:** Deep Learning for Time Series, 2026  
**Authors:** Zié COULIBALY · Clijo JOSE · Oumouhani ELVILALY

---

## Overview

This project tackles **14-class astronomical object classification** on the LSST/PLAsTiCC dataset (`N × T=36 × C=6`), adapting a pre-trained time-series foundation model (Setting 1).

**Best result: Ensemble Acc = 0.6448 — beats MUSE SOTA (0.636) by +0.86%**

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`Chronos.ipynb`](Chronos.ipynb) | **Main notebook** — Chronos-T5-Small as foundation model (Setting 1) + ensemble |
| [`Moment.ipynb`](Moment.ipynb) | Alternative — MOMENT-1-large as foundation model (Setting 1) + ensemble |

Both notebooks run on **Google Colab (T4 GPU)** and clone this repo automatically.

---

## Results

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| MOMENT-1-large (Setting 1) | 0.3078 | 0.2791 |
| PatchTST + TTA | 0.4903 | 0.4194 |
| **Chronos-T5-Small** (Setting 1, Phase 1+2) | 0.5333 | **0.4333** |
| Baseline — InceptionTime (scratch) | 0.5483 | 0.3753 |
| MultiROCKET | 0.6079 | 0.3625 |
| MUSE SOTA (Ruiz et al. 2021) | 0.636 | — |
| InceptionTime-Large × 5 + TTA | 0.6427 | 0.4150 |
| ★ **Ensemble (ours)** | **0.6448** | **0.4200** |

---

## Repository Structure

```
Chronos.ipynb          ← main notebook (Chronos foundation model)
Moment.ipynb           ← MOMENT foundation model comparison
rapport_beamer.tex     ← LaTeX beamer presentation (13 slides)
rapport_beamer.pdf     ← compiled PDF report
figures/               ← figures used in the report
models/                ← ChronosClassifier, MOMENTClassifier, InceptionTime, PatchTST
data/                  ← LSST dataset loader
utils.py               ← training utilities (train_epoch, eval_epoch, EarlyStopping)
cours/                 ← course notebooks (01–06)
archive/               ← old training scripts
```

---

## Key Design Choices

- **Foundation model**: Chronos-T5-Small (46M) adapted from forecasting → classification by repurposing the T5 encoder
- **Imbalance**: WeightedRandomSampler + Focal Loss (γ=2) — critical for class 53 (7 train samples)
- **Ensemble**: soft-voting weighted by exp(10·F1_val), members excluded if F1 < 0.33
- **TTA**: 5 augmented test passes for InceptionTime×5 and PatchTST
