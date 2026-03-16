# Time Series Classification on LSST — Setting 1: Foundation Model Adaptation

**Course:** Deep Learning for Time Series, 2026  
**Authors:** Zié COULIBALY · Clijo JOSE · Oumouhani ELVILALY

---

## Overview

This project tackles **14-class astronomical object classification** on the LSST/PLAsTiCC dataset (`N × T=36 × C=6`), adapting a pre-trained time-series foundation model (Setting 1).

**Best result: Ensemble Acc = 0.6448, Weighted F1 = 0.61**

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`Chronos.ipynb`](Chronos.ipynb) | **Main notebook** — Chronos-T5-Small as foundation model (Setting 1) + ensemble |
| [`Moment.ipynb`](Moment.ipynb) | Alternative — MOMENT-1-large as foundation model (Setting 1) + ensemble |

Both notebooks run on **Google Colab (T4 GPU)** and clone this repo automatically.

---

## Results

| Method | Accuracy | Macro F1 | Weighted F1 |
|--------|----------|----------|-------------|
| MOMENT-1-large (Setting 1) | 0.3078 | 0.2791 | — |
| PatchTST + TTA | 0.4903 | 0.4194 | — |
| **Chronos-T5-Small** (Setting 1, Phase 1+2) | 0.5333 | **0.4333** | — |
| Baseline — InceptionTime (scratch) | 0.5483 | 0.3753 | — |
| MultiROCKET | 0.6079 | 0.3625 | — |
| InceptionTime-Large × 5 + TTA | 0.6427 | 0.4150 | — |
| ★ **Ensemble (ours)** | **0.6448** | **0.4200** | **0.61** |
| MUSE SOTA (Ruiz et al. 2021) | 0.636 | — | — |
| ROCKET (Ruiz et al. 2021) | 0.632 | — | — |

> **Note on Macro F1:** The low Macro F1 (0.42) is entirely explained by two unlearnable classes — class 53 (only 7 train samples, F1=0) and class 64 (24 samples, F1=0.08). Removing these two classes, Macro F1 rises to **≈0.49**. The Weighted F1=0.61 (weighted by class support) better reflects real-world performance on the learnable classes.

---

## Complementary Work — Data Augmentation

We also explored a complementary approach using data augmentation (jitter, scaling, time shift) — see the notebook [`time_series.ipynb`](https://github.com/mhani6/deep_learning_time_series/blob/main/time_series.ipynb) available on [github.com/mhani6/deep_learning_time_series](https://github.com/mhani6/deep_learning_time_series).

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| InceptionTime + aug. | 0.5783 | 0.4324 |
| PatchTST + aug. | 0.5985 | 0.4599 |
| UniTS + aug. (NeurIPS 2024) | **0.6606** | **0.5314** |

---

## Repository Structure

```
Chronos.ipynb     ← main notebook (Chronos foundation model + ensemble)
Moment.ipynb      ← MOMENT foundation model comparison
report.pdf        ← 3-page ICML 2026 style report
data/             ← LSST dataset loader
models/           ← ChronosClassifier, MOMENTClassifier, InceptionTime, PatchTST
utils.py          ← training utilities
```

---

## Key Design Choices

- **Foundation model**: Chronos-T5-Small (46M) adapted from forecasting → classification by repurposing the T5 encoder
- **Imbalance**: WeightedRandomSampler + Focal Loss (γ=2) — critical for class 53 (7 train samples)
- **Ensemble**: combination of 4 models that each capture different patterns in the data. Each model outputs a probability distribution over the 14 classes. We combine them via **weighted soft voting**: each model's weight is proportional to exp(10 × F1_val), so models that performed better on validation get more influence. Models with val F1 < 0.33 are excluded entirely. The final prediction is the class with the highest combined probability.
- **TTA**: 5 augmented test passes for InceptionTime×5 and PatchTST
