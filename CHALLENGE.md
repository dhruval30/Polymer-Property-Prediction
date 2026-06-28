# ANRF AISEHack 2.0 — Polymer Property Prediction

A Kaggle competition focused on predicting key polymer properties directly from chemical structure (SMILES) to accelerate sustainable materials discovery.

- **Competition page:** https://kaggle.com/competitions/aisehack-2-0
- **Host:** ANRF AISEHack 2.0 (organizers: LaksmanN, Rahulsundar, Rohit Batra IITM, Shreya Kumari, VIJITH P)

---

## Overview

Can your machine learning model help to uncover the secrets of polymers? In this competition, you're tasked with predicting the properties of polymers to speed up the development of new materials. Your contributions will help researchers innovate faster, paving the way for more sustainable and biocompatible materials that can positively impact our planet.

---

## Description

Polymers have transformed modern society by enabling materials that are lightweight, durable, flexible, and highly customizable. They are essential to technologies and products ranging from medicine, electronics, transportation, energy storage, packaging, and sustainable manufacturing — making them one of the most important classes of materials in the modern world. The search for better and more sustainable materials is becoming increasingly important, particularly with the growing focus on the environment. However, progress is still limited by the difficulty of identifying polymer chemistries that achieve the required mechanical, thermal, or electronic properties while also being recyclable, sustainable, and environmentally responsible.

**Your mission:** predict a polymer's real-world performance directly from its chemical structure. You'll be provided with the polymer's structure as a simple text string (SMILES), and your challenge is to build a model that can accurately forecast two different polymer key metrics:

| Target | Meaning | Unit |
|---|---|---|
| **Tg** | Glass transition temperature — measure of thermal stability | °C |
| **Egc** | Chain band gap — measure of electronic property | eV |

Your contributions have the potential to redefine polymer discovery, accelerating sustainable polymer research through virtual screening and driving significant advancements in materials science.

---

## Evaluation

The evaluation metric is the **mean coefficient of determination (R²)** across the two targets, Tg and Egc:

$$
\text{Score} = \frac{R^2_{T_g} + R^2_{E_{gc}}}{2}
$$

where R² is defined as:

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

- $y_i$ = ground truth value
- $\hat{y}_i$ = predicted value
- $\bar{y}$ = mean of the ground truth values

---

## Submission Format

The submission file must be a CSV. For each `id` in the test set, predict the `target` value (interpreted according to its `target_type`). The file must contain a header and follow this format:

```csv
id,target
1,220
2,2.3
3,110
4,70
```

- File **must** be named `submission.csv`.
- One row per `id` from `test.csv`.
- A single `target` column — its meaning (Tg vs. Egc) is determined by the `target_type` of the corresponding test row.

---

## Timeline

- **Start Date:** 24 June 2026
- **Final Submission Deadline:** 24 July 2026

---

## Requirements

- Submissions must be made through **Kaggle Notebooks**. In order for the "Submit" button to be active after a commit, the standard Kaggle notebook competition conditions apply.
- **No publicly available external data is allowed**, including pre-trained models.
- The submission file must be named `submission.csv`.

---

## Dataset

The training dataset consists of **8,171 data points** combining both target properties, where:

- Tg values are reported in **°C**
- Egc values are reported in **eV**
- The `target_type` column identifies which of the two properties each row belongs to (`tg` or `egc`).

### Leaderboard and Baseline Model

Final rankings will be determined using a **hidden private test set**, which serves as the official evaluation dataset. Performance on the private set will be used to determine the teams that advance to the next stage of the competition.

To provide live feedback during the competition, a subset of the test data is used for the **public leaderboard**:

| Split | Size |
|---|---|
| Public test set | 1,543 polymers |
| Private test set | 2,572 polymers |

A sample submission file is provided to illustrate the required prediction format. A baseline notebook is also provided demonstrating a full ML workflow: it uses the open-source **RDKit** library to extract molecular descriptors from polymer SMILES, applies basic feature engineering, and trains a predictive model (Ridge Regression) for both targets. Participants are encouraged to build upon and improve this baseline.

### Files

Located under `data/` in this repo:

| File | Description |
|---|---|
| `train.csv` | Training set — columns: `smiles`, `target`, `target_type` |
| `test.csv` | Test set — columns: `id`, `smiles`, `target_type` |
| `sample_submission.csv` | Example submission in the correct format (`id`, `target`) |
| `base_line_model.ipynb` | Baseline Ridge Regression model demonstrating RDKit-based cheminformatics features, model training, and generation of `submission.csv` |

### Columns in detail

**`train.csv`**
- `smiles` — SMILES string representation of the polymer structure.
- `target` — Property value (either Tg in °C or Egc in eV, indicated by `target_type`).
- `target_type` — `tg` or `egc`.

**`test.csv`**
- `id` — Unique identifier for each polymer.
- `smiles` — SMILES string representation of the polymer structure.
- `target_type` — `tg` or `egc` (tells you what property to predict for that row).

> Note: SMILES strings in this dataset use `*` to denote polymer repeat-unit attachment points (e.g. `*Oc1ccc(cc1)...*`). Standard RDKit parsing handles this as a wildcard/dummy atom.

---

## Current Leaderboard Snapshot (as of 2026-06-28)

| # | Team | Score | Entries |
|---|---|---|---|
| 1 | MEGALODON | 0.904 | 11 |
| 2 | The AI Alchemists | 0.908 | 15 |
| 3 | Kuch Bhi | 0.900 | 4 |
| 4 | Runtime Rebel's | 0.900 | 12 |
| 5 | Thiru321 | 0.900 | 4 |
| 6 | Cross Linkers | 0.898 | 2 |
| 7 | Team X | 0.898 | 3 |
| 8 | Prime Polymers | 0.897 | 9 |
| 9 | Cosmic | 0.896 | 3 |
| 10 | akshit0943x | 0.896 | 2 |
| 11 | Vikas_23f1001674 | 0.896 | 2 |
| 12 | Aniruddha_20241065 | 0.896 | 7 |
| 13 | PATA NAHI | 0.896 | 6 |
| 14 | Five Aces | 0.895 | 3 |
| 15 | Hackaholics | 0.895 | 10 |
| 16 | InfinityLoop | 0.895 | 16 |
| 17 | 1nf1n1ty | 0.895 | 16 |
| 18 | Gommies | 0.895 | 3 |
| 19 | Coding Brigades | 0.894 | 10 |
| 20 | Melwin Joseph | 0.894 | 1 |
| 21 | PranjalSrivastava888 | 0.894 | 2 |
| 22 | Breaking Bonds | 0.894 | 12 |
| 23 | The Alchemists | 0.893 | 15 |
| 24 | Akshit | 0.893 | 6 |
| 25 | TRH | 0.892 | 4 |
| 26 | GAZMasters | 0.892 | 11 |
| 27 | Team XD | 0.892 | 11 |
| 28 | The Debuggers | 0.890 | 14 |
| 29 | The Team | 0.890 | 1 |
| 30 | gogomegoa | 0.889 | 2 |
| 31 | Atharva Kulkarni | 0.888 | 7 |
| 32 | Alwin Joseph07 | 0.886 | 2 |
| 33 | counter | 0.883 | 6 |
| 34 | Himanshu Dhiman | 0.882 | 1 |
| 35 | dkforuse tokearn | 0.882 | 1 |
| 36 | Sudhir2023 | 0.882 | 1 |
| 37 | amol | 0.882 | 1 |
| 38 | Kartik Soni | 0.882 | 1 |
| 39 | Aoit Kumar Nayak | 0.882 | 2 |
| 40 | Siddhu Jaykay | 0.877 | 5 |
| 41 | Ronit Raj1 | 0.870 | 3 |
| 42 | Rishi202411059 | 0.865 | 4 |
| 43 | Sarjamw | 0.854 | 2 |
| 44 | AASHISH NIRANJAN BARA THYAKANAN | 0.846 | 1 |
| 45 | Sinh David | 0.799 | 1 |
| 46 | SAMSUL HABIB | 0.665 | 1 |
| 47 | Prakash | 0.651 | 4 |

> Rankings may differ from on-screen due to score sorting vs. tiebreakers; numbers transcribed from the public leaderboard snapshot.

---

## Citation

> LaksmanN, Rahulsundar, Rohit Batra IITM, Shreya Kumari, and VIJITH P. *ANRF AISEHack 2.0 — Polymer Property Prediction.* https://kaggle.com/competitions/aisehack-2-0, 2026. Kaggle.

---

## Quick Start (this repo)

1. The dataset already lives under `data/`:
   - `data/train.csv` (8,171 rows)
   - `data/test.csv` (4,115 rows)
   - `data/sample_submission.csv`
   - `data/base_line_model.ipynb`
2. Open the baseline notebook to see the RDKit-descriptor + Ridge Regression workflow.
3. Iterate on features / models. The headline metric is the **mean R²** over the Tg and Egc splits of the test set.
4. Produce `submission.csv` in the format above and submit via Kaggle Notebooks.
