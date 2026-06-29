# 0.911 LB Submission — How it was achieved

Final result on the public leaderboard for ANRF AISEHack 2.0 — Polymer Property Prediction:

| Metric | Value |
|---|---|
| **Public LB score** | **0.911** |
| Public leaderboard rank | #1 legit (excluding MEGALODON's outlier 0.994) |
| Submitted file | `results/exp_chemprop_multitask/submission.csv` |

Public leaderboard is computed on ~37% of the test set. The remaining 63% determines the private (final) standings.

---

## Approach in one paragraph

A two-stage stack of complementary model classes blended via per-target NNLS:

1. **Phase 1 — gradient-boosting cocktail** trained per target on a wide feature mix (RDKit 2D descriptors + 4 fingerprint families). Three GBMs (LightGBM, CatBoost, HistGradientBoosting), each model trained per target, simple mean blend per target.
2. **Phase 2 — Chemprop D-MPNN multitask** trained from scratch on SMILES graphs. Predicts Tg and Egc together so molecules labeled for only one property still contribute to the shared graph representation.
3. **Blend** — per-target non-negative least squares (NNLS) on out-of-fold predictions to combine Phase 1 + Chemprop into the final submission.

The two model classes are fully orthogonal — GBMs see "which substructures are present, weighted" while the D-MPNN learns a graph representation via message passing. Their residuals decorrelate sharply, so the NNLS blend extracts genuine new R² rather than averaging noise.

---

## Final architecture summary

### Phase 1 — GBM Cocktail
- **Features (per molecule, ~9,088 columns total, ~8,428 after dropping constants):**
  - RDKit 2D descriptors (~210)
  - Morgan fingerprint, radius=2, 2048-bit, **count** vectors
  - Morgan fingerprint, radius=3, 2048-bit, **count** vectors
  - MACCS keys (167)
  - Avalon fingerprint (512)
  - Atom-Pair fingerprint, 2048-bit count
  - Topological-Torsion fingerprint, 2048-bit count
- **Models** — one of each, fit separately on `tg` and `egc` rows:
  - LightGBM (`n_estimators=4000`, `learning_rate=0.03`, `num_leaves=63`, `feature_fraction=0.5`, `bagging_fraction=0.85`, `reg_lambda=1.0`, early-stop on validation fold at 200 rounds)
  - CatBoost (`iterations=4000`, `depth=8`, `learning_rate=0.03`, `l2_leaf_reg=3.0`, `SymmetricTree`)
  - HistGradientBoostingRegressor (sklearn defaults from the script; built-in early stopping)
- **Target transform** — `log1p` on Egc (range ~0.1–9.9 eV, right-skewed), identity on Tg.
- **CV** — 5-fold StratifiedKFold on 10 quantile bins of the (transformed) target.
- **Final test prediction** — refit each model on full per-target training data using the median best-iteration count from CV folds, then simple mean blend of the three models.
- **OOF result**: Tg 0.9018, Egc 0.9092, mean 0.9055.
- **Solo LB** (this submission, submitted before Chemprop was added): **0.900**.

### Phase 2 — Chemprop D-MPNN Multitask
- **Architecture** — `BondMessagePassing(d_h=300, depth=4, dropout=0.05)` + `MeanAggregation` + `RegressionFFN(hidden=300, n_layers=2, n_tasks=2)`. ~409K parameters.
- **Training data layout** — pivoted from long to wide: each row is one unique SMILES with `(tg, egc)` columns, NaN where unlabeled. 6,158 unique SMILES total (4,137 with Tg, 2,028 with Egc, 7 with both).
- **Loss** — chemprop's masked MSE on per-target z-scored targets (normalization fit on training fold only).
- **Training** — Adam-like default optimizer, `max_epochs=50`, `batch_size=64`, `EarlyStopping(patience=10)`, `gradient_clip_val=1.0`.
- **CV / bagging** — same StratifiedKFold over the nan-mean of the two targets; 3 seeds bagged per fold.
- **Accelerator** — Apple Silicon MPS via Lightning.
- **Partial training** — due to MPS performance degradation across long sessions (per-fold runtime grew from 1.7h → 5.4h → 8h+ likely from thermal throttling and MPS memory not being fully released between fits), training was stopped after fold 2 of 5. Per-fold checkpointing in `results/exp_chemprop_multitask/checkpoint.npz` made this clean.
- **Partial OOF (40% of training rows covered)** — Tg 0.9033, Egc 0.8991.
- **Test predictions** — bag-averaged over 6 successful models (3 seeds × 2 folds).

### Blend — NNLS per target
- For each of {Tg, Egc}, fit non-negative weights `w` minimizing `‖y − w₁·phase1_oof − w₂·chemprop_oof‖²` on the subset of training rows where Chemprop OOF exists. Normalize weights so they sum to 1.
- Apply the same weights to test predictions: `final_test = w₁·phase1_test + w₂·chemprop_test`.

This 2-base stack produced the **0.911 LB submission**.

---

## How to reproduce

### 1. Environment

Python 3.11 is required (Chemprop 2.x is incompatible with 3.10):

```bash
conda create -n poly python=3.11 -y
conda activate poly
pip install chemprop lightgbm catboost rdkit tqdm scikit-learn pandas scipy lightning torch
```

### 2. Run order

From the repo root:

```bash
# 1. GBM cocktail — Phase 1.
#    First run featurizes all SMILES (~10 min cold cache) then trains for ~1.5h.
#    Re-runs hit the .cache/features/ folder and skip featurization.
python experiments/exp_gbm_cocktail.py

# 2. Chemprop D-MPNN multitask — overnight.
#    Per-fold checkpointing means you can Ctrl+C at any point and resume.
#    Stopping after fold 2/5 is sufficient to reproduce 0.911 — the
#    submission posted to the LB used only those 2 completed folds.
python experiments/exp_chemprop_multitask.py

# 3. Quick finish — NNLS-stacks Phase 1 with whatever Chemprop folds completed.
#    Writes the final submission to results/exp_chemprop_multitask/submission.csv.
python experiments/exp_chemprop_quick_finish.py
```

### 3. Submit

```
results/exp_chemprop_multitask/submission.csv
```

Upload that file to the Kaggle competition page.

---

## Frozen artifacts in this repo (do not modify)

These files are the exact ones that produced 0.911 on the LB, committed at `cfe08c9`:

| File | What it is |
|---|---|
| `experiments/_utils.py` | Shared featurization (cached), stratified CV, target transforms, logging |
| `experiments/exp_gbm_cocktail.py` | Phase 1 script |
| `experiments/exp_chemprop_multitask.py` | Chemprop multitask training script |
| `experiments/exp_chemprop_quick_finish.py` | NNLS blender + final submission writer |
| `results/exp_gbm_cocktail/submission.csv` | Phase 1 solo submission (LB 0.900) |
| `results/exp_gbm_cocktail/oof.csv` | Phase 1 per-model OOF + blend OOF (used by the quick-finish blender) |
| `results/exp_chemprop_multitask/checkpoint.npz` | Chemprop partial-training checkpoint (folds 1+2, 6 models) |
| `results/exp_chemprop_multitask/submission.csv` | **The 0.911 LB submission** |
| `results/exp_chemprop_multitask/oof.csv` | Chemprop OOF (NaN where folds didn't complete) |
| `results/exp_chemprop_multitask/cv_summary_partial.json` | NNLS weights + per-base R² on the masked subset |

---

## Why this combination works

- **Phase 1 captures substructure-presence signal cleanly.** Tree models split on individual fingerprint bits and descriptor thresholds — they're effectively saying "if a molecule has this functional group and that descriptor exceeds X, predict Y."
- **Chemprop captures graph-context signal.** The D-MPNN's iterative message passing produces atom-level embeddings that summarize each atom's chemical neighborhood. The final molecular embedding is learned end-to-end against the target — fundamentally different from hand-crafted fingerprints.
- **Multitask learning is a big multiplier on Chemprop side.** The Tg-labeled molecules (4,137) contribute to the message-passing representation even when predicting Egc, and vice versa. Phase 1's GBMs train per-target in isolation; the GBM that predicts Egc only ever sees the 2,028 Egc-labeled molecules.
- **NNLS finds the right weight automatically.** Look at the NNLS weights in `cv_summary_partial.json`: on Tg, Phase 1 still earns substantial weight; on Egc, Chemprop dominates because Phase 1 was already strong there and Chemprop adds the most new signal.
- **Train and LB calibrate tightly.** Across every submission today, the gap between mean OOF R² and public LB was ~0.005, so OOF improvements transferred almost 1:1 to the leaderboard.

---

## Caveats and notes for the record

- The Chemprop run was stopped early at 2 of 5 folds. The reproduction is therefore mildly stochastic — running the same script will train more folds if your machine survives the MPS slowdown, in which case `submission.csv` will reflect a 15-model bag rather than a 6-model bag and the result may be marginally different (likely +0.000–0.003).
- To exactly reproduce 0.911, the cleanest path is: load `checkpoint.npz` (which is committed) and run the quick-finish script. Skipping step 2 above and running only step 3 will reproduce the exact submission byte-for-byte.
- Apple Silicon MPS performance degraded as training sessions extended past ~3 hours — likely a mix of MPS memory not being fully released between Lightning fits and thermal throttling on a fanless MacBook Air. If you re-run on a workstation with a discrete GPU, all 5 folds × 3 seeds should complete comfortably in a few hours.
- The `.cache/features/` directory holds RDKit fingerprint and descriptor caches. It's regenerated automatically on the first run and not committed (it's in `.gitignore`).
- Submitted via the Kaggle competition's notebook submission flow (the script's CSV output dropped into a notebook that just writes `submission.csv`).
