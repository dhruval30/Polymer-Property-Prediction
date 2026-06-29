"""Phase 4: Chemprop D-MPNN, multitask Tg+Egc, MPS-accelerated, overnight run.

Architecture: directed message-passing neural network with shared molecular
representation and two regression heads (Tg, Egc). Trained from scratch — no
pretrained weights, no external data. Multitask exploits the full ~6171
training SMILES (not just per-target subsets), since SMILES with only Tg
still contribute to the shared message-passing representation that the Egc
head uses, and vice versa.

CV: 5-fold StratifiedKFold over the (NaN-mean) target.
Bag: 3 seeds per fold = 15 trained models total.
Checkpointing: per-fold .npz snapshot so a mid-run crash leaves recoverable
state (OOF + accumulated test predictions for completed folds).

Final output: an NNLS-stacked submission combining Phase 1 cocktail blend +
multi-KRR stack + this Chemprop model. The raw Chemprop-only submission is
also written as `submission_chemprop_only.csv` for diagnostic comparison.

Run:
    pip install chemprop lightning
    python experiments/exp_chemprop_multitask.py
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls
from sklearn.metrics import r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from lightning import pytorch as pl
except ImportError:
    try:
        import pytorch_lightning as pl  # type: ignore
    except ImportError as e:
        raise ImportError("Install lightning: pip install lightning") from e

try:
    from chemprop import data as cdata
    from chemprop import featurizers as cfeat
    from chemprop import models as cmodels
    from chemprop import nn as cnn
except ImportError as e:
    raise ImportError("Install chemprop: pip install chemprop") from e

from _utils import (
    DATA_DIR,
    RESULTS_DIR,
    prepare_run_dir,
    setup_logging,
    stratified_quantile_split,
    transform_target,
)

setup_logging()
log = logging.getLogger("polymer")

RUN_NAME = "exp_chemprop_multitask"
SEED = 42
N_FOLDS = 5
N_QUANTILE_BINS = 10
BAG_SEEDS = [42, 1337, 7]
TARGETS = ["tg", "egc"]
TARGET_TRANSFORMS = {"tg": "identity", "egc": "log1p"}

# Model architecture
MP_HIDDEN = 300
MP_DEPTH = 4
FFN_HIDDEN = 300
FFN_DEPTH = 2
DROPOUT = 0.05

# Training
MAX_EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 10
LR = 1e-3

# For stacking at the end
PHASE1_RUN = "exp_gbm_cocktail"
KRR_STACK_RUN = "exp_stack_multi_krr"


# ---------------------------------------------------------------------------
# per-epoch logger callback (so the user can see training progress)
# ---------------------------------------------------------------------------

class EpochLogger(pl.Callback):
    """Logs all callback metrics once per epoch via our `polymer` logger."""

    def __init__(self, label: str, seed: int):
        super().__init__()
        self.label = label
        self.seed = seed
        self._epoch_t = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        parts: list[str] = []
        for k in ("train_loss", "val_loss"):
            v = metrics.get(k)
            if v is not None:
                try:
                    parts.append(f"{k}={float(v):.4f}")
                except Exception:
                    pass
        # also surface any extra metrics chemprop logs
        for k, v in metrics.items():
            if k in ("train_loss", "val_loss"):
                continue
            try:
                parts.append(f"{k}={float(v):.4f}")
            except Exception:
                pass
        dt = time.time() - self._epoch_t
        log.info("[%s seed=%d] epoch %02d  %s  (%.1fs)",
                 self.label, self.seed, epoch, " ".join(parts) or "(no metrics)", dt)


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

def lightning_accelerator() -> tuple[str, int]:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps", 1
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


# ---------------------------------------------------------------------------
# data prep — long → multitask wide
# ---------------------------------------------------------------------------

def long_to_wide(train: pd.DataFrame) -> pd.DataFrame:
    """One row per unique SMILES with (tg, egc) columns; NaN where unlabeled."""
    rows: dict[str, dict] = {}
    for smi, t, y in zip(train["smiles"].values, train["target_type"].values, train["target"].values):
        if smi not in rows:
            rows[smi] = {"smiles": smi, "tg": np.nan, "egc": np.nan}
        rows[smi][t] = y
    wide = pd.DataFrame(list(rows.values()))[["smiles", "tg", "egc"]]
    return wide


def apply_target_transforms(wide_orig: pd.DataFrame) -> pd.DataFrame:
    out = wide_orig.copy()
    for t in TARGETS:
        y_t, _ = transform_target(wide_orig[t].values, TARGET_TRANSFORMS[t])
        out[t] = y_t  # NaN stays NaN through log1p
    return out


def inverse_targets(transformed: np.ndarray) -> np.ndarray:
    """Apply per-task inverse to a (N, 2) array of transformed predictions."""
    out = np.empty_like(transformed)
    for ti, t in enumerate(TARGETS):
        _, inv = transform_target(np.array([0.0]), TARGET_TRANSFORMS[t])
        out[:, ti] = inv(transformed[:, ti])
    return out


# ---------------------------------------------------------------------------
# Chemprop model + training
# ---------------------------------------------------------------------------

def build_mpnn(seed: int) -> "cmodels.MPNN":
    pl.seed_everything(seed)
    mp = cnn.BondMessagePassing(d_h=MP_HIDDEN, depth=MP_DEPTH, dropout=DROPOUT)
    agg = cnn.MeanAggregation()
    predictor = cnn.RegressionFFN(
        input_dim=MP_HIDDEN,
        hidden_dim=FFN_HIDDEN,
        n_layers=FFN_DEPTH,
        dropout=DROPOUT,
        n_tasks=len(TARGETS),
    )
    return cmodels.MPNN(mp, agg, predictor, batch_norm=True)


def make_datasets(
    smis_train: np.ndarray, ys_train_z: np.ndarray,
    smis_val: np.ndarray, ys_val_z: np.ndarray,
):
    featurizer = cfeat.SimpleMoleculeMolGraphFeaturizer()
    tr_dps = [
        cdata.MoleculeDatapoint.from_smi(s, y=y.astype(np.float32))
        for s, y in zip(smis_train, ys_train_z)
    ]
    va_dps = [
        cdata.MoleculeDatapoint.from_smi(s, y=y.astype(np.float32))
        for s, y in zip(smis_val, ys_val_z)
    ]
    tr_dset = cdata.MoleculeDataset(tr_dps, featurizer)
    va_dset = cdata.MoleculeDataset(va_dps, featurizer)
    tr_loader = cdata.build_dataloader(tr_dset, batch_size=BATCH_SIZE, num_workers=0)
    va_loader = cdata.build_dataloader(va_dset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    return tr_loader, va_loader


def make_predict_loader(smis: list[str]):
    featurizer = cfeat.SimpleMoleculeMolGraphFeaturizer()
    dps = [cdata.MoleculeDatapoint.from_smi(s) for s in smis]
    dset = cdata.MoleculeDataset(dps, featurizer)
    return cdata.build_dataloader(dset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)


def train_and_predict_one_model(
    *, tr_smis: np.ndarray, tr_ys: np.ndarray,
    va_smis: np.ndarray, va_ys: np.ndarray,
    test_smis: list[str],
    seed: int, label: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train one MPNN, return (val_preds_in_z_space, test_preds_in_z_space, info).

    `tr_ys` and `va_ys` are in TRANSFORMED scale (log1p for egc) but NOT
    yet standardized. This function standardizes per-target using train-fold
    statistics, trains the model on standardized targets, predicts in
    standardized space, then un-standardizes back to transformed space.
    """
    # per-target standardization (compute on train fold only, ignore NaN)
    mu = np.nanmean(tr_ys, axis=0).astype(np.float32)
    sd = np.nanstd(tr_ys, axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd)

    tr_ys_z = (tr_ys - mu) / sd
    va_ys_z = (va_ys - mu) / sd

    tr_loader, va_loader = make_datasets(tr_smis, tr_ys_z, va_smis, va_ys_z)

    model = build_mpnn(seed)
    acc, dev = lightning_accelerator()

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min",
                                     check_finite=False),
        EpochLogger(label=label, seed=seed),
    ]

    trainer = pl.Trainer(
        accelerator=acc,
        devices=dev,
        max_epochs=MAX_EPOCHS,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=callbacks,
        deterministic=False,
        gradient_clip_val=1.0,
    )

    t_train = time.time()
    trainer.fit(model, tr_loader, va_loader)
    train_time = time.time() - t_train

    # val predictions (z space)
    val_loader_eval = cdata.build_dataloader(
        cdata.MoleculeDataset(
            [cdata.MoleculeDatapoint.from_smi(s) for s in va_smis],
            cfeat.SimpleMoleculeMolGraphFeaturizer(),
        ),
        batch_size=BATCH_SIZE, num_workers=0, shuffle=False,
    )
    val_preds_z = torch.cat(trainer.predict(model, val_loader_eval), dim=0).cpu().numpy()

    # test predictions (z space)
    test_loader = make_predict_loader(test_smis)
    test_preds_z = torch.cat(trainer.predict(model, test_loader), dim=0).cpu().numpy()

    # un-standardize back to transformed scale
    val_preds_t = val_preds_z * sd + mu
    test_preds_t = test_preds_z * sd + mu

    info = {
        "seed": seed,
        "n_train": int(len(tr_smis)),
        "n_val": int(len(va_smis)),
        "train_time_sec": round(train_time, 1),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
    }
    log.info("[%s seed=%d] trained in %.1fs", label, seed, train_time)
    return val_preds_t, test_preds_t, info


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    run_dir = prepare_run_dir(RUN_NAME)
    acc, _ = lightning_accelerator()

    log.info("=== %s ===", RUN_NAME)
    log.info("results dir: %s", run_dir)
    log.info("lightning accelerator: %s", acc)
    log.info("seed=%d  n_folds=%d  bag_seeds=%s", SEED, N_FOLDS, BAG_SEEDS)
    log.info("arch: MP(hidden=%d, depth=%d, dropout=%.2f) + FFN(hidden=%d, depth=%d)",
             MP_HIDDEN, MP_DEPTH, DROPOUT, FFN_HIDDEN, FFN_DEPTH)
    log.info("training: max_epochs=%d batch=%d patience=%d lr=%.0e",
             MAX_EPOCHS, BATCH_SIZE, PATIENCE, LR)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)

    # --- Long → wide multitask format ---
    wide_orig = long_to_wide(train)
    wide_t = apply_target_transforms(wide_orig)
    test_unique_smis = test["smiles"].drop_duplicates().tolist()

    log.info("train unique SMILES: %d", len(wide_orig))
    has_tg = wide_orig["tg"].notna().sum()
    has_egc = wide_orig["egc"].notna().sum()
    has_both = (wide_orig["tg"].notna() & wide_orig["egc"].notna()).sum()
    log.info("  has tg labeled : %d", has_tg)
    log.info("  has egc labeled: %d", has_egc)
    log.info("  has BOTH       : %d", has_both)
    log.info("test unique SMILES: %d", len(test_unique_smis))

    train_smis_all = wide_orig["smiles"].values
    ys_all_t = wide_t[TARGETS].values.astype(np.float32)  # (n, 2) — NaN where missing

    # Stratification target: nanmean across the 2 columns
    strat_target = np.nanmean(ys_all_t, axis=1)
    folds = stratified_quantile_split(
        strat_target, n_folds=N_FOLDS, n_bins=N_QUANTILE_BINS, seed=SEED,
    )

    # OOF storage in transformed scale
    oof_t = np.full((len(wide_orig), len(TARGETS)), np.nan, dtype=np.float64)
    # Test predictions accumulator (transformed scale)
    test_acc_t = np.zeros((len(test_unique_smis), len(TARGETS)), dtype=np.float64)
    n_models_added_to_test = 0

    fold_records: list[dict] = []

    # Checkpoint resume — if a previous run wrote progress, load it
    ckpt_path = run_dir / "checkpoint.npz"
    completed_folds = 0
    if ckpt_path.exists():
        try:
            ck = np.load(ckpt_path, allow_pickle=False)
            oof_t = ck["oof_t"]
            test_acc_t = ck["test_acc_t"]
            n_models_added_to_test = int(ck["n_models_added_to_test"])
            completed_folds = int(ck["completed_folds"])
            log.info("Resumed from checkpoint: %d folds done, %d models in test accum",
                     completed_folds, n_models_added_to_test)
        except Exception as e:
            log.warning("Failed to load checkpoint, starting fresh: %s", e)
            completed_folds = 0

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        if fold_idx < completed_folds:
            log.info("Skipping fold %d/%d (already done)", fold_idx + 1, N_FOLDS)
            continue
        log.info("─" * 64)
        log.info("FOLD %d/%d  (train=%d val=%d)", fold_idx + 1, N_FOLDS, len(tr_idx), len(va_idx))
        t_fold = time.time()

        tr_smis = train_smis_all[tr_idx]
        va_smis = train_smis_all[va_idx]
        tr_ys = ys_all_t[tr_idx]
        va_ys = ys_all_t[va_idx]

        seed_preds_val: list[np.ndarray] = []
        seed_preds_test: list[np.ndarray] = []
        seed_infos: list[dict] = []

        for seed in BAG_SEEDS:
            try:
                val_p, test_p, info = train_and_predict_one_model(
                    tr_smis=tr_smis, tr_ys=tr_ys,
                    va_smis=va_smis, va_ys=va_ys,
                    test_smis=test_unique_smis,
                    seed=seed, label=f"fold{fold_idx + 1}",
                )
                seed_preds_val.append(val_p)
                seed_preds_test.append(test_p)
                seed_infos.append(info)
            except Exception as e:
                log.exception("[fold %d seed %d] FAILED: %s — continuing", fold_idx + 1, seed, e)

        if not seed_preds_val:
            log.error("[fold %d] ALL seeds failed, skipping fold", fold_idx + 1)
            continue

        # Bag average within this fold
        val_pred_bag_t = np.mean(np.stack(seed_preds_val, axis=0), axis=0)
        test_pred_bag_t = np.mean(np.stack(seed_preds_test, axis=0), axis=0)

        oof_t[va_idx] = val_pred_bag_t
        test_acc_t += test_pred_bag_t * len(seed_preds_val)
        n_models_added_to_test += len(seed_preds_val)

        # Per-target fold R^2 on original scale
        val_pred_orig = inverse_targets(val_pred_bag_t)
        y_va_orig = wide_orig.iloc[va_idx][TARGETS].values
        fold_r2 = {}
        for ti, t in enumerate(TARGETS):
            mask = ~np.isnan(y_va_orig[:, ti])
            if mask.sum() > 0:
                fold_r2[t] = float(r2_score(y_va_orig[mask, ti], val_pred_orig[mask, ti]))
            else:
                fold_r2[t] = float("nan")
        log.info("[fold %d] OOF R^2  tg=%.4f  egc=%.4f  (%d seeds, %.1fs)",
                 fold_idx + 1, fold_r2["tg"], fold_r2["egc"],
                 len(seed_preds_val), time.time() - t_fold)

        fold_records.append({
            "fold": fold_idx + 1,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "fold_r2": fold_r2,
            "seeds": seed_infos,
        })

        # Checkpoint
        np.savez(ckpt_path,
                 oof_t=oof_t, test_acc_t=test_acc_t,
                 n_models_added_to_test=n_models_added_to_test,
                 completed_folds=fold_idx + 1)
        log.info("[fold %d] checkpoint saved", fold_idx + 1)

    # --- Aggregate test predictions (transformed scale) ---
    if n_models_added_to_test == 0:
        log.error("No successful models — cannot produce submission. Aborting.")
        return
    test_t = test_acc_t / n_models_added_to_test
    log.info("Aggregated test predictions over %d successful models", n_models_added_to_test)

    # --- OOF R^2 on original scale ---
    oof_orig = inverse_targets(oof_t)
    test_orig = inverse_targets(test_t)
    y_all_orig = wide_orig[TARGETS].values

    summary: dict = {
        "run_name": RUN_NAME,
        "seed": SEED, "n_folds": N_FOLDS, "bag_seeds": BAG_SEEDS,
        "accelerator": acc,
        "arch": {"mp_hidden": MP_HIDDEN, "mp_depth": MP_DEPTH,
                  "ffn_hidden": FFN_HIDDEN, "ffn_depth": FFN_DEPTH,
                  "dropout": DROPOUT},
        "training": {"max_epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE,
                      "patience": PATIENCE, "lr": LR},
        "target_transforms": TARGET_TRANSFORMS,
        "per_target": {},
        "fold_records": fold_records,
        "n_successful_models": n_models_added_to_test,
    }
    log.info("═" * 64)
    chemprop_r2s = {}
    for ti, t in enumerate(TARGETS):
        mask = ~np.isnan(y_all_orig[:, ti])
        r2 = float(r2_score(y_all_orig[mask, ti], oof_orig[mask, ti]))
        chemprop_r2s[t] = r2
        summary["per_target"][t] = {"chemprop_oof_r2": r2, "n_labeled": int(mask.sum())}
        log.info("[%s] Chemprop OOF R^2 = %.4f (n=%d)", t, r2, mask.sum())
    mean_chemprop_r2 = float(np.mean(list(chemprop_r2s.values())))
    summary["mean_chemprop_oof_r2"] = mean_chemprop_r2
    log.info("Mean Chemprop OOF R^2 = %.4f", mean_chemprop_r2)
    log.info("═" * 64)

    # --- Raw Chemprop submission (per-row lookup by (smiles, target_type)) ---
    test_smi_to_idx = {s: i for i, s in enumerate(test_unique_smis)}
    chemprop_sub_rows = []
    for _, row in test.iterrows():
        smi = row["smiles"]; tt = row["target_type"]
        ti = TARGETS.index(tt)
        chemprop_sub_rows.append({
            "id": int(row["id"]),
            "target": float(test_orig[test_smi_to_idx[smi], ti]),
        })
    chemprop_sub = pd.DataFrame(chemprop_sub_rows).sort_values("id").reset_index(drop=True)
    chemprop_sub.to_csv(run_dir / "submission_chemprop_only.csv", index=False)
    log.info("Wrote %s", run_dir / "submission_chemprop_only.csv")

    # --- OOF long-format aligned with train.csv ---
    smi_to_oof_idx = {s: i for i, s in enumerate(train_smis_all)}
    oof_rows = []
    for _, row in train.iterrows():
        smi = row["smiles"]; tt = row["target_type"]
        ti = TARGETS.index(tt)
        oof_rows.append({
            "smiles": smi, "target_type": tt, "target": float(row["target"]),
            "oof_chemprop": float(oof_orig[smi_to_oof_idx[smi], ti]),
        })
    chemprop_oof = pd.DataFrame(oof_rows)
    chemprop_oof.to_csv(run_dir / "oof.csv", index=False)
    log.info("Wrote %s", run_dir / "oof.csv")

    # ---------------------------------------------------------------
    # FINAL STACK: NNLS over [phase1_blend, krr_stack, chemprop] per target
    # ---------------------------------------------------------------
    log.info("─" * 64)
    log.info("Building final NNLS stack with prior runs...")
    try:
        phase1_oof = pd.read_csv(RESULTS_DIR / PHASE1_RUN / "oof.csv")
        phase1_sub = pd.read_csv(RESULTS_DIR / PHASE1_RUN / "submission.csv")
        krr_oof = pd.read_csv(RESULTS_DIR / KRR_STACK_RUN / "oof.csv")
        krr_sub = pd.read_csv(RESULTS_DIR / KRR_STACK_RUN / "submission.csv")
        log.info("loaded phase1: oof=%s sub=%s", phase1_oof.shape, phase1_sub.shape)
        log.info("loaded krr stack: oof=%s sub=%s", krr_oof.shape, krr_sub.shape)
    except FileNotFoundError as e:
        log.warning("Could not load prior runs for stacking: %s", e)
        log.warning("Falling back: submission.csv will be raw Chemprop predictions.")
        chemprop_sub.to_csv(run_dir / "submission.csv", index=False)
        summary["final_stack_status"] = "skipped (prior runs missing)"
        summary["runtime_sec"] = round(time.time() - t0, 1)
        (run_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Done in %.1fs", time.time() - t0)
        return

    final_oof_frames: list[pd.DataFrame] = []
    final_test_frames: list[pd.DataFrame] = []

    for t in TARGETS:
        tr_mask = (train["target_type"] == t).values
        te_mask = (test["target_type"] == t).values
        train_smis_t = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        y_tr = train.loc[tr_mask, "target"].reset_index(drop=True).values
        ids_te = test.loc[te_mask, "id"].reset_index(drop=True).values

        p1_t = phase1_oof[phase1_oof["target_type"] == t].reset_index(drop=True)
        krr_t = krr_oof[krr_oof["target_type"] == t].reset_index(drop=True)
        cp_t = chemprop_oof[chemprop_oof["target_type"] == t].reset_index(drop=True)

        assert (p1_t["smiles"].values == train_smis_t).all(), f"[{t}] phase1 OOF misaligned"
        assert (krr_t["smiles"].values == train_smis_t).all(), f"[{t}] krr OOF misaligned"
        assert (cp_t["smiles"].values == train_smis_t).all(), f"[{t}] chemprop OOF misaligned"

        oofs_3 = np.column_stack([
            p1_t["oof_blend"].values,
            krr_t["oof_stack"].values,
            cp_t["oof_chemprop"].values,
        ]).astype(np.float64)

        w_raw, _ = nnls(oofs_3, y_tr.astype(np.float64))
        w_sum = w_raw.sum()
        if w_sum < 1e-9:
            log.warning("[%s] final NNLS all zeros → phase1-only", t)
            w_norm = np.array([1.0, 0.0, 0.0])
        else:
            w_norm = w_raw / w_sum

        log.info("[%s] FINAL stack weights (raw → norm, sum=%.4f):", t, w_sum)
        for name, wr, wn in zip(["phase1", "krr_stack", "chemprop"], w_raw, w_norm):
            log.info("    %-10s raw=%.4f  norm=%.4f", name, wr, wn)

        per_base_r2 = {
            "phase1": float(r2_score(y_tr, p1_t["oof_blend"].values)),
            "krr_stack": float(r2_score(y_tr, krr_t["oof_stack"].values)),
            "chemprop": float(r2_score(y_tr, cp_t["oof_chemprop"].values)),
        }
        log.info("[%s] per-base OOF R^2: %s",
                 t, {k: f"{v:.4f}" for k, v in per_base_r2.items()})

        final_oof = oofs_3 @ w_norm
        final_r2 = float(r2_score(y_tr, final_oof))
        log.info("[%s] FINAL STACK OOF R^2 = %.4f", t, final_r2)

        # Apply weights to test predictions
        p1_test = phase1_sub.set_index("id").loc[ids_te, "target"].values
        krr_test = krr_sub.set_index("id").loc[ids_te, "target"].values
        cp_test = chemprop_sub.set_index("id").loc[ids_te, "target"].values
        final_test = w_norm[0] * p1_test + w_norm[1] * krr_test + w_norm[2] * cp_test

        final_test_frames.append(pd.DataFrame({"id": ids_te, "target": final_test}))
        final_oof_frames.append(pd.DataFrame({
            "smiles": train_smis_t, "target_type": t, "target": y_tr,
            "oof_phase1": p1_t["oof_blend"].values,
            "oof_krr_stack": krr_t["oof_stack"].values,
            "oof_chemprop": cp_t["oof_chemprop"].values,
            "oof_final": final_oof,
        }))

        summary["per_target"][t].update({
            "per_base_oof_r2": per_base_r2,
            "final_stack_oof_r2": final_r2,
            "final_nnls_weights_raw": {k: float(v) for k, v in zip(["phase1", "krr_stack", "chemprop"], w_raw)},
            "final_nnls_weights_normalized": {k: float(v) for k, v in zip(["phase1", "krr_stack", "chemprop"], w_norm)},
        })

    final_sub = (
        pd.concat(final_test_frames, axis=0).sort_values("id").reset_index(drop=True)
    )
    final_sub.to_csv(run_dir / "submission.csv", index=False)
    log.info("Wrote %s  (%d rows)", run_dir / "submission.csv", len(final_sub))

    final_oof_all = pd.concat(final_oof_frames, axis=0).reset_index(drop=True)
    final_oof_all.to_csv(run_dir / "oof_stacked.csv", index=False)
    log.info("Wrote %s", run_dir / "oof_stacked.csv")

    mean_final_r2 = float(np.mean(
        [summary["per_target"][t]["final_stack_oof_r2"] for t in TARGETS]
    ))
    summary["mean_final_stack_oof_r2"] = mean_final_r2
    summary["final_stack_status"] = "ok"

    log.info("═" * 64)
    log.info("FINAL Mean OOF R^2 (stack) = %.4f", mean_final_r2)
    log.info("Chemprop solo mean OOF R^2 = %.4f", mean_chemprop_r2)
    log.info("═" * 64)

    summary["runtime_sec"] = round(time.time() - t0, 1)
    (run_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", run_dir / "cv_summary.json")
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
