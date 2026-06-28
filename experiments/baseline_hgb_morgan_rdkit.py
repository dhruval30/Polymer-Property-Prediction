"""Baseline: HistGradientBoosting on RDKit descriptors + Morgan fingerprints.

Trains separate HGB models for Tg (glass transition temperature, deg C) and
Egc (chain band gap, eV). Reports 5-fold OOF R^2 per target and their mean
(the competition metric). Writes outputs to results/<RUN_NAME>/.

Run:
    python experiments/baseline_hgb_morgan_rdkit.py
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from _utils import (
    DATA_DIR,
    clean_inf,
    cv_oof,
    drop_constant_cols,
    featurize_smiles,
    prepare_run_dir,
    setup_logging,
    split_by_target_type,
)

setup_logging()
log = logging.getLogger("polymer")

RUN_NAME = "baseline_hgb_morgan_rdkit"
SEED = 42
N_FOLDS = 5
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

HGB_PARAMS = dict(
    max_iter=1000,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=SEED,
)


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**HGB_PARAMS)


def main() -> None:
    t0 = time.time()
    run_dir = prepare_run_dir(RUN_NAME)
    log.info("=== %s ===", RUN_NAME)
    log.info("results dir: %s", run_dir)
    log.info("seed=%d  n_folds=%d  morgan(radius=%d, bits=%d)",
             SEED, N_FOLDS, MORGAN_RADIUS, MORGAN_BITS)

    log.info("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)
    log.info("train target_type: %s", dict(train["target_type"].value_counts()))
    log.info("test  target_type: %s", dict(test["target_type"].value_counts()))

    train_feat = clean_inf(featurize_smiles(
        train["smiles"].tolist(),
        morgan_radius=MORGAN_RADIUS, morgan_bits=MORGAN_BITS,
        desc="train feats",
    ))
    log.info("train feature matrix: %s", train_feat.shape)

    test_feat = clean_inf(featurize_smiles(
        test["smiles"].tolist(),
        morgan_radius=MORGAN_RADIUS, morgan_bits=MORGAN_BITS,
        desc="test feats",
    ))
    log.info("test feature matrix: %s", test_feat.shape)

    train_feat, test_feat, keep_cols = drop_constant_cols(train_feat, test_feat)
    log.info("non-constant features kept: %d", len(keep_cols))

    cv_summary: dict = {
        "run_name": RUN_NAME,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "feature_count": len(keep_cols),
        "morgan": {"radius": MORGAN_RADIUS, "bits": MORGAN_BITS},
        "hgb_params": {k: v for k, v in HGB_PARAMS.items()},
        "per_target": {},
    }

    targets = ["tg", "egc"]
    oof_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for t in targets:
        log.info("─" * 60)
        log.info("Training %s", t.upper())

        X_tr, y_tr, X_te, ids_te = split_by_target_type(
            train, test, train_feat, test_feat, t
        )
        log.info("[%s] train rows: %d | test rows: %d | y range: [%.3f, %.3f]",
                 t, len(X_tr), len(X_te), float(y_tr.min()), float(y_tr.max()))

        oof, fold_scores, oof_r2 = cv_oof(
            X_tr, y_tr, t, make_model, n_folds=N_FOLDS, seed=SEED
        )
        log.info("[%s] OOF R^2 = %.4f  (per-fold: %s)",
                 t, oof_r2, [f"{s:.4f}" for s in fold_scores])
        cv_summary["per_target"][t] = {
            "oof_r2": oof_r2,
            "fold_r2": fold_scores,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        }

        oof_frames.append(pd.DataFrame({
            "smiles": train.loc[train["target_type"] == t, "smiles"].values,
            "target_type": t,
            "target": y_tr.values,
            "oof_pred": oof,
        }))

        log.info("[%s] refitting on full %d training rows...", t, len(X_tr))
        final = make_model()
        final.fit(X_tr, y_tr)
        log.info("[%s] final model iters: %d", t, final.n_iter_)
        test_frames.append(pd.DataFrame({
            "id": ids_te.values, "target": final.predict(X_te),
        }))

    mean_r2 = float(np.mean([cv_summary["per_target"][t]["oof_r2"] for t in targets]))
    cv_summary["mean_r2"] = mean_r2
    log.info("═" * 60)
    log.info("Mean OOF R^2 (competition metric) = %.4f", mean_r2)
    log.info("═" * 60)

    submission = (
        pd.concat(test_frames, axis=0)
        .sort_values("id")
        .reset_index(drop=True)
    )
    submission.to_csv(run_dir / "submission.csv", index=False)
    log.info("Wrote %s  (%d rows)", run_dir / "submission.csv", len(submission))

    oof_all = pd.concat(oof_frames, axis=0).reset_index(drop=True)
    oof_all.to_csv(run_dir / "oof.csv", index=False)
    log.info("Wrote %s  (%d rows)", run_dir / "oof.csv", len(oof_all))

    cv_summary["runtime_sec"] = round(time.time() - t0, 1)
    (run_dir / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2))
    log.info("Wrote %s", run_dir / "cv_summary.json")
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
