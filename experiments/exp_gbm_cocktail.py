"""Phase 1: GBM ensemble (LightGBM + CatBoost + HGB) on a feature cocktail.

Features (concatenated):
- RDKit 2D descriptors  (~210)
- Morgan count r=2, 2048 bits
- Morgan count r=3, 2048 bits
- MACCS keys (167)
- Avalon FP (512)
- Atom-Pair count (2048)
- Topological-Torsion count (2048)
- Mordred 2D descriptors (~1600, if mordred installed)

Models: LightGBM + CatBoost + HistGradientBoosting, simple mean blend per target.
Target transforms: log1p on Egc (range ~0.1–9.9 eV), identity on Tg.
CV: StratifiedKFold on quantile bins of the (transformed) target.

Run:
    pip install lightgbm catboost mordred
    python experiments/exp_gbm_cocktail.py
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("This script requires lightgbm. Install with: pip install lightgbm") from e

try:
    from catboost import CatBoostRegressor
except ImportError as e:
    raise ImportError("This script requires catboost. Install with: pip install catboost") from e

from _utils import (
    DATA_DIR,
    clean_inf,
    compute_atom_pair_fp,
    compute_avalon_fp,
    compute_maccs,
    compute_mordred_descriptors,
    compute_morgan_fp,
    compute_rdkit_descriptors,
    compute_topological_torsion_fp,
    drop_constant_cols,
    prepare_run_dir,
    sanitize_columns,
    setup_logging,
    split_by_target_type,
    stratified_quantile_split,
    transform_target,
)

setup_logging()
log = logging.getLogger("polymer")

RUN_NAME = "exp_gbm_cocktail"
SEED = 42
N_FOLDS = 5
N_QUANTILE_BINS = 10

TARGET_TRANSFORMS = {"tg": "identity", "egc": "log1p"}

LGB_PARAMS = dict(
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=10,
    feature_fraction=0.5,
    bagging_fraction=0.85,
    bagging_freq=5,
    reg_lambda=1.0,
    objective="regression",
    metric="rmse",
    verbosity=-1,
    random_state=SEED,
    n_jobs=-1,
)

CAT_PARAMS = dict(
    iterations=4000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    grow_policy="SymmetricTree",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
)

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


# ---------------------------------------------------------------------------
# feature assembly
# ---------------------------------------------------------------------------

def build_features(smiles: list[str], *, split_label: str) -> pd.DataFrame:
    log.info("=== building feature cocktail: %s (n=%d) ===", split_label, len(smiles))
    parts = [
        compute_rdkit_descriptors(smiles, desc=f"{split_label}/rdk-desc"),
        compute_morgan_fp(smiles, radius=2, n_bits=2048, count=True,
                          desc=f"{split_label}/morgan2-cnt"),
        compute_morgan_fp(smiles, radius=3, n_bits=2048, count=True,
                          desc=f"{split_label}/morgan3-cnt"),
        compute_maccs(smiles, desc=f"{split_label}/maccs"),
        compute_avalon_fp(smiles, n_bits=512, desc=f"{split_label}/avalon"),
        compute_atom_pair_fp(smiles, n_bits=2048, count=True,
                              desc=f"{split_label}/atompair-cnt"),
        compute_topological_torsion_fp(smiles, n_bits=2048, count=True,
                                        desc=f"{split_label}/torsion-cnt"),
        compute_mordred_descriptors(smiles, ignore_3d=True,
                                     desc=f"{split_label}/mordred"),
    ]
    parts = [p.reset_index(drop=True) for p in parts if p.shape[1] > 0]
    X = pd.concat(parts, axis=1)
    X = clean_inf(X)
    X = sanitize_columns(X)
    log.info("[%s] combined feature matrix: %s", split_label, X.shape)
    return X


# ---------------------------------------------------------------------------
# per-fold trainers (note: va_idx is used for early stopping — slightly
# optimistic CV, but standard Kaggle practice; calibrate against LB)
# ---------------------------------------------------------------------------

def train_lgb(X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    return model, int(model.best_iteration_ or LGB_PARAMS["n_estimators"])


def train_cat(X_tr, y_tr, X_va, y_va):
    model = CatBoostRegressor(**CAT_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=200,
        verbose=False,
    )
    return model, int(model.get_best_iteration() or CAT_PARAMS["iterations"])


def train_hgb(X_tr, y_tr, X_va=None, y_va=None):
    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(X_tr, y_tr)
    return model, int(model.n_iter_)


MODELS = {"lgb": train_lgb, "cat": train_cat, "hgb": train_hgb}


def fit_full(model_name: str, X, y, n_iters: int):
    if model_name == "lgb":
        p = dict(LGB_PARAMS); p["n_estimators"] = max(int(n_iters * 1.10), 200)
        m = lgb.LGBMRegressor(**p); m.fit(X, y); return m
    if model_name == "cat":
        p = dict(CAT_PARAMS); p["iterations"] = max(int(n_iters * 1.10), 200)
        m = CatBoostRegressor(**p); m.fit(X, y, verbose=False); return m
    if model_name == "hgb":
        m = HistGradientBoostingRegressor(**HGB_PARAMS); m.fit(X, y); return m
    raise ValueError(model_name)


# ---------------------------------------------------------------------------
# CV per target
# ---------------------------------------------------------------------------

def cv_per_target(X: pd.DataFrame, y_orig: np.ndarray, target_name: str, transform_kind: str):
    y_t, inv = transform_target(y_orig, transform_kind)
    log.info("[%s] target transform: %s | y_t range: [%.3f, %.3f]",
             target_name, transform_kind, float(y_t.min()), float(y_t.max()))

    oof = {k: np.zeros(len(X), dtype=np.float64) for k in MODELS}
    best_iters = {k: [] for k in MODELS}

    folds = stratified_quantile_split(y_t, n_folds=N_FOLDS, n_bins=N_QUANTILE_BINS, seed=SEED)
    fold_bar = tqdm(folds, desc=f"{target_name} CV", unit="fold")
    for fold, (tr_idx, va_idx) in enumerate(fold_bar):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_t[tr_idx], y_t[va_idx]
        for k, train_fn in MODELS.items():
            t0 = time.time()
            model, n_it = train_fn(X_tr, y_tr, X_va, y_va)
            pred_orig = inv(model.predict(X_va))
            oof[k][va_idx] = pred_orig
            r2 = r2_score(y_orig[va_idx], pred_orig)
            best_iters[k].append(n_it)
            log.info("[%s/%s] fold %d/%d  R^2=%.4f  iters=%d  (%.1fs)",
                     target_name, k, fold + 1, N_FOLDS, r2, n_it, time.time() - t0)
        fold_bar.set_postfix(
            **{k: f"{r2_score(y_orig[va_idx], oof[k][va_idx]):.3f}" for k in MODELS}
        )

    per_model_r2 = {k: float(r2_score(y_orig, oof[k])) for k in MODELS}
    blend_oof = np.mean(np.stack([oof[k] for k in MODELS], axis=0), axis=0)
    blend_r2 = float(r2_score(y_orig, blend_oof))
    log.info("[%s] OOF R^2 per model: %s",
             target_name, {k: f"{v:.4f}" for k, v in per_model_r2.items()})
    log.info("[%s] OOF R^2 (mean blend) = %.4f", target_name, blend_r2)
    return oof, blend_oof, per_model_r2, blend_r2, best_iters


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    run_dir = prepare_run_dir(RUN_NAME)
    log.info("=== %s ===", RUN_NAME)
    log.info("results dir: %s", run_dir)
    log.info("seed=%d  n_folds=%d  bins=%d", SEED, N_FOLDS, N_QUANTILE_BINS)
    log.info("target transforms: %s", TARGET_TRANSFORMS)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)
    log.info("train target_type: %s", dict(train["target_type"].value_counts()))
    log.info("test  target_type: %s", dict(test["target_type"].value_counts()))

    X_train_full = build_features(train["smiles"].tolist(), split_label="train")
    X_test_full = build_features(test["smiles"].tolist(), split_label="test")

    # column alignment (Mordred may include cols computed only on one split)
    common = sorted(set(X_train_full.columns) & set(X_test_full.columns))
    X_train_full = X_train_full[common]
    X_test_full = X_test_full[common]
    log.info("columns after intersection: %d (train:%d test:%d)",
             len(common), X_train_full.shape[1], X_test_full.shape[1])

    X_train_full, X_test_full, kept = drop_constant_cols(X_train_full, X_test_full)
    log.info("features after drop-constant: %d", len(kept))

    summary: dict = {
        "run_name": RUN_NAME,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_quantile_bins": N_QUANTILE_BINS,
        "feature_count": len(kept),
        "target_transforms": TARGET_TRANSFORMS,
        "models": list(MODELS.keys()),
        "lgb_params": LGB_PARAMS,
        "cat_params": CAT_PARAMS,
        "hgb_params": HGB_PARAMS,
        "per_target": {},
    }

    oof_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for t in ["tg", "egc"]:
        log.info("─" * 64)
        log.info("Training %s", t.upper())

        X_tr, y_tr_s, X_te, ids_te = split_by_target_type(
            train, test, X_train_full, X_test_full, t,
        )
        y_tr = y_tr_s.values
        log.info("[%s] train rows: %d | test rows: %d | y range: [%.3f, %.3f]",
                 t, len(X_tr), len(X_te), float(y_tr.min()), float(y_tr.max()))

        oof, blend_oof, per_model_r2, blend_r2, best_iters = cv_per_target(
            X_tr, y_tr, t, TARGET_TRANSFORMS[t],
        )

        summary["per_target"][t] = {
            "per_model_oof_r2": per_model_r2,
            "blend_oof_r2": blend_r2,
            "median_best_iters": {k: int(np.median(v)) for k, v in best_iters.items()},
            "fold_best_iters": {k: [int(x) for x in v] for k, v in best_iters.items()},
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        }

        oof_frames.append(pd.DataFrame({
            "smiles": train.loc[train["target_type"] == t, "smiles"].values,
            "target_type": t,
            "target": y_tr,
            **{f"oof_{k}": oof[k] for k in MODELS},
            "oof_blend": blend_oof,
        }))

        # Final refit per model on full target-split training data
        log.info("[%s] refitting on full %d rows...", t, len(X_tr))
        y_t_full, inv = transform_target(y_tr, TARGET_TRANSFORMS[t])
        test_preds: dict[str, np.ndarray] = {}
        for k in MODELS:
            n_it = int(np.median(best_iters[k]))
            t_fit = time.time()
            model = fit_full(k, X_tr, y_t_full, n_it)
            test_preds[k] = inv(model.predict(X_te))
            log.info("[%s/%s] refit done (iters=%d, %.1fs)",
                     t, k, n_it, time.time() - t_fit)

        blend_test = np.mean(np.stack(list(test_preds.values()), axis=0), axis=0)
        test_frames.append(pd.DataFrame({"id": ids_te.values, "target": blend_test}))

    mean_r2 = float(np.mean([summary["per_target"][t]["blend_oof_r2"] for t in ["tg", "egc"]]))
    summary["mean_blend_oof_r2"] = mean_r2

    log.info("═" * 64)
    log.info("Per-target OOF R^2 (blend): tg=%.4f  egc=%.4f",
             summary["per_target"]["tg"]["blend_oof_r2"],
             summary["per_target"]["egc"]["blend_oof_r2"])
    log.info("Mean OOF R^2 (competition metric) = %.4f", mean_r2)
    log.info("═" * 64)

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

    summary["runtime_sec"] = round(time.time() - t0, 1)
    (run_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", run_dir / "cv_summary.json")
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
