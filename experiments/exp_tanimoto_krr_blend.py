"""Fast: Tanimoto-kernel ridge regression blended with Phase 1 cocktail.

What it does:
- Loads cached Morgan-r2-count fingerprints, thresholds to binary.
- Computes the train-train Tanimoto kernel (one GEMM, ~1s for 4k mols).
- Trains KernelRidge per target with 5-fold CV and a small alpha grid.
- Loads `results/exp_gbm_cocktail/oof.csv` and `submission.csv`.
- NNLS-fits per-target blend weights on OOF to combine cocktail blend + KRR.
- Applies the same weights to test predictions, writes a new submission.

Why this should help:
- Tanimoto on Morgan FPs measures structural similarity in a way that's
  completely different from tree splits — KRR generalizes by smoothly
  interpolating between known similar molecules. Residuals decorrelate from
  GBMs, so the blend picks up R² without needing the solo model to be best.

Runtime: ~3-5 minutes total. No new heavy featurization, no retraining of
the existing GBM ensemble.

Run:
    python experiments/exp_tanimoto_krr_blend.py
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from tqdm import tqdm

from _utils import (
    DATA_DIR,
    RESULTS_DIR,
    compute_morgan_fp,
    prepare_run_dir,
    setup_logging,
    stratified_quantile_split,
    transform_target,
)

setup_logging()
log = logging.getLogger("polymer")

RUN_NAME = "exp_tanimoto_krr_blend"
PHASE1_RUN = "exp_gbm_cocktail"
SEED = 42
N_FOLDS = 5
N_QUANTILE_BINS = 10
TARGET_TRANSFORMS = {"tg": "identity", "egc": "log1p"}
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


# ---------------------------------------------------------------------------
# Tanimoto kernel (binary fingerprints)
# ---------------------------------------------------------------------------

def tanimoto_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """T(a, b) = |a ∩ b| / |a ∪ b| for binary fp matrices.

    Vectorized as XY / (|X|_row + |Y|_row^T - XY).
    """
    X = X.astype(np.float32, copy=False)
    if Y is None:
        Y = X
    else:
        Y = Y.astype(np.float32, copy=False)
    XY = X @ Y.T
    X_sum = X.sum(axis=1, keepdims=True)
    Y_sum = Y.sum(axis=1, keepdims=True)
    denom = X_sum + Y_sum.T - XY
    np.maximum(denom, 1e-9, out=denom)
    return (XY / denom).astype(np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    run_dir = prepare_run_dir(RUN_NAME)
    log.info("=== %s ===", RUN_NAME)
    log.info("results dir: %s", run_dir)
    log.info("blending against Phase 1 run: %s", PHASE1_RUN)

    # --- Load data + Phase 1 outputs ---
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)

    phase1_dir = RESULTS_DIR / PHASE1_RUN
    if not (phase1_dir / "oof.csv").exists():
        raise FileNotFoundError(
            f"Phase 1 OOF not found at {phase1_dir / 'oof.csv'}. "
            f"Run experiments/exp_gbm_cocktail.py first."
        )
    phase1_oof = pd.read_csv(phase1_dir / "oof.csv")
    phase1_sub = pd.read_csv(phase1_dir / "submission.csv")
    log.info("phase1 oof: %s | submission: %s", phase1_oof.shape, phase1_sub.shape)

    # --- Morgan-r2-bit (reuse cached count, threshold to binary) ---
    log.info("loading Morgan-r2-count fingerprints (cached)...")
    m2c_train = compute_morgan_fp(train["smiles"].tolist(), radius=2, n_bits=2048,
                                    count=True, desc="train/m2-cnt").values
    m2c_test = compute_morgan_fp(test["smiles"].tolist(), radius=2, n_bits=2048,
                                    count=True, desc="test/m2-cnt").values
    fp_train = (m2c_train > 0).astype(np.float32)
    fp_test = (m2c_test > 0).astype(np.float32)
    log.info("binary FPs: train=%s test=%s | mean density: %.3f",
             fp_train.shape, fp_test.shape, float(fp_train.mean()))

    summary: dict = {
        "run_name": RUN_NAME,
        "phase1_run": PHASE1_RUN,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_quantile_bins": N_QUANTILE_BINS,
        "target_transforms": TARGET_TRANSFORMS,
        "alpha_grid": ALPHA_GRID,
        "per_target": {},
    }

    oof_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    phase1_lb_per_target_r2: dict[str, float] = {}

    for t in ["tg", "egc"]:
        log.info("─" * 64)
        log.info("Tanimoto-KRR for %s", t.upper())

        tr_mask = (train["target_type"] == t).values
        te_mask = (test["target_type"] == t).values
        X_tr = fp_train[tr_mask]
        X_te = fp_test[te_mask]
        y_tr = train.loc[tr_mask, "target"].values
        ids_te = test.loc[te_mask, "id"].values
        log.info("[%s] train rows: %d | test rows: %d", t, len(X_tr), len(X_te))

        # --- Tanimoto kernel (train-train, one-shot) ---
        t_k = time.time()
        K_tt = tanimoto_kernel(X_tr)
        log.info("[%s] K_tt %s computed in %.2fs", t, K_tt.shape, time.time() - t_k)

        # --- Target transform ---
        y_t, inv = transform_target(y_tr, TARGET_TRANSFORMS[t])

        # --- 5-fold OOF KRR with per-fold alpha pick ---
        oof_krr = np.zeros(len(X_tr), dtype=np.float64)
        chosen_alphas: list[float] = []
        folds = stratified_quantile_split(
            y_t, n_folds=N_FOLDS, n_bins=N_QUANTILE_BINS, seed=SEED,
        )
        for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc=f"{t} KRR", unit="fold")):
            K_tr = K_tt[np.ix_(tr_idx, tr_idx)]
            K_va = K_tt[np.ix_(va_idx, tr_idx)]
            y_tr_fold = y_t[tr_idx]

            best_alpha, best_r2, best_pred = ALPHA_GRID[0], -np.inf, None
            for alpha in ALPHA_GRID:
                model = KernelRidge(alpha=alpha, kernel="precomputed")
                model.fit(K_tr, y_tr_fold)
                pred_t = model.predict(K_va)
                pred_orig = inv(pred_t)
                r2 = r2_score(y_tr[va_idx], pred_orig)
                if r2 > best_r2:
                    best_r2, best_alpha, best_pred = r2, alpha, pred_orig
            oof_krr[va_idx] = best_pred
            chosen_alphas.append(best_alpha)
            log.info("[%s/krr] fold %d/%d  R^2=%.4f  alpha=%.3g",
                     t, fold + 1, N_FOLDS, best_r2, best_alpha)

        krr_oof_r2 = float(r2_score(y_tr, oof_krr))
        log.info("[%s] KRR OOF R^2 = %.4f", t, krr_oof_r2)

        # --- Pull Phase 1 OOF for this target (preserves original order) ---
        p1_target = phase1_oof[phase1_oof["target_type"] == t].reset_index(drop=True)
        # alignment check: smiles should match positionally
        train_smiles_t = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        assert (p1_target["smiles"].values == train_smiles_t).all(), (
            f"[{t}] Phase 1 OOF SMILES order doesn't match train.csv filtering. "
            f"Re-run Phase 1 against the same data."
        )
        p1_blend_oof = p1_target["oof_blend"].values
        p1_oof_r2 = float(r2_score(y_tr, p1_blend_oof))
        phase1_lb_per_target_r2[t] = p1_oof_r2
        log.info("[%s] Phase 1 cocktail OOF R^2 = %.4f", t, p1_oof_r2)

        # --- NNLS blend on OOF ---
        A = np.column_stack([p1_blend_oof, oof_krr]).astype(np.float64)
        w_raw, _ = nnls(A, y_tr.astype(np.float64))
        w_sum = w_raw.sum()
        if w_sum < 1e-9:
            log.warning("[%s] NNLS gave all-zero weights, falling back to phase1-only", t)
            w_norm = np.array([1.0, 0.0])
        else:
            w_norm = w_raw / w_sum
        log.info("[%s] NNLS raw weights: phase1=%.4f krr=%.4f (sum=%.4f)",
                 t, w_raw[0], w_raw[1], w_sum)
        log.info("[%s] normalized:       phase1=%.4f krr=%.4f",
                 t, w_norm[0], w_norm[1])

        blend_oof = A @ w_norm
        blend_r2 = float(r2_score(y_tr, blend_oof))
        delta = blend_r2 - p1_oof_r2
        log.info("[%s] BLEND OOF R^2 = %.4f  (Δ over Phase 1: %+.4f)",
                 t, blend_r2, delta)

        summary["per_target"][t] = {
            "krr_oof_r2": krr_oof_r2,
            "phase1_oof_r2": p1_oof_r2,
            "blend_oof_r2": blend_r2,
            "delta_over_phase1": delta,
            "nnls_weights_raw": {"phase1": float(w_raw[0]), "krr": float(w_raw[1])},
            "nnls_weights_normalized": {"phase1": float(w_norm[0]), "krr": float(w_norm[1])},
            "chosen_alphas_per_fold": chosen_alphas,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        }

        # --- Refit KRR on full target-split training data, predict test ---
        t_r = time.time()
        from collections import Counter
        alpha_final = Counter(chosen_alphas).most_common(1)[0][0]
        log.info("[%s] final KRR alpha (mode of folds) = %.3g", t, alpha_final)
        log.info("[%s] computing K_test_train kernel...", t)
        K_test_train = tanimoto_kernel(X_te, X_tr)
        model = KernelRidge(alpha=alpha_final, kernel="precomputed")
        model.fit(K_tt, y_t)
        krr_test_pred = inv(model.predict(K_test_train))
        log.info("[%s] KRR refit + test predict done in %.2fs", t, time.time() - t_r)

        # --- Pull Phase 1 test predictions for this target's ids ---
        p1_sub_t = phase1_sub.set_index("id").loc[ids_te, "target"].values

        # --- Apply same NNLS weights to test predictions ---
        blend_test = w_norm[0] * p1_sub_t + w_norm[1] * krr_test_pred

        test_frames.append(pd.DataFrame({"id": ids_te, "target": blend_test}))
        oof_frames.append(pd.DataFrame({
            "smiles": train_smiles_t,
            "target_type": t,
            "target": y_tr,
            "oof_phase1_blend": p1_blend_oof,
            "oof_krr": oof_krr,
            "oof_blend": blend_oof,
        }))

    # --- Summarize ---
    mean_blend_r2 = float(np.mean(
        [summary["per_target"][t]["blend_oof_r2"] for t in ["tg", "egc"]]
    ))
    mean_phase1_r2 = float(np.mean(list(phase1_lb_per_target_r2.values())))
    summary["mean_blend_oof_r2"] = mean_blend_r2
    summary["mean_phase1_oof_r2"] = mean_phase1_r2
    summary["delta_over_phase1"] = mean_blend_r2 - mean_phase1_r2

    log.info("═" * 64)
    log.info("Per-target blend OOF R^2: tg=%.4f  egc=%.4f",
             summary["per_target"]["tg"]["blend_oof_r2"],
             summary["per_target"]["egc"]["blend_oof_r2"])
    log.info("Mean OOF R^2 (blend)   = %.4f", mean_blend_r2)
    log.info("Mean OOF R^2 (phase 1) = %.4f", mean_phase1_r2)
    log.info("Δ over Phase 1         = %+.4f", mean_blend_r2 - mean_phase1_r2)
    log.info("═" * 64)

    submission = (
        pd.concat(test_frames, axis=0).sort_values("id").reset_index(drop=True)
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
