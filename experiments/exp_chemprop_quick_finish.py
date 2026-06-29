"""Finish the Chemprop run early using whatever checkpoint we have.

Loads results/exp_chemprop_multitask/checkpoint.npz and:
- Reconstructs partial Chemprop OOF (only completed folds have predictions)
- Averages test predictions over completed (fold, seed) models
- NNLS-stacks against Phase 1 + KRR stack using ONLY rows where Chemprop OOF
  exists, then applies the same weights to ALL test predictions

The NNLS weights are fit on a subset of training data (~40% if 2 of 5 folds
done). That subset is still stratified across the target distribution (because
each fold is a quantile-stratified sample), so the weights should generalize
reasonably to the rest of the data, but expect slightly more variance than a
full 5-fold OOF.

Run AFTER killing the main training run:
    python experiments/exp_chemprop_quick_finish.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import r2_score

from _utils import (
    DATA_DIR,
    RESULTS_DIR,
    prepare_run_dir,
    setup_logging,
    transform_target,
)

setup_logging()
log = logging.getLogger("polymer")

RUN_NAME = "exp_chemprop_multitask"  # outputs go into the existing dir
PHASE1_RUN = "exp_gbm_cocktail"
KRR_STACK_RUN = "exp_stack_multi_krr"
TARGETS = ["tg", "egc"]
TARGET_TRANSFORMS = {"tg": "identity", "egc": "log1p"}


def long_to_wide(train: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for smi, t, y in zip(train["smiles"].values, train["target_type"].values, train["target"].values):
        if smi not in rows:
            rows[smi] = {"smiles": smi, "tg": np.nan, "egc": np.nan}
        rows[smi][t] = y
    return pd.DataFrame(list(rows.values()))[["smiles", "tg", "egc"]]


def inverse_targets(transformed: np.ndarray) -> np.ndarray:
    out = np.empty_like(transformed)
    for ti, t in enumerate(TARGETS):
        _, inv = transform_target(np.array([0.0]), TARGET_TRANSFORMS[t])
        out[:, ti] = inv(transformed[:, ti])
    return out


def main() -> None:
    t0 = time.time()
    run_dir = prepare_run_dir(RUN_NAME)
    log.info("=== Chemprop quick-finish (using partial checkpoint) ===")
    log.info("results dir: %s", run_dir)

    ckpt_path = run_dir / "checkpoint.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} — run the main training first")

    ck = np.load(ckpt_path, allow_pickle=False)
    oof_t = ck["oof_t"]
    test_acc_t = ck["test_acc_t"]
    n_models = int(ck["n_models_added_to_test"])
    completed_folds = int(ck["completed_folds"])
    log.info("checkpoint: %d completed folds, %d models in test accum",
             completed_folds, n_models)
    if n_models == 0:
        raise RuntimeError("No models in test accumulator — checkpoint is empty")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)

    wide_orig = long_to_wide(train)
    test_unique_smis = test["smiles"].drop_duplicates().tolist()
    train_smis_all = wide_orig["smiles"].values

    log.info("train unique SMILES: %d  | oof_t shape: %s", len(wide_orig), oof_t.shape)
    log.info("test unique SMILES: %d  | test_acc_t shape: %s",
             len(test_unique_smis), test_acc_t.shape)

    # Average test predictions over all completed models
    test_t = test_acc_t / n_models
    test_orig = inverse_targets(test_t)
    oof_orig = inverse_targets(oof_t)

    # Identify rows where Chemprop OOF is populated (folds 1..completed_folds rows)
    has_oof = ~np.isnan(oof_t).any(axis=1)
    log.info("rows with Chemprop OOF: %d / %d (%.1f%%)",
             int(has_oof.sum()), len(has_oof), 100 * has_oof.mean())

    # Chemprop solo R^2 on completed-fold rows
    y_all_orig = wide_orig[TARGETS].values
    chemprop_r2_partial: dict[str, float] = {}
    for ti, t in enumerate(TARGETS):
        mask = (~np.isnan(y_all_orig[:, ti])) & has_oof
        if mask.sum() > 0:
            r2 = float(r2_score(y_all_orig[mask, ti], oof_orig[mask, ti]))
            chemprop_r2_partial[t] = r2
            log.info("[%s] Chemprop partial OOF R^2 = %.4f (n=%d)", t, r2, int(mask.sum()))

    # Raw Chemprop test predictions (one row per test id)
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
    log.info("Wrote %s  (%d rows)", run_dir / "submission_chemprop_only.csv", len(chemprop_sub))

    # Long-format OOF aligned with train.csv; NaN for rows not yet in OOF
    smi_to_oof_idx = {s: i for i, s in enumerate(train_smis_all)}
    oof_rows = []
    for _, row in train.iterrows():
        smi = row["smiles"]; tt = row["target_type"]
        ti = TARGETS.index(tt)
        idx = smi_to_oof_idx[smi]
        oof_rows.append({
            "smiles": smi,
            "target_type": tt,
            "target": float(row["target"]),
            "oof_chemprop": float(oof_orig[idx, ti]) if has_oof[idx] else np.nan,
            "has_oof": bool(has_oof[idx]),
        })
    chemprop_oof = pd.DataFrame(oof_rows)
    chemprop_oof.to_csv(run_dir / "oof.csv", index=False)
    log.info("Wrote %s  (%d rows)", run_dir / "oof.csv", len(chemprop_oof))

    # ---- NNLS stack ----
    log.info("─" * 64)
    log.info("Loading prior runs for NNLS stack...")
    phase1_oof = pd.read_csv(RESULTS_DIR / PHASE1_RUN / "oof.csv")
    phase1_sub = pd.read_csv(RESULTS_DIR / PHASE1_RUN / "submission.csv")
    krr_oof_path = RESULTS_DIR / KRR_STACK_RUN / "oof.csv"
    krr_sub_path = RESULTS_DIR / KRR_STACK_RUN / "submission.csv"
    if krr_oof_path.exists() and krr_sub_path.exists():
        krr_oof = pd.read_csv(krr_oof_path)
        krr_sub = pd.read_csv(krr_sub_path)
        have_krr = True
        log.info("KRR stack loaded (3-base stacking)")
    else:
        krr_oof = krr_sub = None
        have_krr = False
        log.warning("KRR stack not found at %s — falling back to phase1+chemprop only",
                     RESULTS_DIR / KRR_STACK_RUN)

    summary: dict = {
        "run_name": RUN_NAME,
        "mode": "partial_finish",
        "checkpoint_completed_folds": completed_folds,
        "checkpoint_n_models": n_models,
        "chemprop_oof_rows": int(has_oof.sum()),
        "chemprop_oof_pct": float(has_oof.mean()),
        "chemprop_partial_oof_r2": chemprop_r2_partial,
        "per_target": {},
    }

    final_test_frames: list[pd.DataFrame] = []

    for t in TARGETS:
        log.info("─" * 32)
        log.info("Stacking %s", t.upper())
        tr_mask = (train["target_type"] == t).values
        te_mask = (test["target_type"] == t).values
        train_smis_t = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        y_tr = train.loc[tr_mask, "target"].reset_index(drop=True).values
        ids_te = test.loc[te_mask, "id"].reset_index(drop=True).values

        p1_t = phase1_oof[phase1_oof["target_type"] == t].reset_index(drop=True)
        cp_t = chemprop_oof[chemprop_oof["target_type"] == t].reset_index(drop=True)
        assert (p1_t["smiles"].values == train_smis_t).all()
        assert (cp_t["smiles"].values == train_smis_t).all()

        mask = cp_t["has_oof"].values
        log.info("[%s] using %d / %d rows where Chemprop OOF exists",
                 t, int(mask.sum()), len(mask))

        p1_blend = p1_t["oof_blend"].values
        cp_oof_arr = cp_t["oof_chemprop"].values
        p1_test = phase1_sub.set_index("id").loc[ids_te, "target"].values
        cp_test_v = chemprop_sub.set_index("id").loc[ids_te, "target"].values

        if have_krr:
            krr_t = krr_oof[krr_oof["target_type"] == t].reset_index(drop=True)
            assert (krr_t["smiles"].values == train_smis_t).all()
            krr_stack = krr_t["oof_stack"].values
            krr_test = krr_sub.set_index("id").loc[ids_te, "target"].values
        else:
            krr_stack = krr_test = None

        if mask.sum() < 50:
            log.warning("[%s] not enough rows with Chemprop OOF — falling back to non-chemprop blend", t)
            if have_krr:
                A = np.column_stack([p1_blend, krr_stack]).astype(np.float64)
                w_raw, _ = nnls(A, y_tr.astype(np.float64))
                w_sum = w_raw.sum()
                w_norm = (w_raw / w_sum) if w_sum > 1e-9 else np.array([1.0, 0.0])
                blend_oof = A @ w_norm
                blend_test = w_norm[0] * p1_test + w_norm[1] * krr_test
                bases = ["phase1", "krr_stack"]
            else:
                w_norm = np.array([1.0])
                blend_oof = p1_blend
                blend_test = p1_test
                bases = ["phase1"]
            blend_r2 = float(r2_score(y_tr, blend_oof))
            log.info("[%s] fallback weights: %s  → OOF R^2=%.4f",
                     t, {b: float(w) for b, w in zip(bases, w_norm)}, blend_r2)
            summary["per_target"][t] = {
                "stack_type": "+".join(bases),
                "weights_normalized": {b: float(w) for b, w in zip(bases, w_norm)},
                "stack_oof_r2_full": blend_r2,
            }
            final_test_frames.append(pd.DataFrame({"id": ids_te, "target": blend_test}))
            continue

        # NNLS over [phase1, (krr), chemprop], fit on the mask subset
        if have_krr:
            A_full = np.column_stack([p1_blend, krr_stack, cp_oof_arr]).astype(np.float64)
            test_mat = np.column_stack([p1_test, krr_test, cp_test_v]).astype(np.float64)
            bases = ["phase1", "krr_stack", "chemprop"]
        else:
            A_full = np.column_stack([p1_blend, cp_oof_arr]).astype(np.float64)
            test_mat = np.column_stack([p1_test, cp_test_v]).astype(np.float64)
            bases = ["phase1", "chemprop"]

        A_sub = A_full[mask]
        y_sub = y_tr[mask]

        w_raw, _ = nnls(A_sub, y_sub.astype(np.float64))
        w_sum = w_raw.sum()
        w_norm = (w_raw / w_sum) if w_sum > 1e-9 else np.eye(len(bases))[0]

        log.info("[%s] NNLS weights (raw → norm, sum=%.4f):", t, w_sum)
        for name, wr, wn in zip(bases, w_raw, w_norm):
            log.info("    %-10s raw=%.4f  norm=%.4f", name, wr, wn)

        blend_oof_sub = A_sub @ w_norm
        blend_r2_sub = float(r2_score(y_sub, blend_oof_sub))

        per_base: dict[str, float] = {
            "phase1_full_r2": float(r2_score(y_tr, p1_blend)),
            "phase1_subset_r2": float(r2_score(y_sub, p1_blend[mask])),
            "chemprop_subset_r2": float(r2_score(y_sub, cp_oof_arr[mask])),
        }
        if have_krr:
            per_base["krr_subset_r2"] = float(r2_score(y_sub, krr_stack[mask]))
        log.info("[%s] per-base subset OOF R^2: %s",
                 t, {k: f"{v:.4f}" for k, v in per_base.items()})
        log.info("[%s] STACK OOF R^2 (subset, %d rows) = %.4f",
                 t, int(mask.sum()), blend_r2_sub)

        # Apply weights to test predictions
        blend_test = test_mat @ w_norm

        final_test_frames.append(pd.DataFrame({"id": ids_te, "target": blend_test}))
        summary["per_target"][t] = {
            "stack_type": "+".join(bases),
            "weights_normalized": {b: float(w) for b, w in zip(bases, w_norm)},
            "weights_raw": {b: float(w) for b, w in zip(bases, w_raw)},
            "per_base_subset_r2": per_base,
            "stack_oof_r2_subset": blend_r2_sub,
            "n_rows_used_for_nnls": int(mask.sum()),
            "n_rows_total": int(len(mask)),
        }

    final_sub = pd.concat(final_test_frames, axis=0).sort_values("id").reset_index(drop=True)
    final_sub.to_csv(run_dir / "submission.csv", index=False)
    log.info("Wrote %s  (%d rows)", run_dir / "submission.csv", len(final_sub))

    summary["runtime_sec"] = round(time.time() - t0, 1)
    (run_dir / "cv_summary_partial.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote cv_summary_partial.json")
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
