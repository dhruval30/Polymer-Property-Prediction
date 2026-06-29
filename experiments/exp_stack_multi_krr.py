"""Fast: stack 4 Tanimoto-KRR bases (different fingerprints) with Phase 1 blend.

Bases:
- Phase 1 cocktail blend (LGB + CAT + HGB on the feature cocktail)
- KRR-Tanimoto on Morgan-r2-bit  (small-radius substructure similarity)
- KRR-Tanimoto on Morgan-r3-bit  (medium-radius motifs)
- KRR-Tanimoto on AtomPair-bit   (long-range topological distance)
- KRR-Tanimoto on Torsion-bit    (4-atom rotational patterns)

Stack via per-target NNLS on OOF. Apply same weights to test predictions.

Runtime: ~5-10 minutes. Reuses cached fingerprints + Phase 1 OOF/submission.

Run:
    python experiments/exp_stack_multi_krr.py
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from tqdm import tqdm

from _utils import (
    DATA_DIR,
    RESULTS_DIR,
    compute_atom_pair_fp,
    compute_morgan_fp,
    compute_topological_torsion_fp,
    prepare_run_dir,
    setup_logging,
    stratified_quantile_split,
    transform_target,
)

setup_logging()
log = logging.getLogger("polymer")

RUN_NAME = "exp_stack_multi_krr"
PHASE1_RUN = "exp_gbm_cocktail"
SEED = 42
N_FOLDS = 5
N_QUANTILE_BINS = 10
TARGET_TRANSFORMS = {"tg": "identity", "egc": "log1p"}
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]

KRR_FP_BASES = ["m2", "m3", "ap", "tt"]


def tanimoto_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
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


def krr_cv_and_test(
    fp_train: np.ndarray, fp_test: np.ndarray,
    y_orig: np.ndarray, y_t: np.ndarray, inv,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    K_tt = tanimoto_kernel(fp_train)
    folds = stratified_quantile_split(y_t, n_folds=N_FOLDS, n_bins=N_QUANTILE_BINS, seed=SEED)
    oof = np.zeros(len(fp_train), dtype=np.float64)
    chosen_alphas: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc=label, unit="fold", leave=False)):
        K_tr = K_tt[np.ix_(tr_idx, tr_idx)]
        K_va = K_tt[np.ix_(va_idx, tr_idx)]
        best_a, best_r2, best_pred = ALPHA_GRID[0], -np.inf, None
        for alpha in ALPHA_GRID:
            m = KernelRidge(alpha=alpha, kernel="precomputed")
            m.fit(K_tr, y_t[tr_idx])
            pred = inv(m.predict(K_va))
            r2 = r2_score(y_orig[va_idx], pred)
            if r2 > best_r2:
                best_r2, best_a, best_pred = r2, alpha, pred
        oof[va_idx] = best_pred
        chosen_alphas.append(best_a)

    alpha_final = Counter(chosen_alphas).most_common(1)[0][0]
    K_test_train = tanimoto_kernel(fp_test, fp_train)
    m = KernelRidge(alpha=alpha_final, kernel="precomputed")
    m.fit(K_tt, y_t)
    test_pred = inv(m.predict(K_test_train))
    return oof, test_pred, alpha_final


def main() -> None:
    t0 = time.time()
    run_dir = prepare_run_dir(RUN_NAME)
    log.info("=== %s ===", RUN_NAME)
    log.info("results dir: %s", run_dir)
    log.info("stacking against Phase 1 run: %s", PHASE1_RUN)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)

    phase1_dir = RESULTS_DIR / PHASE1_RUN
    phase1_oof = pd.read_csv(phase1_dir / "oof.csv")
    phase1_sub = pd.read_csv(phase1_dir / "submission.csv")
    log.info("phase1 oof: %s | submission: %s", phase1_oof.shape, phase1_sub.shape)

    log.info("loading + binarizing cached fingerprints...")
    train_smi = train["smiles"].tolist()
    test_smi = test["smiles"].tolist()

    def _bit(df):
        return (df.values > 0).astype(np.float32)

    fps_train = {
        "m2": _bit(compute_morgan_fp(train_smi, radius=2, n_bits=2048, count=True, desc="train/m2-cnt")),
        "m3": _bit(compute_morgan_fp(train_smi, radius=3, n_bits=2048, count=True, desc="train/m3-cnt")),
        "ap": _bit(compute_atom_pair_fp(train_smi, n_bits=2048, count=True, desc="train/ap-cnt")),
        "tt": _bit(compute_topological_torsion_fp(train_smi, n_bits=2048, count=True, desc="train/tt-cnt")),
    }
    fps_test = {
        "m2": _bit(compute_morgan_fp(test_smi, radius=2, n_bits=2048, count=True, desc="test/m2-cnt")),
        "m3": _bit(compute_morgan_fp(test_smi, radius=3, n_bits=2048, count=True, desc="test/m3-cnt")),
        "ap": _bit(compute_atom_pair_fp(test_smi, n_bits=2048, count=True, desc="test/ap-cnt")),
        "tt": _bit(compute_topological_torsion_fp(test_smi, n_bits=2048, count=True, desc="test/tt-cnt")),
    }
    for k in KRR_FP_BASES:
        log.info("  %s: train=%s test=%s density=%.3f",
                 k, fps_train[k].shape, fps_test[k].shape, float(fps_train[k].mean()))

    summary: dict = {
        "run_name": RUN_NAME, "phase1_run": PHASE1_RUN,
        "seed": SEED, "n_folds": N_FOLDS, "n_quantile_bins": N_QUANTILE_BINS,
        "target_transforms": TARGET_TRANSFORMS, "alpha_grid": ALPHA_GRID,
        "krr_fp_bases": KRR_FP_BASES, "per_target": {},
    }

    oof_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for t in ["tg", "egc"]:
        log.info("─" * 64)
        log.info("STACK: %s", t.upper())

        tr_mask = (train["target_type"] == t).values
        te_mask = (test["target_type"] == t).values
        y_tr = train.loc[tr_mask, "target"].values
        ids_te = test.loc[te_mask, "id"].values
        log.info("[%s] train rows: %d | test rows: %d", t, len(y_tr), te_mask.sum())

        y_t, inv = transform_target(y_tr, TARGET_TRANSFORMS[t])

        krr_oofs: dict[str, np.ndarray] = {}
        krr_tests: dict[str, np.ndarray] = {}
        krr_alphas: dict[str, float] = {}

        for fp_key in KRR_FP_BASES:
            tk = time.time()
            log.info("[%s] KRR-%s training...", t, fp_key)
            oof, tp, alpha = krr_cv_and_test(
                fps_train[fp_key][tr_mask],
                fps_test[fp_key][te_mask],
                y_tr, y_t, inv,
                label=f"{t}/krr-{fp_key}",
            )
            krr_oofs[fp_key] = oof
            krr_tests[fp_key] = tp
            krr_alphas[fp_key] = float(alpha)
            log.info("[%s] KRR-%s OOF R^2 = %.4f  alpha=%.3g  (%.1fs)",
                     t, fp_key, r2_score(y_tr, oof), alpha, time.time() - tk)

        p1_target = phase1_oof[phase1_oof["target_type"] == t].reset_index(drop=True)
        train_smiles_t = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        assert (p1_target["smiles"].values == train_smiles_t).all(), (
            f"[{t}] Phase 1 OOF SMILES order mismatch — re-run Phase 1."
        )
        p1_oof = p1_target["oof_blend"].values
        p1_test = phase1_sub.set_index("id").loc[ids_te, "target"].values

        bases = ["phase1"] + [f"krr_{k}" for k in KRR_FP_BASES]
        A_oof = np.column_stack([
            p1_oof, *[krr_oofs[k] for k in KRR_FP_BASES],
        ]).astype(np.float64)
        A_test = np.column_stack([
            p1_test, *[krr_tests[k] for k in KRR_FP_BASES],
        ]).astype(np.float64)

        w_raw, _ = nnls(A_oof, y_tr.astype(np.float64))
        w_sum = w_raw.sum()
        if w_sum < 1e-9:
            log.warning("[%s] NNLS gave all-zero weights, defaulting to phase1-only", t)
            w_norm = np.zeros_like(w_raw); w_norm[0] = 1.0
        else:
            w_norm = w_raw / w_sum

        log.info("[%s] NNLS weights (raw → norm, sum=%.4f):", t, w_sum)
        for b, wr, wn in zip(bases, w_raw, w_norm):
            log.info("    %-10s raw=%.4f  norm=%.4f", b, wr, wn)

        per_base_r2 = {
            "phase1": float(r2_score(y_tr, p1_oof)),
            **{f"krr_{k}": float(r2_score(y_tr, krr_oofs[k])) for k in KRR_FP_BASES},
        }
        log.info("[%s] per-base OOF R^2: %s",
                 t, {k: f"{v:.4f}" for k, v in per_base_r2.items()})

        blend_oof = A_oof @ w_norm
        blend_test = A_test @ w_norm
        blend_r2 = float(r2_score(y_tr, blend_oof))
        delta = blend_r2 - per_base_r2["phase1"]
        log.info("[%s] STACK OOF R^2 = %.4f  (Δ over Phase 1: %+.4f)",
                 t, blend_r2, delta)

        summary["per_target"][t] = {
            "per_base_oof_r2": per_base_r2,
            "stack_oof_r2": blend_r2,
            "delta_over_phase1": delta,
            "nnls_weights_raw": {b: float(w) for b, w in zip(bases, w_raw)},
            "nnls_weights_normalized": {b: float(w) for b, w in zip(bases, w_norm)},
            "krr_alphas": krr_alphas,
            "n_train": int(len(y_tr)),
            "n_test": int(te_mask.sum()),
        }

        test_frames.append(pd.DataFrame({"id": ids_te, "target": blend_test}))
        oof_record = {
            "smiles": train_smiles_t,
            "target_type": t,
            "target": y_tr,
            "oof_phase1": p1_oof,
            **{f"oof_krr_{k}": krr_oofs[k] for k in KRR_FP_BASES},
            "oof_stack": blend_oof,
        }
        oof_frames.append(pd.DataFrame(oof_record))

    mean_stack_r2 = float(np.mean(
        [summary["per_target"][t]["stack_oof_r2"] for t in ["tg", "egc"]]
    ))
    mean_phase1_r2 = float(np.mean(
        [summary["per_target"][t]["per_base_oof_r2"]["phase1"] for t in ["tg", "egc"]]
    ))
    summary["mean_stack_oof_r2"] = mean_stack_r2
    summary["mean_phase1_oof_r2"] = mean_phase1_r2
    summary["delta_over_phase1"] = mean_stack_r2 - mean_phase1_r2

    log.info("═" * 64)
    log.info("Per-target stack OOF: tg=%.4f  egc=%.4f",
             summary["per_target"]["tg"]["stack_oof_r2"],
             summary["per_target"]["egc"]["stack_oof_r2"])
    log.info("Mean OOF (stack)     = %.4f", mean_stack_r2)
    log.info("Mean OOF (phase 1)   = %.4f", mean_phase1_r2)
    log.info("Δ over Phase 1       = %+.4f", mean_stack_r2 - mean_phase1_r2)
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
