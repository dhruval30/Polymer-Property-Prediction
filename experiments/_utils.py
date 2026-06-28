"""Shared utilities for polymer property prediction experiments.

Every experiment script in this directory should:
- `from _utils import ...` for featurization, CV, and IO helpers
- define a module-level `RUN_NAME` and call `prepare_run_dir(RUN_NAME)`
- write its `submission.csv`, `oof.csv`, `cv_summary.json` into that run dir

Feature computation is cached under `.cache/features/` keyed by SMILES list +
config, so repeated experiments don't repeat ~5 minutes of RDKit work.
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"
CACHE_DIR = REPO / ".cache" / "features"

log = logging.getLogger("polymer")


# ---------------------------------------------------------------------------
# logging + run dirs
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """One-shot global logging setup. Idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def prepare_run_dir(run_name: str) -> Path:
    """Create and return results/<run_name>/."""
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# featurization
# ---------------------------------------------------------------------------

def featurize_smiles(
    smiles_list: list[str],
    *,
    include_descriptors: bool = True,
    include_morgan: bool = True,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    desc: str = "featurizing",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Compute RDKit descriptors and/or Morgan fingerprints for each SMILES.

    Returns a wide DataFrame with descriptor columns then `fp_0..fp_{N-1}`.
    Cached on disk by (smiles_list, config) hash under .cache/features/.
    """
    cfg = (include_descriptors, include_morgan, morgan_radius, morgan_bits)
    cache_path = CACHE_DIR / f"feat_{_features_cache_key(smiles_list, cfg)}.pkl.gz"

    if use_cache and cache_path.exists():
        log.info("[%s] cache hit: %s", desc, cache_path.name)
        return _load_pickle(cache_path)

    probe = Chem.MolFromSmiles("CCO")
    desc_keys = list(Descriptors.CalcMolDescriptors(probe).keys())
    nan_desc = {k: np.nan for k in desc_keys}

    desc_rows: list[dict] = []
    fp_arr = (
        np.zeros((len(smiles_list), morgan_bits), dtype=np.int8)
        if include_morgan else None
    )
    n_failed = 0

    for i, smi in enumerate(tqdm(smiles_list, desc=desc, unit="mol")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            if include_descriptors:
                desc_rows.append(dict(nan_desc))
            n_failed += 1
            continue
        if include_descriptors:
            try:
                desc_rows.append(Descriptors.CalcMolDescriptors(mol))
            except Exception:
                desc_rows.append(dict(nan_desc))
                n_failed += 1
        if include_morgan:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, morgan_radius, nBits=morgan_bits
            )
            ConvertToNumpyArray(fp, fp_arr[i])

    if n_failed:
        log.warning(
            "[%s] %d / %d SMILES failed parsing or descriptor computation",
            desc, n_failed, len(smiles_list),
        )

    frames: list[pd.DataFrame] = []
    if include_descriptors:
        frames.append(pd.DataFrame(desc_rows, columns=desc_keys).reset_index(drop=True))
    if include_morgan:
        frames.append(pd.DataFrame(
            fp_arr, columns=[f"fp_{j}" for j in range(morgan_bits)]
        ).reset_index(drop=True))
    df = pd.concat(frames, axis=1)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _save_pickle(df, cache_path)
        log.info(
            "[%s] cached features → %s (%.1f MB)",
            desc, cache_path.name, cache_path.stat().st_size / 1e6,
        )
    return df


def _features_cache_key(smiles_list: list[str], cfg: tuple) -> str:
    h = hashlib.sha256()
    h.update(repr(cfg).encode())
    h.update(b"\n".join(s.encode() for s in smiles_list))
    return h.hexdigest()[:16]


def _save_pickle(obj, path: Path) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def clean_inf(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan)


def drop_constant_cols(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    keep = X_train.columns[X_train.nunique(dropna=False) > 1].tolist()
    return X_train[keep], X_test[keep], keep


def split_by_target_type(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    target_type: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Slice (features, target) for one of {'tg', 'egc'}."""
    tr_mask = (train["target_type"] == target_type).values
    te_mask = (test["target_type"] == target_type).values
    X_tr = train_feat[tr_mask].reset_index(drop=True)
    y_tr = train.loc[tr_mask, "target"].reset_index(drop=True)
    X_te = test_feat[te_mask].reset_index(drop=True)
    ids_te = test.loc[te_mask, "id"].reset_index(drop=True)
    return X_tr, y_tr, X_te, ids_te


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------

def cv_oof(
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    model_factory: Callable[[], object],
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, list[float], float]:
    """K-fold OOF predictions. `model_factory()` returns a fresh estimator per fold."""
    oof = np.zeros(len(X), dtype=np.float64)
    fold_scores: list[float] = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_iter = tqdm(list(kf.split(X)), desc=f"{name} CV", unit="fold", leave=True)
    for fold, (tr_idx, va_idx) in enumerate(fold_iter):
        t0 = time.time()
        model = model_factory()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[va_idx])
        oof[va_idx] = pred
        r2 = float(r2_score(y.iloc[va_idx], pred))
        fold_scores.append(r2)
        dt = time.time() - t0
        n_iters = getattr(model, "n_iter_", None)
        log.info(
            "[%s] fold %d/%d  R^2=%.4f%s  (%.1fs)",
            name, fold + 1, n_folds, r2,
            f"  iters={n_iters}" if n_iters is not None else "",
            dt,
        )
        fold_iter.set_postfix(R2=f"{r2:.4f}")
    return oof, fold_scores, float(r2_score(y, oof))
