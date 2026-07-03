"""Single-file reproduction of the 0.911 LB submission.

Runs the full pipeline that produced the 0.911 public-LB score on the ANRF
AISEHack 2.0 Polymer Property Prediction competition:

    Phase 1 — GBM cocktail (LightGBM + CatBoost + HGB) over a wide feature mix
              (RDKit 2D descriptors + 6 fingerprint families), per target, mean
              blended. Trained under 5-fold stratified-quantile CV.

    Phase 2 — Chemprop D-MPNN multitask (Tg + Egc jointly), 3-seed bag per
              fold. On this machine the training was stopped after 2 of 5
              folds because MPS slowed dramatically past ~3h; the resulting
              checkpoint.npz is committed and this script reuses it.

    Blend  — Per-target non-negative least squares over
              [Phase-1-blend, Chemprop-bag]. Weights fit on the subset of
              training rows where Chemprop OOF is populated (i.e. the rows in
              the 2 completed folds), then applied to all test rows.

FAST REPRODUCTION PATH (recommended)
------------------------------------
If the committed artifacts under `results/exp_gbm_cocktail/` and
`results/exp_chemprop_multitask/checkpoint.npz` exist (they do on this repo),
this script skips Phase 1 training and Chemprop training entirely and just
re-runs the blend. Runtime: seconds. Output byte-identical to the LB
submission.

FULL FROM-SCRATCH PATH
----------------------
Delete `results/exp_gbm_cocktail/` and `results/exp_chemprop_multitask/` and
re-run this script. It will train Phase 1 (~90 min once features are cached)
and Chemprop (long — expect the full 5-fold × 3-seed schedule to take many
hours on Apple Silicon). Result is stochastic across runs but should land in
0.905-0.913 LB range.

OUTPUT
------
    results/reproduce_0911/submission.csv    ← upload this to Kaggle
    results/reproduce_0911/oof_final.csv     ← per-row diagnostics
    results/reproduce_0911/summary.json      ← weights, R^2, config

Environment
-----------
    conda create -n poly python=3.11 -y && conda activate poly
    pip install rdkit lightgbm catboost scikit-learn pandas scipy tqdm \
                chemprop lightning torch
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pickle
import re
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.DataStructs import ConvertToNumpyArray
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"
CACHE_DIR = REPO / ".cache" / "features"

PHASE1_DIR = RESULTS_DIR / "exp_gbm_cocktail"
CHEMPROP_DIR = RESULTS_DIR / "exp_chemprop_multitask"
OUT_DIR = RESULTS_DIR / "reproduce_0911"

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

SEED = 42
N_FOLDS = 5
N_QUANTILE_BINS = 10
TARGETS = ["tg", "egc"]
TARGET_TRANSFORMS = {"tg": "identity", "egc": "log1p"}

LGB_PARAMS = dict(
    n_estimators=4000, learning_rate=0.03, num_leaves=63, min_child_samples=10,
    feature_fraction=0.5, bagging_fraction=0.85, bagging_freq=5, reg_lambda=1.0,
    objective="regression", metric="rmse", verbosity=-1, random_state=SEED, n_jobs=-1,
)
CAT_PARAMS = dict(
    iterations=4000, depth=8, learning_rate=0.03, l2_leaf_reg=3.0,
    grow_policy="SymmetricTree", random_seed=SEED, verbose=False,
    allow_writing_files=False,
)
HGB_PARAMS = dict(
    max_iter=1000, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20,
    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, random_state=SEED,
)

# Chemprop model / training
MP_HIDDEN, MP_DEPTH, FFN_HIDDEN, FFN_DEPTH, DROPOUT = 300, 4, 300, 2, 0.05
MAX_EPOCHS, BATCH_SIZE, PATIENCE, LR = 50, 64, 10, 1e-3
BAG_SEEDS = [42, 1337, 7]

log = logging.getLogger("polymer")


# ---------------------------------------------------------------------------
# logging + IO helpers
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _save_pickle(obj, path: Path) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _cache_path(category: str, smiles_list: list[str], extra: tuple = ()) -> Path:
    h = hashlib.sha256()
    h.update(category.encode())
    h.update(repr(extra).encode())
    h.update(b"\n".join(s.encode() for s in smiles_list))
    return CACHE_DIR / f"{category}_{h.hexdigest()[:16]}.pkl.gz"


def _cached_compute(category: str, smiles_list: list[str], compute_fn,
                    *, extra: tuple = (), desc: str = "") -> pd.DataFrame:
    path = _cache_path(category, smiles_list, extra)
    label = desc or category
    if path.exists():
        log.info("[%s] cache hit: %s", label, path.name)
        return _load_pickle(path)
    df = compute_fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _save_pickle(df, path)
    log.info("[%s] cached → %s (%.1f MB)", label, path.name, path.stat().st_size / 1e6)
    return df


def _smiles_to_mols(smiles_list: list[str], desc: str) -> list:
    mols, n_failed = [], 0
    for smi in tqdm(smiles_list, desc=f"{desc} parse", unit="mol", leave=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
        mols.append(mol)
    if n_failed:
        log.warning("[%s] %d / %d SMILES failed to parse", desc, n_failed, len(smiles_list))
    return mols


# ---------------------------------------------------------------------------
# Phase-1 featurizers (RDKit descriptors + 6 fingerprint families)
# ---------------------------------------------------------------------------

def compute_rdkit_descriptors(smiles_list: list[str], *, desc: str = "rdk-desc") -> pd.DataFrame:
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, desc)
        keys = list(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles("CCO")).keys())
        nan = {k: np.nan for k in keys}
        rows = []
        for mol in tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False):
            if mol is None:
                rows.append(dict(nan)); continue
            try:
                rows.append(Descriptors.CalcMolDescriptors(mol))
            except Exception:
                rows.append(dict(nan))
        return pd.DataFrame(rows, columns=keys).add_prefix("rdk_")
    return _cached_compute("rdk_desc_v1", smiles_list, _go, desc=desc)


def compute_morgan_fp(smiles_list: list[str], *, radius: int = 2, n_bits: int = 2048,
                      count: bool = False, desc: str | None = None) -> pd.DataFrame:
    label = desc or f"morgan{radius}-{'cnt' if count else 'bit'}-{n_bits}"
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, label)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        dtype = np.int16 if count else np.int8
        arr = np.zeros((len(mols), n_bits), dtype=dtype)
        for i, mol in enumerate(tqdm(mols, desc=f"{label} compute", unit="mol", leave=False)):
            if mol is None: continue
            arr[i] = (gen.GetCountFingerprintAsNumPy(mol) if count
                       else gen.GetFingerprintAsNumPy(mol))
        prefix = f"m{radius}{'c' if count else 'b'}"
        return pd.DataFrame(arr, columns=[f"{prefix}_{j}" for j in range(n_bits)])
    return _cached_compute(
        f"morgan_r{radius}_{'cnt' if count else 'bit'}_{n_bits}_v1", smiles_list, _go, desc=label,
    )


def compute_maccs(smiles_list: list[str], *, desc: str = "maccs") -> pd.DataFrame:
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, desc)
        arr = np.zeros((len(mols), 167), dtype=np.int8)
        for i, mol in enumerate(tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False)):
            if mol is None: continue
            ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr[i])
        return pd.DataFrame(arr, columns=[f"maccs_{j}" for j in range(167)])
    return _cached_compute("maccs_v1", smiles_list, _go, desc=desc)


def compute_avalon_fp(smiles_list: list[str], *, n_bits: int = 512,
                       desc: str = "avalon") -> pd.DataFrame:
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, desc)
        arr = np.zeros((len(mols), n_bits), dtype=np.int8)
        for i, mol in enumerate(tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False)):
            if mol is None: continue
            ConvertToNumpyArray(GetAvalonFP(mol, nBits=n_bits), arr[i])
        return pd.DataFrame(arr, columns=[f"avlon_{j}" for j in range(n_bits)])
    return _cached_compute(f"avalon_{n_bits}_v1", smiles_list, _go, desc=desc, extra=(n_bits,))


def compute_atom_pair_fp(smiles_list: list[str], *, n_bits: int = 2048,
                          count: bool = True, desc: str | None = None) -> pd.DataFrame:
    label = desc or f"atompair-{'cnt' if count else 'bit'}-{n_bits}"
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, label)
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
        dtype = np.int16 if count else np.int8
        arr = np.zeros((len(mols), n_bits), dtype=dtype)
        for i, mol in enumerate(tqdm(mols, desc=f"{label} compute", unit="mol", leave=False)):
            if mol is None: continue
            arr[i] = (gen.GetCountFingerprintAsNumPy(mol) if count
                       else gen.GetFingerprintAsNumPy(mol))
        prefix = f"ap{'c' if count else 'b'}"
        return pd.DataFrame(arr, columns=[f"{prefix}_{j}" for j in range(n_bits)])
    return _cached_compute(
        f"atom_pair_{'cnt' if count else 'bit'}_{n_bits}_v1", smiles_list, _go, desc=label,
    )


def compute_topological_torsion_fp(smiles_list: list[str], *, n_bits: int = 2048,
                                     count: bool = True, desc: str | None = None) -> pd.DataFrame:
    label = desc or f"torsion-{'cnt' if count else 'bit'}-{n_bits}"
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, label)
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
        dtype = np.int16 if count else np.int8
        arr = np.zeros((len(mols), n_bits), dtype=dtype)
        for i, mol in enumerate(tqdm(mols, desc=f"{label} compute", unit="mol", leave=False)):
            if mol is None: continue
            arr[i] = (gen.GetCountFingerprintAsNumPy(mol) if count
                       else gen.GetFingerprintAsNumPy(mol))
        prefix = f"tt{'c' if count else 'b'}"
        return pd.DataFrame(arr, columns=[f"{prefix}_{j}" for j in range(n_bits)])
    return _cached_compute(
        f"torsion_{'cnt' if count else 'bit'}_{n_bits}_v1", smiles_list, _go, desc=label,
    )


def compute_mordred_descriptors(smiles_list: list[str], *, desc: str = "mordred") -> pd.DataFrame:
    """Mordred 2D descriptors. Returns empty DF if mordred is not installed
    (the 0.911 submission was NOT dependent on mordred — it was included in the
    Phase 1 feature mix if available, but leaving it out costs at most ~0.001)."""
    try:
        from mordred import Calculator, descriptors  # type: ignore
    except ImportError:
        log.info("[%s] mordred not installed → skipping (feature mix will still work)", desc)
        return pd.DataFrame(index=range(len(smiles_list)))

    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, desc)
        fallback = Chem.MolFromSmiles("C")
        safe_mols = [m if m is not None else fallback for m in mols]
        log.info("[%s] running Mordred (this takes ~15-25 min on 6k mols)...", desc)
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(safe_mols, quiet=True, nproc=1)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.add_prefix("mord_")
    return _cached_compute("mordred_2d_v1", smiles_list, _go, desc=desc, extra=(True,))


# ---------------------------------------------------------------------------
# preprocessing helpers
# ---------------------------------------------------------------------------

def clean_inf(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan)


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in df.columns]
    if df.columns.duplicated().any():
        new_cols, seen = [], {}
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}__{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
    return df


def drop_constant_cols(X_train: pd.DataFrame, X_test: pd.DataFrame):
    keep = X_train.columns[X_train.nunique(dropna=False) > 1].tolist()
    return X_train[keep], X_test[keep], keep


def split_by_target_type(train, test, X_train, X_test, target_type: str):
    tr_mask = (train["target_type"] == target_type).values
    te_mask = (test["target_type"] == target_type).values
    return (
        X_train[tr_mask].reset_index(drop=True),
        train.loc[tr_mask, "target"].reset_index(drop=True),
        X_test[te_mask].reset_index(drop=True),
        test.loc[te_mask, "id"].reset_index(drop=True),
    )


def stratified_quantile_split(y, *, n_folds: int = 5, n_bins: int = 10, seed: int = 42):
    y = np.asarray(y)
    n_bins_eff = min(n_bins, max(2, len(np.unique(y))))
    try:
        bins = pd.qcut(y, q=n_bins_eff, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(y, bins=n_bins_eff, labels=False, include_lowest=True)
    bins = np.asarray(bins)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros((len(y), 1)), bins))


def transform_target(y, kind: str) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    y = np.asarray(y, dtype=np.float64)
    if kind == "log1p":
        return np.log1p(y), np.expm1
    if kind == "identity":
        return y, (lambda x: x)
    raise ValueError(f"unknown target transform: {kind!r}")


# ---------------------------------------------------------------------------
# Phase 1 — GBM cocktail
# ---------------------------------------------------------------------------

def _build_features(smiles: list[str], *, split_label: str) -> pd.DataFrame:
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
        compute_mordred_descriptors(smiles, desc=f"{split_label}/mordred"),
    ]
    parts = [p.reset_index(drop=True) for p in parts if p.shape[1] > 0]
    X = pd.concat(parts, axis=1)
    X = sanitize_columns(clean_inf(X))
    log.info("[%s] combined feature matrix: %s", split_label, X.shape)
    return X


def _train_lgb(X_tr, y_tr, X_va, y_va):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                     lgb.log_evaluation(period=0)])
    return m, int(m.best_iteration_ or LGB_PARAMS["n_estimators"])


def _train_cat(X_tr, y_tr, X_va, y_va):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**CAT_PARAMS)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=200, verbose=False)
    return m, int(m.get_best_iteration() or CAT_PARAMS["iterations"])


def _train_hgb(X_tr, y_tr, X_va=None, y_va=None):
    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    m.fit(X_tr, y_tr)
    return m, int(m.n_iter_)


PHASE1_MODELS = {"lgb": _train_lgb, "cat": _train_cat, "hgb": _train_hgb}


def _fit_full(model_name: str, X, y, n_iters: int):
    import lightgbm as lgb
    if model_name == "lgb":
        p = dict(LGB_PARAMS); p["n_estimators"] = max(int(n_iters * 1.10), 200)
        m = lgb.LGBMRegressor(**p); m.fit(X, y); return m
    if model_name == "cat":
        from catboost import CatBoostRegressor
        p = dict(CAT_PARAMS); p["iterations"] = max(int(n_iters * 1.10), 200)
        m = CatBoostRegressor(**p); m.fit(X, y, verbose=False); return m
    if model_name == "hgb":
        m = HistGradientBoostingRegressor(**HGB_PARAMS); m.fit(X, y); return m
    raise ValueError(model_name)


def _cv_per_target(X, y_orig, target_name, transform_kind):
    y_t, inv = transform_target(y_orig, transform_kind)
    log.info("[%s] transform=%s | y_t range [%.3f, %.3f]",
             target_name, transform_kind, float(y_t.min()), float(y_t.max()))
    oof = {k: np.zeros(len(X), dtype=np.float64) for k in PHASE1_MODELS}
    best_iters = {k: [] for k in PHASE1_MODELS}
    folds = stratified_quantile_split(y_t, n_folds=N_FOLDS, n_bins=N_QUANTILE_BINS, seed=SEED)
    fold_bar = tqdm(folds, desc=f"{target_name} CV", unit="fold")
    for fold, (tr_idx, va_idx) in enumerate(fold_bar):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_t[tr_idx], y_t[va_idx]
        for k, fn in PHASE1_MODELS.items():
            t0 = time.time()
            model, n_it = fn(X_tr, y_tr, X_va, y_va)
            pred = inv(model.predict(X_va))
            oof[k][va_idx] = pred
            r2 = r2_score(y_orig[va_idx], pred)
            best_iters[k].append(n_it)
            log.info("[%s/%s] fold %d/%d  R^2=%.4f  iters=%d  (%.1fs)",
                     target_name, k, fold + 1, N_FOLDS, r2, n_it, time.time() - t0)
    per_model_r2 = {k: float(r2_score(y_orig, oof[k])) for k in PHASE1_MODELS}
    blend_oof = np.mean(np.stack([oof[k] for k in PHASE1_MODELS], axis=0), axis=0)
    blend_r2 = float(r2_score(y_orig, blend_oof))
    log.info("[%s] per-model OOF R^2: %s", target_name,
             {k: f"{v:.4f}" for k, v in per_model_r2.items()})
    log.info("[%s] blend OOF R^2 = %.4f", target_name, blend_r2)
    return oof, blend_oof, best_iters


def run_phase1(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (phase1_oof_df long-format, phase1_submission_df)."""
    oof_path = PHASE1_DIR / "oof.csv"
    sub_path = PHASE1_DIR / "submission.csv"
    if oof_path.exists() and sub_path.exists():
        log.info("Phase 1 artifacts already present → skipping training")
        return pd.read_csv(oof_path), pd.read_csv(sub_path)

    log.info("═" * 64)
    log.info("PHASE 1 — GBM cocktail training from scratch")
    log.info("═" * 64)
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)

    X_train = _build_features(train["smiles"].tolist(), split_label="train")
    X_test = _build_features(test["smiles"].tolist(), split_label="test")

    common = sorted(set(X_train.columns) & set(X_test.columns))
    X_train, X_test = X_train[common], X_test[common]
    X_train, X_test, kept = drop_constant_cols(X_train, X_test)
    log.info("features after intersect + drop-constant: %d", len(kept))

    oof_frames, test_frames = [], []
    for t in TARGETS:
        log.info("─" * 64)
        log.info("Training %s", t.upper())
        X_tr, y_tr_s, X_te, ids_te = split_by_target_type(train, test, X_train, X_test, t)
        y_tr = y_tr_s.values
        log.info("[%s] train=%d test=%d y=[%.3f, %.3f]",
                 t, len(X_tr), len(X_te), float(y_tr.min()), float(y_tr.max()))

        oof, blend_oof, best_iters = _cv_per_target(X_tr, y_tr, t, TARGET_TRANSFORMS[t])

        oof_frames.append(pd.DataFrame({
            "smiles": train.loc[train["target_type"] == t, "smiles"].values,
            "target_type": t,
            "target": y_tr,
            **{f"oof_{k}": oof[k] for k in PHASE1_MODELS},
            "oof_blend": blend_oof,
        }))

        # Refit each model on full training data using median best iters
        log.info("[%s] refitting on full %d rows...", t, len(X_tr))
        y_t_full, inv = transform_target(y_tr, TARGET_TRANSFORMS[t])
        test_preds = {}
        for k in PHASE1_MODELS:
            n_it = int(np.median(best_iters[k]))
            t_fit = time.time()
            model = _fit_full(k, X_tr, y_t_full, n_it)
            test_preds[k] = inv(model.predict(X_te))
            log.info("[%s/%s] refit done (iters=%d, %.1fs)",
                     t, k, n_it, time.time() - t_fit)
        blend_test = np.mean(np.stack(list(test_preds.values()), axis=0), axis=0)
        test_frames.append(pd.DataFrame({"id": ids_te.values, "target": blend_test}))

    oof_all = pd.concat(oof_frames).reset_index(drop=True)
    sub_all = pd.concat(test_frames).sort_values("id").reset_index(drop=True)
    oof_all.to_csv(oof_path, index=False)
    sub_all.to_csv(sub_path, index=False)
    log.info("Wrote %s and %s", oof_path, sub_path)
    return oof_all, sub_all


# ---------------------------------------------------------------------------
# Phase 2 — Chemprop D-MPNN multitask
# ---------------------------------------------------------------------------

def long_to_wide(train: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for smi, t, y in zip(train["smiles"].values, train["target_type"].values, train["target"].values):
        if smi not in rows:
            rows[smi] = {"smiles": smi, "tg": np.nan, "egc": np.nan}
        rows[smi][t] = y
    return pd.DataFrame(list(rows.values()))[["smiles", "tg", "egc"]]


def _apply_target_transforms_wide(wide_orig: pd.DataFrame) -> pd.DataFrame:
    out = wide_orig.copy()
    for t in TARGETS:
        y_t, _ = transform_target(wide_orig[t].values, TARGET_TRANSFORMS[t])
        out[t] = y_t
    return out


def _inverse_targets_wide(transformed: np.ndarray) -> np.ndarray:
    out = np.empty_like(transformed)
    for ti, t in enumerate(TARGETS):
        _, inv = transform_target(np.array([0.0]), TARGET_TRANSFORMS[t])
        out[:, ti] = inv(transformed[:, ti])
    return out


def _train_chemprop_from_scratch(train: pd.DataFrame, test: pd.DataFrame,
                                  run_dir: Path) -> Path:
    """Train Chemprop end-to-end and write checkpoint.npz to run_dir. Slow."""
    import torch
    try:
        from lightning import pytorch as pl
    except ImportError:
        import pytorch_lightning as pl  # type: ignore
    from chemprop import data as cdata, featurizers as cfeat, models as cmodels, nn as cnn

    log.info("═" * 64)
    log.info("PHASE 2 — Chemprop training from scratch (long)")
    log.info("═" * 64)

    def _accel():
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps", 1
        if torch.cuda.is_available():
            return "gpu", 1
        return "cpu", 1

    def _build_mpnn(seed):
        pl.seed_everything(seed)
        mp = cnn.BondMessagePassing(d_h=MP_HIDDEN, depth=MP_DEPTH, dropout=DROPOUT)
        agg = cnn.MeanAggregation()
        pred = cnn.RegressionFFN(input_dim=MP_HIDDEN, hidden_dim=FFN_HIDDEN,
                                  n_layers=FFN_DEPTH, dropout=DROPOUT, n_tasks=len(TARGETS))
        return cmodels.MPNN(mp, agg, pred, batch_norm=True)

    def _loaders(tr_smis, tr_ys_z, va_smis, va_ys_z):
        feat = cfeat.SimpleMoleculeMolGraphFeaturizer()
        tr_dps = [cdata.MoleculeDatapoint.from_smi(s, y=y.astype(np.float32))
                  for s, y in zip(tr_smis, tr_ys_z)]
        va_dps = [cdata.MoleculeDatapoint.from_smi(s, y=y.astype(np.float32))
                  for s, y in zip(va_smis, va_ys_z)]
        return (
            cdata.build_dataloader(cdata.MoleculeDataset(tr_dps, feat), batch_size=BATCH_SIZE, num_workers=0),
            cdata.build_dataloader(cdata.MoleculeDataset(va_dps, feat), batch_size=BATCH_SIZE, num_workers=0, shuffle=False),
        )

    def _predict_loader(smis):
        feat = cfeat.SimpleMoleculeMolGraphFeaturizer()
        dps = [cdata.MoleculeDatapoint.from_smi(s) for s in smis]
        return cdata.build_dataloader(cdata.MoleculeDataset(dps, feat), batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    wide_orig = long_to_wide(train)
    wide_t = _apply_target_transforms_wide(wide_orig)
    test_smis = test["smiles"].drop_duplicates().tolist()
    train_smis = wide_orig["smiles"].values
    ys = wide_t[TARGETS].values.astype(np.float32)
    strat = np.nanmean(ys, axis=1)
    folds = stratified_quantile_split(strat, n_folds=N_FOLDS, n_bins=N_QUANTILE_BINS, seed=SEED)

    oof_t = np.full((len(wide_orig), len(TARGETS)), np.nan, dtype=np.float64)
    test_acc_t = np.zeros((len(test_smis), len(TARGETS)), dtype=np.float64)
    n_models = 0
    ckpt_path = run_dir / "checkpoint.npz"

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        log.info("─" * 64)
        log.info("Chemprop FOLD %d/%d", fold_idx + 1, N_FOLDS)
        tr_smis_f, va_smis_f = train_smis[tr_idx], train_smis[va_idx]
        tr_ys_f, va_ys_f = ys[tr_idx], ys[va_idx]
        mu = np.nanmean(tr_ys_f, axis=0).astype(np.float32)
        sd = np.nanstd(tr_ys_f, axis=0).astype(np.float32)
        sd = np.where(sd < 1e-6, 1.0, sd)
        tr_z, va_z = (tr_ys_f - mu) / sd, (va_ys_f - mu) / sd

        seed_val, seed_test = [], []
        for seed in BAG_SEEDS:
            try:
                tr_loader, va_loader = _loaders(tr_smis_f, tr_z, va_smis_f, va_z)
                model = _build_mpnn(seed)
                acc, dev = _accel()
                trainer = pl.Trainer(
                    accelerator=acc, devices=dev, max_epochs=MAX_EPOCHS,
                    enable_progress_bar=False, enable_checkpointing=False, logger=False,
                    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                            mode="min", check_finite=False)],
                    gradient_clip_val=1.0,
                )
                t0 = time.time()
                trainer.fit(model, tr_loader, va_loader)
                val_preds_z = torch.cat(trainer.predict(model, _predict_loader(va_smis_f.tolist())), dim=0).cpu().numpy()
                test_preds_z = torch.cat(trainer.predict(model, _predict_loader(test_smis)), dim=0).cpu().numpy()
                seed_val.append(val_preds_z * sd + mu)
                seed_test.append(test_preds_z * sd + mu)
                log.info("[fold%d seed=%d] done in %.1fs", fold_idx + 1, seed, time.time() - t0)
            except Exception as e:
                log.exception("[fold%d seed=%d] failed: %s", fold_idx + 1, seed, e)
        if not seed_val:
            continue
        oof_t[va_idx] = np.mean(np.stack(seed_val), axis=0)
        test_acc_t += np.mean(np.stack(seed_test), axis=0) * len(seed_val)
        n_models += len(seed_val)
        np.savez(ckpt_path, oof_t=oof_t, test_acc_t=test_acc_t,
                  n_models_added_to_test=n_models, completed_folds=fold_idx + 1)
        log.info("[fold %d] checkpoint saved", fold_idx + 1)
    return ckpt_path


def get_chemprop_artifacts(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (chemprop_oof_df long, chemprop_submission_df, meta).

    Uses committed checkpoint if present, otherwise trains from scratch.
    """
    ckpt_path = CHEMPROP_DIR / "checkpoint.npz"
    if not ckpt_path.exists():
        CHEMPROP_DIR.mkdir(parents=True, exist_ok=True)
        _train_chemprop_from_scratch(train, test, CHEMPROP_DIR)

    ck = np.load(ckpt_path, allow_pickle=False)
    oof_t = ck["oof_t"]
    test_acc_t = ck["test_acc_t"]
    n_models = int(ck["n_models_added_to_test"])
    completed_folds = int(ck["completed_folds"])
    if n_models == 0:
        raise RuntimeError(f"Empty checkpoint at {ckpt_path}")
    log.info("Chemprop checkpoint: %d folds, %d models", completed_folds, n_models)

    wide_orig = long_to_wide(train)
    test_unique = test["smiles"].drop_duplicates().tolist()
    test_t = test_acc_t / n_models
    test_orig = _inverse_targets_wide(test_t)
    oof_orig = _inverse_targets_wide(oof_t)
    has_oof = ~np.isnan(oof_t).any(axis=1)
    log.info("Chemprop OOF coverage: %d / %d train rows (%.1f%%)",
             int(has_oof.sum()), len(has_oof), 100 * has_oof.mean())

    test_smi_to_idx = {s: i for i, s in enumerate(test_unique)}
    sub_rows = [{"id": int(r["id"]),
                  "target": float(test_orig[test_smi_to_idx[r["smiles"]], TARGETS.index(r["target_type"])])}
                 for _, r in test.iterrows()]
    sub_df = pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True)

    smi_to_oof_idx = {s: i for i, s in enumerate(wide_orig["smiles"].values)}
    oof_rows = []
    for _, r in train.iterrows():
        idx = smi_to_oof_idx[r["smiles"]]
        ti = TARGETS.index(r["target_type"])
        oof_rows.append({
            "smiles": r["smiles"], "target_type": r["target_type"], "target": float(r["target"]),
            "oof_chemprop": float(oof_orig[idx, ti]) if has_oof[idx] else np.nan,
            "has_oof": bool(has_oof[idx]),
        })
    oof_df = pd.DataFrame(oof_rows)

    return oof_df, sub_df, {"completed_folds": completed_folds, "n_models": n_models}


# ---------------------------------------------------------------------------
# NNLS blend
# ---------------------------------------------------------------------------

def blend_and_submit(train: pd.DataFrame, test: pd.DataFrame,
                      phase1_oof: pd.DataFrame, phase1_sub: pd.DataFrame,
                      chemprop_oof: pd.DataFrame, chemprop_sub: pd.DataFrame,
                      chemprop_meta: dict, out_dir: Path) -> dict:
    log.info("═" * 64)
    log.info("BLEND — per-target NNLS over [Phase 1, Chemprop]")
    log.info("═" * 64)

    summary: dict = {"chemprop_meta": chemprop_meta, "per_target": {}}
    final_test_frames, final_oof_frames = [], []

    for t in TARGETS:
        log.info("─" * 32)
        log.info("Stacking %s", t.upper())
        tr_mask = (train["target_type"] == t).values
        te_mask = (test["target_type"] == t).values
        smis_t = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        y_tr = train.loc[tr_mask, "target"].reset_index(drop=True).values
        ids_te = test.loc[te_mask, "id"].reset_index(drop=True).values

        p1_t = phase1_oof[phase1_oof["target_type"] == t].reset_index(drop=True)
        cp_t = chemprop_oof[chemprop_oof["target_type"] == t].reset_index(drop=True)
        assert (p1_t["smiles"].values == smis_t).all(), f"[{t}] phase1 OOF misaligned"
        assert (cp_t["smiles"].values == smis_t).all(), f"[{t}] chemprop OOF misaligned"

        p1_blend = p1_t["oof_blend"].values
        cp_oof = cp_t["oof_chemprop"].values
        mask = cp_t["has_oof"].values

        p1_test = phase1_sub.set_index("id").loc[ids_te, "target"].values
        cp_test = chemprop_sub.set_index("id").loc[ids_te, "target"].values

        if mask.sum() < 50:
            log.warning("[%s] Chemprop OOF too sparse — falling back to Phase 1 only", t)
            blend_test = p1_test
            weights_norm = np.array([1.0, 0.0])
        else:
            A_sub = np.column_stack([p1_blend[mask], cp_oof[mask]]).astype(np.float64)
            y_sub = y_tr[mask].astype(np.float64)
            w_raw, _ = nnls(A_sub, y_sub)
            w_sum = w_raw.sum()
            weights_norm = (w_raw / w_sum) if w_sum > 1e-9 else np.array([1.0, 0.0])
            log.info("[%s] NNLS weights (raw → norm, sum=%.4f):", t, w_sum)
            for name, wr, wn in zip(["phase1", "chemprop"], w_raw, weights_norm):
                log.info("    %-10s raw=%.4f  norm=%.4f", name, wr, wn)

            stack_oof_sub = A_sub @ weights_norm
            stack_r2_sub = float(r2_score(y_sub, stack_oof_sub))
            log.info("[%s] per-base subset R^2: phase1=%.4f  chemprop=%.4f",
                     t, r2_score(y_sub, p1_blend[mask]), r2_score(y_sub, cp_oof[mask]))
            log.info("[%s] STACK OOF R^2 (subset, %d rows) = %.4f",
                     t, int(mask.sum()), stack_r2_sub)
            blend_test = weights_norm[0] * p1_test + weights_norm[1] * cp_test

        final_test_frames.append(pd.DataFrame({"id": ids_te, "target": blend_test}))
        final_oof_frames.append(pd.DataFrame({
            "smiles": smis_t, "target_type": t, "target": y_tr,
            "oof_phase1": p1_blend, "oof_chemprop": cp_oof,
            "has_chemprop_oof": mask,
        }))
        summary["per_target"][t] = {
            "weights_normalized": {b: float(w) for b, w in zip(["phase1", "chemprop"], weights_norm)},
            "n_rows_used_for_nnls": int(mask.sum()),
        }

    final_sub = pd.concat(final_test_frames).sort_values("id").reset_index(drop=True)
    final_oof = pd.concat(final_oof_frames).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_sub.to_csv(out_dir / "submission.csv", index=False)
    final_oof.to_csv(out_dir / "oof_final.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s (%d rows)", out_dir / "submission.csv", len(final_sub))
    log.info("Wrote %s and %s", out_dir / "oof_final.csv", out_dir / "summary.json")
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    t0 = time.time()
    log.info("=" * 64)
    log.info("reproduce_0911.py — end-to-end reproduction")
    log.info("=" * 64)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)

    phase1_oof, phase1_sub = run_phase1(train, test)
    chemprop_oof, chemprop_sub, chemprop_meta = get_chemprop_artifacts(train, test)
    blend_and_submit(train, test, phase1_oof, phase1_sub,
                      chemprop_oof, chemprop_sub, chemprop_meta, OUT_DIR)

    log.info("═" * 64)
    log.info("DONE — submit %s", OUT_DIR / "submission.csv")
    log.info("Total runtime: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
