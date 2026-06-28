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


# ---------------------------------------------------------------------------
# Modular feature layers (Phase 1 cocktail)
# ---------------------------------------------------------------------------

def _cache_path(category: str, smiles_list: list[str], extra: tuple = ()) -> Path:
    h = hashlib.sha256()
    h.update(category.encode())
    h.update(repr(extra).encode())
    h.update(b"\n".join(s.encode() for s in smiles_list))
    return CACHE_DIR / f"{category}_{h.hexdigest()[:16]}.pkl.gz"


def _cached_compute(category: str, smiles_list: list[str], compute_fn, *,
                    extra: tuple = (), desc: str = "") -> pd.DataFrame:
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
    mols = []
    n_failed = 0
    for smi in tqdm(smiles_list, desc=f"{desc} parse", unit="mol", leave=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
        mols.append(mol)
    if n_failed:
        log.warning("[%s] %d / %d SMILES failed to parse", desc, n_failed, len(smiles_list))
    return mols


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
    from rdkit.Chem import rdFingerprintGenerator
    label = desc or f"morgan{radius}-{'cnt' if count else 'bit'}-{n_bits}"
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, label)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        dtype = np.int16 if count else np.int8
        arr = np.zeros((len(mols), n_bits), dtype=dtype)
        for i, mol in enumerate(tqdm(mols, desc=f"{label} compute", unit="mol", leave=False)):
            if mol is None:
                continue
            if count:
                arr[i] = gen.GetCountFingerprintAsNumPy(mol)
            else:
                arr[i] = gen.GetFingerprintAsNumPy(mol)
        prefix = f"m{radius}{'c' if count else 'b'}"
        return pd.DataFrame(arr, columns=[f"{prefix}_{j}" for j in range(n_bits)])
    return _cached_compute(
        f"morgan_r{radius}_{'cnt' if count else 'bit'}_{n_bits}_v1",
        smiles_list, _go, desc=label,
    )


def compute_maccs(smiles_list: list[str], *, desc: str = "maccs") -> pd.DataFrame:
    from rdkit.Chem import MACCSkeys
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, desc)
        n_bits = 167
        arr = np.zeros((len(mols), n_bits), dtype=np.int8)
        for i, mol in enumerate(tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False)):
            if mol is None:
                continue
            ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr[i])
        return pd.DataFrame(arr, columns=[f"maccs_{j}" for j in range(n_bits)])
    return _cached_compute("maccs_v1", smiles_list, _go, desc=desc)


def compute_avalon_fp(smiles_list: list[str], *, n_bits: int = 512,
                      desc: str = "avalon") -> pd.DataFrame:
    from rdkit.Avalon.pyAvalonTools import GetAvalonFP
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, desc)
        arr = np.zeros((len(mols), n_bits), dtype=np.int8)
        for i, mol in enumerate(tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False)):
            if mol is None:
                continue
            ConvertToNumpyArray(GetAvalonFP(mol, nBits=n_bits), arr[i])
        return pd.DataFrame(arr, columns=[f"avlon_{j}" for j in range(n_bits)])
    return _cached_compute(f"avalon_{n_bits}_v1", smiles_list, _go, desc=desc,
                            extra=(n_bits,))


def compute_atom_pair_fp(smiles_list: list[str], *, n_bits: int = 2048,
                          count: bool = True, desc: str | None = None) -> pd.DataFrame:
    from rdkit.Chem import rdFingerprintGenerator
    label = desc or f"atompair-{'cnt' if count else 'bit'}-{n_bits}"
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, label)
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
        dtype = np.int16 if count else np.int8
        arr = np.zeros((len(mols), n_bits), dtype=dtype)
        for i, mol in enumerate(tqdm(mols, desc=f"{label} compute", unit="mol", leave=False)):
            if mol is None:
                continue
            if count:
                arr[i] = gen.GetCountFingerprintAsNumPy(mol)
            else:
                arr[i] = gen.GetFingerprintAsNumPy(mol)
        prefix = f"ap{'c' if count else 'b'}"
        return pd.DataFrame(arr, columns=[f"{prefix}_{j}" for j in range(n_bits)])
    return _cached_compute(
        f"atom_pair_{'cnt' if count else 'bit'}_{n_bits}_v1",
        smiles_list, _go, desc=label,
    )


def compute_topological_torsion_fp(smiles_list: list[str], *, n_bits: int = 2048,
                                    count: bool = True, desc: str | None = None) -> pd.DataFrame:
    from rdkit.Chem import rdFingerprintGenerator
    label = desc or f"torsion-{'cnt' if count else 'bit'}-{n_bits}"
    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, label)
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
        dtype = np.int16 if count else np.int8
        arr = np.zeros((len(mols), n_bits), dtype=dtype)
        for i, mol in enumerate(tqdm(mols, desc=f"{label} compute", unit="mol", leave=False)):
            if mol is None:
                continue
            if count:
                arr[i] = gen.GetCountFingerprintAsNumPy(mol)
            else:
                arr[i] = gen.GetFingerprintAsNumPy(mol)
        prefix = f"tt{'c' if count else 'b'}"
        return pd.DataFrame(arr, columns=[f"{prefix}_{j}" for j in range(n_bits)])
    return _cached_compute(
        f"torsion_{'cnt' if count else 'bit'}_{n_bits}_v1",
        smiles_list, _go, desc=label,
    )


def compute_mordred_descriptors(smiles_list: list[str], *, ignore_3d: bool = True,
                                 nproc: int = 1, desc: str = "mordred") -> pd.DataFrame:
    """Mordred 2D descriptors (~1600 features). Empty DataFrame if mordred not installed."""
    try:
        from mordred import Calculator, descriptors  # type: ignore
    except ImportError:
        log.warning("[%s] mordred not installed → skipping (pip install mordred)", desc)
        return pd.DataFrame(index=range(len(smiles_list)))

    def _go() -> pd.DataFrame:
        mols = _smiles_to_mols(smiles_list, desc)
        # Mordred needs valid Mol objects; substitute methane for failed parses
        fallback = Chem.MolFromSmiles("C")
        safe_mols = [m if m is not None else fallback for m in mols]
        log.info("[%s] running Mordred (this takes ~15-25 min on 6k mols)...", desc)
        calc = Calculator(descriptors, ignore_3D=ignore_3d)
        df = calc.pandas(safe_mols, quiet=True, nproc=nproc)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.add_prefix("mord_")
    return _cached_compute(
        "mordred_2d_v1" if ignore_3d else "mordred_3d_v1",
        smiles_list, _go, desc=desc, extra=(ignore_3d,),
    )


# ---------------------------------------------------------------------------
# Column hygiene
# ---------------------------------------------------------------------------

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-alphanumeric chars in column names (LightGBM-safe)."""
    import re
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in df.columns]
    # de-dupe collisions
    if df.columns.duplicated().any():
        new_cols = []
        seen: dict[str, int] = {}
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}__{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# Stratified CV + target transforms
# ---------------------------------------------------------------------------

def stratified_quantile_split(
    y: np.ndarray, *, n_folds: int = 5, n_bins: int = 10, seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """KFold splits over quantile bins of a continuous target."""
    from sklearn.model_selection import StratifiedKFold
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
    """Return (y_transformed, inverse_fn) for target preprocessing."""
    y = np.asarray(y, dtype=np.float64)
    if kind == "log1p":
        return np.log1p(y), np.expm1
    if kind == "identity":
        return y, (lambda x: x)
    raise ValueError(f"unknown target transform: {kind!r}")


# ---------------------------------------------------------------------------
# Polymer-aware SMILES transformations
# ---------------------------------------------------------------------------

def cap_polymer_smiles_list(smiles_list: list[str], *, cap_atomic_num: int = 6) -> list[str]:
    """Replace each polymer wildcard atom (`*`) with a real atom (default carbon).

    This converts a polymer-style SMILES (`*CCCC*`) into a capped, fully-defined
    molecule (`CCCCCC`). Many RDKit descriptors handle wildcards as a degenerate
    dummy atom; replacing with real atoms gives the model meaningful descriptor
    signal in those slots. Falls back to the original SMILES on parse failure.
    """
    capped: list[str] = []
    n_changed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            capped.append(smi)
            continue
        rw = Chem.RWMol(mol)
        touched = False
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(cap_atomic_num)
                touched = True
        if not touched:
            capped.append(smi)
            continue
        try:
            m = rw.GetMol()
            Chem.SanitizeMol(m)
            capped.append(Chem.MolToSmiles(m))
            n_changed += 1
        except Exception:
            capped.append(smi)
    log.info("cap_polymer_smiles_list: %d / %d SMILES had wildcards replaced",
             n_changed, len(smiles_list))
    return capped


# ---------------------------------------------------------------------------
# MPS / CUDA / CPU device selection (for PyTorch experiments)
# ---------------------------------------------------------------------------

def torch_device():
    """Return the best available torch device: MPS > CUDA > CPU."""
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
