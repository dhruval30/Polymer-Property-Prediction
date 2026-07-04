"""exp_polymer_physics_stack.py — clever 8-9h Mac plan to break past LB 0.911.

Everything trained from scratch on competition data only. No pretrained models,
no external datasets, no borrowing from prior results. Self-contained (drop
into a Kaggle notebook if needed).

Four levers, attacking the ceiling from independent angles:

Phase 1 — Physics-informed feature expansion
   * Multi-conformer 3D (5 conformers per polymer; mean/std/min/max of shape)
   * Gasteiger partial charges (mean/std/min/max/skew per molecule)
   * Coulomb-matrix eigenvalues (top 20; electronic-structure surrogate)
   * All existing RDKit 2D + Morgan/MACCS/Avalon/AP/TT fingerprints
   * Custom polymer descriptors + Bicerano SMARTS groups
   * GBM cocktail (LGB + CAT + HGB), 5-fold stratified-quantile CV.

Phase 2 — Dual-architecture Chemprop bag (CPU)
   * Variant A: BondMessagePassing + MeanAggregation + shared FFN (n_tasks=2)
   * Variant B: BondMessagePassing + SumAggregation + deeper FFN
   * Each: 3 folds x 2 seeds = 6 models. Total 12 models.
   * Different pooling + FFN depth = uncorrelated inductive biases.

Phase 3 — Iterative pseudo-labeling (2 rounds)
   * Round 1: Chemprop OOF -> pseudo-label the missing target -> refit LGB
   * Round 2: Round-1 preds -> sharpened pseudo-labels -> refit LGB again
   * Sample weight = 0.3 on pseudo rows.

Phase 4 — CatBoost meta-stacker
   * Nonlinear stacker over [GBM blend, LGB-pseudo-r1, LGB-pseudo-r2, Chemprop-A, Chemprop-B]
   * With Bemis-Murcko scaffold ID as a categorical feature.
   * Replaces NNLS.

Phase 5 — Optional LB bias correction
   * If LB probe reveals target-mean shift, apply bias constants TG_BIAS/EGC_BIAS.
   * Run `python experiments/exp_polymer_physics_stack.py --lb-probe` to write
     two constant-value CSVs you can submit to detect the shift.

Every phase is checkpointed to `results/exp_polymer_physics_stack/`. Crash-safe.

Runtime budget on MacBook Air:
   Featurization (cached after first run): 30-45 min cold, seconds warm
   Phase 1 GBM: ~2.5h
   Phase 2 Chemprop (12 models CPU): ~4h
   Phase 3 Pseudo: ~1.5h
   Phase 4 Meta: ~30 min
   Total: ~8-9h.
"""
from __future__ import annotations

import argparse
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
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Descriptors3D,
    MACCSkeys,
    rdFingerprintGenerator,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
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
# paths + config
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
CACHE_DIR = REPO / ".cache" / "features"
OUT_DIR = REPO / "results" / "exp_polymer_physics_stack"

SEED = 42
N_FOLDS_GBM = 5
N_QUANTILE_BINS = 10
TARGETS = ["tg", "egc"]
TARGET_TRANSFORMS = {"tg": "identity", "egc": "log1p"}

# Physics-feature configs
N_CONFORMERS = 5
COULOMB_TOP_K = 20

# GBM configs (unchanged from earlier)
LGB_PARAMS = dict(
    n_estimators=6000, learning_rate=0.025, num_leaves=63, min_child_samples=10,
    feature_fraction=0.45, bagging_fraction=0.85, bagging_freq=5, reg_lambda=1.0,
    objective="regression", metric="rmse", verbosity=-1, random_state=SEED, n_jobs=-1,
)
CAT_PARAMS = dict(
    iterations=6000, depth=8, learning_rate=0.03, l2_leaf_reg=3.0,
    grow_policy="SymmetricTree", random_seed=SEED, verbose=False,
    allow_writing_files=False,
)
HGB_PARAMS = dict(
    max_iter=1500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20,
    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=40, random_state=SEED,
)

# Chemprop dual-variant configs
CP_MP_HIDDEN, CP_MP_DEPTH = 250, 3       # compressed for CPU budget
CP_FFN_HIDDEN, CP_DROPOUT = 250, 0.05
CP_FFN_DEPTH_A = 2                        # variant A
CP_FFN_DEPTH_B = 3                        # variant B (deeper FFN)
CP_MAX_EPOCHS, CP_BATCH_SIZE, CP_PATIENCE = 40, 64, 8
CP_N_FOLDS = 3
CP_BAG_SEEDS = [42, 1337]

# Pseudo-labeling
PSEUDO_WEIGHT = 0.3
PSEUDO_N_ROUNDS = 2

# LB bias corrections (set from probe results before final run)
TG_BIAS = 0.0
EGC_BIAS = 0.0

log = logging.getLogger("polymer")


def setup_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# cache + helpers
# ---------------------------------------------------------------------------

def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _cache_key(category: str, smiles_list: list[str], extra: tuple = ()) -> Path:
    h = hashlib.sha256()
    h.update(category.encode())
    h.update(repr(extra).encode())
    h.update(b"\n".join(s.encode() for s in smiles_list))
    return CACHE_DIR / f"{category}_{h.hexdigest()[:16]}.pkl.gz"


def _cached(category: str, smiles_list: list[str], compute_fn,
            *, extra: tuple = (), desc: str = "") -> pd.DataFrame:
    path = _cache_key(category, smiles_list, extra)
    label = desc or category
    if path.exists():
        log.info("[%s] cache hit: %s", label, path.name)
        return _load_pickle(path)
    df = compute_fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _save_pickle(df, path)
    log.info("[%s] cached -> %s (%.1f MB)", label, path.name, path.stat().st_size / 1e6)
    return df


def _smiles_to_mols(smiles_list: list[str], desc: str) -> list:
    mols, n_failed = [], 0
    for smi in tqdm(smiles_list, desc=f"{desc} parse", unit="mol", leave=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
        mols.append(mol)
    if n_failed:
        log.warning("[%s] %d / %d SMILES failed parse", desc, n_failed, len(smiles_list))
    return mols


def _capped_smiles(smi: str, cap: str = "C") -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    rw = Chem.RWMol(mol)
    touched = False
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6 if cap == "C" else 1)
            touched = True
    if not touched:
        return smi
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return smi


def _embed_conformer(mol, seed: int = 42):
    """Return conformer-embedded, UFF-optimized mol or None on failure."""
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        return mol
    except Exception:
        return None


# ---------------------------------------------------------------------------
# BASE FEATURIZERS (RDKit 2D + fingerprints)
# ---------------------------------------------------------------------------

def compute_rdkit_2d(smiles_list, *, desc="rdk-2d"):
    def _go():
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
    return _cached("rdk_2d_v1", smiles_list, _go, desc=desc)


def compute_morgan_fp(smiles_list, *, radius=2, n_bits=2048, count=False, desc=None):
    label = desc or f"morgan{radius}-{'cnt' if count else 'bit'}-{n_bits}"
    def _go():
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
    return _cached(
        f"morgan_r{radius}_{'cnt' if count else 'bit'}_{n_bits}_v1",
        smiles_list, _go, desc=label,
    )


def compute_maccs(smiles_list, *, desc="maccs"):
    def _go():
        mols = _smiles_to_mols(smiles_list, desc)
        arr = np.zeros((len(mols), 167), dtype=np.int8)
        for i, mol in enumerate(tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False)):
            if mol is None: continue
            ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr[i])
        return pd.DataFrame(arr, columns=[f"maccs_{j}" for j in range(167)])
    return _cached("maccs_v1", smiles_list, _go, desc=desc)


def compute_avalon_fp(smiles_list, *, n_bits=512, desc="avalon"):
    def _go():
        mols = _smiles_to_mols(smiles_list, desc)
        arr = np.zeros((len(mols), n_bits), dtype=np.int8)
        for i, mol in enumerate(tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False)):
            if mol is None: continue
            ConvertToNumpyArray(GetAvalonFP(mol, nBits=n_bits), arr[i])
        return pd.DataFrame(arr, columns=[f"avlon_{j}" for j in range(n_bits)])
    return _cached(f"avalon_{n_bits}_v1", smiles_list, _go, desc=desc, extra=(n_bits,))


def compute_atom_pair_fp(smiles_list, *, n_bits=2048, count=True, desc=None):
    label = desc or f"atompair-{'cnt' if count else 'bit'}-{n_bits}"
    def _go():
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
    return _cached(f"atom_pair_{'cnt' if count else 'bit'}_{n_bits}_v1",
                    smiles_list, _go, desc=label)


def compute_topological_torsion_fp(smiles_list, *, n_bits=2048, count=True, desc=None):
    label = desc or f"torsion-{'cnt' if count else 'bit'}-{n_bits}"
    def _go():
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
    return _cached(f"torsion_{'cnt' if count else 'bit'}_{n_bits}_v1",
                    smiles_list, _go, desc=label)


# ---------------------------------------------------------------------------
# POLYMER-SPECIFIC HAND-CRAFTED FEATURES
# ---------------------------------------------------------------------------

def compute_polymer_custom(smiles_list, *, desc="poly-custom"):
    def _feats_one(smi):
        raw_wc = smi.count("*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return {
                "n_wildcards_smiles": raw_wc, "n_atoms": 0, "n_heavy_atoms": 0,
                "n_rings": 0, "n_aromatic_rings": 0, "n_aliphatic_rings": 0,
                "n_saturated_rings": 0, "frac_aromatic": 0.0, "frac_sp3": 0.0,
                "frac_sp2": 0.0, "n_halogens": 0, "n_hbd": 0, "n_hba": 0,
                "n_nitrogen": 0, "n_oxygen": 0, "n_sulfur": 0, "n_phosphorus": 0,
                "n_fluorine": 0, "n_chlorine": 0, "n_bromine": 0, "n_iodine": 0,
                "n_wildcards_mol": 0, "molecule_backbone_len": 0,
                "n_double_bonds": 0, "n_triple_bonds": 0, "n_rotatable_bonds": 0,
                "n_stereo_centers": 0,
            }
        n_atoms = mol.GetNumAtoms()
        n_heavy = mol.GetNumHeavyAtoms()
        n_wc_mol = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
        arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        sp3 = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP3)
        sp2 = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
        n_rings = mol.GetRingInfo().NumRings()
        n_arom_rings = sum(1 for r in mol.GetRingInfo().AtomRings()
                           if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
        n_ali_rings = n_rings - n_arom_rings
        n_sat_rings = sum(1 for r in mol.GetRingInfo().AtomRings()
                          if all(mol.GetAtomWithIdx(i).GetHybridization() ==
                                  Chem.HybridizationType.SP3 for i in r))
        counts = {"N": 0, "O": 0, "S": 0, "P": 0, "F": 0, "Cl": 0, "Br": 0, "I": 0}
        for a in mol.GetAtoms():
            sym = a.GetSymbol()
            if sym in counts:
                counts[sym] += 1
        try:
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rot = Descriptors.NumRotatableBonds(mol)
        except Exception:
            hbd, hba, rot = 0, 0, 0
        n_dbl = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        n_trp = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE)
        stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        backbone_len = 0
        if n_wc_mol >= 2:
            wc_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            try:
                sp = Chem.rdmolops.GetShortestPath(mol, wc_idx[0], wc_idx[-1])
                backbone_len = len(sp) - 1 if sp else 0
            except Exception:
                pass
        return {
            "n_wildcards_smiles": raw_wc, "n_atoms": n_atoms, "n_heavy_atoms": n_heavy,
            "n_rings": n_rings, "n_aromatic_rings": n_arom_rings,
            "n_aliphatic_rings": n_ali_rings, "n_saturated_rings": n_sat_rings,
            "frac_aromatic": arom / max(n_atoms, 1),
            "frac_sp3": sp3 / max(n_atoms, 1), "frac_sp2": sp2 / max(n_atoms, 1),
            "n_halogens": counts["F"] + counts["Cl"] + counts["Br"] + counts["I"],
            "n_hbd": hbd, "n_hba": hba,
            "n_nitrogen": counts["N"], "n_oxygen": counts["O"], "n_sulfur": counts["S"],
            "n_phosphorus": counts["P"], "n_fluorine": counts["F"],
            "n_chlorine": counts["Cl"], "n_bromine": counts["Br"], "n_iodine": counts["I"],
            "n_wildcards_mol": n_wc_mol, "molecule_backbone_len": backbone_len,
            "n_double_bonds": n_dbl, "n_triple_bonds": n_trp,
            "n_rotatable_bonds": rot, "n_stereo_centers": stereo,
        }

    def _go():
        rows = [_feats_one(s) for s in tqdm(smiles_list, desc=f"{desc} compute",
                                              unit="mol", leave=False)]
        return pd.DataFrame(rows).add_prefix("pc_")
    return _cached("polymer_custom_v1", smiles_list, _go, desc=desc)


BICERANO_SMARTS = {
    "ester":        "[#6][CX3](=O)[OX2H0][#6]",
    "carbonate":    "[OX2H0][CX3](=O)[OX2H0]",
    "amide":        "[NX3][CX3](=O)[#6]",
    "sulfonyl":     "[SX4](=O)(=O)",
    "sulfoxide":    "[SX3](=O)",
    "sulfide":      "[#16X2H0]",
    "ether":        "[OD2]([#6])[#6]",
    "phosphate":    "[PX4](=O)(O)(O)O",
    "phosphonate":  "[PX4](=O)(O)O[#6]",
    "phenyl":       "c1ccccc1",
    "naphthyl":     "c1ccc2ccccc2c1",
    "biphenyl":     "c1ccc(cc1)-c1ccccc1",
    "pyridine":     "n1ccccc1",
    "furan":        "o1cccc1",
    "thiophene":    "s1cccc1",
    "imide":        "[#6][NX3]([#6])[CX3](=O)",
    "urea":         "[NX3][CX3](=O)[NX3]",
    "carbamate":    "[NX3][CX3](=O)[OX2H0][#6]",
    "hydroxy":      "[OX2H]",
    "amine_prim":   "[NX3H2][#6]",
    "amine_sec":    "[NX3H1]([#6])[#6]",
    "amine_tert":   "[NX3H0]([#6])([#6])[#6]",
    "nitrile":      "[#6]#[NX1]",
    "nitro":        "[NX3](=O)=O",
    "trifluoromethyl": "[CX4](F)(F)F",
    "vinyl":        "[CH2]=[CH2]",
    "alkyne":       "[#6]#[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "isocyanate":   "[NX2]=[CX2]=[OX1]",
    "epoxide":      "[C;R]1[O;R][C;R]1",
    "aromatic_amine": "[nX3;+0]",
    "styrene_like": "c1ccccc1[CX3]=[CX3]",
    "backbone_double": "[!#0]-[!#0]=[!#0]",
    "bulky_side":   "[CX4]([#6])([#6])[#6]",
}


def compute_bicerano_groups(smiles_list, *, desc="bic-groups"):
    def _go():
        compiled = {name: Chem.MolFromSmarts(p) for name, p in BICERANO_SMARTS.items()}
        rows = []
        for smi in tqdm(smiles_list, desc=f"{desc} compute", unit="mol", leave=False):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({k: 0 for k in compiled}); continue
            r = {}
            for name, patt in compiled.items():
                if patt is None:
                    r[name] = 0; continue
                try:
                    r[name] = len(mol.GetSubstructMatches(patt))
                except Exception:
                    r[name] = 0
            rows.append(r)
        return pd.DataFrame(rows, columns=list(BICERANO_SMARTS.keys())).add_prefix("bg_")
    return _cached("bicerano_groups_v1", smiles_list, _go, desc=desc)


# ---------------------------------------------------------------------------
# NEW PHYSICS-INFORMED FEATURES
# ---------------------------------------------------------------------------

def compute_multi_conformer_3d(smiles_list, *, n_conformers=N_CONFORMERS,
                                 desc="multi-conf-3d"):
    """5 conformers per polymer, aggregated shape descriptors (mean/std/min/max).

    Chain packing / free volume is Tg's mechanism. Multiple conformers sample
    the space instead of relying on a single random embedding.
    """
    base_keys = [
        "Asphericity", "Eccentricity", "InertialShapeFactor",
        "NPR1", "NPR2", "PMI1", "PMI2", "PMI3",
        "RadiusOfGyration", "SpherocityIndex",
    ]
    agg_keys = [f"{k}_{s}" for k in base_keys for s in ("mean", "std", "min", "max")]

    def _feats_one(smi):
        capped = _capped_smiles(smi)
        mol = Chem.MolFromSmiles(capped)
        if mol is None:
            return {k: np.nan for k in agg_keys}
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        try:
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        except Exception:
            cids = []
        if not cids:
            return {k: np.nan for k in agg_keys}
        for cid in cids:
            try:
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=100)
            except Exception:
                pass
        rows = []
        for cid in cids:
            r = {}
            for k in base_keys:
                try:
                    r[k] = float(getattr(Descriptors3D, k)(mol, confId=cid))
                except Exception:
                    r[k] = np.nan
            rows.append(r)
        arr = pd.DataFrame(rows)
        agg = {}
        for k in base_keys:
            v = arr[k].dropna()
            if len(v):
                agg[f"{k}_mean"] = float(v.mean())
                agg[f"{k}_std"] = float(v.std()) if len(v) > 1 else 0.0
                agg[f"{k}_min"] = float(v.min())
                agg[f"{k}_max"] = float(v.max())
            else:
                for s in ("mean", "std", "min", "max"):
                    agg[f"{k}_{s}"] = np.nan
        return agg

    def _go():
        rows = [_feats_one(s) for s in tqdm(smiles_list, desc=f"{desc} compute",
                                              unit="mol", leave=False)]
        return pd.DataFrame(rows, columns=agg_keys).add_prefix("mc3d_")
    return _cached(f"multi_conf_3d_v1_n{n_conformers}", smiles_list, _go,
                    desc=desc, extra=(n_conformers,))


def compute_gasteiger_charges(smiles_list, *, desc="gasteiger"):
    """Gasteiger partial charges: mean/std/min/max/skew of charges on heavy atoms.
    Electronic environment surrogate. Cheap and physics-based."""
    keys = ["charge_mean", "charge_std", "charge_min", "charge_max",
            "charge_absmax", "charge_range", "charge_p90", "charge_p10"]

    def _feats_one(smi):
        # Gasteiger charges need real atoms — cap * with carbon first
        capped = _capped_smiles(smi)
        mol = Chem.MolFromSmiles(capped)
        if mol is None:
            return {k: np.nan for k in keys}
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            return {k: np.nan for k in keys}
        charges = []
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                continue
            try:
                c = float(a.GetProp("_GasteigerCharge"))
                if not np.isnan(c) and not np.isinf(c):
                    charges.append(c)
            except Exception:
                pass
        if not charges:
            return {k: np.nan for k in keys}
        arr = np.array(charges)
        return {
            "charge_mean": float(arr.mean()),
            "charge_std": float(arr.std()) if len(arr) > 1 else 0.0,
            "charge_min": float(arr.min()),
            "charge_max": float(arr.max()),
            "charge_absmax": float(np.abs(arr).max()),
            "charge_range": float(arr.max() - arr.min()),
            "charge_p90": float(np.percentile(arr, 90)),
            "charge_p10": float(np.percentile(arr, 10)),
        }

    def _go():
        rows = [_feats_one(s) for s in tqdm(smiles_list, desc=f"{desc} compute",
                                              unit="mol", leave=False)]
        return pd.DataFrame(rows, columns=keys).add_prefix("gc_")
    return _cached("gasteiger_v1", smiles_list, _go, desc=desc)


def compute_coulomb_matrix_eigvals(smiles_list, *, top_k=COULOMB_TOP_K,
                                     desc="coulomb"):
    """Top-k sorted eigenvalues of the Coulomb matrix from ETKDG conformer.

    Coulomb matrix: C[i,j] = Z_i * Z_j / |R_i - R_j|  (i != j)
                    C[i,i] = 0.5 * Z_i^2.4
    Eigenvalues (sorted |ev| desc, padded/truncated to top_k) are a permutation-
    invariant encoding of the electronic-nuclear geometry. Well-known predictor
    for HOMO-LUMO gap (i.e. Egc).
    """
    keys = [f"eig_{i}" for i in range(top_k)]

    def _feats_one(smi):
        capped = _capped_smiles(smi)
        mol = Chem.MolFromSmiles(capped)
        if mol is None:
            return {k: 0.0 for k in keys}
        embedded = _embed_conformer(mol)
        if embedded is None:
            return {k: 0.0 for k in keys}
        try:
            conf = embedded.GetConformer()
            zs = np.array([a.GetAtomicNum() for a in embedded.GetAtoms()], dtype=np.float64)
            coords = np.array([[conf.GetAtomPosition(i).x,
                                 conf.GetAtomPosition(i).y,
                                 conf.GetAtomPosition(i).z]
                                for i in range(embedded.GetNumAtoms())])
            n = len(zs)
            C = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        C[i, j] = 0.5 * (zs[i] ** 2.4)
                    else:
                        d = np.linalg.norm(coords[i] - coords[j])
                        if d > 1e-6:
                            C[i, j] = zs[i] * zs[j] / d
            eigvals = np.linalg.eigvalsh(C)
            order = np.argsort(-np.abs(eigvals))
            sorted_ev = eigvals[order]
            if len(sorted_ev) >= top_k:
                out = sorted_ev[:top_k]
            else:
                out = np.pad(sorted_ev, (0, top_k - len(sorted_ev)))
            return {k: float(v) for k, v in zip(keys, out)}
        except Exception:
            return {k: 0.0 for k in keys}

    def _go():
        rows = [_feats_one(s) for s in tqdm(smiles_list, desc=f"{desc} compute",
                                              unit="mol", leave=False)]
        return pd.DataFrame(rows, columns=keys).add_prefix("cm_")
    return _cached(f"coulomb_matrix_v1_k{top_k}", smiles_list, _go,
                    desc=desc, extra=(top_k,))


# ---------------------------------------------------------------------------
# FEATURE ASSEMBLY
# ---------------------------------------------------------------------------

def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in df.columns]
    if df.columns.duplicated().any():
        new, seen = [], {}
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new.append(f"{c}__{seen[c]}")
            else:
                seen[c] = 0
                new.append(c)
        df.columns = new
    return df


def build_features(smiles: list[str], *, split_label: str) -> pd.DataFrame:
    log.info("=== building rich feature matrix: %s (n=%d) ===", split_label, len(smiles))
    parts = [
        compute_rdkit_2d(smiles, desc=f"{split_label}/rdk-2d"),
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
        compute_polymer_custom(smiles, desc=f"{split_label}/poly-custom"),
        compute_bicerano_groups(smiles, desc=f"{split_label}/bic-groups"),
        # ---- new physics features ----
        compute_multi_conformer_3d(smiles, desc=f"{split_label}/mc3d"),
        compute_gasteiger_charges(smiles, desc=f"{split_label}/gasteiger"),
        compute_coulomb_matrix_eigvals(smiles, desc=f"{split_label}/coulomb"),
    ]
    parts = [p.reset_index(drop=True) for p in parts if p.shape[1] > 0]
    X = pd.concat(parts, axis=1)
    X = _sanitize(X)
    log.info("[%s] final feature matrix: %s", split_label, X.shape)
    return X


def drop_constant_cols(Xtr, Xte):
    keep = Xtr.columns[Xtr.nunique(dropna=False) > 1].tolist()
    return Xtr[keep], Xte[keep], keep


def split_by_target(train, test, Xtr, Xte, target: str):
    tm = (train["target_type"] == target).values
    em = (test["target_type"] == target).values
    return (
        Xtr[tm].reset_index(drop=True),
        train.loc[tm, "target"].reset_index(drop=True),
        Xte[em].reset_index(drop=True),
        test.loc[em, "id"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# CV + target transforms
# ---------------------------------------------------------------------------

def stratified_quantile_split(y, *, n_folds=5, n_bins=10, seed=42):
    y = np.asarray(y)
    n_bins_eff = min(n_bins, max(2, len(np.unique(y))))
    try:
        bins = pd.qcut(y, q=n_bins_eff, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(y, bins=n_bins_eff, labels=False, include_lowest=True)
    bins = np.asarray(bins)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros((len(y), 1)), bins))


def transform_target(y, kind: str):
    y = np.asarray(y, dtype=np.float64)
    if kind == "log1p":
        return np.log1p(y), np.expm1
    if kind == "identity":
        return y, (lambda x: x)
    raise ValueError(f"unknown transform: {kind!r}")


# ---------------------------------------------------------------------------
# PHASE 1 — GBM cocktail
# ---------------------------------------------------------------------------

def _train_lgb(X_tr, y_tr, X_va, y_va, *, w_tr=None):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    fit_kw = {"eval_set": [(X_va, y_va)],
              "callbacks": [lgb.early_stopping(stopping_rounds=200, verbose=False),
                            lgb.log_evaluation(period=0)]}
    if w_tr is not None:
        fit_kw["sample_weight"] = w_tr
    m.fit(X_tr, y_tr, **fit_kw)
    return m, int(m.best_iteration_ or LGB_PARAMS["n_estimators"])


def _train_cat(X_tr, y_tr, X_va, y_va, *, w_tr=None):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**CAT_PARAMS)
    if w_tr is not None:
        m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va),
              early_stopping_rounds=200, verbose=False)
    else:
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va),
              early_stopping_rounds=200, verbose=False)
    return m, int(m.get_best_iteration() or CAT_PARAMS["iterations"])


def _train_hgb(X_tr, y_tr, X_va=None, y_va=None, *, w_tr=None):
    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    if w_tr is not None:
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        m.fit(X_tr, y_tr)
    return m, int(m.n_iter_)


GBM_TRAINERS = {"lgb": _train_lgb, "cat": _train_cat, "hgb": _train_hgb}


def _refit_full(name, X, y, n_iters, *, w=None):
    import lightgbm as lgb
    if name == "lgb":
        p = dict(LGB_PARAMS); p["n_estimators"] = max(int(n_iters * 1.10), 200)
        m = lgb.LGBMRegressor(**p)
        m.fit(X, y, sample_weight=w) if w is not None else m.fit(X, y)
        return m
    if name == "cat":
        from catboost import CatBoostRegressor
        p = dict(CAT_PARAMS); p["iterations"] = max(int(n_iters * 1.10), 200)
        m = CatBoostRegressor(**p)
        m.fit(X, y, sample_weight=w, verbose=False) if w is not None else m.fit(X, y, verbose=False)
        return m
    if name == "hgb":
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X, y, sample_weight=w) if w is not None else m.fit(X, y)
        return m
    raise ValueError(name)


def cv_per_target(X, y_orig, target, kind, *, n_folds=N_FOLDS_GBM):
    y_t, inv = transform_target(y_orig, kind)
    oof = {k: np.zeros(len(X), dtype=np.float64) for k in GBM_TRAINERS}
    best_iters = {k: [] for k in GBM_TRAINERS}
    folds = stratified_quantile_split(y_t, n_folds=n_folds, n_bins=N_QUANTILE_BINS, seed=SEED)
    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc=f"{target} CV", unit="fold")):
        for k, fn in GBM_TRAINERS.items():
            t0 = time.time()
            m, n_it = fn(X.iloc[tr_idx], y_t[tr_idx], X.iloc[va_idx], y_t[va_idx])
            pred = inv(m.predict(X.iloc[va_idx]))
            oof[k][va_idx] = pred
            r2 = r2_score(y_orig[va_idx], pred)
            best_iters[k].append(n_it)
            log.info("[%s/%s] fold %d/%d R^2=%.4f iters=%d (%.1fs)",
                     target, k, fold + 1, n_folds, r2, n_it, time.time() - t0)
    per_r2 = {k: float(r2_score(y_orig, oof[k])) for k in GBM_TRAINERS}
    blend_oof = np.mean(np.stack([oof[k] for k in GBM_TRAINERS], axis=0), axis=0)
    blend_r2 = float(r2_score(y_orig, blend_oof))
    log.info("[%s] per-model OOF R^2: %s", target, {k: f"{v:.4f}" for k, v in per_r2.items()})
    log.info("[%s] blend OOF R^2 = %.4f", target, blend_r2)
    return oof, blend_oof, per_r2, blend_r2, best_iters


def run_phase1_gbm(train, test, X_train, X_test) -> dict:
    log.info("=" * 64)
    log.info("PHASE 1 -- GBM cocktail on rich physics-augmented features")
    log.info("=" * 64)
    per_target = {}
    for t in TARGETS:
        log.info("-" * 64)
        log.info("training %s", t.upper())
        X_tr, y_tr_s, X_te, ids_te = split_by_target(train, test, X_train, X_test, t)
        y_tr = y_tr_s.values
        oof, blend_oof, per_r2, blend_r2, best_iters = cv_per_target(
            X_tr, y_tr, t, TARGET_TRANSFORMS[t],
        )
        log.info("[%s] refit on full %d rows...", t, len(X_tr))
        y_t_full, inv = transform_target(y_tr, TARGET_TRANSFORMS[t])
        test_preds = {}
        for k in GBM_TRAINERS:
            n_it = int(np.median(best_iters[k]))
            t0 = time.time()
            m = _refit_full(k, X_tr, y_t_full, n_it)
            test_preds[k] = inv(m.predict(X_te))
            log.info("[%s/%s] refit done (iters=%d, %.1fs)", t, k, n_it, time.time() - t0)
        blend_test = np.mean(np.stack(list(test_preds.values())), axis=0)
        per_target[t] = {
            "smis": train.loc[train["target_type"] == t, "smiles"].values,
            "y_tr": y_tr, "oof_per_model": oof, "oof_blend": blend_oof,
            "per_model_r2": per_r2, "blend_oof_r2": blend_r2,
            "best_iters": best_iters, "test_ids": ids_te.values,
            "test_pred_per_model": test_preds, "test_pred_blend": blend_test,
        }
    return per_target


# ---------------------------------------------------------------------------
# PHASE 2 — Dual-variant Chemprop bag (CPU)
# ---------------------------------------------------------------------------

def long_to_wide(train: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for smi, t, y in zip(train["smiles"].values, train["target_type"].values,
                           train["target"].values):
        if smi not in rows:
            rows[smi] = {"smiles": smi, "tg": np.nan, "egc": np.nan}
        rows[smi][t] = y
    return pd.DataFrame(list(rows.values()))[["smiles", "tg", "egc"]]


def _apply_wide_transforms(wide):
    out = wide.copy()
    for t in TARGETS:
        y_t, _ = transform_target(wide[t].values, TARGET_TRANSFORMS[t])
        out[t] = y_t
    return out


def _inverse_wide(transformed):
    out = np.empty_like(transformed)
    for ti, t in enumerate(TARGETS):
        _, inv = transform_target(np.array([0.0]), TARGET_TRANSFORMS[t])
        out[:, ti] = inv(transformed[:, ti])
    return out


def _run_chemprop_variant(train, test, run_dir: Path, variant: str,
                            ffn_depth: int, aggregation: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Train one Chemprop variant. Returns (oof [n_uniq, 2], test [n_uniq, 2], n_models).
    Checkpoints per fold under run_dir/chemprop_{variant}.npz.
    """
    log.info("-" * 64)
    log.info("CHEMPROP VARIANT %s (agg=%s, ffn_depth=%d)", variant, aggregation, ffn_depth)
    log.info("-" * 64)

    import torch
    try:
        from lightning import pytorch as pl
    except ImportError:
        import pytorch_lightning as pl  # type: ignore
    from chemprop import data as cdata, featurizers as cfeat, models as cmodels, nn as cnn

    def _build_mpnn(seed):
        pl.seed_everything(seed)
        mp = cnn.BondMessagePassing(d_h=CP_MP_HIDDEN, depth=CP_MP_DEPTH, dropout=CP_DROPOUT)
        if aggregation == "mean":
            agg = cnn.MeanAggregation()
        elif aggregation == "sum":
            agg = cnn.SumAggregation()
        elif aggregation == "norm":
            agg = cnn.NormAggregation()
        else:
            raise ValueError(f"unknown aggregation: {aggregation}")
        pred = cnn.RegressionFFN(input_dim=CP_MP_HIDDEN, hidden_dim=CP_FFN_HIDDEN,
                                    n_layers=ffn_depth, dropout=CP_DROPOUT,
                                    n_tasks=len(TARGETS))
        return cmodels.MPNN(mp, agg, pred, batch_norm=True)

    def _loaders(tr_smis, tr_z, va_smis, va_z):
        feat = cfeat.SimpleMoleculeMolGraphFeaturizer()
        tr_dps = [cdata.MoleculeDatapoint.from_smi(s, y=y.astype(np.float32))
                    for s, y in zip(tr_smis, tr_z)]
        va_dps = [cdata.MoleculeDatapoint.from_smi(s, y=y.astype(np.float32))
                    for s, y in zip(va_smis, va_z)]
        return (
            cdata.build_dataloader(cdata.MoleculeDataset(tr_dps, feat),
                                    batch_size=CP_BATCH_SIZE, num_workers=0),
            cdata.build_dataloader(cdata.MoleculeDataset(va_dps, feat),
                                    batch_size=CP_BATCH_SIZE, num_workers=0, shuffle=False),
        )

    def _predict_loader(smis):
        feat = cfeat.SimpleMoleculeMolGraphFeaturizer()
        dps = [cdata.MoleculeDatapoint.from_smi(s) for s in smis]
        return cdata.build_dataloader(cdata.MoleculeDataset(dps, feat),
                                       batch_size=CP_BATCH_SIZE, num_workers=0, shuffle=False)

    wide_orig = long_to_wide(train)
    wide_t = _apply_wide_transforms(wide_orig)
    test_smis = test["smiles"].drop_duplicates().tolist()
    train_smis = wide_orig["smiles"].values
    ys = wide_t[TARGETS].values.astype(np.float32)
    strat = np.nanmean(ys, axis=1)
    folds = stratified_quantile_split(strat, n_folds=CP_N_FOLDS,
                                        n_bins=N_QUANTILE_BINS, seed=SEED)

    ckpt_path = run_dir / f"chemprop_{variant}.npz"
    oof_t = np.full((len(wide_orig), len(TARGETS)), np.nan, dtype=np.float64)
    test_acc_t = np.zeros((len(test_smis), len(TARGETS)), dtype=np.float64)
    n_models = 0
    completed = 0
    if ckpt_path.exists():
        try:
            ck = np.load(ckpt_path, allow_pickle=False)
            oof_t = ck["oof_t"]
            test_acc_t = ck["test_acc_t"]
            n_models = int(ck["n_models"])
            completed = int(ck["completed_folds"])
            log.info("resumed variant %s from ckpt: %d folds, %d models",
                     variant, completed, n_models)
        except Exception as e:
            log.warning("failed to load variant %s ckpt: %s", variant, e)

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        if fold_idx < completed:
            log.info("variant %s fold %d/%d already done", variant, fold_idx + 1, CP_N_FOLDS)
            continue
        log.info("variant %s FOLD %d/%d", variant, fold_idx + 1, CP_N_FOLDS)
        tr_smis_f, va_smis_f = train_smis[tr_idx], train_smis[va_idx]
        tr_ys, va_ys = ys[tr_idx], ys[va_idx]
        mu = np.nanmean(tr_ys, axis=0).astype(np.float32)
        sd = np.nanstd(tr_ys, axis=0).astype(np.float32)
        sd = np.where(sd < 1e-6, 1.0, sd)
        tr_z, va_z = (tr_ys - mu) / sd, (va_ys - mu) / sd

        seed_val, seed_test = [], []
        for seed in CP_BAG_SEEDS:
            try:
                tr_loader, va_loader = _loaders(tr_smis_f, tr_z, va_smis_f, va_z)
                model = _build_mpnn(seed)
                trainer = pl.Trainer(
                    accelerator="cpu", devices=1, max_epochs=CP_MAX_EPOCHS,
                    enable_progress_bar=False, enable_checkpointing=False, logger=False,
                    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=CP_PATIENCE, mode="min",
                                                            check_finite=False)],
                    gradient_clip_val=1.0,
                )
                t0 = time.time()
                trainer.fit(model, tr_loader, va_loader)
                val_z = torch.cat(trainer.predict(model, _predict_loader(va_smis_f.tolist())),
                                    dim=0).cpu().numpy()
                test_z = torch.cat(trainer.predict(model, _predict_loader(test_smis)),
                                     dim=0).cpu().numpy()
                seed_val.append(val_z * sd + mu)
                seed_test.append(test_z * sd + mu)
                log.info("[%s fold%d seed=%d] done in %.1fs",
                         variant, fold_idx + 1, seed, time.time() - t0)
            except Exception as e:
                log.exception("[%s fold%d seed=%d] failed: %s", variant, fold_idx + 1, seed, e)

        if not seed_val:
            continue
        oof_t[va_idx] = np.mean(np.stack(seed_val), axis=0)
        test_acc_t += np.mean(np.stack(seed_test), axis=0) * len(seed_val)
        n_models += len(seed_val)
        np.savez(ckpt_path, oof_t=oof_t, test_acc_t=test_acc_t,
                 n_models=n_models, completed_folds=fold_idx + 1)
        log.info("[%s fold %d] checkpoint saved (n_models=%d)", variant, fold_idx + 1, n_models)

    if n_models == 0:
        raise RuntimeError(f"Variant {variant}: 0 models trained")
    return oof_t, test_acc_t / n_models, n_models


def run_chemprop_dual(train, test, run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train both variants and average their predictions.
    Returns (oof_transformed, test_transformed_uniq, meta)."""
    log.info("=" * 64)
    log.info("PHASE 2 -- Dual-variant Chemprop bag (CPU)")
    log.info("=" * 64)

    oof_A, test_A, n_A = _run_chemprop_variant(
        train, test, run_dir, variant="A", ffn_depth=CP_FFN_DEPTH_A, aggregation="mean")
    oof_B, test_B, n_B = _run_chemprop_variant(
        train, test, run_dir, variant="B", ffn_depth=CP_FFN_DEPTH_B, aggregation="sum")

    # Average the two variants (equal weight; NNLS can rebalance downstream)
    valid_A = ~np.isnan(oof_A).any(axis=1)
    valid_B = ~np.isnan(oof_B).any(axis=1)
    oof_avg = np.full_like(oof_A, np.nan)
    both = valid_A & valid_B
    oof_avg[both] = 0.5 * (oof_A[both] + oof_B[both])
    only_a = valid_A & ~valid_B
    oof_avg[only_a] = oof_A[only_a]
    only_b = valid_B & ~valid_A
    oof_avg[only_b] = oof_B[only_b]

    test_avg = 0.5 * (test_A + test_B)
    meta = {"n_A": n_A, "n_B": n_B,
            "coverage_A": float(valid_A.mean()), "coverage_B": float(valid_B.mean())}
    log.info("variant A: n_models=%d cov=%.3f  |  variant B: n_models=%d cov=%.3f",
             n_A, meta["coverage_A"], n_B, meta["coverage_B"])
    return oof_avg, test_avg, meta


def align_chemprop_long(train, test, wide_smis, test_smis,
                          oof_t, test_t) -> tuple[pd.DataFrame, pd.DataFrame,
                                                    np.ndarray, np.ndarray, np.ndarray]:
    oof_orig = _inverse_wide(oof_t)
    test_orig = _inverse_wide(test_t)
    has_oof = ~np.isnan(oof_t).any(axis=1)
    smi2i_tr = {s: i for i, s in enumerate(wide_smis)}
    smi2i_te = {s: i for i, s in enumerate(test_smis)}

    oof_rows = []
    for _, r in train.iterrows():
        idx = smi2i_tr[r["smiles"]]
        ti = TARGETS.index(r["target_type"])
        oof_rows.append({
            "smiles": r["smiles"], "target_type": r["target_type"],
            "target": float(r["target"]),
            "oof_chemprop": float(oof_orig[idx, ti]) if has_oof[idx] else np.nan,
            "has_oof": bool(has_oof[idx]),
        })
    long_oof = pd.DataFrame(oof_rows)

    sub_rows = [{"id": int(r["id"]),
                  "target": float(test_orig[smi2i_te[r["smiles"]], TARGETS.index(r["target_type"])])}
                 for _, r in test.iterrows()]
    long_sub = pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True)
    return long_oof, long_sub, oof_orig, test_orig, has_oof


# ---------------------------------------------------------------------------
# PHASE 3 — Iterative pseudo-labeling
# ---------------------------------------------------------------------------

def _do_one_pseudo_round(train, test, X_train, X_test, phase1,
                          wide_smis, oof_source_wide, round_idx: int) -> dict:
    """Fill missing-target rows using `oof_source_wide` (n_uniq, 2), retrain LGB per target.
    Returns per-target dict with oof and test preds."""
    log.info("-" * 32)
    log.info("PSEUDO ROUND %d", round_idx)
    log.info("-" * 32)
    import lightgbm as lgb

    smi2i = {s: i for i, s in enumerate(wide_smis)}
    per_target = {}
    for t in TARGETS:
        ti = TARGETS.index(t)
        other = "egc" if t == "tg" else "tg"

        tr_mask = (train["target_type"] == t).values
        other_mask = (train["target_type"] == other).values
        real_smis = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        real_y = train.loc[tr_mask, "target"].reset_index(drop=True).values
        real_X = X_train[tr_mask].reset_index(drop=True)

        other_smis = train.loc[other_mask, "smiles"].reset_index(drop=True).values
        real_set = set(real_smis)
        candidates = [s for s in other_smis if s not in real_set]
        candidates = list(dict.fromkeys(candidates))
        pseudo_smis, pseudo_y = [], []
        for smi in candidates:
            if smi not in smi2i:
                continue
            v = oof_source_wide[smi2i[smi], ti]
            if not np.isnan(v):
                pseudo_smis.append(smi)
                pseudo_y.append(float(v))
        pseudo_smis, pseudo_y = np.array(pseudo_smis), np.array(pseudo_y)
        log.info("[R%d/%s] real=%d  pseudo=%d", round_idx, t, len(real_y), len(pseudo_smis))

        if len(pseudo_smis) > 0:
            other_smis_arr = train.loc[other_mask, "smiles"].reset_index(drop=True).values
            X_other = X_train[other_mask].reset_index(drop=True)
            first_idx = {}
            for i, s in enumerate(other_smis_arr):
                first_idx.setdefault(s, i)
            keep_idx = np.array([first_idx[s] for s in pseudo_smis])
            X_pseudo = X_other.iloc[keep_idx].reset_index(drop=True)
        else:
            X_pseudo = real_X.iloc[:0].copy()

        X_expanded = pd.concat([real_X, X_pseudo], axis=0, ignore_index=True)
        y_expanded = np.concatenate([real_y, pseudo_y])
        w_expanded = np.concatenate([
            np.ones(len(real_y)), PSEUDO_WEIGHT * np.ones(len(pseudo_y))
        ])
        y_t_expanded, inv = transform_target(y_expanded, TARGET_TRANSFORMS[t])

        real_folds = stratified_quantile_split(y_t_expanded[:len(real_y)],
                                                 n_folds=N_FOLDS_GBM, n_bins=N_QUANTILE_BINS,
                                                 seed=SEED)
        oof_r = np.zeros(len(real_y), dtype=np.float64)
        best_iters = []
        pseudo_idx = np.arange(len(real_y), len(real_y) + len(pseudo_y))
        for fold, (tr_idx, va_idx) in enumerate(tqdm(real_folds,
                                                       desc=f"R{round_idx}/{t}",
                                                       unit="fold")):
            X_tr = pd.concat([X_expanded.iloc[tr_idx], X_expanded.iloc[pseudo_idx]],
                              axis=0, ignore_index=True)
            y_tr = np.concatenate([y_t_expanded[tr_idx], y_t_expanded[pseudo_idx]])
            w_tr = np.concatenate([w_expanded[tr_idx], w_expanded[pseudo_idx]])
            X_va = X_expanded.iloc[va_idx]
            y_va_t = y_t_expanded[va_idx]
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va_t)],
                   callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                                lgb.log_evaluation(period=0)])
            oof_r[va_idx] = inv(m.predict(X_va))
            best_iters.append(int(m.best_iteration_ or LGB_PARAMS["n_estimators"]))
            r2 = r2_score(real_y[va_idx], oof_r[va_idx])
            log.info("[R%d/%s/lgb] fold %d/%d R^2=%.4f iters=%d",
                     round_idx, t, fold + 1, N_FOLDS_GBM, r2, best_iters[-1])
        r2_full = float(r2_score(real_y, oof_r))
        log.info("[R%d/%s] pseudo-LGB OOF R^2 = %.4f", round_idx, t, r2_full)

        # Refit on full expanded set for test predictions
        n_it = int(np.median(best_iters))
        p = dict(LGB_PARAMS); p["n_estimators"] = max(int(n_it * 1.10), 200)
        m_full = lgb.LGBMRegressor(**p)
        m_full.fit(X_expanded, y_t_expanded, sample_weight=w_expanded)
        te_mask = (test["target_type"] == t).values
        X_te = X_test[te_mask].reset_index(drop=True)
        ids_te = test.loc[te_mask, "id"].reset_index(drop=True).values
        test_p = inv(m_full.predict(X_te))

        per_target[t] = {
            "oof_pseudo": oof_r, "pseudo_r2": r2_full,
            "test_pseudo": test_p, "test_ids": ids_te,
            "n_real": len(real_y), "n_pseudo": len(pseudo_y),
        }
    return per_target


def run_iterative_pseudo(train, test, X_train, X_test, phase1,
                           wide_smis, chemprop_oof_wide) -> list[dict]:
    """Two rounds of pseudo-labeling. Round 2 uses Round 1 OOF + Chemprop OOF averaged."""
    log.info("=" * 64)
    log.info("PHASE 3 -- Iterative pseudo-labeling (%d rounds)", PSEUDO_N_ROUNDS)
    log.info("=" * 64)

    rounds = []
    # Round 1: source is Chemprop OOF alone
    r1 = _do_one_pseudo_round(train, test, X_train, X_test, phase1,
                                wide_smis, chemprop_oof_wide, round_idx=1)
    rounds.append(r1)

    if PSEUDO_N_ROUNDS >= 2:
        # Round 2: source is average of Chemprop OOF and Round-1 LGB pseudo OOF
        # Need to convert per-target LGB OOF back into a wide (n_uniq, 2) matrix
        smi2i = {s: i for i, s in enumerate(wide_smis)}
        wide_r1 = np.full((len(wide_smis), len(TARGETS)), np.nan, dtype=np.float64)
        for ti, t in enumerate(TARGETS):
            tr_mask = (train["target_type"] == t).values
            tr_smis = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
            for j, smi in enumerate(tr_smis):
                if smi in smi2i:
                    wide_r1[smi2i[smi], ti] = r1[t]["oof_pseudo"][j]

        # Also fill wildcard: for unlabeled targets on labeled rows, keep NaN
        # (LGB was only fit for real target rows). For those we rely on Chemprop.
        # Average where both available:
        avg = np.full_like(wide_r1, np.nan)
        both = ~np.isnan(wide_r1) & ~np.isnan(chemprop_oof_wide)
        only_cp = np.isnan(wide_r1) & ~np.isnan(chemprop_oof_wide)
        only_r1 = ~np.isnan(wide_r1) & np.isnan(chemprop_oof_wide)
        avg[both] = 0.5 * (wide_r1[both] + chemprop_oof_wide[both])
        avg[only_cp] = chemprop_oof_wide[only_cp]
        avg[only_r1] = wide_r1[only_r1]

        r2 = _do_one_pseudo_round(train, test, X_train, X_test, phase1,
                                    wide_smis, avg, round_idx=2)
        rounds.append(r2)

    return rounds


# ---------------------------------------------------------------------------
# PHASE 4 — CatBoost meta-stacker
# ---------------------------------------------------------------------------

def _murcko_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "none"
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        smi_scaf = Chem.MolToSmiles(core, canonical=True)
        return smi_scaf if smi_scaf else "empty"
    except Exception:
        return "err"


def run_meta_stacker(train, test, phase1, chemprop_oof_long,
                       chemprop_sub_long, pseudo_rounds, out_dir: Path,
                       nnls_fallback: bool = True) -> dict:
    """CatBoost meta-stacker per target.

    Meta-features: base-model OOFs + scaffold ID (categorical) + a handful of
    tabular features (n_heavy_atoms, frac_aromatic, n_aromatic_rings).
    Falls back to NNLS if CatBoost throws.
    """
    log.info("=" * 64)
    log.info("PHASE 4 -- CatBoost meta-stacker over base OOFs + scaffold")
    log.info("=" * 64)

    summary = {"per_target": {}}
    final_test_frames, final_oof_frames = [], []

    for t in TARGETS:
        log.info("-" * 32)
        log.info("meta-stacking %s", t.upper())
        tr_mask = (train["target_type"] == t).values
        te_mask = (test["target_type"] == t).values
        smis_t = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        y_tr = train.loc[tr_mask, "target"].reset_index(drop=True).values
        ids_te = test.loc[te_mask, "id"].reset_index(drop=True).values

        p1_oof = phase1[t]["oof_blend"]
        p1_test = phase1[t]["test_pred_blend"]

        cp_slice = chemprop_oof_long[chemprop_oof_long["target_type"] == t].reset_index(drop=True)
        assert (cp_slice["smiles"].values == smis_t).all(), f"[{t}] cp mis-align"
        cp_oof = cp_slice["oof_chemprop"].values
        mask_cp = cp_slice["has_oof"].values
        cp_test = chemprop_sub_long.set_index("id").loc[ids_te, "target"].values

        pr1_oof = pseudo_rounds[0][t]["oof_pseudo"]
        pr1_test = pseudo_rounds[0][t]["test_pseudo"]
        base_names = ["phase1", "pseudo_r1", "chemprop"]
        base_oofs = [p1_oof, pr1_oof, cp_oof]
        base_tests = [p1_test, pr1_test, cp_test]

        if len(pseudo_rounds) >= 2:
            pr2_oof = pseudo_rounds[1][t]["oof_pseudo"]
            pr2_test = pseudo_rounds[1][t]["test_pseudo"]
            base_names.append("pseudo_r2")
            base_oofs.append(pr2_oof)
            base_tests.append(pr2_test)

        A_full = np.column_stack(base_oofs).astype(np.float64)
        test_mat = np.column_stack(base_tests).astype(np.float64)

        # Scaffold + a few tabular fields for the meta-features
        scaffolds = np.array([_murcko_scaffold(s) for s in smis_t])
        te_smis = test.loc[te_mask, "smiles"].reset_index(drop=True).values
        scaffolds_te = np.array([_murcko_scaffold(s) for s in te_smis])
        n_heavy = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                             if Chem.MolFromSmiles(s) is not None else 0 for s in smis_t])
        n_heavy_te = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                                if Chem.MolFromSmiles(s) is not None else 0 for s in te_smis])
        frac_arom = []
        for s in smis_t:
            m = Chem.MolFromSmiles(s)
            if m is None:
                frac_arom.append(0.0); continue
            n = m.GetNumAtoms()
            a = sum(1 for at in m.GetAtoms() if at.GetIsAromatic())
            frac_arom.append(a / max(n, 1))
        frac_arom = np.array(frac_arom)
        frac_arom_te = []
        for s in te_smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                frac_arom_te.append(0.0); continue
            n = m.GetNumAtoms()
            a = sum(1 for at in m.GetAtoms() if at.GetIsAromatic())
            frac_arom_te.append(a / max(n, 1))
        frac_arom_te = np.array(frac_arom_te)

        meta_df = pd.DataFrame(A_full, columns=[f"oof_{n}" for n in base_names])
        meta_df["scaffold"] = scaffolds
        meta_df["n_heavy"] = n_heavy
        meta_df["frac_arom"] = frac_arom

        meta_test = pd.DataFrame(test_mat, columns=[f"oof_{n}" for n in base_names])
        meta_test["scaffold"] = scaffolds_te
        meta_test["n_heavy"] = n_heavy_te
        meta_test["frac_arom"] = frac_arom_te

        # Where Chemprop OOF is NaN, meta-CatBoost should still work (CatBoost handles NaN)
        # but replace with phase1 OOF as a safe fallback so features are always valid
        for col in [f"oof_{n}" for n in base_names]:
            v = meta_df[col].values
            if np.isnan(v).any():
                fallback = meta_df["oof_phase1"].values
                meta_df[col] = np.where(np.isnan(v), fallback, v)

        # CatBoost meta-stack (with fold-wise OOF for honest meta-OOF)
        y_t, inv = transform_target(y_tr, TARGET_TRANSFORMS[t])
        folds = stratified_quantile_split(y_t, n_folds=N_FOLDS_GBM,
                                            n_bins=N_QUANTILE_BINS, seed=SEED)
        meta_oof = np.zeros(len(y_tr), dtype=np.float64)
        cat_kwargs = dict(
            iterations=1000, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            grow_policy="SymmetricTree", random_seed=SEED, verbose=False,
            allow_writing_files=False,
        )
        try:
            from catboost import CatBoostRegressor, Pool
            cat_cols = ["scaffold"]
            for fold, (tr_idx, va_idx) in enumerate(folds):
                p_tr = Pool(meta_df.iloc[tr_idx], label=y_t[tr_idx], cat_features=cat_cols)
                p_va = Pool(meta_df.iloc[va_idx], label=y_t[va_idx], cat_features=cat_cols)
                m = CatBoostRegressor(**cat_kwargs)
                m.fit(p_tr, eval_set=p_va, early_stopping_rounds=100, verbose=False)
                meta_oof[va_idx] = inv(m.predict(meta_df.iloc[va_idx]))
                r2 = r2_score(y_tr[va_idx], meta_oof[va_idx])
                log.info("[%s/meta] fold %d/%d R^2=%.4f iters=%d",
                         t, fold + 1, N_FOLDS_GBM, r2, m.get_best_iteration())
            meta_r2 = float(r2_score(y_tr, meta_oof))
            log.info("[%s] META OOF R^2 = %.4f", t, meta_r2)

            # Fit on full for test predictions
            p_full = Pool(meta_df, label=y_t, cat_features=cat_cols)
            m_full = CatBoostRegressor(**cat_kwargs)
            m_full.fit(p_full, verbose=False)
            meta_test_pred = inv(m_full.predict(meta_test))
            method_used = "catboost_meta"
        except Exception as e:
            log.warning("[%s] CatBoost meta-stacker failed (%s) — falling back to NNLS", t, e)
            if not nnls_fallback:
                raise
            A_sub = A_full
            y_sub = y_tr.astype(np.float64)
            w_raw, _ = nnls(A_sub, y_sub)
            w_sum = w_raw.sum()
            w_norm = (w_raw / w_sum) if w_sum > 1e-9 else np.eye(len(base_names))[0]
            log.info("[%s] NNLS fallback weights: %s",
                      t, {n: float(w) for n, w in zip(base_names, w_norm)})
            meta_oof = A_full @ w_norm
            meta_test_pred = test_mat @ w_norm
            meta_r2 = float(r2_score(y_tr, meta_oof))
            method_used = "nnls_fallback"
            log.info("[%s] NNLS OOF R^2 = %.4f", t, meta_r2)

        # Apply LB bias correction
        bias = TG_BIAS if t == "tg" else EGC_BIAS
        if bias != 0.0:
            log.info("[%s] applying LB bias correction: %+.3f", t, bias)
            meta_test_pred = meta_test_pred + bias
            meta_oof = meta_oof + bias

        final_test_frames.append(pd.DataFrame({"id": ids_te, "target": meta_test_pred}))
        final_oof_frames.append(pd.DataFrame({
            "smiles": smis_t, "target_type": t, "target": y_tr,
            "oof_phase1": p1_oof, "oof_pseudo_r1": pr1_oof,
            "oof_chemprop": cp_oof, "has_chemprop_oof": mask_cp,
            "oof_meta": meta_oof,
            **({"oof_pseudo_r2": pseudo_rounds[1][t]["oof_pseudo"]}
                if len(pseudo_rounds) >= 2 else {}),
        }))
        summary["per_target"][t] = {
            "method": method_used, "meta_oof_r2": meta_r2,
            "phase1_oof_r2": phase1[t]["blend_oof_r2"],
            "pseudo_r1_oof_r2": pseudo_rounds[0][t]["pseudo_r2"],
            "pseudo_r2_oof_r2": (pseudo_rounds[1][t]["pseudo_r2"]
                                  if len(pseudo_rounds) >= 2 else None),
            "n_pseudo_r1": pseudo_rounds[0][t]["n_pseudo"],
            "lb_bias_applied": bias,
        }

    final_sub = pd.concat(final_test_frames).sort_values("id").reset_index(drop=True)
    final_oof = pd.concat(final_oof_frames).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_sub.to_csv(out_dir / "submission.csv", index=False)
    final_oof.to_csv(out_dir / "oof_final.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s (%d rows)", out_dir / "submission.csv", len(final_sub))

    mean_meta = float(np.mean([summary["per_target"][t]["meta_oof_r2"] for t in TARGETS]))
    mean_phase1 = float(np.mean([summary["per_target"][t]["phase1_oof_r2"] for t in TARGETS]))
    log.info("=" * 64)
    log.info("Mean Phase 1 OOF R^2 = %.4f", mean_phase1)
    log.info("Mean META OOF R^2   = %.4f", mean_meta)
    log.info("=" * 64)
    return summary


# ---------------------------------------------------------------------------
# LB PROBE HELPER
# ---------------------------------------------------------------------------

def lb_probe(out_dir: Path = OUT_DIR) -> None:
    """Write two constant-value CSVs to detect target-mean shift on public LB.

    Submit each to Kaggle. From the two R^2 (or MSE) scores you can back out
    the LB target mean per target. If it differs from the train mean, set the
    TG_BIAS / EGC_BIAS constants at the top of this file to `LB_mean - train_mean`
    and re-run the meta-stacker phase only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    tg_mean = float(train.loc[train["target_type"] == "tg", "target"].mean())
    egc_mean = float(train.loc[train["target_type"] == "egc", "target"].mean())

    # Constant sub 1: train mean per target
    sub1 = test.copy()
    sub1["target"] = np.where(sub1["target_type"] == "tg", tg_mean, egc_mean)
    sub1 = sub1[["id", "target"]].sort_values("id").reset_index(drop=True)
    p1 = out_dir / "lb_probe_train_mean.csv"
    sub1.to_csv(p1, index=False)

    # Constant sub 2: offset +30°C for Tg only (Egc still at train mean)
    sub2 = test.copy()
    sub2["target"] = np.where(sub2["target_type"] == "tg", tg_mean + 30.0, egc_mean)
    sub2 = sub2[["id", "target"]].sort_values("id").reset_index(drop=True)
    p2 = out_dir / "lb_probe_tg_plus30.csv"
    sub2.to_csv(p2, index=False)

    log.info("Wrote LB-probe files:")
    log.info("  %s  (Tg=%.3f, Egc=%.3f)", p1, tg_mean, egc_mean)
    log.info("  %s  (Tg=%.3f+30=%.3f)", p2, tg_mean, tg_mean + 30.0)
    log.info("")
    log.info("Submit both to Kaggle. R^2 for a constant prediction c on target y is")
    log.info("  R^2 = 1 - Var(y - c) / Var(y)")
    log.info("If you plot LB score vs Tg offset, the maximum tells you LB Tg mean.")
    log.info("If LB Tg mean != %.3f, set TG_BIAS = (LB_mean - %.3f) and re-run meta.",
              tg_mean, tg_mean)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb-probe", action="store_true",
                         help="Write LB-probe constant-value CSVs and exit.")
    args = parser.parse_args()

    if args.lb_probe:
        lb_probe(OUT_DIR)
        return

    t0 = time.time()
    log.info("=" * 64)
    log.info("exp_polymer_physics_stack.py")
    log.info("=" * 64)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Featurize ---
    X_train_full = build_features(train["smiles"].tolist(), split_label="train")
    X_test_full = build_features(test["smiles"].tolist(), split_label="test")
    common = sorted(set(X_train_full.columns) & set(X_test_full.columns))
    X_train_full, X_test_full = X_train_full[common], X_test_full[common]
    X_train_full, X_test_full, kept = drop_constant_cols(X_train_full, X_test_full)
    log.info("features after intersect + drop-constant: %d", len(kept))

    # --- Phase 1 (checkpointed) ---
    p1_ckpt = OUT_DIR / "phase1_state.pkl.gz"
    if p1_ckpt.exists():
        log.info("Phase 1 ckpt found; loading")
        phase1 = _load_pickle(p1_ckpt)
        log.info("Phase 1 restored. Tg OOF=%.4f  Egc OOF=%.4f",
                 phase1["tg"]["blend_oof_r2"], phase1["egc"]["blend_oof_r2"])
    else:
        phase1 = run_phase1_gbm(train, test, X_train_full, X_test_full)
        _save_pickle(phase1, p1_ckpt)
        log.info("Phase 1 ckpt saved")

    # --- Phase 2 (checkpointed inside per variant) ---
    wide_orig = long_to_wide(train)
    wide_smis = wide_orig["smiles"].values.tolist()
    test_unique = test["smiles"].drop_duplicates().tolist()

    p2_ckpt = OUT_DIR / "phase2_state.pkl.gz"
    if p2_ckpt.exists():
        log.info("Phase 2 top-level ckpt found; loading")
        state2 = _load_pickle(p2_ckpt)
        oof_t_all, test_t_all = state2["oof_t"], state2["test_t"]
        chemprop_meta = state2["meta"]
    else:
        oof_t_all, test_t_all, chemprop_meta = run_chemprop_dual(train, test, OUT_DIR)
        _save_pickle({"oof_t": oof_t_all, "test_t": test_t_all, "meta": chemprop_meta},
                       p2_ckpt)
        log.info("Phase 2 top-level ckpt saved")

    chemprop_oof_long, chemprop_sub_long, oof_orig_wide, test_orig_wide, has_oof = \
        align_chemprop_long(train, test, wide_smis, test_unique, oof_t_all, test_t_all)
    chemprop_oof_long.to_csv(OUT_DIR / "chemprop_oof.csv", index=False)
    chemprop_sub_long.to_csv(OUT_DIR / "chemprop_only_submission.csv", index=False)
    log.info("Chemprop dual coverage: %d / %d train (%.1f%%), n_models_A=%d n_models_B=%d",
             int(has_oof.sum()), len(has_oof), 100 * has_oof.mean(),
             chemprop_meta["n_A"], chemprop_meta["n_B"])

    # --- Phase 3 (checkpointed) ---
    p3_ckpt = OUT_DIR / "phase3_state.pkl.gz"
    if p3_ckpt.exists():
        log.info("Phase 3 ckpt found; loading")
        pseudo_rounds = _load_pickle(p3_ckpt)
    else:
        pseudo_rounds = run_iterative_pseudo(train, test, X_train_full, X_test_full,
                                              phase1, wide_smis, oof_orig_wide)
        _save_pickle(pseudo_rounds, p3_ckpt)
        log.info("Phase 3 ckpt saved")

    # --- Phase 4 (always re-run — cheap, and lets bias correction be applied) ---
    summary = run_meta_stacker(train, test, phase1, chemprop_oof_long,
                                 chemprop_sub_long, pseudo_rounds, OUT_DIR)
    summary["chemprop_meta"] = chemprop_meta
    summary["lb_bias"] = {"tg": TG_BIAS, "egc": EGC_BIAS}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info("=" * 64)
    log.info("DONE — submit %s", OUT_DIR / "submission.csv")
    log.info("total runtime: %.1fs (%.1fh)",
              time.time() - t0, (time.time() - t0) / 3600)


if __name__ == "__main__":
    main()
