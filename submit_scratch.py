"""submit_scratch.py — From-scratch polymer property prediction pipeline.

Everything is trained from scratch on the competition data only.
No pretrained models. No external labeled datasets. No borrowed artifacts.
All features are deterministic transformations of the input SMILES.

Strategy for pushing past the 0.911 LB ceiling:

1. Richer, more physics-informed feature engineering
   - RDKit 2D descriptors (~210)
   - RDKit 3D descriptors (~40, from ETKDG conformers) — new, captures chain packing
   - Count/binary Morgan (r=2, r=3), MACCS, Avalon, Atom-Pair, Topological-Torsion
   - Custom polymer descriptors — new (backbone length, wildcards, ring/aromatic fractions)
   - Bicerano-style SMARTS group counts — new (chemistry-aware Tg prior)
   - Optional: GFN2-xTB HOMO-LUMO gap (if `tblite` is installed) — Egc physics prior

2. GBM cocktail per target (LightGBM + CatBoost + HistGradientBoosting), 5-fold
   stratified-quantile CV, log1p on Egc, identity on Tg. Mean-blended.

3. Chemprop D-MPNN multitask trained from scratch on CPU (avoids the MPS
   thermal-throttling that killed the previous run). Compressed to
   3-fold x 2-seed = 6 bagged models to stay under ~5h.

4. Pseudo-label round: fill missing per-target labels using the Chemprop
   cross-target predictions (sample_weight=0.3), refit GBMs on real+pseudo.

5. Blend: NNLS over [Phase-1-blend, Phase-1-pseudo-blend, Chemprop-bag]
   per target, applied to test predictions. Writes final submission.

Output: results/submit_scratch/submission.csv
        results/submit_scratch/oof_final.csv
        results/submit_scratch/summary.json

The whole thing is self-contained: paste this into a single Kaggle notebook
cell, ensure `data/train.csv` and `data/test.csv` are in the working
directory, and it will run end-to-end.

Environment:
    pip install rdkit lightgbm catboost scikit-learn pandas scipy tqdm \
                chemprop lightning torch
    # optional for xTB HOMO-LUMO features:
    pip install tblite
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
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, MACCSkeys, rdFingerprintGenerator
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

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"
CACHE_DIR = REPO / ".cache" / "features"
OUT_DIR = REPO / "results" / "submit_scratch"

SEED = 42
N_FOLDS_GBM = 5
N_QUANTILE_BINS = 10
TARGETS = ["tg", "egc"]
TARGET_TRANSFORMS = {"tg": "identity", "egc": "log1p"}

# --- GBM configs ---
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

# --- Chemprop configs ---
CP_MP_HIDDEN, CP_MP_DEPTH = 300, 4
CP_FFN_HIDDEN, CP_FFN_DEPTH, CP_DROPOUT = 300, 2, 0.05
CP_MAX_EPOCHS, CP_BATCH_SIZE, CP_PATIENCE = 50, 64, 10
CP_N_FOLDS = 3
CP_BAG_SEEDS = [42, 1337]  # 3 folds * 2 seeds = 6 models

# --- pseudo-labeling ---
PSEUDO_WEIGHT = 0.3

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
# cache helpers — features get computed once and pickled under .cache/features/
# ---------------------------------------------------------------------------

def _save_pickle(obj, path: Path) -> None:
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
# featurizers (all deterministic, all in-competition)
# ---------------------------------------------------------------------------

def _capped_smiles(smi: str, cap: str = "C") -> str:
    """Replace polymer wildcards `*` with a real atom for 3D + xTB calculations."""
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


def compute_rdkit_2d(smiles_list: list[str], *, desc: str = "rdk-2d") -> pd.DataFrame:
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


def compute_rdkit_3d(smiles_list: list[str], *, desc: str = "rdk-3d") -> pd.DataFrame:
    """RDKit 3D descriptors from ETKDG + UFF conformer. Polymer wildcards capped
    with carbon so ETKDG can embed the molecule.

    Adds ~40 shape/geometry features: Asphericity, Eccentricity, InertialShapeFactor,
    RadiusOfGyration, PMI1/2/3, NPR1/2, SpherocityIndex, ...
    """
    keys_3d = [
        "Asphericity", "Eccentricity", "InertialShapeFactor",
        "NPR1", "NPR2", "PMI1", "PMI2", "PMI3",
        "RadiusOfGyration", "SpherocityIndex",
    ]

    def _go():
        rows = []
        for smi in tqdm(smiles_list, desc=f"{desc} compute", unit="mol", leave=False):
            capped = _capped_smiles(smi)
            mol = Chem.MolFromSmiles(capped)
            if mol is None:
                rows.append({k: np.nan for k in keys_3d}); continue
            embedded = _embed_conformer(mol)
            if embedded is None:
                rows.append({k: np.nan for k in keys_3d}); continue
            row = {}
            for k in keys_3d:
                try:
                    row[k] = float(getattr(Descriptors3D, k)(embedded))
                except Exception:
                    row[k] = np.nan
            rows.append(row)
        return pd.DataFrame(rows, columns=keys_3d).add_prefix("d3_")
    return _cached("rdk_3d_v1", smiles_list, _go, desc=desc)


def compute_morgan_fp(smiles_list: list[str], *, radius: int = 2, n_bits: int = 2048,
                       count: bool = False, desc: str | None = None) -> pd.DataFrame:
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


def compute_maccs(smiles_list: list[str], *, desc: str = "maccs") -> pd.DataFrame:
    def _go():
        mols = _smiles_to_mols(smiles_list, desc)
        arr = np.zeros((len(mols), 167), dtype=np.int8)
        for i, mol in enumerate(tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False)):
            if mol is None: continue
            ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr[i])
        return pd.DataFrame(arr, columns=[f"maccs_{j}" for j in range(167)])
    return _cached("maccs_v1", smiles_list, _go, desc=desc)


def compute_avalon_fp(smiles_list: list[str], *, n_bits: int = 512,
                       desc: str = "avalon") -> pd.DataFrame:
    def _go():
        mols = _smiles_to_mols(smiles_list, desc)
        arr = np.zeros((len(mols), n_bits), dtype=np.int8)
        for i, mol in enumerate(tqdm(mols, desc=f"{desc} compute", unit="mol", leave=False)):
            if mol is None: continue
            ConvertToNumpyArray(GetAvalonFP(mol, nBits=n_bits), arr[i])
        return pd.DataFrame(arr, columns=[f"avlon_{j}" for j in range(n_bits)])
    return _cached(f"avalon_{n_bits}_v1", smiles_list, _go, desc=desc, extra=(n_bits,))


def compute_atom_pair_fp(smiles_list: list[str], *, n_bits: int = 2048,
                          count: bool = True, desc: str | None = None) -> pd.DataFrame:
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
    return _cached(
        f"atom_pair_{'cnt' if count else 'bit'}_{n_bits}_v1",
        smiles_list, _go, desc=label,
    )


def compute_topological_torsion_fp(smiles_list: list[str], *, n_bits: int = 2048,
                                     count: bool = True, desc: str | None = None) -> pd.DataFrame:
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
    return _cached(
        f"torsion_{'cnt' if count else 'bit'}_{n_bits}_v1",
        smiles_list, _go, desc=label,
    )


# --- polymer-specific hand-crafted features --------------------------------

def compute_polymer_custom(smiles_list: list[str], *, desc: str = "poly-custom") -> pd.DataFrame:
    """Chain-aware descriptors: wildcards, aromatic fraction, ring counts,
    sp2/sp3 balance, halogen counts, heteroatom fractions, backbone length
    proxy. Everything derivable from the SMILES itself."""

    def _feats_one(smi):
        raw_wildcards = smi.count("*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return {
                "n_wildcards_smiles": raw_wildcards,
                "n_atoms": 0, "n_heavy_atoms": 0, "n_rings": 0,
                "n_aromatic_rings": 0, "n_aliphatic_rings": 0, "n_saturated_rings": 0,
                "frac_aromatic": 0.0, "frac_sp3": 0.0, "frac_sp2": 0.0,
                "n_halogens": 0, "n_hbd": 0, "n_hba": 0,
                "n_nitrogen": 0, "n_oxygen": 0, "n_sulfur": 0, "n_phosphorus": 0,
                "n_fluorine": 0, "n_chlorine": 0, "n_bromine": 0, "n_iodine": 0,
                "n_wildcards_mol": 0, "molecule_backbone_len": 0,
                "n_double_bonds": 0, "n_triple_bonds": 0, "n_rotatable_bonds": 0,
                "n_stereo_centers": 0,
            }
        n_atoms = mol.GetNumAtoms()
        n_heavy = mol.GetNumHeavyAtoms()
        n_wc_mol = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
        aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
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
        n_halogens = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
        try:
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rot = Descriptors.NumRotatableBonds(mol)
        except Exception:
            hbd, hba, rot = 0, 0, 0
        n_double = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        n_triple = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE)
        stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        # backbone length proxy: shortest path between two wildcards, if two exist
        backbone_len = 0
        if n_wc_mol >= 2:
            wc_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            try:
                sp = Chem.rdmolops.GetShortestPath(mol, wc_idx[0], wc_idx[-1])
                backbone_len = len(sp) - 1 if sp else 0
            except Exception:
                backbone_len = 0
        return {
            "n_wildcards_smiles": raw_wildcards,
            "n_atoms": n_atoms, "n_heavy_atoms": n_heavy, "n_rings": n_rings,
            "n_aromatic_rings": n_arom_rings, "n_aliphatic_rings": n_ali_rings,
            "n_saturated_rings": n_sat_rings,
            "frac_aromatic": aromatic_atoms / max(n_atoms, 1),
            "frac_sp3": sp3 / max(n_atoms, 1),
            "frac_sp2": sp2 / max(n_atoms, 1),
            "n_halogens": n_halogens, "n_hbd": hbd, "n_hba": hba,
            "n_nitrogen": counts["N"], "n_oxygen": counts["O"], "n_sulfur": counts["S"],
            "n_phosphorus": counts["P"], "n_fluorine": counts["F"], "n_chlorine": counts["Cl"],
            "n_bromine": counts["Br"], "n_iodine": counts["I"],
            "n_wildcards_mol": n_wc_mol, "molecule_backbone_len": backbone_len,
            "n_double_bonds": n_double, "n_triple_bonds": n_triple,
            "n_rotatable_bonds": rot, "n_stereo_centers": stereo,
        }

    def _go():
        rows = [_feats_one(s) for s in tqdm(smiles_list, desc=f"{desc} compute",
                                              unit="mol", leave=False)]
        return pd.DataFrame(rows).add_prefix("pc_")
    return _cached("polymer_custom_v1", smiles_list, _go, desc=desc)


# --- Bicerano-style group contribution counts ------------------------------
#
# Each pattern below matches a chemically distinct functional group whose
# presence is known to correlate with Tg (glass transition) — either through
# main-chain stiffness (aromatic rings, imides) or through cohesive energy
# (H-bond donors/acceptors, polar groups). These are chemistry-informed
# priors that supplement the raw Morgan bits.
#
# Column names are the group name; values are the substructure match count.

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


def compute_bicerano_groups(smiles_list: list[str], *, desc: str = "bic-groups") -> pd.DataFrame:
    def _go():
        compiled = {name: Chem.MolFromSmarts(patt) for name, patt in BICERANO_SMARTS.items()}
        rows = []
        for smi in tqdm(smiles_list, desc=f"{desc} compute", unit="mol", leave=False):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({k: 0 for k in compiled}); continue
            r = {}
            for name, patt in compiled.items():
                if patt is None:
                    r[name] = 0
                    continue
                try:
                    r[name] = len(mol.GetSubstructMatches(patt))
                except Exception:
                    r[name] = 0
            rows.append(r)
        return pd.DataFrame(rows, columns=list(BICERANO_SMARTS.keys())).add_prefix("bg_")
    return _cached("bicerano_groups_v1", smiles_list, _go, desc=desc)


# --- optional: GFN2-xTB HOMO-LUMO gap --------------------------------------

def compute_xtb_features(smiles_list: list[str], *, desc: str = "xtb") -> pd.DataFrame:
    """GFN2-xTB electronic descriptors: HOMO, LUMO, HOMO-LUMO gap, total energy.
    Skipped gracefully if `tblite` isn't installed. Runtime ~1-3s per molecule.

    Physics-based (semi-empirical DFT), no learned ML parameters.
    """
    try:
        from tblite.interface import Calculator  # type: ignore
        import numpy as _np
    except ImportError:
        log.info("[%s] tblite not installed → skipping xTB features", desc)
        return pd.DataFrame(index=range(len(smiles_list)))

    def _go():
        keys = ["homo", "lumo", "gap", "total_energy", "dipole_norm"]
        nan = {k: np.nan for k in keys}
        rows = []
        for smi in tqdm(smiles_list, desc=f"{desc} compute", unit="mol", leave=False):
            capped = _capped_smiles(smi)
            mol = Chem.MolFromSmiles(capped)
            if mol is None:
                rows.append(dict(nan)); continue
            embedded = _embed_conformer(mol)
            if embedded is None:
                rows.append(dict(nan)); continue
            try:
                conf = embedded.GetConformer()
                positions = _np.array([[conf.GetAtomPosition(i).x,
                                          conf.GetAtomPosition(i).y,
                                          conf.GetAtomPosition(i).z]
                                         for i in range(embedded.GetNumAtoms())]) / 0.529177  # Å → bohr
                numbers = _np.array([a.GetAtomicNum() for a in embedded.GetAtoms()])
                calc = Calculator("GFN2-xTB", numbers, positions)
                calc.set("verbosity", 0)
                res = calc.singlepoint()
                orb = res.get("orbital-energies")
                occ = res.get("orbital-occupations")
                homo_idx = int(_np.max(_np.where(occ > 0.5)[0]))
                lumo_idx = homo_idx + 1
                homo = float(orb[homo_idx])
                lumo = float(orb[lumo_idx]) if lumo_idx < len(orb) else float("nan")
                gap = lumo - homo
                dip = res.get("dipole")
                dip_norm = float(_np.linalg.norm(dip))
                total = float(res.get("energy"))
                rows.append({"homo": homo, "lumo": lumo, "gap": gap,
                              "total_energy": total, "dipole_norm": dip_norm})
            except Exception:
                rows.append(dict(nan))
        return pd.DataFrame(rows, columns=keys).add_prefix("xtb_")
    return _cached("xtb_gfn2_v1", smiles_list, _go, desc=desc)


# ---------------------------------------------------------------------------
# feature assembly
# ---------------------------------------------------------------------------

def build_features(smiles: list[str], *, split_label: str) -> pd.DataFrame:
    log.info("=== building feature cocktail: %s (n=%d) ===", split_label, len(smiles))
    parts = [
        compute_rdkit_2d(smiles, desc=f"{split_label}/rdk-2d"),
        compute_rdkit_3d(smiles, desc=f"{split_label}/rdk-3d"),
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
        compute_xtb_features(smiles, desc=f"{split_label}/xtb"),
    ]
    parts = [p.reset_index(drop=True) for p in parts if p.shape[1] > 0]
    X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    # sanitize column names for LightGBM
    X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in X.columns]
    if X.columns.duplicated().any():
        new_cols, seen = [], {}
        for c in X.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}__{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        X.columns = new_cols
    log.info("[%s] combined feature matrix: %s", split_label, X.shape)
    return X


def drop_constant_cols(X_train, X_test):
    keep = X_train.columns[X_train.nunique(dropna=False) > 1].tolist()
    return X_train[keep], X_test[keep], keep


def split_by_target(train, test, X_tr, X_te, target: str):
    tm = (train["target_type"] == target).values
    em = (test["target_type"] == target).values
    return (
        X_tr[tm].reset_index(drop=True),
        train.loc[tm, "target"].reset_index(drop=True),
        X_te[em].reset_index(drop=True),
        test.loc[em, "id"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# CV + target transforms
# ---------------------------------------------------------------------------

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
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=200, verbose=False)
    return m, int(m.get_best_iteration() or CAT_PARAMS["iterations"])


def _train_hgb(X_tr, y_tr, X_va=None, y_va=None, *, w_tr=None):
    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    if w_tr is not None:
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        m.fit(X_tr, y_tr)
    return m, int(m.n_iter_)


GBM_TRAINERS = {"lgb": _train_lgb, "cat": _train_cat, "hgb": _train_hgb}


def _refit_full(model_name: str, X, y, n_iters: int, *, w=None):
    import lightgbm as lgb
    if model_name == "lgb":
        p = dict(LGB_PARAMS); p["n_estimators"] = max(int(n_iters * 1.10), 200)
        m = lgb.LGBMRegressor(**p)
        m.fit(X, y, sample_weight=w) if w is not None else m.fit(X, y)
        return m
    if model_name == "cat":
        from catboost import CatBoostRegressor
        p = dict(CAT_PARAMS); p["iterations"] = max(int(n_iters * 1.10), 200)
        m = CatBoostRegressor(**p)
        if w is not None:
            m.fit(X, y, sample_weight=w, verbose=False)
        else:
            m.fit(X, y, verbose=False)
        return m
    if model_name == "hgb":
        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X, y, sample_weight=w) if w is not None else m.fit(X, y)
        return m
    raise ValueError(model_name)


def cv_per_target(X, y_orig, target_name, transform_kind, *, n_folds=N_FOLDS_GBM):
    y_t, inv = transform_target(y_orig, transform_kind)
    log.info("[%s] transform=%s | y_t range [%.3f, %.3f]",
              target_name, transform_kind, float(y_t.min()), float(y_t.max()))
    oof = {k: np.zeros(len(X), dtype=np.float64) for k in GBM_TRAINERS}
    best_iters = {k: [] for k in GBM_TRAINERS}
    folds = stratified_quantile_split(y_t, n_folds=n_folds, n_bins=N_QUANTILE_BINS, seed=SEED)
    fold_bar = tqdm(folds, desc=f"{target_name} CV", unit="fold")
    for fold, (tr_idx, va_idx) in enumerate(fold_bar):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_t[tr_idx], y_t[va_idx]
        for k, fn in GBM_TRAINERS.items():
            t0 = time.time()
            model, n_it = fn(X_tr, y_tr, X_va, y_va)
            pred = inv(model.predict(X_va))
            oof[k][va_idx] = pred
            r2 = r2_score(y_orig[va_idx], pred)
            best_iters[k].append(n_it)
            log.info("[%s/%s] fold %d/%d R^2=%.4f iters=%d (%.1fs)",
                     target_name, k, fold + 1, n_folds, r2, n_it, time.time() - t0)
    per_model_r2 = {k: float(r2_score(y_orig, oof[k])) for k in GBM_TRAINERS}
    blend_oof = np.mean(np.stack([oof[k] for k in GBM_TRAINERS], axis=0), axis=0)
    blend_r2 = float(r2_score(y_orig, blend_oof))
    log.info("[%s] per-model OOF R^2: %s", target_name,
             {k: f"{v:.4f}" for k, v in per_model_r2.items()})
    log.info("[%s] mean-blend OOF R^2 = %.4f", target_name, blend_r2)
    return oof, blend_oof, per_model_r2, blend_r2, best_iters


def run_phase1_gbm(train, test, X_train, X_test) -> dict:
    log.info("═" * 64)
    log.info("PHASE 1 — GBM cocktail on rich features")
    log.info("═" * 64)
    per_target = {}
    for t in TARGETS:
        log.info("─" * 64)
        log.info("training %s", t.upper())
        X_tr, y_tr_s, X_te, ids_te = split_by_target(train, test, X_train, X_test, t)
        y_tr = y_tr_s.values
        log.info("[%s] train=%d test=%d y=[%.3f, %.3f]",
                 t, len(X_tr), len(X_te), float(y_tr.min()), float(y_tr.max()))
        oof, blend_oof, per_model_r2, blend_r2, best_iters = cv_per_target(
            X_tr, y_tr, t, TARGET_TRANSFORMS[t],
        )
        log.info("[%s] refitting on full %d rows...", t, len(X_tr))
        y_t_full, inv = transform_target(y_tr, TARGET_TRANSFORMS[t])
        test_preds = {}
        for k in GBM_TRAINERS:
            n_it = int(np.median(best_iters[k]))
            t_fit = time.time()
            model = _refit_full(k, X_tr, y_t_full, n_it)
            test_preds[k] = inv(model.predict(X_te))
            log.info("[%s/%s] refit done (iters=%d, %.1fs)",
                     t, k, n_it, time.time() - t_fit)
        blend_test = np.mean(np.stack(list(test_preds.values()), axis=0), axis=0)
        per_target[t] = {
            "smis": train.loc[train["target_type"] == t, "smiles"].values,
            "y_tr": y_tr,
            "oof_per_model": oof,
            "oof_blend": blend_oof,
            "per_model_r2": per_model_r2,
            "blend_oof_r2": blend_r2,
            "best_iters": best_iters,
            "test_ids": ids_te.values,
            "test_pred_per_model": test_preds,
            "test_pred_blend": blend_test,
            "X_tr_index": np.where(train["target_type"] == t)[0],
            "X_te_index": np.where(test["target_type"] == t)[0],
        }
    return per_target


# ---------------------------------------------------------------------------
# Phase 2 — Chemprop D-MPNN multitask (CPU, 3-fold × 2-seed)
# ---------------------------------------------------------------------------

def long_to_wide(train: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for smi, t, y in zip(train["smiles"].values, train["target_type"].values, train["target"].values):
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


def run_chemprop_from_scratch(train, test, run_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (oof_transformed [n_train_uniq, 2], test_transformed [n_test_uniq, 2], n_models)."""
    log.info("═" * 64)
    log.info("PHASE 2 — Chemprop D-MPNN multitask (CPU, from scratch)")
    log.info("═" * 64)

    import torch
    try:
        from lightning import pytorch as pl
    except ImportError:
        import pytorch_lightning as pl  # type: ignore
    from chemprop import data as cdata, featurizers as cfeat, models as cmodels, nn as cnn

    def _build_mpnn(seed):
        pl.seed_everything(seed)
        mp = cnn.BondMessagePassing(d_h=CP_MP_HIDDEN, depth=CP_MP_DEPTH, dropout=CP_DROPOUT)
        agg = cnn.MeanAggregation()
        pred = cnn.RegressionFFN(input_dim=CP_MP_HIDDEN, hidden_dim=CP_FFN_HIDDEN,
                                    n_layers=CP_FFN_DEPTH, dropout=CP_DROPOUT, n_tasks=len(TARGETS))
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
    folds = stratified_quantile_split(strat, n_folds=CP_N_FOLDS, n_bins=N_QUANTILE_BINS, seed=SEED)

    ckpt_path = run_dir / "chemprop_checkpoint.npz"
    oof_t = np.full((len(wide_orig), len(TARGETS)), np.nan, dtype=np.float64)
    test_acc_t = np.zeros((len(test_smis), len(TARGETS)), dtype=np.float64)
    n_models = 0
    completed_folds = 0
    if ckpt_path.exists():
        try:
            ck = np.load(ckpt_path, allow_pickle=False)
            oof_t = ck["oof_t"]
            test_acc_t = ck["test_acc_t"]
            n_models = int(ck["n_models_added_to_test"])
            completed_folds = int(ck["completed_folds"])
            log.info("resumed Chemprop from ckpt: %d folds, %d models",
                      completed_folds, n_models)
        except Exception as e:
            log.warning("failed to load Chemprop ckpt, starting fresh: %s", e)

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        if fold_idx < completed_folds:
            log.info("Chemprop fold %d/%d already done, skipping", fold_idx + 1, CP_N_FOLDS)
            continue
        log.info("─" * 64)
        log.info("Chemprop FOLD %d/%d", fold_idx + 1, CP_N_FOLDS)
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
                log.info("[fold%d seed=%d] done in %.1fs",
                         fold_idx + 1, seed, time.time() - t0)
            except Exception as e:
                log.exception("[fold%d seed=%d] failed: %s", fold_idx + 1, seed, e)

        if not seed_val:
            continue
        oof_t[va_idx] = np.mean(np.stack(seed_val), axis=0)
        test_acc_t += np.mean(np.stack(seed_test), axis=0) * len(seed_val)
        n_models += len(seed_val)
        np.savez(ckpt_path, oof_t=oof_t, test_acc_t=test_acc_t,
                 n_models_added_to_test=n_models, completed_folds=fold_idx + 1)
        log.info("[fold%d] Chemprop checkpoint saved (n_models=%d)", fold_idx + 1, n_models)

    if n_models == 0:
        raise RuntimeError("Chemprop produced 0 models — training failed")
    return oof_t, test_acc_t / n_models, n_models


def align_chemprop_to_long(train, test, wide_smis, test_smis, oof_t_all, test_t_all):
    """Convert Chemprop wide (unique-SMILES) predictions back to per-row format
    aligned with train.csv and test.csv row order.

    Returns:
        chemprop_oof_long : pd.DataFrame with cols smiles, target_type, target,
                            oof_chemprop, has_oof
        chemprop_sub_long : pd.DataFrame with cols id, target
    """
    oof_orig = _inverse_wide(oof_t_all)
    test_orig = _inverse_wide(test_t_all)
    has_oof = ~np.isnan(oof_t_all).any(axis=1)
    smi_to_idx_train = {s: i for i, s in enumerate(wide_smis)}
    smi_to_idx_test = {s: i for i, s in enumerate(test_smis)}

    oof_rows = []
    for _, r in train.iterrows():
        idx = smi_to_idx_train[r["smiles"]]
        ti = TARGETS.index(r["target_type"])
        oof_rows.append({
            "smiles": r["smiles"], "target_type": r["target_type"],
            "target": float(r["target"]),
            "oof_chemprop": float(oof_orig[idx, ti]) if has_oof[idx] else np.nan,
            "has_oof": bool(has_oof[idx]),
        })
    chemprop_oof_long = pd.DataFrame(oof_rows)

    sub_rows = []
    for _, r in test.iterrows():
        idx = smi_to_idx_test[r["smiles"]]
        ti = TARGETS.index(r["target_type"])
        sub_rows.append({"id": int(r["id"]), "target": float(test_orig[idx, ti])})
    chemprop_sub_long = pd.DataFrame(sub_rows).sort_values("id").reset_index(drop=True)

    return chemprop_oof_long, chemprop_sub_long, oof_orig, test_orig, has_oof


# ---------------------------------------------------------------------------
# Phase 3 — pseudo-labeling with Chemprop's cross-target predictions
# ---------------------------------------------------------------------------

def run_pseudo_labeling(train, test, X_train, X_test, phase1: dict,
                         wide_smis, oof_orig_wide, test_orig_wide) -> dict:
    """For each target, add rows where Chemprop predicts the target for
    molecules that have the *other* target labeled. Refit LightGBM on
    real (weight=1.0) + pseudo (weight=PSEUDO_WEIGHT)."""
    log.info("═" * 64)
    log.info("PHASE 3 — Pseudo-labeling with cross-target Chemprop preds")
    log.info("═" * 64)
    import lightgbm as lgb

    wide_smi_to_idx = {s: i for i, s in enumerate(wide_smis)}
    per_target = {}

    for t in TARGETS:
        log.info("─" * 32)
        log.info("pseudo-labeling %s", t.upper())
        ti = TARGETS.index(t)
        # Real labels
        tr_mask = (train["target_type"] == t).values
        real_smis = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        real_y = train.loc[tr_mask, "target"].reset_index(drop=True).values
        real_X = X_train[tr_mask].reset_index(drop=True)

        # Molecules with only the OTHER target labeled → candidates for pseudo
        other = "egc" if t == "tg" else "tg"
        other_mask = (train["target_type"] == other).values
        other_smis = train.loc[other_mask, "smiles"].reset_index(drop=True).values
        # dedupe against real_smis (some molecules have both targets)
        real_set = set(real_smis)
        pseudo_candidates = [s for s in other_smis if s not in real_set]
        # dedupe among candidates
        pseudo_candidates = list(dict.fromkeys(pseudo_candidates))
        log.info("[%s] real=%d  candidate pseudo=%d", t, len(real_smis), len(pseudo_candidates))

        # Get Chemprop pseudo-label for each candidate (only rows where OOF exists)
        pseudo_smis, pseudo_y = [], []
        for smi in pseudo_candidates:
            if smi not in wide_smi_to_idx:
                continue
            idx = wide_smi_to_idx[smi]
            v = oof_orig_wide[idx, ti]
            if not np.isnan(v):
                pseudo_smis.append(smi)
                pseudo_y.append(float(v))
        pseudo_smis = np.array(pseudo_smis)
        pseudo_y = np.array(pseudo_y)
        log.info("[%s] usable pseudo (has Chemprop OOF): %d", t, len(pseudo_smis))

        # Build pseudo-feature matrix by matching SMILES rows in the OTHER target subset
        if len(pseudo_smis) > 0:
            other_smis_arr = train.loc[other_mask, "smiles"].reset_index(drop=True).values
            X_other = X_train[other_mask].reset_index(drop=True)
            # deduped index lookup
            first_idx = {}
            for i, s in enumerate(other_smis_arr):
                if s not in first_idx:
                    first_idx[s] = i
            keep_idx = np.array([first_idx[s] for s in pseudo_smis])
            X_pseudo = X_other.iloc[keep_idx].reset_index(drop=True)
        else:
            X_pseudo = real_X.iloc[:0].copy()

        # Assemble expanded training set: real (w=1.0) + pseudo (w=PSEUDO_WEIGHT)
        X_expanded = pd.concat([real_X, X_pseudo], axis=0, ignore_index=True)
        y_expanded = np.concatenate([real_y, pseudo_y])
        w_expanded = np.concatenate([
            np.ones(len(real_y)), PSEUDO_WEIGHT * np.ones(len(pseudo_y))
        ])
        log.info("[%s] expanded train rows: %d (real=%d pseudo=%d)",
                 t, len(X_expanded), len(real_y), len(pseudo_y))

        # 5-fold CV on REAL labels only (pseudo never in val)
        y_t_expanded, inv = transform_target(y_expanded, TARGET_TRANSFORMS[t])
        real_folds = stratified_quantile_split(y_t_expanded[:len(real_y)],
                                                 n_folds=N_FOLDS_GBM, n_bins=N_QUANTILE_BINS,
                                                 seed=SEED)
        oof_pseudo = np.zeros(len(real_y), dtype=np.float64)
        best_iters = []
        pseudo_idx = np.arange(len(real_y), len(real_y) + len(pseudo_y))
        for fold, (tr_idx, va_idx) in enumerate(tqdm(real_folds,
                                                      desc=f"{t}/pseudo CV", unit="fold")):
            X_tr = pd.concat([X_expanded.iloc[tr_idx],
                              X_expanded.iloc[pseudo_idx]],
                              axis=0, ignore_index=True)
            y_tr = np.concatenate([y_t_expanded[tr_idx], y_t_expanded[pseudo_idx]])
            w_tr = np.concatenate([w_expanded[tr_idx], w_expanded[pseudo_idx]])
            X_va = X_expanded.iloc[va_idx]
            y_va_t = y_t_expanded[va_idx]
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va_t)],
                   callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                                lgb.log_evaluation(period=0)])
            oof_pseudo[va_idx] = inv(m.predict(X_va))
            best_iters.append(int(m.best_iteration_ or LGB_PARAMS["n_estimators"]))
            r2 = r2_score(real_y[va_idx], oof_pseudo[va_idx])
            log.info("[%s/pseudo/lgb] fold %d/%d R^2=%.4f iters=%d",
                     t, fold + 1, N_FOLDS_GBM, r2, best_iters[-1])
        pseudo_r2 = float(r2_score(real_y, oof_pseudo))
        log.info("[%s] Pseudo-LGB OOF R^2 = %.4f", t, pseudo_r2)

        # Refit on full expanded set, predict test
        n_it = int(np.median(best_iters))
        p = dict(LGB_PARAMS); p["n_estimators"] = max(int(n_it * 1.10), 200)
        m_full = lgb.LGBMRegressor(**p)
        m_full.fit(X_expanded, y_t_expanded, sample_weight=w_expanded)
        te_mask = (test["target_type"] == t).values
        X_te = X_test[te_mask].reset_index(drop=True)
        ids_te = test.loc[te_mask, "id"].reset_index(drop=True).values
        test_pseudo = inv(m_full.predict(X_te))

        per_target[t] = {
            "oof_pseudo": oof_pseudo,
            "pseudo_r2": pseudo_r2,
            "test_pseudo": test_pseudo,
            "test_ids": ids_te,
            "n_real": len(real_y),
            "n_pseudo": len(pseudo_y),
        }

    return per_target


# ---------------------------------------------------------------------------
# blending — NNLS per target over base OOFs
# ---------------------------------------------------------------------------

def blend_and_submit(train, test, phase1: dict, phase3: dict,
                      chemprop_oof_long, chemprop_sub_long, out_dir: Path) -> dict:
    log.info("═" * 64)
    log.info("PHASE 4 — Per-target NNLS blend")
    log.info("═" * 64)

    summary = {"per_target": {}, "chemprop_meta": {}}
    final_test_frames, final_oof_frames = [], []

    for t in TARGETS:
        log.info("─" * 32)
        log.info("blending %s", t.upper())
        tr_mask = (train["target_type"] == t).values
        te_mask = (test["target_type"] == t).values
        smis_t = train.loc[tr_mask, "smiles"].reset_index(drop=True).values
        y_tr = train.loc[tr_mask, "target"].reset_index(drop=True).values
        ids_te = test.loc[te_mask, "id"].reset_index(drop=True).values

        p1_oof = phase1[t]["oof_blend"]
        p3_oof = phase3[t]["oof_pseudo"]
        p1_test = phase1[t]["test_pred_blend"]
        p3_test = phase3[t]["test_pseudo"]

        cp_slice = chemprop_oof_long[chemprop_oof_long["target_type"] == t].reset_index(drop=True)
        assert (cp_slice["smiles"].values == smis_t).all(), f"[{t}] cp OOF misaligned"
        cp_oof = cp_slice["oof_chemprop"].values
        mask = cp_slice["has_oof"].values
        cp_test = chemprop_sub_long.set_index("id").loc[ids_te, "target"].values

        base_oofs = {"phase1": p1_oof, "phase1_pseudo": p3_oof, "chemprop": cp_oof}
        base_tests = {"phase1": p1_test, "phase1_pseudo": p3_test, "chemprop": cp_test}

        # NNLS fits on subset where cp_oof is populated (typically ~2464 / 6158 unique molecules)
        A_sub = np.column_stack([p1_oof[mask], p3_oof[mask], cp_oof[mask]]).astype(np.float64)
        y_sub = y_tr[mask].astype(np.float64)
        if len(y_sub) < 50:
            log.warning("[%s] not enough Chemprop-OOF rows (%d) — falling back to phase1/pseudo only", t, len(y_sub))
            A_sub = np.column_stack([p1_oof, p3_oof]).astype(np.float64)
            y_sub = y_tr.astype(np.float64)
            base_names = ["phase1", "phase1_pseudo"]
            test_mat = np.column_stack([p1_test, p3_test]).astype(np.float64)
        else:
            base_names = ["phase1", "phase1_pseudo", "chemprop"]
            test_mat = np.column_stack([p1_test, p3_test, cp_test]).astype(np.float64)

        w_raw, _ = nnls(A_sub, y_sub)
        w_sum = w_raw.sum()
        w_norm = (w_raw / w_sum) if w_sum > 1e-9 else np.eye(len(base_names))[0]

        log.info("[%s] NNLS weights (raw → norm, sum=%.4f):", t, w_sum)
        for name, wr, wn in zip(base_names, w_raw, w_norm):
            log.info("    %-14s raw=%.4f norm=%.4f", name, wr, wn)

        stack_oof = A_sub @ w_norm
        stack_r2 = float(r2_score(y_sub, stack_oof))
        log.info("[%s] STACK OOF R^2 (subset, n=%d) = %.4f", t, len(y_sub), stack_r2)

        # Also compute per-base OOF R² on the same subset for diagnostic
        per_base_r2 = {}
        for name, oof_arr in zip(base_names, A_sub.T):
            per_base_r2[name] = float(r2_score(y_sub, oof_arr))
        log.info("[%s] per-base subset R^2: %s",
                  t, {k: f"{v:.4f}" for k, v in per_base_r2.items()})

        blend_test = test_mat @ w_norm

        final_test_frames.append(pd.DataFrame({"id": ids_te, "target": blend_test}))
        final_oof_frames.append(pd.DataFrame({
            "smiles": smis_t, "target_type": t, "target": y_tr,
            "oof_phase1": p1_oof, "oof_phase1_pseudo": p3_oof, "oof_chemprop": cp_oof,
            "has_chemprop_oof": mask,
        }))
        summary["per_target"][t] = {
            "n_train": int(tr_mask.sum()),
            "n_test": int(te_mask.sum()),
            "weights_normalized": {b: float(w) for b, w in zip(base_names, w_norm)},
            "per_base_subset_r2": per_base_r2,
            "stack_oof_r2_subset": stack_r2,
            "phase1_full_oof_r2": phase1[t]["blend_oof_r2"],
            "pseudo_full_oof_r2": phase3[t]["pseudo_r2"],
            "phase1_per_model_r2": phase1[t]["per_model_r2"],
        }

    final_sub = pd.concat(final_test_frames).sort_values("id").reset_index(drop=True)
    final_oof = pd.concat(final_oof_frames).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_sub.to_csv(out_dir / "submission.csv", index=False)
    final_oof.to_csv(out_dir / "oof_final.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s (%d rows)", out_dir / "submission.csv", len(final_sub))

    mean_stack = float(np.mean([summary["per_target"][t]["stack_oof_r2_subset"] for t in TARGETS]))
    mean_phase1 = float(np.mean([summary["per_target"][t]["phase1_full_oof_r2"] for t in TARGETS]))
    log.info("═" * 64)
    log.info("Mean Phase 1 OOF R^2       = %.4f", mean_phase1)
    log.info("Mean Stack OOF R^2 (subset) = %.4f", mean_stack)
    log.info("═" * 64)
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    t0 = time.time()
    log.info("=" * 64)
    log.info("submit_scratch.py — from-scratch pipeline")
    log.info("=" * 64)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    log.info("train: %s | test: %s", train.shape, test.shape)
    log.info("train target_type: %s", dict(train["target_type"].value_counts()))
    log.info("test  target_type: %s", dict(test["target_type"].value_counts()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Featurize ---
    X_train_full = build_features(train["smiles"].tolist(), split_label="train")
    X_test_full = build_features(test["smiles"].tolist(), split_label="test")
    common = sorted(set(X_train_full.columns) & set(X_test_full.columns))
    X_train_full, X_test_full = X_train_full[common], X_test_full[common]
    X_train_full, X_test_full, kept = drop_constant_cols(X_train_full, X_test_full)
    log.info("features after intersect + drop-constant: %d", len(kept))

    # --- Phase 1: GBM cocktail (checkpointed to avoid re-training on restart) ---
    phase1_ckpt = OUT_DIR / "phase1_state.pkl.gz"
    if phase1_ckpt.exists():
        log.info("Phase 1 checkpoint found → loading %s", phase1_ckpt.name)
        phase1 = _load_pickle(phase1_ckpt)
        log.info("Phase 1 restored. Tg OOF R^2=%.4f  Egc OOF R^2=%.4f",
                 phase1["tg"]["blend_oof_r2"], phase1["egc"]["blend_oof_r2"])
    else:
        phase1 = run_phase1_gbm(train, test, X_train_full, X_test_full)
        _save_pickle(phase1, phase1_ckpt)
        log.info("Phase 1 checkpoint saved → %s", phase1_ckpt.name)

    # --- Phase 2: Chemprop D-MPNN ---
    wide_orig = long_to_wide(train)
    wide_smis = wide_orig["smiles"].values.tolist()
    test_unique = test["smiles"].drop_duplicates().tolist()
    oof_t_all, test_t_all, n_chemprop_models = run_chemprop_from_scratch(train, test, OUT_DIR)

    chemprop_oof_long, chemprop_sub_long, oof_orig_wide, test_orig_wide, has_oof = \
        align_chemprop_to_long(train, test, wide_smis, test_unique, oof_t_all, test_t_all)
    chemprop_oof_long.to_csv(OUT_DIR / "chemprop_oof.csv", index=False)
    chemprop_sub_long.to_csv(OUT_DIR / "chemprop_only_submission.csv", index=False)
    log.info("Chemprop coverage: %d / %d train rows (%.1f%%), n_models=%d",
              int(has_oof.sum()), len(has_oof), 100 * has_oof.mean(), n_chemprop_models)

    # --- Phase 3: Pseudo-labeling (checkpointed too) ---
    phase3_ckpt = OUT_DIR / "phase3_state.pkl.gz"
    if phase3_ckpt.exists():
        log.info("Phase 3 checkpoint found → loading %s", phase3_ckpt.name)
        phase3 = _load_pickle(phase3_ckpt)
    else:
        phase3 = run_pseudo_labeling(train, test, X_train_full, X_test_full,
                                       phase1, wide_smis, oof_orig_wide, test_orig_wide)
        _save_pickle(phase3, phase3_ckpt)
        log.info("Phase 3 checkpoint saved → %s", phase3_ckpt.name)

    # --- Phase 4: Blend ---
    summary = blend_and_submit(train, test, phase1, phase3,
                                chemprop_oof_long, chemprop_sub_long, OUT_DIR)
    summary["chemprop_meta"] = {
        "n_models": n_chemprop_models,
        "n_folds_completed": int(has_oof.sum() / len(has_oof) * CP_N_FOLDS),
        "coverage": float(has_oof.mean()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info("═" * 64)
    log.info("DONE — submit %s", OUT_DIR / "submission.csv")
    log.info("total runtime: %.1fs (%.1fh)",
              time.time() - t0, (time.time() - t0) / 3600)


if __name__ == "__main__":
    main()
