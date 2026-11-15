"""MOE-type per-molecule descriptors and the 3,780-column pair builder.

Pair feature definition (documented hypothesis, confirmed by the Brick 5 check):
  3,600 cross products  drug_i * food_j   (i, j in 1..60)
     60 sums             drug_i + food_i
     60 absolute diffs   |drug_i - food_i|
     60 ratios           drug_i / food_i
  -------------------------------------------------
  3,780 columns
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from PyBioMed.PyMolecule.moe import GetMOE

RDLogger.DisableLog("rdApp.*")

EPS = 1e-6
RATIO_CLIP = 1e6


def descriptor_names() -> list[str]:
    mol = Chem.MolFromSmiles("CCO")
    return list(GetMOE(mol).keys())


DESC_NAMES = descriptor_names()
N_DESC = len(DESC_NAMES)


def pair_feature_names(desc_names: Sequence[str] = DESC_NAMES) -> list[str]:
    names = [f"{a}*{b}" for a in desc_names for b in desc_names]
    names += [f"{a}+{a}" for a in desc_names]
    names += [f"abs({a}-{a})" for a in desc_names]
    names += [f"{a}/{a}" for a in desc_names]
    if len(set(names)) != len(names):
        raise ValueError("pair feature names are not unique")
    return names


ALL_PAIR_NAMES = pair_feature_names()


def compute_moe_table(molecule_ids: Iterable, smiles: Iterable[str]) -> tuple[pd.DataFrame, list[dict]]:
    rows, failures = [], []
    for mid, smi in zip(molecule_ids, smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append({"molecule_id": str(mid), "reason": "parse"})
            continue
        try:
            values = GetMOE(mol)
        except Exception as exc:
            failures.append({"molecule_id": str(mid), "reason": f"moe:{type(exc).__name__}"})
            continue
        arr = np.array([values[k] for k in DESC_NAMES], dtype=np.float64)
        if not np.isfinite(arr).all():
            failures.append({"molecule_id": str(mid), "reason": "nonfinite"})
            continue
        rows.append({"molecule_id": mid, **{k: float(v) for k, v in zip(DESC_NAMES, arr)}})
    return pd.DataFrame(rows), failures


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    mask = np.abs(den) > EPS
    out[mask] = num[mask] / den[mask]
    return np.clip(out, -RATIO_CLIP, RATIO_CLIP)


def _parse_name(name: str) -> tuple[str, int, int]:
    for op, kind in (("abs(", "absdiff"),):
        if name.startswith(op) and name.endswith(")"):
            body = name[len(op):-1]
            a, b = body.split("-", 1)
            return kind, DESC_NAMES.index(a), DESC_NAMES.index(b)
    if "*" in name:
        a, b = name.split("*", 1)
        return "product", DESC_NAMES.index(a), DESC_NAMES.index(b)
    if "+" in name:
        a, b = name.split("+", 1)
        return "sum", DESC_NAMES.index(a), DESC_NAMES.index(b)
    if "/" in name:
        a, b = name.split("/", 1)
        return "ratio", DESC_NAMES.index(a), DESC_NAMES.index(b)
    raise ValueError(f"unknown pair feature {name}")


def pair_matrix(
    drug_values: np.ndarray,
    food_values: np.ndarray,
    drug_index: np.ndarray,
    food_index: np.ndarray,
    names: Sequence[str] | None = None,
    chunk_size: int = 5000,
    dtype=np.float32,
):
    """Yield (start_row, chunk) matrices for the requested pair features."""
    names = list(ALL_PAIR_NAMES if names is None else names)
    use_all = names == ALL_PAIR_NAMES
    parsed = None if use_all else [_parse_name(n) for n in names]
    for start in range(0, len(drug_index), chunk_size):
        di = drug_index[start:start + chunk_size]
        fi = food_index[start:start + chunk_size]
        d = drug_values[di]
        f = food_values[fi]
        out = np.empty((len(di), len(names)), dtype=dtype)
        if use_all:
            for i in range(N_DESC):
                out[:, i * N_DESC:(i + 1) * N_DESC] = d[:, [i]] * f
            base = N_DESC * N_DESC
            out[:, base:base + N_DESC] = d + f
            out[:, base + N_DESC:base + 2 * N_DESC] = np.abs(d - f)
            out[:, base + 2 * N_DESC:base + 3 * N_DESC] = _ratio(d, f)
        else:
            for k, (kind, i, j) in enumerate(parsed):
                if kind == "product":
                    out[:, k] = d[:, i] * f[:, j]
                elif kind == "sum":
                    out[:, k] = d[:, i] + f[:, j]
                elif kind == "absdiff":
                    out[:, k] = np.abs(d[:, i] - f[:, j])
                else:
                    out[:, k] = _ratio(d[:, i], f[:, j])
        yield start, out


def pair_frame_for_ids(
    drug_df: pd.DataFrame,
    food_df: pd.DataFrame,
    drug_ids: Sequence,
    food_ids: Sequence,
    names: Sequence[str] | None = None,
    chunk_size: int = 5000,
    dtype=np.float32,
) -> pd.DataFrame:
    """Build features for aligned drug/food ID lists (not the full cross product)."""
    names = list(ALL_PAIR_NAMES if names is None else names)
    drug_df = drug_df.set_index("molecule_id")
    food_df = food_df.set_index("molecule_id")
    dvals = drug_df.loc[list(drug_ids), DESC_NAMES].to_numpy(dtype=np.float64)
    fvals = food_df.loc[list(food_ids), DESC_NAMES].to_numpy(dtype=np.float64)
    parts = []
    for _, chunk in pair_matrix(
        dvals, fvals, np.arange(len(drug_ids)), np.arange(len(food_ids)),
        names=names, chunk_size=chunk_size, dtype=dtype,
    ):
        parts.append(chunk)
    return pd.DataFrame(np.vstack(parts), columns=names)
