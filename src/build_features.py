"""Brick 5 - per-molecule MOE descriptors, pair builder and library parity check.

Usage: python src/brick5_features.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PyBioMed.Pyinteraction import CalculateInteraction2, CalculateInteraction3

from common import dataframe_to_markdown, load_config, save_json, setup_stdout
from features import (
    ALL_PAIR_NAMES, DESC_NAMES, N_DESC, compute_moe_table, pair_matrix,
)


def dedupe_descriptors(df: pd.DataFrame, id_col: str = "molecule_id") -> tuple[pd.DataFrame, list]:
    hashes = pd.util.hash_pandas_object(df[DESC_NAMES].round(6), index=False)
    dup = hashes.duplicated(keep="first")
    removed = df.loc[dup, id_col].tolist()
    return df.loc[~dup].reset_index(drop=True), removed


def verify_against_library(drug_moe, food_moe, n=500, seed=42):
    rng = np.random.default_rng(seed)
    drug_ids = drug_moe["molecule_id"].to_numpy()
    food_ids = food_moe["molecule_id"].to_numpy()
    d_idx = rng.integers(0, len(drug_ids), n)
    f_idx = rng.integers(0, len(food_ids), n)
    dvals = drug_moe[DESC_NAMES].to_numpy()
    fvals = food_moe[DESC_NAMES].to_numpy()

    max_prod_lib = 0.0
    max_sum_lib = 0.0
    max_prod_direct = 0.0
    for start, chunk in pair_matrix(dvals, fvals, d_idx, f_idx, names=None, chunk_size=100, dtype=np.float64):
        for k in range(len(chunk)):
            a = {name: float(dvals[d_idx[start + k], i]) for i, name in enumerate(DESC_NAMES)}
            b = {name: float(fvals[f_idx[start + k], i]) for i, name in enumerate(DESC_NAMES)}
            ref2 = CalculateInteraction2(a, b)
            ref3 = CalculateInteraction3(a, b)
            for i, di in enumerate(DESC_NAMES):
                for j, fj in enumerate(DESC_NAMES):
                    lib = ref2[f"{di}*{fj}"]
                    ours = float(chunk[k, i * N_DESC + j])
                    max_prod_lib = max(max_prod_lib, abs(ours - lib))
                    direct = a[di] * b[fj]
                    max_prod_direct = max(max_prod_direct, abs(ours - direct))
                max_sum_lib = max(max_sum_lib, abs(float(chunk[k, N_DESC * N_DESC + i]) - ref3[f"{di}+{di}"]))
    return {
        "pairs_checked": int(n),
        "columns_ours": len(ALL_PAIR_NAMES),
        "max_abs_diff_vs_pybio_products": float(max_prod_lib),
        "max_abs_diff_vs_pybio_sums": float(max_sum_lib),
        "max_abs_diff_vs_direct_arithmetic": float(max_prod_direct),
        "library_rounds_to_3dp": True,
        "pass_library_tolerance_1e-3": bool(max(max_prod_lib, max_sum_lib) < 1e-3),
        "pass_direct_tolerance_1e-9": bool(max_prod_direct < 1e-9),
    }


def main() -> None:
    setup_stdout()
    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])

    drugs = pd.read_parquet(interim / "drugs_clean.parquet")
    foods = pd.read_parquet(interim / "foods_clean.parquet")
    organic = foods.loc[foods["is_organic"]].copy()

    t0 = time.time()
    drug_path = interim / "drug_moe.parquet"
    food_path = interim / "food_moe.parquet"
    if drug_path.exists() and food_path.exists():
        drug_moe = pd.read_parquet(drug_path)
        food_moe = pd.read_parquet(food_path)
        drug_fail = food_fail = []
    else:
        drug_moe, drug_fail = compute_moe_table(drugs["drug_id"], drugs["smiles"])
        drug_moe.to_parquet(drug_path, index=False)
        food_moe, food_fail = compute_moe_table(organic["compound_id"], organic["smiles"])
        food_moe.to_parquet(food_path, index=False)

    drug_kept, drug_dups = dedupe_descriptors(drug_moe)
    food_kept, food_dups = dedupe_descriptors(food_moe)
    drug_kept.to_parquet(drug_path, index=False)
    food_kept.to_parquet(food_path, index=False)

    verify = verify_against_library(drug_kept, food_kept, n=500, seed=cfg["seed"])
    n_finite = bool(
        np.isfinite(drug_kept[DESC_NAMES].to_numpy()).all()
        and np.isfinite(food_kept[DESC_NAMES].to_numpy()).all()
    )

    schema = {
        "n_molecule_descriptors": N_DESC,
        "descriptor_names": DESC_NAMES,
        "pair_columns": len(ALL_PAIR_NAMES),
        "terms": {"products": N_DESC * N_DESC, "sums": N_DESC, "abs_diffs": N_DESC, "ratios": N_DESC},
        "first_columns": ALL_PAIR_NAMES[:5],
        "last_columns": ALL_PAIR_NAMES[-5:],
    }
    save_json(schema, Path(cfg["paths"]["reports"]) / "pair_feature_schema.json")
    save_json(
        {
            "drug_molecules": int(len(drug_kept)),
            "food_molecules": int(len(food_kept)),
            "drug_descriptor_failures": drug_fail,
            "food_descriptor_failures": food_fail,
            "drug_duplicate_vectors_removed": drug_dups,
            "food_duplicate_vectors_removed": food_dups,
            "all_values_finite": n_finite,
            "verification": verify,
            "seconds": round(time.time() - t0, 1),
        },
        Path(cfg["paths"]["reports"]) / "brick5_features.json",
    )

    lines = [
        "# Brick 5 - feature extraction",
        "",
        f"- MOE descriptors per molecule: **{N_DESC}**.",
        f"- Pair columns: **{len(ALL_PAIR_NAMES):,}** = "
        f"{N_DESC*N_DESC:,} products + {N_DESC} sums + {N_DESC} abs-diffs + {N_DESC} ratios.",
        f"- Drug descriptors kept: {len(drug_kept)} (unique vectors); duplicate vectors removed: {len(drug_dups)}.",
        f"- Food descriptors kept: {len(food_kept)} (unique vectors); duplicate vectors removed: {len(food_dups)}.",
        f"- All values finite: {n_finite}.",
        "",
        "## Verification on 500 random pairs",
        "",
        dataframe_to_markdown(pd.DataFrame([verify]).T.rename(columns={0: "value"})),
        "",
        "PyBioMed rounds interaction values to 3 decimals, so parity is judged at 1e-3; "
        "the unrounded builder is also checked against direct arithmetic at 1e-9 (float64 verification). "
        "Production pair matrices are stored as float32.",
        "",
        f"Descriptor and pair-feature failures are listed in `brick5_features.json`.",
    ]
    (Path(cfg["paths"]["reports"]) / "brick5_features.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
