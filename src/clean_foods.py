"""Brick 3 - clean food compounds, build compound-food links, Tanimoto dedupe.

Parses each unique compound once (13k) rather than every food-link row (838k).

Usage: python src/brick3_clean_foods.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.csv as pc
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from common import dataframe_to_markdown, load_config, norm_name, save_json, setup_stdout

RDLogger.DisableLog("rdApp.*")

FOOD_COLS = [
    "food_id", "id", "name", "moldb_smiles", "SMILES", "Food_Name",
    "orig_food_common_name", "kingdom", "superklass", "klass", "subklass",
]


def is_inorganic(mol: Chem.Mol, metals: set[str]) -> bool:
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    if symbols & metals:
        return True
    return "C" not in symbols


def main() -> None:
    setup_stdout()
    cfg = load_config()
    protected = {norm_name(x) for x in cfg["data"]["protected_compounds"]}
    metals = {m.strip() for group in cfg["data"]["metals"] for m in group.split(",")}

    food = pc.read_csv(
        cfg["paths"]["foods_raw"],
        convert_options=pc.ConvertOptions(include_columns=FOOD_COLS),
        parse_options=pc.ParseOptions(newlines_in_values=True),
    ).to_pandas()
    funnel = {"rows_loaded": len(food)}

    # Link table is built from every row before any structural filtering.
    links_raw = (
        food[["id", "food_id", "Food_Name"]]
        .rename(columns={"id": "compound_id", "Food_Name": "food_name"})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    funnel["raw_links"] = len(links_raw)

    # One row per compound; groupby.first() takes the first non-null value.
    uniq = food.groupby("id", sort=True, as_index=False).first()
    funnel["unique_compounds"] = len(uniq)
    funnel["moldb_smiles_equals_SMILES"] = int((uniq["moldb_smiles"].astype(str) == uniq["SMILES"].astype(str)).sum())

    uniq["smiles_src"] = uniq["SMILES"].astype(str).str.strip()
    empty = uniq["smiles_src"].isin(["", "nan", "None"])
    funnel["no_smiles"] = int(empty.sum())
    uniq = uniq[~empty].copy()

    records = []
    parse_fail = 0
    for rec in uniq.itertuples(index=False):
        mol = Chem.MolFromSmiles(rec.smiles_src)
        if mol is None:
            parse_fail += 1
            continue
        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            inchikey = Chem.MolToInchiKey(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, cfg["data"]["morgan_radius"], nBits=cfg["data"]["morgan_bits"]
            )
        except Exception:
            parse_fail += 1
            continue
        if not inchikey:
            parse_fail += 1
            continue
        records.append(
            {
                "compound_id": int(rec.id),
                "name": str(rec.name),
                "smiles": smiles,
                "inchikey": inchikey,
                "orig_food": str(rec.orig_food_common_name),
                "kingdom": rec.kingdom,
                "superklass": rec.superklass,
                "klass": rec.klass,
                "subklass": rec.subklass,
                "fp": fp,
            }
        )
    funnel["parse_failures"] = parse_fail
    funnel["parsed_unique_compounds"] = len(records)
    parsed = pd.DataFrame(records)

    before = len(parsed)
    parsed = parsed.sort_values("compound_id").drop_duplicates("inchikey", keep="first").reset_index(drop=True)
    funnel["duplicate_inchikey_removed"] = int(before - len(parsed))

    parsed["is_protected"] = parsed["name"].map(lambda x: norm_name(x) in protected)
    parsed.loc[parsed["name"].map(lambda x: "vitamin k" in norm_name(x)), "is_protected"] = True
    funnel["protected_present_before_similarity"] = int(parsed["is_protected"].sum())

    kept_rows = []
    kept_fps = []
    removed_similarity = 0
    for rec in parsed.sort_values("compound_id").itertuples(index=False):
        if rec.is_protected or not kept_fps:
            keep = True
        else:
            sims = DataStructs.BulkTanimotoSimilarity(rec.fp, kept_fps)
            keep = max(sims) < cfg["data"]["tanimoto_cutoff"]
        if keep:
            kept_rows.append(rec)
            kept_fps.append(rec.fp)
        else:
            removed_similarity += 1
    funnel["near_duplicate_tanimoto_removed"] = removed_similarity
    funnel["kept_after_tanimoto"] = len(kept_rows)

    clean = pd.DataFrame(
        [
            {
                "compound_id": r.compound_id,
                "name": r.name,
                "smiles": r.smiles,
                "inchikey": r.inchikey,
                "is_organic": not is_inorganic(Chem.MolFromSmiles(r.smiles), metals),
                "is_protected": bool(r.is_protected),
                "kingdom": r.kingdom,
                "superklass": r.superklass,
                "klass": r.klass,
                "subklass": r.subklass,
            }
            for r in kept_rows
        ]
    ).reset_index(drop=True)

    funnel["organic_compounds"] = int(clean["is_organic"].sum())
    funnel["inorganic_compounds"] = int((~clean["is_organic"]).sum())
    funnel["protected_present_final"] = int(clean["is_protected"].sum())
    kept_ids = set(clean["compound_id"])
    links = links_raw[links_raw["compound_id"].isin(kept_ids)].drop_duplicates().reset_index(drop=True)
    funnel["final_links"] = len(links)

    out_foods = Path(cfg["paths"]["interim"]) / "foods_clean.parquet"
    out_links = Path(cfg["paths"]["interim"]) / "food_links.parquet"
    clean.to_parquet(out_foods, index=False)
    links.to_parquet(out_links, index=False)
    save_json(funnel, Path(cfg["paths"]["reports"]) / "brick3_funnel.json")

    protected_found = clean.loc[clean["is_protected"], "name"].tolist()
    missing_protected = sorted(protected - {norm_name(x) for x in protected_found})
    lines = [
        "# Brick 3 - food cleaning funnel",
        "",
        dataframe_to_markdown(pd.DataFrame([funnel]).T.rename(columns={0: "count"})),
        "",
        f"Protected compounds found: {', '.join(protected_found) if protected_found else 'none'}.",
        "",
        f"Protected names not found in the food table: {', '.join(missing_protected) if missing_protected else 'none'}.",
        "",
        f"Outputs: `{out_foods}` ({len(clean)} compounds) and `{out_links}` ({len(links)} links).",
        "",
        "Inorganic and metal-containing compounds are kept in the table but flagged `is_organic=False`; "
        "they are excluded from model training and handled by a name rule at inference.",
        "",
        "Note: the file header is shifted, so `moldb_smiles` contains InChIKeys. The actual structures "
        "are in the `SMILES` column, which is what this brick uses.",
    ]
    (Path(cfg["paths"]["reports"]) / "brick3_funnel.md").write_text("\n".join(lines), encoding="utf-8")
    print(clean.head(5).to_string())
    print(funnel)
    print("protected found:", protected_found)
    print("missing protected:", missing_protected)


if __name__ == "__main__":
    main()
