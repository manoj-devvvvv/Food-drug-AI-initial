"""Brick 2 - clean drugs: SMILES parse, salt strip, InChIKey14 dedupe.

Usage: python src/brick2_clean_drugs.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from common import dataframe_to_markdown, first_int, load_config, save_json, setup_stdout

RDLogger.DisableLog("rdApp.*")


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def main() -> None:
    setup_stdout()
    cfg = load_config()
    cols = [
        "FHDI_Drug_ID", "Drug_Name", "Synonyms", "Drug_Type", "Canonical_SMILES",
        "InChIKey", "DrugBank_ID", "PubChem_Compound_ID",
    ]
    df = pd.read_csv(cfg["paths"]["drugs_raw"], encoding="utf-8-sig", usecols=cols)
    funnel = {"loaded": len(df)}

    db_from_syn = df["Synonyms"].astype(str).str.extract(r"\b(DB\d{5})\b", expand=False)
    df["db_id"] = df["DrugBank_ID"].fillna(db_from_syn)
    funnel["with_drugbank_id"] = int(df["db_id"].notna().sum())
    funnel["db_id_recovered_from_synonyms"] = int(
        (df["DrugBank_ID"].isna() & df["db_id"].notna()).sum()
    )

    rows = []
    parse_fail = 0
    for rec in df.itertuples(index=False):
        if not isinstance(rec.Canonical_SMILES, str) or not rec.Canonical_SMILES.strip():
            parse_fail += 1
            continue
        mol = Chem.MolFromSmiles(rec.Canonical_SMILES)
        if mol is None:
            parse_fail += 1
            continue
        mol = largest_fragment(mol)
        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            parse_fail += 1
            continue
        if not inchikey:
            parse_fail += 1
            continue
        rows.append(
            {
                "drug_id": rec.FHDI_Drug_ID,
                "cid": first_int(rec.PubChem_Compound_ID),
                "db_id": rec.db_id if isinstance(rec.db_id, str) else None,
                "name": rec.Drug_Name,
                "drug_type": rec.Drug_Type,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchikey14": inchikey[:14],
            }
        )
    funnel["parse_failures_or_empty"] = parse_fail
    clean = pd.DataFrame(rows)
    funnel["parsed"] = len(clean)

    clean["_has_db"] = clean["db_id"].notna().astype(int)
    clean["_cid"] = clean["cid"].fillna(10**12)
    clean = clean.sort_values(["inchikey14", "_has_db", "_cid"], ascending=[True, False, True])
    dup_mask = clean.duplicated("inchikey14", keep="first")
    funnel["duplicate_inchikey14_removed"] = int(dup_mask.sum())
    clean = clean[~dup_mask].drop(columns=["_has_db", "_cid"]).reset_index(drop=True)

    bad = 0
    for s in clean["smiles"]:
        if Chem.MolFromSmiles(s) is None:
            bad += 1
    if bad:
        raise SystemExit(f"{bad} kept SMILES failed to re-parse")
    funnel["final_drugs"] = len(clean)
    funnel["final_with_db_id"] = int(clean["db_id"].notna().sum())

    out = Path(cfg["paths"]["interim"]) / "drugs_clean.parquet"
    clean.to_parquet(out, index=False)
    save_json(funnel, Path(cfg["paths"]["reports"]) / "brick2_funnel.json")

    lines = [
        "# Brick 2 - drug cleaning funnel",
        "",
        dataframe_to_markdown(pd.DataFrame([funnel]).T.rename(columns={0: "count"})),
        "",
        f"Output: `{out}` with columns {list(clean.columns)}.",
        "",
        f"All {len(clean)} kept SMILES re-parse in RDKit.",
    ]
    (Path(cfg["paths"]["reports"]) / "brick2_funnel.md").write_text("\n".join(lines), encoding="utf-8")
    print(clean.head(5).to_string())
    print(funnel)


if __name__ == "__main__":
    main()
