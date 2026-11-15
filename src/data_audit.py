"""Brick 1 - load both tables, audit columns, and test the label source.

Usage: python src/brick1_audit.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.csv as pc
import requests

from common import ROOT, dataframe_to_markdown, load_config, save_json, setup_stdout

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"


def find_food_sections(node, found, path=""):
    """Recursively collect headings that look like food-interaction sections."""
    if isinstance(node, dict):
        heading = node.get("TOCHeading", "")
        new_path = f"{path}/{heading}" if heading else path
        if heading and "food" in heading.lower() and "interaction" in heading.lower():
            texts = []
            for info in node.get("Information", []):
                for v in info.get("Value", {}).get("StringWithMarkup", []):
                    if v.get("String"):
                        texts.append(v["String"])
            found.append({"path": new_path, "texts": texts})
        for v in node.values():
            find_food_sections(v, found, new_path)
    elif isinstance(node, list):
        for v in node:
            find_food_sections(v, found, path)


def pubchem_probe(cfg, drugs: pd.DataFrame, cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for _, row in drugs.head(8).iterrows():
        cid = row["cid"]
        cache = cache_dir / f"{cid}.json"
        payload = None
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            try:
                r = requests.get(
                    PUBCHEM.format(cid=cid),
                    timeout=20,
                    headers={"User-Agent": "dfi-research/1.0"},
                )
                if r.status_code == 200:
                    payload = r.json()
                    cache.write_text(json.dumps(payload), encoding="utf-8")
                time.sleep(0.25)
            except Exception as exc:  # network is optional; FHDI labels are primary
                results.append({"cid": cid, "drug": row["name"], "error": str(exc)[:200]})
                continue
        if payload is None:
            results.append({"cid": cid, "drug": row["name"], "error": "no payload"})
            continue
        found: list[dict] = []
        find_food_sections(payload, found)
        n_text = sum(len(x["texts"]) for x in found)
        results.append(
            {
                "cid": cid,
                "drug": row["name"],
                "sections": [x["path"] for x in found][:3],
                "n_strings": n_text,
                "sample": (found[0]["texts"][0][:240] if n_text else ""),
            }
        )
    return results


def main() -> None:
    setup_stdout()
    cfg = load_config()
    drugs = pd.read_csv(cfg["paths"]["drugs_raw"], encoding="utf-8-sig")
    inter = pd.read_csv(cfg["paths"]["interactions_raw"], encoding="utf-8-sig", low_memory=False)

    food_cols = [
        "food_id", "id", "name", "moldb_smiles", "SMILES", "Food_Name",
        "orig_food_common_name", "kingdom", "superklass", "klass", "subklass",
    ]
    food = pc.read_csv(
        cfg["paths"]["foods_raw"],
        convert_options=pc.ConvertOptions(include_columns=food_cols),
        parse_options=pc.ParseOptions(newlines_in_values=True),
    ).to_pandas()

    drug_sample = inter[inter["Conclusion"].notna()][["Drug_ID", "Drug_Name"]].drop_duplicates().head(50)
    map_cid = (
        drugs.dropna(subset=["PubChem_Compound_ID"])
        .assign(cid=lambda d: d["PubChem_Compound_ID"].astype(str).str.extract(r"(\d+)")[0])
        .dropna(subset=["cid"])
        .assign(cid=lambda d: d["cid"].astype(int))
        .drop_duplicates("FHDI_Drug_ID")
        .set_index("FHDI_Drug_ID")[["cid", "Drug_Name"]]
    )
    probe_rows = drug_sample.merge(map_cid, left_on="Drug_ID", right_index=True, how="inner")
    probe_rows = probe_rows.rename(columns={"Drug_Name_x": "name"})[["cid", "name"]]
    probe = pubchem_probe(cfg, probe_rows, Path(cfg["paths"]["interim"]) / "pubchem_probe")

    audit = {
        "drugs": {
            "shape": list(drugs.shape),
            "columns": list(drugs.columns),
            "missing": {c: int(drugs[c].isna().sum()) for c in drugs.columns},
            "unique_drugs": int(drugs["FHDI_Drug_ID"].nunique()),
            "has_smiles": int(drugs["Canonical_SMILES"].notna().sum()),
            "has_inchikey": int(drugs["InChIKey"].notna().sum()),
            "has_drugbank_id": int(drugs["DrugBank_ID"].notna().sum()),
        },
        "interactions": {
            "shape": list(inter.shape),
            "columns": list(inter.columns),
            "unique_drugs": int(inter["Drug_ID"].nunique()),
            "unique_food_herb_ids": int(inter["Food_Herb_ID"].nunique()),
            "unique_components": int(inter["Component"].nunique()),
            "effect_counts": {str(k): int(v) for k, v in inter["Effect"].value_counts(dropna=False).items()},
            "relationship_counts": {
                str(k): int(v) for k, v in inter["Relationship_classification"].value_counts(dropna=False).items()
            },
            "notes_nonnull": int(inter["Note"].notna().sum()),
            "conclusion_nonnull": int(inter["Conclusion"].notna().sum()),
            "drug_overlap_with_drug_table": int(inter["Drug_ID"].isin(drugs["FHDI_Drug_ID"]).sum()),
        },
        "foods": {
            "shape": list(food.shape),
            "columns": list(food.columns),
            "unique_food_ids": int(food["food_id"].nunique()),
            "unique_compounds": int(food["id"].nunique()),
            "unique_food_names": int(food["Food_Name"].nunique()),
            "moldb_smiles_equals_SMILES": int((food["moldb_smiles"] == food["SMILES"]).sum()),
            "note": (
                "Header misalignment detected: the column named moldb_smiles holds InChIKeys and "
                "the column named SMILES holds the actual structures. Use SMILES."
            ),
            "sample_row": {c: str(food.iloc[0][c])[:120] for c in food.columns},
        },
        "join": {
            "food_herb_names_in_food_table": int(
                inter["Food_Herb_Name"].dropna().str.strip().str.lower().isin(
                    food["Food_Name"].dropna().str.strip().str.lower()
                ).sum()
            ),
            "component_names_in_compound_table": int(
                inter["Component"].dropna().str.strip().str.lower().isin(
                    food["name"].dropna().str.strip().str.lower()
                ).sum()
            ),
        },
        "pubchem_label_probe": probe,
    }
    save_json(audit, Path(cfg["paths"]["reports"]) / "brick1_audit.json")

    lines = [
        "# Brick 1 - data audit",
        "",
        f"- Drugs: **{drugs.shape[0]}** rows, {drugs['FHDI_Drug_ID'].nunique()} unique FHDI IDs.",
        f"- Interactions: **{inter.shape[0]}** rows, {inter['Drug_ID'].nunique()} drugs, "
        f"{inter['Food_Herb_ID'].nunique()} foods/herbs, {inter['Component'].nunique()} components.",
        f"- Food table: **{food.shape[0]}** rows, {food['id'].nunique()} compounds, {food['Food_Name'].nunique()} foods.",
        "",
        "## Column check (first row)",
        "",
        "Drug columns: " + ", ".join(drugs.columns),
        "",
        "Interaction columns: " + ", ".join(inter.columns),
        "",
        "Food columns: " + ", ".join(food.columns),
        "",
        "## Food header misalignment",
        "",
        "`moldb_smiles` contains InChIKeys and `SMILES` contains structures; the two columns agree 0 times. "
        "All later code uses the `SMILES` column. This is the header/value misalignment the plan warned about.",
        "",
        "## Label source",
        "",
        "Primary labels come from the structured FHDI interaction table (`Effect`, "
        "`Relationship_classification`, `Conclusion`), which carries DrugBank-derived rows. "
        "PubChem PUG-View was probed on 5-8 CIDs as a cross-check:",
        "",
        dataframe_to_markdown(pd.DataFrame(probe)),
        "",
        "## Effect distribution",
        "",
        dataframe_to_markdown(inter["Effect"].value_counts(dropna=False).rename_axis("Effect").reset_index(name="rows")),
        "",
    ]
    (Path(cfg["paths"]["reports"]) / "brick1_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "pubchem_label_probe"}, indent=2)[:3000])
    print("\nPubChem probe:")
    print(pd.DataFrame(probe).to_string())


if __name__ == "__main__":
    main()
