"""Brick 11 - held-out case-study table for the recommender.

Usage: python src/brick11_recommender.py
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from common import load_config, save_json, setup_stdout
from recommend import DFIRecommender

CASES = [
    ("tetracycline", "calcium", 0, "paper case: chelation"),
    ("methotrexate", "ethanol", 0, "paper case: hepatotoxicity"),
    ("diazepam", "ethanol", 0, "paper case: CNS depression"),
    ("nitroglycerin", "ethanol", 0, "paper case: hypotension"),
    ("digoxin", "hyperforin", 0, "paper case: St John's Wort induction"),
    ("nisoldipine", "bergamottin", 0, "paper case: CYP3A4 inhibition"),
    ("midazolam", "furanocoumarin", 0, "paper case: furanocoumarin class"),
    ("warfarin", "vitamin k1 2,3 epoxide", 0, "paper case: vitamin K antagonism"),
    ("warfarin", "dimethyl disulfide", 0, "paper case: garlic constituent"),
    ("warfarin", "glucose", 2, "control: inert sugar"),
    ("warfarin", "unobtainium", None, "control: compound absent from the food table"),
    ("nitroglycerin", "geranyl rhamnosylglucoside", None, "paper unseen compound"),
    ("warfarin", "dicoumarol", 0, "paper unseen compound"),
    # Clinical probes: well-documented interactions used to check the rule layer.
    ("warfarin", "grapefruit", 0, "probe: grapefruit juice, CYP2C9/CYP3A4"),
    ("warfarin", "grape juice", 0, "probe: user-reported grape juice case"),
    ("simvastatin", "grapefruit", 0, "probe: CYP3A4 inhibition raises statin exposure"),
    ("midazolam", "grapefruit", 0, "probe: CYP3A4 inhibition, classic index substrate"),
    ("metronidazole", "alcohol", 0, "probe: disulfiram-like reaction"),
    ("warfarin", "cranberry", 0, "probe: cranberry potentiation"),
    ("digoxin", "st john's wort", 0, "probe: P-glycoprotein induction"),
]


def fetch_smiles(name, cache_path: Path):
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if name in cache:
        return cache[name]
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name)}/property/CanonicalSMILES/JSON"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "dfi-research/1.0"})
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            smi = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
        else:
            smi = None
    except Exception:
        smi = None
    cache[name] = smi
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return smi


def main() -> None:
    setup_stdout()
    cfg = load_config()
    rec = DFIRecommender("S")
    cache_path = Path(cfg["paths"]["interim"]) / "pubchem_name_cache.json"
    rows = []
    for drug, food, expected, note in CASES:
        result = rec.recommend(drug, food)
        if result.get("status") == "not_supported":
            smi = fetch_smiles(food, cache_path)
            if smi:
                result = rec.recommend(drug, smi)
        pred = result.get("predicted_class")
        status = result.get("status")
        if expected is None:
            verdict = "informational"
        elif status in ("ok", "rule", "evidence") and pred is not None:
            verdict = "pass" if int(pred) == expected else "fail"
        elif status == "uncertain":
            verdict = "abstained"
        else:
            verdict = "not supported"
        prob = result.get("probabilities") or {}
        rows.append({
            "pair": f"{drug} + {food}",
            "expected": expected,
            "predicted": pred,
            "status": status,
            "verdict": verdict,
            "confidence": result.get("confidence"),
            "p_negative": prob.get(0),
            "p_positive": prob.get(1),
            "p_nonsignificant": prob.get(2),
            "sentence": result.get("sentence"),
            "reason": result.get("reason"),
            "note": note,
            "disclaimer": result.get("disclaimer"),
        })
        print(food, "->", result.get("status"), result.get("predicted_class"), result.get("confidence"))

    table = pd.DataFrame(rows)
    reports = Path(cfg["paths"]["reports"])
    table.to_csv(reports / "case_study_table.csv", index=False, encoding="utf-8-sig")
    table.to_markdown(reports / "case_study_table.md", index=False)
    scored = table[table["verdict"].isin(["pass", "fail"])]
    summary = {
        "cases": len(table),
        "scored_cases": int(len(scored)),
        "passed": int((table["verdict"] == "pass").sum()),
        "failed": int((table["verdict"] == "fail").sum()),
        "abstained": int((table["verdict"] == "abstained").sum()),
        "not_supported": int((table["status"] == "not_supported").sum()),
        "informational": int((table["verdict"] == "informational").sum()),
        "summary_note": (
            "The model is not tuned on these pairs. Failures and abstentions are reported as findings. "
            "The two 'paper unseen compound' rows use PubChem SMILES when available and are informational."
        ),
    }
    save_json(summary, reports / "brick11_case_study.json")

    lines = [
        "# Brick 11 - recommender case studies",
        "",
        f"Scored cases: {summary['scored_cases']}; passed: {summary['passed']}; failed: {summary['failed']}; "
        f"abstained: {summary['abstained']}; not supported: {summary['not_supported']}.",
        "",
        "Nine paper case pairs are held out of all training tables. The four controls check the inert-sugar, "
        "absent-compound and unseen-compound paths; seven clinical probes check the curated rule layer "
        "(warfarin/statins/CYP3A4 substrates + grapefruit, nitroimidazoles + alcohol, cranberry, St John's Wort). "
        "Do not tune on these rows.",
        "",
        table[["pair", "expected", "predicted", "status", "verdict", "confidence"]].to_markdown(index=False),
        "",
        f"Always appended: {table['disclaimer'].iloc[0]}",
    ]
    (reports / "brick11_case_study.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\n", summary)


if __name__ == "__main__":
    main()
