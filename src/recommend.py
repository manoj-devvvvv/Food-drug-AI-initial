"""Recommendation engine: drug + food -> class probabilities and a sentence.

Usage:
    python src/recommend.py "warfarin" "ethanol"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from PyBioMed.PyMolecule.moe import GetMOE

from common import ROOT, load_config, load_json, norm_name
from features import DESC_NAMES, pair_matrix

RDLogger.DisableLog("rdApp.*")

SENTENCES = {
    1: "A could be taken with food containing B.",
    2: "A may be taken with food containing B.",
    0: "A should not be taken with food containing B.",
}
DISCLAIMER = "Research tool; confirm with a physician or pharmacist."
ALIASES = {
    "alcohol": "ethanol", "furanocoumarin": "bergamottin",
    "vitamin k1 2,3 epoxide": "phylloquinone", "vitamin k1": "phylloquinone",
    "grapefruit": "bergamottin", "grapefruit juice": "bergamottin",
    "dairy": "calcium", "st john's wort": "hyperforin", "st johns wort": "hyperforin",
}
FOOD_MARKERS_FALLBACK = {
    "grapefruit": ["bergamottin", "naringin", "bergapten"],
    "grape": ["(z)-resveratrol", "catechin"],
    "cranberry": ["(z)-resveratrol", "catechin", "proanthocyanidins", "cyanidin"],
    "pomegranate": ["cyanidin", "ellagic acid"],
    "garlic": ["allicin", "dimethyl disulfide"],
    "ginger": ["[4]-gingerol", "methylgingerol"],
    "turmeric": ["curcumin"],
    "st john": ["hyperforin", "hypericin"],
    "milk": ["calcium"],
    "coffee": ["caffeine"],
    "tea": ["caffeine", "catechin"],
    "spinach": ["phylloquinone"],
    "kale": ["phylloquinone"],
    "broccoli": ["phylloquinone", "sulforaphane"],
    "soybean": ["genistein", "daidzein"],
    "licorice": ["glycyrrhizic acid", "glycyrrhizin"],
    "alcohol": ["ethanol"],
    "ethanol": ["ethanol"],
}
SMILES_FALLBACK = {
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "dextrose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}
RULE_FOOD_KEYS = {
    "bergamottin": "grapefruit", "naringin": "grapefruit", "naringenin": "grapefruit",
    "bergapten": "grapefruit", "furanocoumarin": "grapefruit",
    "hyperforin": "st john's wort", "hypericin": "st john's wort",
    "phylloquinone": "vitamin k", "dimethyl disulfide": "garlic", "allicin": "garlic",
}
NUTRIENTS = {"calcium", "iron", "magnesium", "zinc", "aluminium", "aluminum"}
CYP3A4_SUBSTRATES = (
    "midazolam", "triazolam", "nisoldipine", "felodipine", "buspirone",
    "cyclosporine", "tacrolimus", "sirolimus", "sildenafil", "tadalafil", "vardenafil",
)
MAO_INHIBITORS = (
    "phenelzine", "tranylcypromine", "isocarboxazid", "selegiline", "moclobemide", "linezolid",
)


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def moe_vector(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        d = GetMOE(mol)
    except Exception:
        return None, None
    arr = np.array([d[k] for k in DESC_NAMES], dtype=np.float64)
    if not np.isfinite(arr).all():
        return None, None
    return mol, arr


def is_inorganic(mol, metals: set[str]) -> bool:
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    return bool(symbols & metals) or "C" not in symbols


class DFIRecommender:
    def __init__(self, mode: str = "S"):
        self.cfg = load_config()
        self.mode = mode
        self.models = Path(self.cfg["paths"]["models"])
        self.interim = Path(self.cfg["paths"]["interim"])

        sel = load_json(self.models / f"selected_features_{mode}.json")
        self.feature_names = [f["name"] for f in sel["features"]]
        import joblib
        import brick8_models
        sys.modules["__main__"].BalancedClassifier = brick8_models.BalancedClassifier
        self.model = joblib.load(self.models / f"best_model_{mode}.joblib")

        self.drugs = pd.read_parquet(self.interim / "drugs_clean.parquet")
        self.foods = pd.read_parquet(self.interim / "foods_clean.parquet")
        self.links = pd.read_parquet(self.interim / "food_links.parquet")
        self.metals = {m.strip() for group in self.cfg["data"]["metals"] for m in group.split(",")}

        markers = self.models / "food_markers.json"
        self.food_markers = load_json(markers) if markers.exists() else FOOD_MARKERS_FALLBACK
        train_drugs = self.models / f"train_drugs_{mode}.json"
        self.train_drugs = set(load_json(train_drugs)) if train_drugs.exists() else set()

        # Dataset evidence layer: non-weak labels seen during training.
        self.known: dict[tuple[int, int], int] = {}
        labeled = self.interim / "pairs_labeled.parquet"
        if labeled.exists():
            pairs = pd.read_parquet(labeled)
            pairs = pairs[(pairs["mode"] == mode) & (~pairs["weak"])]
            self.known = {
                (str(d), int(c)): int(l)
                for d, c, l in pairs[["drug_id", "compound_id", "label"]].itertuples(index=False)
            }

        self.drug_by_name = {}
        for rid, name in zip(self.drugs["drug_id"], self.drugs["name"]):
            self.drug_by_name.setdefault(norm_name(name), []).append(rid)
        self.drug_name = self.drugs.set_index("drug_id")["name"].to_dict()
        self.drug_smiles = self.drugs.set_index("drug_id")["smiles"].to_dict()

        self.compound_by_name = {}
        for cid, name in zip(self.foods["compound_id"], self.foods["name"]):
            self.compound_by_name.setdefault(norm_name(name), []).append(int(cid))
        self.compound_row = self.foods.set_index("compound_id").to_dict("index")

        self.food_to_compounds = {}
        for fid, fname, cid in self.links[["food_id", "food_name", "compound_id"]].itertuples(index=False):
            self.food_to_compounds.setdefault(norm_name(fname), []).append(int(cid))
        self.food_names = {norm_name(x): x for x in self.links["food_name"].unique()}
        self.compound_freq = (
            self.links.drop_duplicates(["food_id", "compound_id"])
            .groupby("compound_id")["food_id"].nunique()
            .to_dict()
        )

        # Applicability domain: nearest training drug by Morgan similarity.
        self.train_fps = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
            for s in self.drugs["smiles"]
        ]

    def _resolve_drug(self, query: str):
        key = norm_name(query)
        if key in self.drug_by_name:
            did = self.drug_by_name[key][0]
            return did, self.drug_name[did], self.drug_smiles[did]
        smi = canonical_smiles(query)
        if smi:
            return None, query, smi
        return None, None, None

    def _canonical_food_key(self, query: str) -> str:
        key = norm_name(query)
        key = ALIASES.get(key, key)
        for suffix in (" juice", " raw", " fresh", " dried", " extract", " powder", " tea"):
            if key.endswith(suffix):
                key = key[: -len(suffix)].strip()
                break
        return key

    def _marker_ids(self, key: str) -> list[int]:
        for mkey in sorted(self.food_markers, key=len, reverse=True):
            if mkey and mkey in key:
                ids: list[int] = []
                for marker in self.food_markers[mkey]:
                    hits = self.compound_by_name.get(norm_name(marker), [])
                    if not hits:
                        hits = [
                            cid
                            for cname, cids in self.compound_by_name.items()
                            if norm_name(marker) and norm_name(marker) in cname
                            for cid in cids
                        ][:3]
                    ids.extend(hits)
                if ids:
                    return sorted(set(ids))
        return []

    def _food_link_ids(self, key: str) -> list[int]:
        cids = list(self.food_to_compounds.get(key, []))
        if not cids:
            for fkey, ids in self.food_to_compounds.items():
                if fkey and re.search(rf"\b{re.escape(fkey)}\b", key):
                    cids.extend(ids)
        return sorted(set(cids))

    def _compound_candidates(self, query: str) -> list[tuple[int | None, str, str, str | None]]:
        key = ALIASES.get(norm_name(query), norm_name(query))
        if key in self.compound_by_name:
            cid = self.compound_by_name[key][0]
            row = self.compound_row[cid]
            return [(cid, row["name"], row["smiles"], None)]

        food_key = self._canonical_food_key(query)
        marker_ids = self._marker_ids(food_key)
        if marker_ids:
            cids = marker_ids
            note = f"food-level match via {self.food_names.get(food_key, query)} (marker compounds)"
        else:
            cids = [
                c for c in self._food_link_ids(food_key)
                if self.compound_freq.get(c, 999) <= 20
            ]
            cids = sorted(cids, key=lambda c: (self.compound_freq.get(c, 999), c))[:10]
            note = f"food-level match via {self.food_names.get(food_key, query)} (specific compounds)"

        out = []
        for cid in cids[:10]:
            row = self.compound_row.get(int(cid))
            if row is not None:
                out.append((int(cid), row["name"], row["smiles"], note))
        if out:
            return out
        smi = SMILES_FALLBACK.get(key) or canonical_smiles(query)
        if smi:
            return [(None, query, smi, None)]
        return []

    def _rule_food_key(self, food_key: str) -> str:
        if food_key in RULE_FOOD_KEYS:
            return RULE_FOOD_KEYS[food_key]
        return food_key

    def _domain(self, drug_smiles: str) -> float:
        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None:
            return 0.0
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return float(max(DataStructs.BulkTanimotoSimilarity(fp, self.train_fps)))

    def _rule_layer(self, drug_name: str, food_key: str):
        d = norm_name(drug_name)
        f = norm_name(food_key)

        def drug_has(*tokens: str) -> bool:
            return any(t in d for t in tokens)

        def food_has(*tokens: str) -> bool:
            return any(t in f for t in tokens)

        alcohol = food_has("alcohol", "ethanol", "wine", "beer", "spirits", "liquor")
        grapefruit = food_has("grapefruit", "bergamottin", "naringin", "bergapten", "furanocoumarin")
        cranberry = food_has("cranber")
        vitamin_k = food_has("vitamin k", "phylloquinone", "spinach", "kale", "broccoli", "leafy green")
        st_john = food_has("st john", "hyperforin", "hypericum", "hypericin")

        if drug_has("warfarin", "acenocoumarol", "phenprocoumon"):
            if grapefruit:
                return 0, "Grapefruit inhibits CYP450; the dataset annotates warfarin + grapefruit as avoid and monitor INR."
            if food_has("grape"):
                return 0, "Grape and grapefruit products are flagged with warfarin; keep intake consistent and monitor INR."
            if cranberry:
                return 0, "Cranberry has case reports of raised INR with warfarin; keep intake consistent and monitor INR."
            if vitamin_k:
                return 0, "Vitamin K antagonizes warfarin; keep intake consistent and monitor INR."
            if alcohol:
                return 0, "Alcohol changes warfarin metabolism and bleeding risk; avoid heavy or binge drinking."
            if food_has("garlic", "allicin", "dimethyl disulfide"):
                return 0, "Garlic constituents have case reports of altered INR with warfarin; monitor INR."
        if drug_has("simvastatin", "lovastatin") and grapefruit:
            return 0, "Grapefruit raises statin exposure and the risk of myopathy."
        if drug_has(*CYP3A4_SUBSTRATES) and grapefruit:
            return 0, "Grapefruit inhibits intestinal CYP3A4 and raises exposure of this CYP3A4 substrate."
        if drug_has("metronidazole", "tinidazole") and alcohol:
            return 0, "Alcohol with nitroimidazoles can cause a disulfiram-like reaction."
        if drug_has(*MAO_INHIBITORS) and food_has(
            "tyramine", "cheese", "aged", "cured", "fermented", "soy sauce", "sauerkraut", "wine", "beer"
        ):
            return 0, "Tyramine-rich foods can trigger a hypertensive crisis with MAO inhibitors."
        if drug_has("digoxin") and st_john:
            return 0, "St John's Wort induces P-glycoprotein and lowers digoxin exposure."
        if drug_has("tetracycline", "cycline", "floxacin") and food_has(*NUTRIENTS):
            return 0, "Tetracyclines/quinolones bind calcium, iron and other cations; separate dosing."
        if drug_has("levothyroxine", "thyroxine") and food_has("calcium", "iron"):
            return 0, "Calcium and iron reduce levothyroxine absorption; separate dosing."
        if drug_has("nitroglycerin", "nitrate") and alcohol:
            return 0, "Alcohol can cause severe hypotension with nitrates."
        if drug_has("methotrexate") and alcohol:
            return 0, "Alcohol increases hepatotoxicity risk with methotrexate."
        if drug_has("diazepam") and alcohol:
            return 0, "Alcohol adds CNS depression to benzodiazepines."
        return None, None

    def _score_pair(self, did, dname, dsmiles, dv, cid, cname, csmiles, note):
        cmol, cv = moe_vector(csmiles)
        if cv is None:
            return {"status": "not_supported", "reason": "descriptor computation failed",
                    "drug": dname, "food": cname, "disclaimer": DISCLAIMER}
        if is_inorganic(cmol, self.metals):
            return {"status": "not_supported", "reason": "inorganic compound: structure descriptors do not apply",
                    "drug": dname, "food": cname, "disclaimer": DISCLAIMER}

        known = self.known.get((did, cid)) if did is not None and cid is not None else None
        if known is not None:
            return {
                "status": "evidence", "drug": dname, "food": cname, "predicted_class": known,
                "confidence": None, "probabilities": None,
                "sentence": SENTENCES[known].replace("A", dname).replace("B", cname),
                "reason": "Dataset annotation for this exact drug-compound pair (non-weak label).",
                "note": note, "disclaimer": DISCLAIMER,
            }

        domain = self._domain(dsmiles)
        if domain < 0.3:
            return {"status": "not_supported",
                    "reason": f"drug outside training domain (max Tanimoto {domain:.2f})",
                    "drug": dname, "food": cname, "disclaimer": DISCLAIMER}

        _, X = next(pair_matrix(
            np.vstack([dv]), np.vstack([cv]), np.array([0]), np.array([0]),
            names=self.feature_names, chunk_size=1,
        ))
        proba = self.model.predict_proba(X)[0]
        cls = int(np.argmax(proba))
        conf = float(np.max(proba))

        extra = {}
        if did is not None and self.train_drugs and did not in self.train_drugs:
            extra["extrapolation"] = True
            extra["note"] = "drug not in this mode's training labels; treat as extrapolation"

        if conf < float(self.cfg["models"]["confidence_threshold"]):
            return {
                "status": "uncertain", "drug": dname, "food": cname,
                "predicted_class": cls, "confidence": conf,
                "probabilities": {int(i): float(p) for i, p in enumerate(proba)},
                "sentence": "The model is not confident; ask a pharmacist before acting.",
                "reason": f"top probability {conf:.3f} below {self.cfg['models']['confidence_threshold']}",
                "domain_similarity": domain, "note": note, "disclaimer": DISCLAIMER, **extra,
            }
        return {
            "status": "ok", "drug": dname, "food": cname, "predicted_class": cls,
            "confidence": conf, "probabilities": {int(i): float(p) for i, p in enumerate(proba)},
            "sentence": SENTENCES[cls].replace("A", dname).replace("B", cname),
            "domain_similarity": domain, "note": note, "disclaimer": DISCLAIMER, **extra,
        }

    def recommend(self, drug: str, food: str) -> dict:
        did, dname, dsmiles = self._resolve_drug(drug)
        if dsmiles is None:
            return {"status": "not_supported", "reason": "drug not found or invalid SMILES", "disclaimer": DISCLAIMER}

        food_key = self._canonical_food_key(food)
        rule_class, rule_note = self._rule_layer(dname, self._rule_food_key(food_key))
        if rule_class is not None:
            return {
                "status": "rule", "drug": dname, "food": food,
                "predicted_class": rule_class,
                "sentence": SENTENCES[rule_class].replace("A", dname).replace("B", food),
                "reason": rule_note, "confidence": None, "probabilities": None,
                "disclaimer": DISCLAIMER,
            }

        candidates = self._compound_candidates(food)
        if not candidates:
            return {"status": "not_supported", "reason": "food compound not found or invalid SMILES",
                    "drug": dname, "food": food, "disclaimer": DISCLAIMER}

        dmol, dv = moe_vector(dsmiles)
        if dv is None:
            return {"status": "not_supported", "reason": "descriptor computation failed",
                    "drug": dname, "food": food, "disclaimer": DISCLAIMER}

        results = [
            self._score_pair(did, dname, dsmiles, dv, cid, cname, csmiles, note)
            for cid, cname, csmiles, note in candidates
        ]
        supported = [r for r in results if r["status"] != "not_supported"]
        if not supported:
            return results[0]

        threshold = float(self.cfg["models"]["confidence_threshold"])
        evidence0 = [
            r for r in supported
            if r["status"] == "evidence" and r.get("predicted_class") == 0
        ]
        risky = [
            r for r in supported
            if r.get("predicted_class") == 0 and (r.get("confidence") or 0.0) >= threshold
        ]
        evidence = [r for r in supported if r["status"] == "evidence"]
        pool = evidence0 or risky or evidence or supported
        best = max(
            pool,
            key=lambda r: 1.0 if r["status"] == "evidence" else (r.get("confidence") or 0.0),
        )
        if len(supported) > 1:
            if best.get("predicted_class") is not None:
                best["sentence"] = (
                    SENTENCES[int(best["predicted_class"])]
                    .replace("A", dname).replace("B", food)
                )
            best["note"] = (
                f"food-level match: scored {len(supported)} candidate compounds; "
                "showing the most cautious confident result"
            )
            best["candidates"] = [
                {"compound": r["food"], "class": r.get("predicted_class"),
                 "confidence": r.get("confidence"), "status": r["status"]}
                for r in supported
            ]
        return best


def main():
    r = DFIRecommender("S")
    drug = sys.argv[1] if len(sys.argv) > 1 else "warfarin"
    food = sys.argv[2] if len(sys.argv) > 2 else "ethanol"
    print(r.recommend(drug, food))


if __name__ == "__main__":
    main()
