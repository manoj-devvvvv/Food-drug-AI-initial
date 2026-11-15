"""Brick 4 - turn FHDI interaction annotations into drug-food pair labels.

Modes:
  P - paper-like: the drug-level label applies to every food compound.
  S - specific: only compounds named by the component/food lexicon are labelled;
      other sampled compounds for the same drug are class 2 and flagged weak.

Usage: python src/brick4_labels.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import dataframe_to_markdown, load_config, norm_name, save_json, setup_stdout

# Paper's sentence rules, kept as auditable regexes.
RULES = {
    0: [
        r"\bavoid\b", r"take .{0,30}separately from (meals|food)",
        r"empty stomach", r"before (a )?meals?",
        r"should not be (taken|used|co-administered)",
        r"do not (take|use|co-administer)", r"contraindicat",
        r"(lower|decrease|reduce|impair|inhibit)[a-z]* .{0,40}(absorption|bioavailability|exposure|concentration)",
        r"limit .{0,30}(intake|consumption)",
        r"serious (drug|food|herb).{0,20}interaction", r"life[- ]threatening",
    ],
    1: [
        r"take with (food|meals?|milk)", r"with (meals?|food)", r"after (a )?meals?",
        r"(increase|enhance|improve)[a-z]* .{0,40}(absorption|bioavailability|exposure|efficacy)",
        r"(reduce|decrease|prevent)[a-z]* .{0,40}(irritation|nausea|gastrointestinal|gi side)",
        r"may be given with (food|milk)",
    ],
    2: [
        r"with or without food", r"consistently with regard to food",
        r"no (significant )?(effect|interaction|change)", r"no difference",
        r"did not (affect|change|alter)", r"not (significantly )?(affected|changed|altered)",
    ],
}
IGNORE = [r"same time (every|each) day"]

# Corrected rules used for Mode S. They fix two problems found by auditing the
# annotations: (1) the structured "Positive" effect is mostly a warning
# ("Avoid grapefruit products", "increases plasma concentrations"), so it is
# only beneficial when the sentence says so; (2) increased absorption is a
# toxicity risk, not a benefit, so it is no longer class 1.
CORRECTED_RULES = {
    2: [
        r"\bno (significant )?(effect|interaction|change|difference)\b",
        r"\bdid not (affect|alter|change|inhibit|increase|decrease)\b",
        r"\bdoes not (affect|alter|change)\b",
        r"\bwith or without food\b",
        r"\bnot (significantly )?(affected|changed|altered)\b",
        r"\bwithout (any )?(significant )?(effect|change)\b",
        r"\bno difference\b",
    ],
    0: [
        r"\bavoid\b",
        r"\btake .{0,30}separately from (meals|food)\b",
        r"\bempty stomach\b",
        r"\bbefore (a )?meals?\b",
        r"\bshould not be (taken|used|co-administered)\b",
        r"\bdo not (take|use|co-administer)\b",
        r"\bcontraindicat",
        r"\b(lower|decrease|reduce|impair|inhibit|prevent)[a-z]* .{0,40}(absorption|bioavailability|exposure|concentration|clearance|metabolism)\b",
        r"\blimit .{0,30}(intake|consumption)\b",
        r"\bserious (drug|food|herb).{0,20}interaction\b",
        r"\blife[- ]threatening\b",
        r"\bmonitor\b",
        r"\bcaution\b",
        r"\badjust (the )?dose\b",
        r"\b(increase|elevate)[a-z]* .{0,30}(plasma|serum|blood|inr)\b",
        r"\b(bleeding|haemorrhage|hemorrhage)\b",
        r"\bpotentiat",
        r"\bantagoniz",
        r"\b(at )?risk (of|for)\b",
        r"\breduce .{0,30}(effectiveness|efficacy)\b",
        r"\bcan (slightly )?affect\b",
    ],
    1: [
        r"\btake with (foods?|meals?|milk)\b",
        r"\bwith (meals?|food)\b",
        r"\bafter (a )?meals?\b",
        r"\bmay be given with (food|milk)\b",
        r"\b(improve|enhance)[a-z]* .{0,40}(efficacy|tolerability)\b",
        r"\b(reduce|decrease|prevent)[a-z]* .{0,40}(irritation|nausea|gastrointestinal|gi side)\b",
        r"\b(increase|enhance)[a-z]* .{0,40}absorption of (iron|vitamin|chromium|calcium)\b",
    ],
}

# Food-level rows (Component is empty) are mapped to pharmacologically relevant
# marker compounds instead of every compound the food contains. Propagating to
# all ~1,000 compounds labelled ubiquitous molecules such as glucose as
# warfarin interactions.
FOOD_MARKERS = {
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


def classify_text_corrected(text: str) -> int | None:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    if any(re.search(p, t) for p in IGNORE):
        return None
    for label in (2, 0, 1):
        if any(re.search(p, t) for p in CORRECTED_RULES[label]):
            return label
    return None

LEXICON = {
    "alcohol": {"compounds": ["ethanol"], "foods": []},
    "alcoholic": {"compounds": ["ethanol"], "foods": []},
    "ethanol": {"compounds": ["ethanol"], "foods": []},
    "grapefruit": {"compounds": ["bergamottin", "naringin", "naringenin"], "foods": ["Grapefruit"]},
    "furanocoumarin": {"compounds": ["bergamottin", "bergapten"], "foods": []},
    "dairy": {"compounds": ["calcium"], "foods": ["Milk"]},
    "calcium": {"compounds": ["calcium"], "foods": []},
    "vitamin k": {"compounds": ["phylloquinone", "vitamin k1"], "foods": []},
    "leafy green": {"compounds": ["phylloquinone"], "foods": ["Spinach", "Kale", "Broccoli"]},
    "tyramine": {"compounds": ["tyramine"], "foods": []},
    "caffeine": {"compounds": ["caffeine"], "foods": ["Arabica Coffee", "Black Tea"]},
    "coffee": {"compounds": ["caffeine"], "foods": ["Arabica Coffee"]},
    "tea": {"compounds": ["caffeine", "catechin"], "foods": ["Black Tea", "Green Tea"]},
    "iron": {"compounds": ["iron"], "foods": []},
    "potassium": {"compounds": ["potassium"], "foods": []},
    "sodium": {"compounds": ["sodium"], "foods": []},
    "vitamin c": {"compounds": ["ascorbic acid", "vitamin c"], "foods": []},
    "vitamin d": {"compounds": ["vitamin d", "cholecalciferol"], "foods": []},
    "omega 3": {"compounds": ["docosahexaenoic acid", "eicosapentaenoic acid"], "foods": []},
    "fish oil": {"compounds": ["docosahexaenoic acid", "eicosapentaenoic acid"], "foods": ["Anchovy", "Atlantic Herring", "Albacore Tuna"]},
    "quercetin": {"compounds": ["quercetin"], "foods": []},
    "resveratrol": {"compounds": ["resveratrol"], "foods": []},
    "curcumin": {"compounds": ["curcumin"], "foods": []},
    "berberine": {"compounds": ["berberine"], "foods": []},
    "hyperforin": {"compounds": ["hyperforin"], "foods": ["St John's Wort"]},
    "st john": {"compounds": ["hyperforin"], "foods": []},
    "licorice": {"compounds": ["glycyrrhizin"], "foods": ["Licorice"]},
    "pomegranate": {"compounds": [], "foods": ["Pomegranate"]},
    "broccoli": {"compounds": ["sulforaphane", "phylloquinone"], "foods": ["Broccoli"]},
    "spinach": {"compounds": ["phylloquinone"], "foods": ["Spinach"]},
    "milk": {"compounds": ["calcium"], "foods": ["Milk"]},
    "cheese": {"compounds": ["calcium", "tyramine"], "foods": ["Cheese"]},
    "soy": {"compounds": ["genistein", "daidzein"], "foods": ["Soybean"]},
    "ginger": {"compounds": ["gingerol"], "foods": ["Ginger"]},
    "garlic": {"compounds": ["allicin"], "foods": ["Garlic"]},
}


def classify_text(text: str) -> int | None:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    if any(re.search(p, t) for p in IGNORE):
        return None
    for label in (0, 1, 2):
        if any(re.search(p, t) for p in RULES[label]):
            return label
    return None


def map_component(component: str, name_index: dict[str, list[int]]) -> list[int]:
    key = norm_name(component)
    if not key:
        return []
    candidates = [key]
    if key.endswith("s"):
        candidates.append(key[:-1])
    if key.endswith("es"):
        candidates.append(key[:-2])
    for c in candidates:
        if c in name_index:
            return name_index[c]
    return []


def name_aliases(name_index: dict[str, list[int]], aliases: dict[str, str]) -> dict[str, list[int]]:
    out = dict(name_index)
    for alias, target in aliases.items():
        cid = name_index.get(norm_name(target))
        if cid:
            out.setdefault(norm_name(alias), cid)
    return out


def match_case_pairs(drug_map, compound_index, cases):
    """Locate the nine held-out case pairs in the cleaned tables."""
    results = []
    for drug_name, compound_name in cases:
        dkey = norm_name(drug_name)
        ckey = norm_name(compound_name)
        drug_ids = drug_map.get(dkey, [])
        comp_ids = compound_index.get(ckey, [])
        if not comp_ids and "furanocoumarin" in ckey:
            comp_ids = compound_index.get("bergamottin", [])
        if not comp_ids and "vitamin k" in ckey:
            comp_ids = compound_index.get("phylloquinone", [])
        results.append(
            {
                "drug": drug_name,
                "compound": compound_name,
                "drug_ids": drug_ids,
                "compound_ids": comp_ids,
                "supported": bool(drug_ids and comp_ids),
            }
        )
    return results


def main() -> None:
    setup_stdout()
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    interim = Path(cfg["paths"]["interim"])

    drugs = pd.read_parquet(interim / "drugs_clean.parquet")
    foods = pd.read_parquet(interim / "foods_clean.parquet")
    links = pd.read_parquet(interim / "food_links.parquet")
    inter = pd.read_csv(cfg["paths"]["interactions_raw"], encoding="utf-8-sig", low_memory=False)

    organics = foods.loc[foods["is_organic"]].reset_index(drop=True)
    organic_ids = organics["compound_id"].to_numpy()
    organic_names = organics.set_index("compound_id")["name"]

    name_index: dict[str, list[int]] = {}
    for cid, name in zip(foods["compound_id"], foods["name"]):
        name_index.setdefault(norm_name(name), []).append(int(cid))
    name_index = name_aliases(name_index, {k: v for k, v in {
        "vitamin c": "ascorbic acid", "vitamin d": "cholecalciferol",
    }.items()})

    # Food-name -> food_id index, with a containment fallback for names like "Grapefruit juice".
    food_name_to_ids: dict[str, set[int]] = {}
    for fid, fname in links[["food_id", "food_name"]].drop_duplicates().itertuples(index=False):
        food_name_to_ids.setdefault(norm_name(fname), set()).add(int(fid))
    food_ids_to_name = links.drop_duplicates("food_id").set_index("food_id")["food_name"].to_dict()
    food_to_compounds: dict[int, list[int]] = {}
    for fid, cid in links[["food_id", "compound_id"]].itertuples(index=False):
        food_to_compounds.setdefault(int(fid), []).append(int(cid))
    compound_freq = (
        links.drop_duplicates(["food_id", "compound_id"])
        .groupby("compound_id")["food_id"].nunique()
        .to_dict()
    )
    organic_set = {int(x) for x in organic_ids}

    marker_ids: dict[str, list[int]] = {}
    for mkey, marker_names in FOOD_MARKERS.items():
        ids: list[int] = []
        for marker in marker_names:
            hits = name_index.get(norm_name(marker), [])
            if not hits:
                hits = [
                    cid
                    for cname, cids in name_index.items()
                    if norm_name(marker) and norm_name(marker) in cname
                    for cid in cids
                ][:3]
            ids.extend(hits)
        marker_ids[mkey] = sorted(set(ids))

    def markers_for_food(food_herb_name: str) -> set[int]:
        key = norm_name(food_herb_name)
        if not key:
            return set()
        # Longest key wins, so "grapefruit" is matched before "grape".
        for mkey in sorted(marker_ids, key=len, reverse=True):
            if mkey and mkey in key:
                return set(marker_ids[mkey])
        return set()

    max_freq = int(cfg["labels"].get("mode_s_food_max_frequency", 20))
    max_compounds = int(cfg["labels"].get("mode_s_food_max_compounds", 25))

    def specific_targets(food_ids: set[int]) -> set[int]:
        cids = {c for fid in food_ids for c in food_to_compounds.get(fid, []) if c in organic_set}
        cids = {c for c in cids if compound_freq.get(c, 999) <= max_freq}
        ordered = sorted(cids, key=lambda c: (compound_freq.get(c, 999), c))
        return set(ordered[:max_compounds])

    def match_foods(food_herb_name: str) -> set[int]:
        key = norm_name(food_herb_name)
        if not key:
            return set()
        if key in food_name_to_ids:
            return food_name_to_ids[key]
        hits = set()
        for fkey, ids in food_name_to_ids.items():
            if fkey and re.search(rf"\b{re.escape(fkey)}\b", key):
                hits.update(ids)
        return hits

    drug_ids_set = set(drugs["drug_id"])
    drug_map: dict[str, list[str]] = {}
    for did, dname in drugs[["drug_id", "name"]].itertuples(index=False):
        drug_map.setdefault(norm_name(dname), []).append(did)

    inter = inter[inter["Drug_ID"].isin(drug_ids_set)].copy()
    inter["effect_class"] = inter["Effect"].map(cfg["labels"]["effect_to_class"])
    inter = inter[inter["effect_class"].notna()].copy()
    inter["effect_class"] = inter["effect_class"].astype(int)
    inter["row_id"] = np.arange(len(inter))

    # "Possible" is an uncertain annotation: let the paper's text rules decide it.
    # Confirmed Negative/Harmful/Positive/No Effect labels stay structured.
    inter["effect_norm"] = inter["Effect"].astype(str).str.strip().str.lower()
    inter["rule_label"] = inter["Conclusion"].map(classify_text)
    inter["rule_corrected"] = inter["Conclusion"].map(classify_text_corrected)

    # Mode P keeps the paper's semantics for comparability.
    possible = inter["effect_norm"].eq("possible")
    inter["label_paper"] = inter["effect_class"]
    use_rule = possible & inter["rule_label"].notna()
    inter.loc[use_rule, "label_paper"] = inter.loc[use_rule, "rule_label"]
    inter["label_paper"] = inter["label_paper"].astype(int)

    # Mode S uses corrected semantics: a "Positive" annotation is a warning
    # unless its sentence is explicitly beneficial, and increased exposure is
    # a risk rather than a benefit.
    effect = inter["effect_norm"]
    label = pd.Series(np.nan, index=inter.index, dtype="float64")
    label[effect.isin(["harmful", "negative"])] = 0
    label[effect.eq("no effect")] = 2
    pos = effect.eq("positive")
    label[pos] = inter.loc[pos, "rule_corrected"].fillna(0)
    prob = effect.eq("possible")
    label[prob] = inter.loc[prob, "rule_corrected"].fillna(2)
    inter["label_specific"] = label.fillna(2).astype(int)
    specific_rule = (pos | prob) & inter["rule_corrected"].notna()
    inter["label_source"] = np.where(specific_rule, "text_rule", "structured")

    # ---- Mode P: one label per drug, applied to every food compound.
    def drug_label(group: pd.DataFrame):
        vals = set(group["label_paper"])
        if 0 in vals:
            return 0
        if 1 in vals:
            return 1
        return 2

    per_drug = inter.groupby("Drug_ID").apply(drug_label, include_groups=False).rename("label").reset_index()
    per_drug = per_drug.rename(columns={"Drug_ID": "drug_id"})
    per_drug["sentence_id"] = per_drug["drug_id"].map(inter.groupby("Drug_ID")["row_id"].first())
    counts_p = per_drug["label"].value_counts().sort_index().to_dict()

    cap_p = int(cfg["labels"]["mode_p_pairs_per_drug"])
    p_rows = []
    for rec in per_drug.itertuples(index=False):
        k = min(cap_p, len(organic_ids))
        picked = rng.choice(organic_ids, size=k, replace=False)
        p_rows.append(
            pd.DataFrame(
                {
                    "drug_id": rec.drug_id,
                    "compound_id": picked,
                    "label": rec.label,
                    "mode": "P",
                    "sentence_id": rec.sentence_id,
                    "weak": False,
                }
            )
        )
    pairs_p = pd.concat(p_rows, ignore_index=True)
    pairs_p["compound_name"] = pairs_p["compound_id"].map(organic_names)

    # ---- Mode S: labels for named compounds or marker compounds of named foods.
    s_map: dict[tuple[str, int], tuple[int, int]] = {}
    mapped_rows = 0
    marker_rows = 0
    specific_rows = 0
    targets_total = 0
    for rec in inter.itertuples(index=False):
        targets: set[int] = set()
        cids = map_component(rec.Component, name_index) if isinstance(rec.Component, str) else []
        targets.update(cids)
        food_ids = match_foods(rec.Food_Herb_Name) if isinstance(rec.Food_Herb_Name, str) else set()
        if not targets and food_ids:
            marker_cids = markers_for_food(rec.Food_Herb_Name) if isinstance(rec.Food_Herb_Name, str) else set()
            if marker_cids:
                targets.update(marker_cids)
                marker_rows += 1
            else:
                targets.update(specific_targets(food_ids))
                specific_rows += 1
        if not targets:
            continue
        mapped_rows += 1
        targets_total += len(targets)
        for cid in targets:
            key = (rec.Drug_ID, int(cid))
            label = int(rec.label_specific)
            if key not in s_map:
                s_map[key] = (label, int(rec.row_id))
            else:
                old_label, old_row = s_map[key]
                # Negative precedence, then positive, then non-significant.
                new_label = label if label < old_label else old_label
                s_map[key] = (new_label, old_row if old_label <= label else int(rec.row_id))

    s_rows = [
        {
            "drug_id": did,
            "compound_id": cid,
            "label": lab,
            "mode": "S",
            "sentence_id": sid,
            "weak": False,
        }
        for (did, cid), (lab, sid) in s_map.items()
    ]
    pairs_s = pd.DataFrame(s_rows)
    weak_min = int(cfg["labels"]["mode_s_weak_pairs_per_drug"])
    weak_ratio = float(cfg["labels"].get("mode_s_weak_ratio", 1.0))
    weak_max = int(cfg["labels"].get("mode_s_weak_max", 2500))
    weak_rows = []
    if len(pairs_s):
        for did, group in pairs_s.groupby("drug_id"):
            used = set(group["compound_id"])
            available = np.array([c for c in organic_ids if c not in used], dtype=int)
            if available.size == 0:
                continue
            informative = int((group["label"] != 2).sum())
            k = int(np.clip(np.ceil(weak_ratio * max(informative, 1)), weak_min, weak_max))
            k = min(k, available.size)
            picked = rng.choice(available, size=k, replace=False)
            weak_rows.append(
                pd.DataFrame(
                    {
                        "drug_id": did,
                        "compound_id": picked,
                        "label": 2,
                        "mode": "S",
                        "sentence_id": -1,
                        "weak": True,
                    }
                )
            )
    if weak_rows:
        pairs_s = pd.concat([pairs_s] + weak_rows, ignore_index=True)
    pairs_s["compound_name"] = pairs_s["compound_id"].map(organic_names)

    # ---- Held-out case studies: remove from every training table.
    cases = [
        ("tetracycline", "calcium"), ("methotrexate", "ethanol"), ("diazepam", "ethanol"),
        ("nitroglycerin", "ethanol"), ("digoxin", "hyperforin"), ("nisoldipine", "bergamottin"),
        ("midazolam", "furanocoumarin"), ("warfarin", "vitamin k1 2,3 epoxide"),
        ("warfarin", "dimethyl disulfide"),
    ]
    probe_cases = [
        ("warfarin", "grapefruit"), ("simvastatin", "grapefruit"),
        ("midazolam", "grapefruit"), ("metronidazole", "alcohol"),
        ("warfarin", "cranberry"), ("digoxin", "st john's wort"),
    ]
    case_locations = match_case_pairs(drug_map, name_index, cases)
    probe_locations = []
    for dname, food in probe_cases:
        dids = drug_map.get(norm_name(dname), [])
        cids = set(markers_for_food(food))
        if not cids:
            cids = set(name_index.get(norm_name(food), []))
        if not cids:
            food_ids = match_foods(food)
            cids = specific_targets(food_ids) if food_ids else set()
        probe_locations.append(
            {"drug": dname, "food": food, "drug_ids": dids, "compound_ids": sorted(int(c) for c in cids)}
        )
    holdout = set()
    for c in case_locations:
        for did in c["drug_ids"]:
            for cid in c["compound_ids"]:
                holdout.add((did, cid))
    for c in probe_locations:
        for did in c["drug_ids"]:
            for cid in c["compound_ids"]:
                holdout.add((did, cid))
    holdout_df = pd.DataFrame(sorted(holdout), columns=["drug_id", "compound_id"])
    for name in ("pairs_p", "pairs_s"):
        frame = locals()[name]
        if len(holdout_df):
            frame = frame.merge(holdout_df, how="left", on=["drug_id", "compound_id"], indicator=True)
            frame = frame[frame["_merge"] == "left_only"].drop(columns="_merge")
        if name == "pairs_p":
            pairs_p = frame.reset_index(drop=True)
        else:
            pairs_s = frame.reset_index(drop=True)

    # ---- Rule audit on 100 sentences.
    audit_pool = inter[inter["Conclusion"].notna()].copy()
    audit = audit_pool.sample(
        n=min(int(cfg["labels"]["audit_sentences"]), len(audit_pool)),
        random_state=cfg["seed"],
    )[
        ["row_id", "Drug_ID", "Food_Herb_Name", "Component", "Conclusion", "Effect",
         "effect_class", "effect_norm", "label_paper", "label_specific", "label_source", "rule_label"]
    ].rename(columns={"rule_label": "pred_rule"})
    audit["agrees_derived"] = audit["pred_rule"] == audit["label_paper"]
    confirmed = audit[~audit["effect_norm"].eq("possible")]
    conf_fired = confirmed[confirmed["pred_rule"].notna()]
    possible_audit = audit[audit["effect_norm"].eq("possible")]
    pos_fired = possible_audit[possible_audit["pred_rule"].notna()]
    fired_all = audit[audit["pred_rule"].notna()]
    audit_summary = {
        "sampled": int(len(audit)),
        "possible_rows": int(len(possible_audit)),
        "confirmed_rows": int(len(confirmed)),
        "rule_fired_all": int(len(fired_all)),
        "rule_coverage_all": float(audit["pred_rule"].notna().mean()),
        "confirmed_rule_fired": int(len(conf_fired)),
        "confirmed_agreement": float(conf_fired["agrees_derived"].mean()) if len(conf_fired) else None,
        "possible_rule_fired": int(len(pos_fired)),
        "possible_rule_coverage": float(possible_audit["pred_rule"].notna().mean()) if len(possible_audit) else None,
        "raw_agreement_vs_effect_on_fired": float((fired_all["pred_rule"] == fired_all["effect_class"]).mean())
        if len(fired_all) else None,
        "note": (
            "Confirmed rows test the rules against experimental annotations. Possible rows are "
            "labelled from the text rules by design, so their agreement is circular; coverage is reported instead. "
            "The CSV in reports/ has an empty manual_label column for the human 100-sentence audit."
        ),
    }
    audit["manual_label"] = ""
    audit.to_csv(Path(cfg["paths"]["reports"]) / "sentence_audit_100.csv", index=False, encoding="utf-8-sig")

    # ---- Outputs
    pairs = pd.concat([pairs_p, pairs_s], ignore_index=True)
    pairs["drug_name"] = pairs["drug_id"].map(drugs.set_index("drug_id")["name"])
    pairs_out = interim / "pairs_labeled.parquet"
    pairs[["drug_id", "compound_id", "drug_name", "compound_name", "label", "mode", "sentence_id", "weak"]].to_parquet(
        pairs_out, index=False
    )

    save_json(RULES, Path(cfg["paths"]["reports"]) / "label_rules.json")
    save_json(CORRECTED_RULES, Path(cfg["paths"]["reports"]) / "label_rules_corrected.json")
    save_json(FOOD_MARKERS, Path(cfg["paths"]["reports"]) / "food_markers.json")
    save_json(FOOD_MARKERS, Path(cfg["paths"]["models"]) / "food_markers.json")
    save_json(LEXICON, Path(cfg["paths"]["reports"]) / "label_lexicon.json")
    audit.to_parquet(interim / "sentence_audit.parquet", index=False)
    save_json(
        {
            "mode_p_pairs": int(len(pairs_p)),
            "mode_s_pairs": int(len(pairs_s)),
            "mode_p_class_counts": {int(k): int(v) for k, v in pairs_p["label"].value_counts().sort_index().items()},
            "mode_s_class_counts": {int(k): int(v) for k, v in pairs_s["label"].value_counts().sort_index().items()},
            "mode_s_weak_pairs": int(pairs_s["weak"].sum()),
            "per_drug_class_counts": {int(k): int(v) for k, v in counts_p.items()},
            "interaction_rows_mapped_mode_s": int(mapped_rows),
            "mode_s_mapping": {
                "marker_rows": int(marker_rows),
                "specific_food_rows": int(specific_rows),
                "targets_total": int(targets_total),
                "mean_targets_per_row": round(targets_total / mapped_rows, 3) if mapped_rows else None,
            },
            "label_source_counts": {str(k): int(v) for k, v in inter["label_source"].value_counts().items()},
            "corrected_effect_distribution": {
                str(k): {str(c): int(n) for c, n in g.items()}
                for k, g in inter.groupby("effect_norm")["label_specific"].value_counts().unstack(fill_value=0).iterrows()
            },
            "holdout_pairs": [list(x) for x in sorted(holdout)],
            "case_study_locations": case_locations,
            "clinical_probe_locations": probe_locations,
            "sentence_audit": audit_summary,
        },
        Path(cfg["paths"]["reports"]) / "brick4_label_report.json",
    )

    lines = [
        "# Brick 4 - labels",
        "",
        f"- Mode P pairs: **{len(pairs_p):,}**; class counts {dict(pairs_p['label'].value_counts().sort_index())} "
        "(paper semantics, kept for comparability).",
        f"- Mode S pairs: **{len(pairs_s):,}**; class counts {dict(pairs_s['label'].value_counts().sort_index())} "
        f"({int(pairs_s['weak'].sum()):,} weak class-2 pairs).",
        f"- Interaction rows resolved to specific Mode S targets: {mapped_rows:,} of {len(inter):,}; "
        f"{marker_rows:,} rows used curated food markers, {specific_rows:,} used food-specific compounds; "
        f"mean targets per row {targets_total / mapped_rows:.1f}." if mapped_rows else "- No rows mapped.",
        f"- Label source (Mode S): {inter['label_source'].value_counts().to_dict()}.",
        "",
        "## Why labels changed",
        "",
        "The structured `Effect` column is ambiguous: 48% of `Positive` annotations say "
        "'avoid' or warn about increased exposure, so `Positive` cannot mean beneficial. "
        "Mode S now labels `Negative`/`Harmful` as 0, `No Effect` as 2, and decides "
        "`Positive`/`Possible` from the sentence: no-effect guard first, then harm, then benefit. "
        "Food-level rows map to curated marker compounds instead of every compound of the food "
        "(which previously labelled ubiquitous molecules such as glucose as warfarin interactions).",
        "",
        "## Rule audit (100 sentences)",
        "",
        dataframe_to_markdown(pd.DataFrame([audit_summary])),
        "",
        "## Held-out case-study pairs",
        "",
        dataframe_to_markdown(pd.DataFrame(case_locations)),
        "",
        "## Held-out clinical probes",
        "",
        dataframe_to_markdown(pd.DataFrame(probe_locations)),
        "",
        f"Pairs removed from every training table: {len(holdout)}.",
        "",
        "Mode P applies one drug-level label to all compounds (paper-like, leakage-prone). "
        "Mode S labels compounds named by the interaction text, curated marker compounds of "
        "the matched food, or food-specific compounds; other sampled compounds for the same "
        "drug are class 2 and flagged weak.",
    ]
    (Path(cfg["paths"]["reports"]) / "brick4_labels.md").write_text("\n".join(lines), encoding="utf-8")
    print(lines[0])
    print("Mode P", len(pairs_p), pairs_p["label"].value_counts().sort_index().to_dict())
    print("Mode S", len(pairs_s), pairs_s["label"].value_counts().sort_index().to_dict())
    print("mapped rows", mapped_rows, "markers", marker_rows, "specific", specific_rows)
    print("audit", audit_summary)
    print(pd.DataFrame(case_locations).to_string())
    print("holdout pairs", len(holdout))


if __name__ == "__main__":
    main()
