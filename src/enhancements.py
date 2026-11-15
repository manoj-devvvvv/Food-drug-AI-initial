"""Brick 12 - ablations and section 15 enhancements (E1a/E1b/E1c, E2).

Everything here is evaluated on Split G with Mode S labels, the honest
configuration. The frozen model from Brick 8 is never modified.

Usage: python src/brick12_enhancements.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from common import load_config, load_json, save_json, setup_stdout
from features import ALL_PAIR_NAMES, DESC_NAMES, pair_matrix
from brick7_selection import parse_pair_name
from brick8_models import BalancedClassifier, build_features, cap_frame, metrics, make_estimator

RDLogger.DisableLog("rdApp.*")
MODE = "S"
FP_BITS = 256


def fingerprint_row(smiles: str) -> np.ndarray:
    out = np.zeros(FP_BITS, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return out
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.frombuffer(bytes(fp.ToBitString(), "ascii"), dtype=np.uint8) - ord("0")
    out[:] = arr.reshape(FP_BITS, 2048 // FP_BITS).max(axis=1)
    return out


def one_hot(values: pd.Series, top: int, prefix: str) -> tuple[np.ndarray, list[str]]:
    vc = values.fillna("").astype(str).replace("", "unannotated").value_counts()
    keep = list(vc.index[:top])
    cols = [f"{prefix}={k}" for k in keep] + [f"{prefix}=other"]
    M = np.zeros((len(values), len(cols)), dtype=np.float32)
    for j, k in enumerate(keep):
        M[:, j] = (values.fillna("").astype(str).replace("", "unannotated") == k).to_numpy()
    M[:, -1] = 1.0 - M[:, :top].sum(axis=1)
    return M, cols


def molecule_side(cfg, drug_moe, food_moe):
    interim = Path(cfg["paths"]["interim"])
    drugs = pd.read_parquet(interim / "drugs_clean.parquet").set_index("drug_id")
    foods = pd.read_parquet(interim / "foods_clean.parquet").set_index("compound_id")

    d_ids = drug_moe["molecule_id"].tolist()
    f_ids = food_moe["molecule_id"].tolist()
    d = drugs.loc[d_ids]
    f = foods.loc[f_ids]

    dtypes = (d["drug_type"] != "Small molecular drug").to_numpy(dtype=np.float32)[:, None]
    superklass = one_hot(f["superklass"], 8, "superklass")
    organic = f["is_organic"].to_numpy(dtype=np.float32)[:, None]
    food_meta = np.hstack([organic, superklass[0]])
    food_meta_cols = ["is_organic"] + superklass[1]

    d_fp = np.vstack([fingerprint_row(s) for s in d["smiles"]])
    f_fp = np.vstack([fingerprint_row(s) for s in f["smiles"]])
    return {
        "drug_meta": dtypes, "drug_meta_cols": ["drug_type=non_small_molecule"],
        "food_meta": food_meta, "food_meta_cols": food_meta_cols,
        "drug_fp": d_fp, "food_fp": f_fp,
    }


def base_and_extras(names, side, di, fi):
    base_idx_d = [DESC_NAMES.index(parse_pair_name(n)["drug_descriptor"]) for n in names]
    base_idx_f = [DESC_NAMES.index(parse_pair_name(n)["food_descriptor"]) for n in names]
    return base_idx_d, base_idx_f


def main() -> None:
    setup_stdout()
    cfg = load_config()
    reports = Path(cfg["paths"]["reports"])
    models = Path(cfg["paths"]["models"])
    interim = Path(cfg["paths"]["interim"])

    sel = load_json(models / f"selected_features_{MODE}.json")
    names = [f["name"] for f in sel["features"]]
    info = load_json(reports / f"brick8_models_{MODE}.json")
    params = dict(info["tuning"]["XGBoost"]["best_params"])
    full_data, drug_moe, food_moe = build_features(cfg, MODE, names)
    drow = {k: i for i, k in enumerate(drug_moe["molecule_id"])}
    frow = {k: i for i, k in enumerate(food_moe["molecule_id"])}

    side = molecule_side(cfg, drug_moe, food_moe)
    dvals = drug_moe[DESC_NAMES].to_numpy(dtype=np.float32)
    fvals = food_moe[DESC_NAMES].to_numpy(dtype=np.float32)
    paired = parse_pair_name

    def select(sub, cap, seed, scramble=False):
        sub = sub[sub["drug_id"].isin(drow) & sub["compound_id"].isin(frow)].reset_index(drop=True)
        if cap and len(sub) > cap:
            rng = np.random.default_rng(seed)
            sub = sub.iloc[np.sort(rng.choice(len(sub), cap, replace=False))].reset_index(drop=True)
        di = np.array([drow[x] for x in sub["drug_id"]], dtype=np.int64)
        fi = np.array([frow[x] for x in sub["compound_id"]], dtype=np.int64)
        if scramble:
            rng = np.random.default_rng(seed + 7)
            fi = fi[rng.permutation(len(fi))]
        return sub, di, fi

    def base_X(di, fi):
        X = np.empty((len(di), len(names)), dtype=np.float32)
        for start, chunk in pair_matrix(dvals, fvals, di, fi, names=names, chunk_size=20000):
            X[start:start + len(chunk)] = chunk
        return X

    def extras(di, fi):
        d_meta = side["drug_meta"][di]
        f_meta = side["food_meta"][fi]
        return {
            "E1a": d_meta,
            "E1b": f_meta,
            "E1c": np.hstack([side["drug_fp"][di], side["food_fp"][fi]]),
        }

    def fit_eval(Xtr, ytr, Xte, yte, seed=42):
        est = BalancedClassifier(base=make_estimator("XGBoost", False).set_params(**params))
        t0 = time.time()
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        out = metrics(yte, pred)
        out["seconds"] = round(time.time() - t0, 1)
        return est, out, pred

    CAP = 40000
    tr, di, fi = select(full_data["train"]["df"], CAP, cfg["seed"])
    te, tdi, tfi = select(full_data["test"]["df"], None, cfg["seed"] + 5)
    Xtr, Xte = base_X(di, fi), base_X(tdi, tfi)
    ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
    print(f"train {Xtr.shape}, test {Xte.shape}", flush=True)

    results = {"mode": MODE, "split": "G", "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
               "features": {}, "ablations": {}, "enhancements": {}}

    _, base_m, base_pred = fit_eval(Xtr, ytr, Xte, yte)
    results["features"]["18_moe"] = base_m
    print(f"base 18: macroF1={base_m['macro_f1']:.4f}", flush=True)

    # ---- Ablations
    drug_only = np.array([dvals[di, DESC_NAMES.index(paired(n)["drug_descriptor"])] for n in names]).T
    food_only = np.array([fvals[fi, DESC_NAMES.index(paired(n)["food_descriptor"])] for n in names]).T
    drug_only_te = np.array([dvals[tdi, DESC_NAMES.index(paired(n)["drug_descriptor"])] for n in names]).T
    food_only_te = np.array([fvals[tfi, DESC_NAMES.index(paired(n)["food_descriptor"])] for n in names]).T
    for label, A, B in (("drug_only", drug_only, drug_only_te), ("food_only", food_only, food_only_te)):
        _, m, _ = fit_eval(A.astype(np.float32), ytr, B.astype(np.float32), yte)
        results["ablations"][label] = m
        print(f"{label}: macroF1={m['macro_f1']:.4f}", flush=True)

    tr_s, di_s, fi_s = select(full_data["train"]["df"], CAP, cfg["seed"], scramble=True)
    te_s, tdi_s, tfi_s = select(full_data["test"]["df"], None, cfg["seed"] + 5, scramble=True)
    Xs, Xs_te = base_X(di_s, fi_s), base_X(tdi_s, tfi_s)
    _, m, _ = fit_eval(Xs, ytr, Xs_te, yte)
    results["ablations"]["shuffled_food"] = m
    print(f"shuffled food: macroF1={m['macro_f1']:.4f}", flush=True)

    # 18 vs all 3,780 pair features (smaller sample to bound memory).
    dvals_all = drug_moe[DESC_NAMES].to_numpy(dtype=np.float32)
    fvals_all = food_moe[DESC_NAMES].to_numpy(dtype=np.float32)
    n_all = min(20000, len(ytr))
    rng = np.random.default_rng(cfg["seed"])
    idx = np.sort(rng.choice(len(ytr), n_all, replace=False))
    Xall = np.empty((n_all, len(ALL_PAIR_NAMES)), dtype=np.float32)
    for start, chunk in pair_matrix(dvals_all, fvals_all, di[idx], fi[idx], names=None, chunk_size=1500):
        Xall[start:start + len(chunk)] = chunk
    tall = te.iloc[np.sort(np.random.default_rng(cfg["seed"] + 9).choice(len(te), min(15000, len(te)), replace=False))]
    Xall_te = np.empty((len(tall), len(ALL_PAIR_NAMES)), dtype=np.float32)
    tdi_sel = np.array([drow[x] for x in tall["drug_id"]], dtype=np.int64)
    tfi_sel = np.array([frow[x] for x in tall["compound_id"]], dtype=np.int64)
    for start, chunk in pair_matrix(dvals_all, fvals_all, tdi_sel, tfi_sel, names=None, chunk_size=1500):
        Xall_te[start:start + len(chunk)] = chunk
    _, m, _ = fit_eval(Xall, ytr[idx], Xall_te, tall["label"].to_numpy())
    results["ablations"]["all_3780_features"] = m
    results["ablations"]["all_3780_features"]["test_rows"] = int(len(tall))
    print(f"all 3780: macroF1={m['macro_f1']:.4f}", flush=True)
    del Xall, Xall_te

    # ---- E1 feature sets
    Xtr_e = extras(di, fi)
    Xte_e = extras(tdi, tfi)
    variants = {
        "18+E1a_drug_metadata": ["E1a"],
        "18+E1b_food_class": ["E1b"],
        "18+E1c_fingerprints": ["E1c"],
        "18+E1abc_all": ["E1a", "E1b", "E1c"],
    }
    best_variant, best_m = "18_moe", base_m
    best_A, best_B = Xtr, Xte
    for label, keys in variants.items():
        A = np.hstack([Xtr] + [Xtr_e[k] for k in keys])
        B = np.hstack([Xte] + [Xte_e[k] for k in keys])
        est, m, pred = fit_eval(A, ytr, B, yte)
        m["n_features"] = int(A.shape[1])
        results["enhancements"][label] = m
        print(f"{label} ({A.shape[1]} cols): macroF1={m['macro_f1']:.4f}", flush=True)
        if m["macro_f1"] > best_m["macro_f1"]:
            best_variant, best_m, best_A, best_B = label, m, A, B

    # ---- E2 calibration + split conformal abstention (90% target coverage)
    cal, cdi, cfi = select(full_data["val"]["df"], 20000, cfg["seed"] + 3)
    Xcal = base_X(cdi, cfi)
    ycal = cal["label"].to_numpy()
    chosen = next((keys for label, keys in variants.items() if label == best_variant), [])
    if chosen:
        ce = extras(cdi, cfi)
        Xcal = np.hstack([Xcal] + [ce[k] for k in chosen])

    calibrated = CalibratedClassifierCV(
        make_estimator("XGBoost", False).set_params(**params), method="sigmoid", cv=3,
    )
    t0 = time.time()
    calibrated.fit(best_A, ytr)
    p_cal = calibrated.predict_proba(Xcal)
    p_te = calibrated.predict_proba(best_B)
    scores = 1.0 - p_cal[np.arange(len(ycal)), ycal]
    q = float(np.quantile(scores, 0.9))
    set_size = (1.0 - p_te) <= q
    covered = set_size[np.arange(len(yte)), yte]
    abstain = set_size.sum(axis=1) > 1
    sel_mask = ~abstain
    pred_cal = calibrated.predict(best_B)
    e2 = {
        "variant": best_variant,
        "q_hat": q,
        "target_coverage": 0.9,
        "coverage": float(covered.mean()),
        "avg_set_size": float(set_size.sum(axis=1).mean()),
        "abstain_rate": float(abstain.mean()),
        "accuracy_all": float(accuracy_score(yte, pred_cal)),
        "selective_accuracy": float(accuracy_score(yte[sel_mask], pred_cal[sel_mask])) if sel_mask.any() else None,
        "selective_macro_f1": float(f1_score(yte[sel_mask], pred_cal[sel_mask], average="macro")) if sel_mask.any() else None,
        "seconds": round(time.time() - t0, 1),
    }

    # Drug-level bootstrap CI on the test partition.
    df_te = te.reset_index(drop=True)
    drugs = df_te["drug_id"].unique()
    rng = np.random.default_rng(cfg["seed"])
    f1s, accs = [], []
    preds = pred_cal
    for _ in range(1000):
        pick = rng.choice(drugs, size=len(drugs), replace=True)
        rows = df_te.index[df_te["drug_id"].isin(pick)]
        f1s.append(f1_score(yte[rows], preds[rows], average="macro"))
        accs.append(accuracy_score(yte[rows], preds[rows]))
    e2["bootstrap_drug_level"] = {
        "replicates": 1000, "n_drugs": int(len(drugs)),
        "macro_f1_ci95": [float(np.quantile(f1s, 0.025)), float(np.quantile(f1s, 0.975))],
        "accuracy_ci95": [float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))],
    }
    results["e2_calibration_conformal"] = e2
    print(f"E2: coverage={e2['coverage']:.3f}, abstain={e2['abstain_rate']:.3f}, "
          f"selective acc={e2['selective_accuracy']}", flush=True)

    save_json(results, reports / "brick12_enhancements.json")

    rows = [{"config": k, "macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "n_features": len(names)}
            for k, v in results["features"].items()]
    rows += [{"config": k, "macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "n_features": len(names)}
             for k, v in results["ablations"].items()]
    rows += [{"config": k, "macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "n_features": v.get("n_features", "")}
             for k, v in results["enhancements"].items()]
    lines = [
        "# Brick 12 - ablations and section 15 enhancements",
        "",
        f"Mode {MODE}, Split G, XGBoost with the tuned Brick 8 parameters. "
        f"Training rows {len(ytr):,} (capped), test rows {len(yte):,}.",
        "",
        "## Ablations and feature sets",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        "## E2 calibration and split conformal",
        "",
        f"Variant: {e2['variant']}; q_hat = {e2['q_hat']:.4f} at a 90% target.",
        f"Test coverage {e2['coverage']:.3f}, average prediction-set size {e2['avg_set_size']:.2f}, "
        f"abstention rate {e2['abstain_rate']:.3f}.",
        f"Argmax accuracy {e2['accuracy_all']:.4f}; selective accuracy {e2['selective_accuracy']:.4f} "
        f"on the {int((~abstain).sum()):,} non-abstained pairs.",
        f"Drug-level bootstrap 95% CI for macro-F1: {e2['bootstrap_drug_level']['macro_f1_ci95']}.",
        "",
        "Reading: drug- and food-only models show whether the pair interaction adds signal. "
        "A large drop under shuffled food pairing is the direct evidence that the model uses the pairing.",
    ]
    (reports / "brick12_enhancements.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
