"""Brick 8 - baselines, XGBoost randomized search, freeze the best model.

Usage: python src/brick8_models.py --mode S
       python src/brick8_models.py --mode P
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GroupKFold, PredefinedSplit, RandomizedSearchCV, StratifiedKFold, cross_val_predict,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from common import load_config, load_json, save_json, setup_stdout
from features import DESC_NAMES, pair_matrix

LABELS = [0, 1, 2]


class BalancedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y, sample_weight=None):
        sw = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight, dtype=float)
        self.base.fit(X, y, sample_weight=sw)
        self.classes_ = self.base.classes_
        return self

    def predict(self, X):
        return self.base.predict(X)

    def predict_proba(self, X):
        return self.base.predict_proba(X)


def build_features(cfg, mode, feature_names):
    processed = Path(cfg["paths"]["processed"])
    interim = Path(cfg["paths"]["interim"])
    splits = pd.read_parquet(processed / f"splits_{mode}.parquet")
    drug_moe = pd.read_parquet(interim / "drug_moe.parquet")
    food_moe = pd.read_parquet(interim / "food_moe.parquet")
    dvals = drug_moe[DESC_NAMES].to_numpy()
    fvals = food_moe[DESC_NAMES].to_numpy()
    drow = {k: i for i, k in enumerate(drug_moe["molecule_id"])}
    frow = {k: i for i, k in enumerate(food_moe["molecule_id"])}
    out = {}
    for split in ("train", "val", "test"):
        sub = splits[splits["split_G"] == split].reset_index(drop=True)
        sub = sub[sub["drug_id"].isin(drow) & sub["compound_id"].isin(frow)].reset_index(drop=True)
        di = np.array([drow[x] for x in sub["drug_id"]], dtype=np.int64)
        fi = np.array([frow[x] for x in sub["compound_id"]], dtype=np.int64)
        X = np.empty((len(sub), len(feature_names)), dtype=np.float32)
        for start, chunk in pair_matrix(dvals, fvals, di, fi, names=feature_names, chunk_size=20000):
            X[start:start + len(chunk)] = chunk
        out[split] = {
            "X": X,
            "y": sub["label"].to_numpy(),
            "weak": sub["weak"].to_numpy(dtype=bool),
            "df": sub,
        }
        print(f"  {split}: {X.shape}")
    return out, drug_moe, food_moe


def metrics(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y, pred, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class": {
            int(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(LABELS)
        },
        "confusion_matrix": cm.tolist(),
    }


def make_estimator(name, balanced):
    if name == "XGBoost":
        base = XGBClassifier(
            tree_method="hist", objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            n_estimators=100, max_depth=6, gamma=0.0, colsample_bytree=1.0,
            min_child_weight=1, learning_rate=0.1, n_jobs=-1, random_state=42,
        )
    elif name == "LightGBM":
        base = LGBMClassifier(
            n_estimators=100, max_depth=-1, num_leaves=31, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbose=-1,
        )
    elif name == "RandomForest":
        base = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    elif name == "ExtraTrees":
        base = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    elif name == "MLP":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42)),
        ])
    else:
        raise ValueError(name)
    if balanced and name != "MLP":
        return BalancedClassifier(base=base)
    return base


def cv_table(cfg, data, split):
    X, y, df = data["X"], data["y"], data["df"]
    groups = df["drug_id"].to_numpy() if split == "G" else None
    cv = GroupKFold(n_splits=cfg["models"]["cv_folds"]) if split == "G" else StratifiedKFold(
        n_splits=cfg["models"]["cv_folds"], shuffle=True, random_state=cfg["seed"]
    )
    rows = []
    for name in ("XGBoost", "LightGBM", "RandomForest", "ExtraTrees", "MLP"):
        for balanced in ((True, False) if name != "MLP" else (False,)):
            t0 = time.time()
            est = make_estimator(name, balanced)
            pred = cross_val_predict(est, X, y, cv=cv, groups=groups, n_jobs=1)
            m = metrics(y, pred)
            rows.append({
                "model": name, "balanced": balanced, "split": f"Split {split}",
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "precision_0": m["per_class"][0]["precision"], "recall_0": m["per_class"][0]["recall"],
                "precision_1": m["per_class"][1]["precision"], "recall_1": m["per_class"][1]["recall"],
                "precision_2": m["per_class"][2]["precision"], "recall_2": m["per_class"][2]["recall"],
                "seconds": round(time.time() - t0, 1),
            })
            print(f"CV {name} balanced={balanced}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
    return rows


def fit_predict_val(name, balanced, Xtr, ytr, Xva, yva):
    est = make_estimator(name, balanced)
    est.fit(Xtr, ytr)
    pred = est.predict(Xva)
    return est, metrics(yva, pred)


def tune(name, Xtr, ytr, wtr, Xva, yva, wva, cfg, n_iter):
    grids = {
        "XGBoost": {
            "n_estimators": [200, 300, 400, 600, 800, 1000],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
            "min_child_weight": [1, 3, 5, 7, 10],
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.5],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        },
        "LightGBM": {
            "n_estimators": [200, 400, 600], "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.02, 0.05, 0.1], "max_depth": [-1, 10, 20],
            "min_child_samples": [5, 20],
        },
        "RandomForest": {
            "n_estimators": [200, 300, 400], "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.5],
        },
        "ExtraTrees": {
            "n_estimators": [200, 300, 400], "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.5],
        },
        "MLP": {
            "mlp__hidden_layer_sizes": [(32, 16), (64, 32), (128,)],
            "mlp__alpha": [1e-4, 1e-3], "mlp__learning_rate_init": [0.001, 0.01],
        },
    }
    base = make_estimator(name, False)
    Xtv = np.vstack([Xtr, Xva])
    ytv = np.r_[ytr, yva]
    weak_tv = np.r_[wtr, wva]
    ps = PredefinedSplit(np.r_[np.full(len(Xtr), -1), np.zeros(len(Xva), dtype=int)])
    use_weights = name != "MLP"
    weights = compute_sample_weight("balanced", ytv) if use_weights else None
    if weights is not None:
        weights = weights * np.where(weak_tv, cfg["models"].get("weak_sample_weight", 1.0), 1.0)
    search = RandomizedSearchCV(
        base, grids[name], n_iter=n_iter, scoring="f1_macro", cv=ps,
        random_state=cfg["seed"], n_jobs=-1, refit=False, verbose=0,
    )
    t0 = time.time()
    if weights is None:
        search.fit(Xtv, ytv)
    else:
        search.fit(Xtv, ytv, sample_weight=weights)
    params = search.best_params_
    final = make_estimator(name, True)
    if name == "MLP":
        final = clone(base).set_params(**params)
        final.fit(Xtr, ytr)
    else:
        final.base.set_params(**params)
        final.fit(Xtr, ytr, sample_weight=np.where(wtr, cfg["models"].get("weak_sample_weight", 1.0), 1.0))
    val = metrics(yva, final.predict(Xva))
    table = pd.DataFrame(search.cv_results_)
    table["model"] = name
    return params, val, table, round(time.time() - t0, 1), final


def cap_frame(d, cap, seed):
    if len(d["y"]) <= cap:
        return d
    rng = np.random.default_rng(seed)
    weak = d["weak"]
    keys = pd.Series([f"{int(a)}|{int(b)}" for a, b in zip(d["y"], weak)])
    alloc = (keys.value_counts(normalize=True) * cap).round().astype(int).clip(lower=1)
    while alloc.sum() > cap:
        alloc[alloc.idxmax()] -= 1
    while alloc.sum() < cap:
        extra = keys.value_counts()
        candidates = extra[extra > alloc].index
        if not len(candidates):
            break
        alloc[candidates[0]] += 1
    picked = []
    for key, group in pd.Series(np.arange(len(keys))).groupby(keys.to_numpy()):
        n = min(int(alloc.get(key, 1)), len(group))
        picked.append(rng.choice(group.to_numpy(), size=n, replace=False))
    idx = np.sort(np.concatenate(picked))
    return {
        "X": d["X"][idx],
        "y": d["y"][idx],
        "weak": weak[idx],
        "df": d["df"].iloc[idx].reset_index(drop=True),
    }


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["P", "S"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    reports = Path(cfg["paths"]["reports"])
    models = Path(cfg["paths"]["models"])

    sel = load_json(models / f"selected_features_{args.mode}.json")
    names = [f["name"] for f in sel["features"]]
    print(f"Mode {args.mode}: {len(names)} selected features")
    data, drug_moe, food_moe = build_features(cfg, args.mode, names)

    cap = int(cfg["models"].get("train_cap", 50000))
    full_data = data
    data = {
        "train": cap_frame(data["train"], cap, cfg["seed"]),
        "val": cap_frame(data["val"], int(cap * 0.8), cfg["seed"] + 1),
        "test": data["test"],
    }
    print(f"training cap {cap:,}: train {len(data['train']['y']):,}, val {len(data['val']['y']):,}", flush=True)

    print("Baseline CV on Split G grouped folds...")
    cv_rows = cv_table(cfg, data["train"], "G")
    pd.DataFrame(cv_rows).to_csv(reports / f"cv_results_{args.mode}.csv", index=False)

    print("Untuned validation scores...")
    val_rows = []
    fitted = {}
    for name in ("XGBoost", "LightGBM", "RandomForest", "ExtraTrees", "MLP"):
        for balanced in ((True, False) if name != "MLP" else (False,)):
            est, m = fit_predict_val(
                name, balanced, data["train"]["X"], data["train"]["y"], data["val"]["X"], data["val"]["y"]
            )
            fitted[(name, balanced)] = est
            val_rows.append({"model": name, "balanced": balanced, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
            print(f"  val {name} balanced={balanced}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

    print("Tuning on validation (PredefinedSplit)...")
    tuning = {}
    tune_rows = []
    tuned_models = {}
    iters = {"XGBoost": int(cfg["models"]["search_iterations"]), "LightGBM": 10,
             "RandomForest": 8, "ExtraTrees": 8, "MLP": 4}
    for name in ("XGBoost", "LightGBM", "RandomForest", "ExtraTrees", "MLP"):
        params, m, table, secs, final = tune(
            name,
            data["train"]["X"], data["train"]["y"], data["train"]["weak"],
            data["val"]["X"], data["val"]["y"], data["val"]["weak"],
            cfg, iters[name],
        )
        tuning[name] = {"best_params": params, "val": m, "seconds": secs}
        tuned_models[name] = final
        table.to_csv(reports / f"tuning_{name}_{args.mode}.csv", index=False)
        tune_rows.append({"model": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "seconds": secs})
        print(f"  tuned {name}: val macroF1={m['macro_f1']:.4f}")

    weak_weight = float(cfg["models"].get("weak_sample_weight", 1.0))
    best_name = max(tuning, key=lambda k: tuning[k]["val"]["macro_f1"])
    best_model = tuned_models[best_name]
    if best_name == "MLP":
        best_model.fit(full_data["train"]["X"], full_data["train"]["y"])
    else:
        best_model.fit(
            full_data["train"]["X"], full_data["train"]["y"],
            sample_weight=np.where(full_data["train"]["weak"], weak_weight, 1.0),
        )
    joblib.dump(best_model, models / f"best_model_{args.mode}.joblib")
    best_val_full = metrics(full_data["val"]["y"], best_model.predict(full_data["val"]["X"]))
    print(f"best={best_name}; full-train refit val macroF1={best_val_full['macro_f1']:.4f}", flush=True)

    xgb = tuned_models["XGBoost"]
    xgb.fit(
        full_data["train"]["X"], full_data["train"]["y"],
        sample_weight=np.where(full_data["train"]["weak"], weak_weight, 1.0),
    )
    xgb_val_full = metrics(full_data["val"]["y"], xgb.predict(full_data["val"]["X"]))
    booster = xgb.base if isinstance(xgb, BalancedClassifier) else xgb
    booster.save_model(str(models / f"xgb_final_{args.mode}.json"))
    reloaded = XGBClassifier()
    reloaded.load_model(str(models / f"xgb_final_{args.mode}.json"))
    p1 = booster.predict_proba(full_data["val"]["X"])
    p2 = reloaded.predict_proba(full_data["val"]["X"])
    reload_diff = float(np.max(np.abs(p1 - p2)))

    result = {
        "mode": args.mode,
        "n_features": len(names),
        "features": names,
        "train_cap": cap,
        "weak_sample_weight": weak_weight,
        "train_class_counts": {
            "all": {int(k): int(v) for k, v in pd.Series(full_data["train"]["y"]).value_counts().sort_index().items()},
            "weak": {int(k): int(v) for k, v in pd.Series(full_data["train"]["y"][full_data["train"]["weak"]]).value_counts().sort_index().items()},
        },
        "cv_split_G": cv_rows,
        "validation_untuned": val_rows,
        "tuning": tuning,
        "best_model": best_name,
        "best_val_macro_f1": tuning[best_name]["val"]["macro_f1"],
        "best_val_full_refit": best_val_full,
        "xgb_val_full_refit": xgb_val_full,
        "xgb_reload_max_prob_diff": reload_diff,
        "tuning_rows": tune_rows,
    }
    train_drugs = sorted({str(x) for x in full_data["train"]["df"]["drug_id"].unique()})
    save_json(train_drugs, models / f"train_drugs_{args.mode}.json")
    result["train_drugs"] = len(train_drugs)
    save_json(result, reports / f"brick8_models_{args.mode}.json")

    lines = [
        f"# Brick 8 - models (Mode {args.mode})",
        "",
        f"Selected features: {len(names)}. Baselines use 5-fold grouped CV on a {cap:,}-pair sample of the Split G training partition.",
        "The winning model and XGBoost are then refit on the full training partition before being saved.",
        "",
        "## Baseline CV",
        "",
        pd.DataFrame(cv_rows)[["model", "balanced", "accuracy", "macro_f1", "recall_0"]].to_markdown(index=False),
        "",
        "## Validation after tuning",
        "",
        pd.DataFrame(tune_rows).to_markdown(index=False),
        "",
        f"Best by validation macro-F1: **{best_name}** ({tuning[best_name]['val']['macro_f1']:.4f} on the sample).",
        f"After the full-training refit: macro-F1 {best_val_full['macro_f1']:.4f} "
        f"(accuracy {best_val_full['accuracy']:.4f}). XGBoost full refit macro-F1 {xgb_val_full['macro_f1']:.4f}.",
        f"XGBoost reload max probability difference: {reload_diff:.2e}.",
        "",
        "Tuned XGBoost validation confusion matrix:",
        "",
        pd.DataFrame(xgb_val_full["confusion_matrix"], index=["true 0", "true 1", "true 2"],
                     columns=["pred 0", "pred 1", "pred 2"]).to_markdown(),
    ]
    (reports / f"brick8_models_{args.mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
