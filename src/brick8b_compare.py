"""Brick 8b - head-to-head comparison of all five classifiers on held-out test data.

Trains XGBoost, LightGBM, RandomForest, ExtraTrees and MLP with the
hyperparameters tuned in Brick 8, then evaluates every model on the exact same
Split G and Split R test partitions. Produces new artifacts only; existing
models and reports are left untouched.

Usage: python src/brick8b_compare.py --mode S
       python src/brick8b_compare.py --mode P
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_config, load_json, save_json, setup_stdout
from features import DESC_NAMES, pair_matrix
import brick8_models as b8

LABELS = [0, 1, 2]
MODEL_ORDER = ["XGBoost", "LightGBM", "RandomForest", "ExtraTrees", "MLP"]


def load_moe(cfg):
    interim = Path(cfg["paths"]["interim"])
    drug_moe = pd.read_parquet(interim / "drug_moe.parquet")
    food_moe = pd.read_parquet(interim / "food_moe.parquet")
    dvals = drug_moe[DESC_NAMES].to_numpy()
    fvals = food_moe[DESC_NAMES].to_numpy()
    drow = {k: i for i, k in enumerate(drug_moe["molecule_id"])}
    frow = {k: i for i, k in enumerate(food_moe["molecule_id"])}
    return drug_moe, food_moe, dvals, fvals, drow, frow


def build_partition(df, feature_names, dvals, fvals, drow, frow):
    df = df[df["drug_id"].isin(drow) & df["compound_id"].isin(frow)].reset_index(drop=True)
    di = np.array([drow[x] for x in df["drug_id"]], dtype=np.int64)
    fi = np.array([frow[x] for x in df["compound_id"]], dtype=np.int64)
    X = np.empty((len(df), len(feature_names)), dtype=np.float32)
    for start, chunk in pair_matrix(dvals, fvals, di, fi, names=feature_names, chunk_size=20000):
        X[start:start + len(chunk)] = chunk
    return X, df["label"].to_numpy(), df["weak"].to_numpy(dtype=bool), df


def evaluate(y, pred, proba, threshold):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y, pred, labels=LABELS)
    conf = proba.max(axis=1)
    keep = conf >= threshold
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "per_class": {
            int(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(LABELS)
        },
        "confusion_matrix": cm.tolist(),
        "predicted_counts": {int(c): int((pred == c).sum()) for c in LABELS},
        "coverage": float(keep.mean()),
        "selective_accuracy": float(accuracy_score(y[keep], pred[keep])) if keep.any() else None,
        "threshold": float(threshold),
    }


def fit_model(name, params, X, y, weak, cfg):
    weak_weight = float(cfg["models"].get("weak_sample_weight", 1.0))
    if name == "MLP":
        est = b8.make_estimator("MLP", False)
        est.set_params(**params)
        est.fit(X, y)
    else:
        est = b8.make_estimator(name, True)
        est.base.set_params(**params)
        est.fit(X, y, sample_weight=np.where(weak, weak_weight, 1.0))
    return est


def row_for(mode, split, name, train_rows, test_rows, m, seconds):
    c0, c1, c2 = m["per_class"][0], m["per_class"][1], m["per_class"][2]
    return {
        "mode": mode,
        "split": f"Split {split}",
        "model": name,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"],
        "neg_precision": c0["precision"],
        "neg_recall": c0["recall"],
        "neg_f1": c0["f1"],
        "pos_f1": c1["f1"],
        "nonsig_precision": c2["precision"],
        "nonsig_recall": c2["recall"],
        "nonsig_f1": c2["f1"],
        "support_0": c0["support"],
        "support_1": c1["support"],
        "support_2": c2["support"],
        "coverage_at_threshold": m["coverage"],
        "selective_accuracy": m["selective_accuracy"],
        "seconds": round(seconds, 1),
    }


def save_figure(rows, mode, figures):
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, split in zip(axes, ["Split G", "Split R"]):
        sub = df[df["split"] == split].set_index("model").loc[MODEL_ORDER]
        colors = ["#2f6fed", "#3fa34d", "#d9822b", "#8e5bd0", "#c0455a"]
        ax.bar(sub.index, sub["macro_f1"], color=colors)
        ax.set_title(f"Mode {mode} - {split} test")
        ax.set_ylim(0, max(0.65, float(sub["macro_f1"].max()) + 0.08))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(sub["macro_f1"]):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].set_ylabel("macro F1")
    fig.tight_layout()
    path = figures / f"model_comparison_{mode}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def markdown_table(df):
    cols = ["model", "accuracy", "balanced_accuracy", "macro_f1", "neg_precision", "neg_recall",
            "nonsig_recall", "coverage_at_threshold", "selective_accuracy", "seconds"]
    table = df[cols].copy()
    for c in cols[1:-1]:
        table[c] = table[c].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v)
    return table.to_markdown(index=False)


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["P", "S"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    reports = Path(cfg["paths"]["reports"])
    figures = Path(cfg["paths"]["figures"])
    models_dir = Path(cfg["paths"]["models"])

    sel = load_json(models_dir / f"selected_features_{args.mode}.json")
    names = [f["name"] for f in sel["features"]]
    info = load_json(reports / f"brick8_models_{args.mode}.json")
    threshold = float(cfg["models"]["confidence_threshold"])
    print(f"Mode {args.mode}: {len(names)} features; threshold {threshold}; models {MODEL_ORDER}")

    splits = pd.read_parquet(Path(cfg["paths"]["processed"]) / f"splits_{args.mode}.parquet")
    drug_moe, food_moe, dvals, fvals, drow, frow = load_moe(cfg)

    rows = []
    details = {"mode": args.mode, "features": names, "threshold": threshold, "splits": {}}
    for split in ("G", "R"):
        train_df = splits[splits[f"split_{split}"] == "train"]
        test_df = splits[splits[f"split_{split}"] == "test"]
        Xtr, ytr, wtr, _ = build_partition(train_df, names, dvals, fvals, drow, frow)
        Xte, yte, _, test_used = build_partition(test_df, names, dvals, fvals, drow, frow)
        print(f"  Split {split}: train {Xtr.shape} test {Xte.shape}", flush=True)
        details["splits"][f"Split {split}"] = {
            "test_rows": int(len(Xte)),
            "test_drugs": int(test_used["drug_id"].nunique()),
            "models": {},
        }
        for name in MODEL_ORDER:
            params = info["tuning"][name]["best_params"]
            t0 = time.time()
            model = fit_model(name, params, Xtr, ytr, wtr, cfg)
            pred = model.predict(Xte)
            proba = model.predict_proba(Xte)
            seconds = time.time() - t0
            m = evaluate(yte, pred, proba, threshold)
            details["splits"][f"Split {split}"]["models"][name] = {"params": params, **m}
            rows.append(row_for(args.mode, split, name, len(Xtr), len(Xte), m, seconds))
            print(
                f"  {name:12s} {split}: acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
                f"macroF1={m['macro_f1']:.4f} neg_recall={m['per_class'][0]['recall']:.4f} "
                f"({seconds:.1f}s)",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(reports / f"model_comparison_{args.mode}.csv", index=False)
    save_json(details, reports / f"model_comparison_{args.mode}.json")
    fig_path = save_figure(rows, args.mode, figures)

    split_g = df[df["split"] == "Split G"].set_index("model").loc[MODEL_ORDER].reset_index()
    split_r = df[df["split"] == "Split R"].set_index("model").loc[MODEL_ORDER].reset_index()
    best_g = split_g.loc[split_g["macro_f1"].idxmax()]
    best_r = split_r.loc[split_r["macro_f1"].idxmax()]

    lines = [
        f"# Brick 8b - all five models on held-out test data (Mode {args.mode})",
        "",
        f"Features: {len(names)} selected pair descriptors. Weak pairs weighted "
        f"{cfg['models'].get('weak_sample_weight', 1.0)}. Non-MLP models use balanced class weights; "
        f"MLP uses its own early stopping. Each model is refit on the full training partition of the split "
        "and scored on that split's test partition, so no tuning-validation rows leak into the test numbers.",
        "",
        f"Split G tests generalization to unseen drugs (headline). Split R is the paper-like random split for "
        f"reference. Selective accuracy only answers rows whose top probability clears {threshold:.2f}.",
        "",
        "## Split G (unseen drugs)",
        "",
        markdown_table(split_g),
        "",
        f"Best on Split G by macro-F1: **{best_g['model']}** ({best_g['macro_f1']:.4f}).",
        "",
        "## Split R (random split)",
        "",
        markdown_table(split_r),
        "",
        f"Best on Split R by macro-F1: **{best_r['model']}** ({best_r['macro_f1']:.4f}).",
        "",
        "Class 1 has very few real pairs, so its F1 is noisy; class 0 (avoid) and class 2 (non-significant) "
        "dominate the macro average.",
        "",
        f"Figure: `{fig_path.relative_to(Path.cwd()) if fig_path.is_relative_to(Path.cwd()) else fig_path}`",
    ]
    (reports / f"model_comparison_{args.mode}.md").write_text("\n".join(lines), encoding="utf-8")

    combined = reports / "model_comparison.csv"
    parts = [reports / f"model_comparison_{m}.csv" for m in ("S", "P")]
    have = [p for p in parts if p.exists()]
    if have:
        pd.concat([pd.read_csv(p) for p in have], ignore_index=True).to_csv(combined, index=False)

    print("\n".join(lines))
    print(f"\nWrote {reports / f'model_comparison_{args.mode}.csv'}, "
          f"{reports / f'model_comparison_{args.mode}.json'}, {fig_path}")


if __name__ == "__main__":
    main()
