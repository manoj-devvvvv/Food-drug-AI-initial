"""Brick 10 - per-class SHAP explanations for the final XGBoost model.

Usage: python src/brick10_shap.py --mode S
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, DMatrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brick9_evaluate import build_matrix
from common import load_config, load_json, save_json, setup_stdout
from features import DESC_NAMES

PAPER_TOP = ["MRVSA0", "PEOEVSA5", "EstateVSA7", "PEOEVSA9", "EstateVSA2"]


def shap_values(model, X, names):
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            return np.stack(sv, axis=-1), "TreeExplainer"
        sv = np.asarray(sv)
        if sv.ndim == 2:
            sv = sv[..., None]
        return sv, "TreeExplainer"
    except Exception:
        dm = DMatrix(X, feature_names=names)
        contrib = model.get_booster().predict(dm, pred_contribs=True)
        n, ncol = contrib.shape
        k = ncol // (len(names) + 1)
        arr = contrib.reshape(n, k, len(names) + 1)[:, :, : len(names)]
        return arr.transpose(0, 2, 1), "xgboost pred_contribs fallback"


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["P", "S"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    models = Path(cfg["paths"]["models"])
    reports = Path(cfg["paths"]["reports"])
    figures = Path(cfg["paths"]["figures"])
    processed = Path(cfg["paths"]["processed"])
    interim = Path(cfg["paths"]["interim"])

    sel = load_json(models / f"selected_features_{args.mode}.json")
    names = [f["name"] for f in sel["features"]]
    model = XGBClassifier()
    model.load_model(str(models / f"xgb_final_{args.mode}.json"))

    splits = pd.read_parquet(processed / f"splits_{args.mode}.parquet")
    test = splits[splits["split_G"] == "test"]
    drug_moe = pd.read_parquet(interim / "drug_moe.parquet")
    food_moe = pd.read_parquet(interim / "food_moe.parquet")
    X, y, _, df = build_matrix(test, names, drug_moe, food_moe)
    if len(X) > 5000:
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(len(X), size=5000, replace=False)
        X, y = X[idx], y[idx]
    values, backend = shap_values(model, X, names)
    print("SHAP backend", backend, "shape", values.shape, "accuracy on sample",
          round(float(accuracy_score(y, model.predict(X))), 4))

    class_names = {0: "negative", 1: "positive", 2: "non_significant"}
    imp_rows = []
    for c in range(values.shape[2]):
        mean_abs = np.abs(values[:, :, c]).mean(axis=0)
        for i, n in enumerate(names):
            imp_rows.append({"class": class_names[c], "feature": n, "mean_abs_shap": float(mean_abs[i])})
        order = np.argsort(-mean_abs)[:18]
        fig = plt.figure(figsize=(8, 5))
        shap.summary_plot(values[:, :, c], X, feature_names=names, show=False, max_display=18)
        plt.title(f"Mode {args.mode} - class {class_names[c]} SHAP")
        plt.tight_layout()
        fig.savefig(figures / f"shap_summary_{args.mode}_{class_names[c]}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(reports / f"shap_importance_{args.mode}.csv", index=False)

    overall = imp.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False)
    top = overall.head(10)
    feature_table = pd.DataFrame(sel["features"])
    feature_table.to_csv(reports / f"feature_table_{args.mode}.csv", index=False)
    overlap = [f for f in top.index if any(p in f for p in PAPER_TOP)]
    result = {
        "mode": args.mode,
        "backend": backend,
        "sample_rows": int(len(X)),
        "top_10_features": {k: float(v) for k, v in top.items()},
        "paper_top_features": PAPER_TOP,
        "paper_features_present_in_top_10": overlap,
        "per_class_top_5": {
            class_names[c]: imp[imp["class"] == class_names[c]].sort_values("mean_abs_shap", ascending=False)["feature"].head(5).tolist()
            for c in range(3)
        },
    }
    save_json(result, reports / f"brick10_shap_{args.mode}.json")

    lines = [
        f"# Brick 10 - SHAP (Mode {args.mode})",
        "",
        f"Backend: {backend}; sample: {len(X):,} test pairs.",
        "",
        "## Overall top 10 (mean |SHAP| across classes)",
        "",
        top.rename_axis("feature").reset_index(name="mean_abs_shap").to_markdown(index=False),
        "",
        "## Paper's top features",
        "",
        f"Paper list: {', '.join(PAPER_TOP)}. Present in our top 10: {', '.join(overlap) if overlap else 'none'}.",
        "",
        "## Per-class top 5",
        "",
        pd.DataFrame(result["per_class_top_5"]).to_markdown(index=False),
        "",
        "## Feature definitions",
        "",
        feature_table[["name", "term", "drug_descriptor_meaning", "food_descriptor_meaning"]].to_markdown(index=False),
        "",
        "Chemical readings of these features are hypotheses, not confirmed mechanisms.",
    ]
    (reports / f"brick10_shap_{args.mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
