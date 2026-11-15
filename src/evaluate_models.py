"""Brick 9 - evaluate the frozen model on Split G and Split R test partitions.

The Split G number is the headline; Split R is retrained with the same tuned
hyperparameters on its own training partition for a like-for-like comparison.

Usage: python src/brick9_evaluate.py --mode S
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, accuracy_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_config, load_json, save_json, setup_stdout
from features import DESC_NAMES, pair_matrix

LABELS = [0, 1, 2]


def build_matrix(df, feature_names, drug_moe, food_moe):
    dvals = drug_moe[DESC_NAMES].to_numpy()
    fvals = food_moe[DESC_NAMES].to_numpy()
    drow = {k: i for i, k in enumerate(drug_moe["molecule_id"])}
    frow = {k: i for i, k in enumerate(food_moe["molecule_id"])}
    df = df[df["drug_id"].isin(drow) & df["compound_id"].isin(frow)].reset_index(drop=True)
    di = np.array([drow[x] for x in df["drug_id"]], dtype=np.int64)
    fi = np.array([frow[x] for x in df["compound_id"]], dtype=np.int64)
    X = np.empty((len(df), len(feature_names)), dtype=np.float32)
    for start, chunk in pair_matrix(dvals, fvals, di, fi, names=feature_names, chunk_size=20000):
        X[start:start + len(chunk)] = chunk
    return X, df["label"].to_numpy(), df["weak"].to_numpy(dtype=bool), df


def evaluate(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y, pred, labels=LABELS)
    row_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class": {
            int(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(LABELS)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": row_norm.round(4).tolist(),
    }


def save_cm_figure(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), ["negative", "positive", "non-sig"])
    ax.set_yticks(range(3), ["negative", "positive", "non-sig"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
    # Models were pickled from brick8's __main__; make the class resolvable here.
    import brick8_models
    sys.modules["__main__"].BalancedClassifier = brick8_models.BalancedClassifier
    best = joblib.load(models / f"best_model_{args.mode}.joblib")
    info = load_json(reports / f"brick8_models_{args.mode}.json")
    best_name = info["best_model"]
    params = info["tuning"][best_name]["best_params"]

    splits = pd.read_parquet(processed / f"splits_{args.mode}.parquet")
    drug_moe = pd.read_parquet(interim / "drug_moe.parquet")
    food_moe = pd.read_parquet(interim / "food_moe.parquet")

    result = {"mode": args.mode, "model": best_name, "features": names, "splits": {}}
    for split in ("G", "R"):
        train_df = splits[splits[f"split_{split}"] == "train"]
        test_df = splits[splits[f"split_{split}"] == "test"]
        Xtr, ytr, wtr, _ = build_matrix(train_df, names, drug_moe, food_moe)
        Xte, yte, _, test_used = build_matrix(test_df, names, drug_moe, food_moe)
        if split == "G":
            model = best
        else:
            model = clone(best)
            try:
                model.set_params(**params)
            except Exception:
                if hasattr(model, "base") and model.base is not None:
                    model.base.set_params(**params)
                else:
                    raise
            model.fit(
                Xtr, ytr,
                sample_weight=np.where(wtr, cfg["models"].get("weak_sample_weight", 1.0), 1.0),
            )
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)
        m = evaluate(yte, pred)

        # Confidence-threshold sweep: what happens if we only answer when sure?
        conf = proba.max(axis=1)
        sweep = []
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            keep = conf >= t
            sweep.append({
                "threshold": t,
                "coverage": float(keep.mean()),
                "answered": int(keep.sum()),
                "accuracy": float(accuracy_score(yte[keep], pred[keep])) if keep.any() else None,
                "negative_recall": float(
                    ((pred[keep] == 0) & (yte[keep] == 0)).sum() / max((yte[keep] == 0).sum(), 1)
                ) if keep.any() else None,
            })

        # Per-drug accuracy spread: are a few drugs driving the score?
        correct = (pred == yte).astype(float)
        per_drug = test_used.assign(correct=correct).groupby("drug_id")["correct"].agg(["mean", "size"])
        drug_spread = {
            "drugs": int(len(per_drug)),
            "median_accuracy": float(per_drug["mean"].median()),
            "p10_accuracy": float(per_drug["mean"].quantile(0.1)),
            "perfect_drugs": int((per_drug["mean"] == 1.0).sum()),
            "zero_drugs": int((per_drug["mean"] == 0.0).sum()),
            "worst": [
                {"drug_id": str(k), "accuracy": float(v["mean"]), "pairs": int(v["size"])}
                for k, v in per_drug.sort_values("mean").head(5).iterrows()
            ],
        }

        result["splits"][split] = {
            "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
            "test_drugs": int(test_used["drug_id"].nunique()),
            "confidence_sweep": sweep,
            "per_drug_spread": drug_spread,
            **m,
        }
        cm = np.array(m["confusion_matrix"])
        save_cm_figure(cm, f"Mode {args.mode} - Split {split}", figures / f"cm_{args.mode}_{split}.png")
        pd.DataFrame({
            "drug_id": test_used["drug_id"].to_numpy(),
            "compound_id": test_used["compound_id"].to_numpy(),
            "label": yte, "pred": pred,
            "p0": proba[:, 0], "p1": proba[:, 1], "p2": proba[:, 2],
        }).to_parquet(reports / f"predictions_{args.mode}_{split}.parquet", index=False)

    result["external_set"] = {
        "status": "not obtained",
        "substitute": "Split G test + the nine held-out case studies in Brick 11",
    }
    result["paper_reference"] = {
        "external_accuracy": 0.9756,
        "internal_negative_recall": 0.8707,
        "internal_negative_precision": 0.9822,
    }
    save_json(result, reports / f"metrics_{args.mode}.json")

    lines = [f"# Brick 9 - evaluation (Mode {args.mode})", "", f"Frozen model: **{best_name}**.", ""]
    for split in ("G", "R"):
        m = result["splits"][split]
        lines += [
            f"## Split {split} (headline: {'yes' if split == 'G' else 'no'})",
            "",
            f"- Test pairs: {m['test_rows']:,}; drugs: {m['test_drugs']}; accuracy: **{m['accuracy']:.4f}**; "
            f"macro-F1: **{m['macro_f1']:.4f}**.",
            f"- Negative recall: {m['per_class'][0]['recall']:.4f}; precision: {m['per_class'][0]['precision']:.4f}.",
            "",
            pd.DataFrame(m["confusion_matrix"], index=["true 0", "true 1", "true 2"],
                         columns=["pred 0", "pred 1", "pred 2"]).to_markdown(),
            "",
            "Confidence-threshold sweep (answer only when the top probability clears the threshold):",
            "",
            pd.DataFrame(m["confidence_sweep"]).to_markdown(index=False),
            "",
            f"Per-drug spread: {m['per_drug_spread']['drugs']} test drugs; median accuracy "
            f"{m['per_drug_spread']['median_accuracy']:.3f}; p10 {m['per_drug_spread']['p10_accuracy']:.3f}; "
            f"perfect {m['per_drug_spread']['perfect_drugs']}; zero {m['per_drug_spread']['zero_drugs']}.",
            "",
        ]
    lines += [
        "Paper reference: external accuracy 0.9756, internal negative recall 0.8707 and precision 0.9822. "
        "The external set was not obtained; numbers are not directly comparable.",
    ]
    (reports / f"brick9_evaluation_{args.mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
