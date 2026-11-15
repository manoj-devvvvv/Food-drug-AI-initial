"""Brick 7 - four-step feature selection (variance, correlation, LASSO, RFE/RFECV).

Usage: python src/brick7_selection.py --mode S
       python src/brick7_selection.py --mode P
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold, f_classif
from sklearn.linear_model import Lasso, LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from common import load_config, save_json, setup_stdout
from features import DESC_NAMES, N_DESC, ALL_PAIR_NAMES, pair_matrix

MEANINGS = {
    "LabuteASA": "Labute approximate surface area",
    "MTPSA": "topological polar surface area (MOE TPSA)",
    "TPSA1": "topological polar surface area (RDKit TPSA)",
    "slogPVSA": "van der Waals surface-area contribution to SlogP (hydrophobicity)",
    "MRVSA": "surface-area contribution to molar refractivity",
    "PEOEVSA": "surface-area contribution to Gasteiger partial charge",
    "EstateVSA": "surface-area contribution to electrotopological state",
    "VSAEstate": "VSA-binned electrotopological state contribution",
}


def descriptor_meaning(name: str) -> str:
    base = re.sub(r"\d+$", "", name)
    return MEANINGS.get(base, base)


def parse_pair_name(name: str) -> dict:
    if name.startswith("abs("):
        a, b = name[4:-1].split("-", 1)
        return {"term": "abs_diff", "drug_descriptor": a, "food_descriptor": b}
    if "*" in name:
        a, b = name.split("*", 1)
        return {"term": "product", "drug_descriptor": a, "food_descriptor": b}
    if "+" in name:
        a, b = name.split("+", 1)
        return {"term": "sum", "drug_descriptor": a, "food_descriptor": b}
    a, b = name.split("/", 1)
    return {"term": "ratio", "drug_descriptor": a, "food_descriptor": b}


def build_sample(cfg, mode, sample_size, seed):
    processed = Path(cfg["paths"]["processed"])
    interim = Path(cfg["paths"]["interim"])
    splits = pd.read_parquet(processed / f"splits_{mode}.parquet")
    train = splits[splits["split_G"] == "train"].reset_index(drop=True)
    drug_moe = pd.read_parquet(interim / "drug_moe.parquet")
    food_moe = pd.read_parquet(interim / "food_moe.parquet")

    drow = {k: i for i, k in enumerate(drug_moe["molecule_id"])}
    frow = {k: i for i, k in enumerate(food_moe["molecule_id"])}
    train = train[train["drug_id"].isin(drow) & train["compound_id"].isin(frow)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if len(train) > sample_size:
        choose = rng.choice(len(train), size=sample_size, replace=False)
        train = train.iloc[choose].reset_index(drop=True)
    di = np.array([drow[x] for x in train["drug_id"]], dtype=np.int64)
    fi = np.array([frow[x] for x in train["compound_id"]], dtype=np.int64)
    X = np.empty((len(train), len(ALL_PAIR_NAMES)), dtype=np.float32)
    for start, chunk in pair_matrix(
        drug_moe[DESC_NAMES].to_numpy(), food_moe[DESC_NAMES].to_numpy(), di, fi,
        names=None, chunk_size=2500, dtype=np.float32,
    ):
        X[start:start + len(chunk)] = chunk
    y = train["label"].to_numpy()
    return X, y, train, drug_moe, food_moe


def variance_step(X, names, threshold):
    vt = VarianceThreshold(threshold).fit(X)
    mask = vt.get_support()
    return X[:, mask], [n for n, m in zip(names, mask) if m]


def correlation_step(X, y, names, cutoff):
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr)
    fval, _ = f_classif(X, y)
    order = np.argsort(-fval)
    kept = []
    for i in order:
        if not kept or np.max(np.abs(corr[i, kept])) < cutoff:
            kept.append(int(i))
    kept = sorted(kept)
    return X[:, kept], [names[i] for i in kept], corr, fval


def lasso_step(X, y, names, alpha):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Lasso(alpha=alpha, max_iter=2000, tol=1e-3, selection="random", random_state=42)
    model.fit(Xs, y)
    coef = model.coef_
    nz = np.flatnonzero(np.abs(coef) > 0)
    return Xs, coef, nz, [names[i] for i in nz]


def logistic_l1(X, y, sample=8000, seed=42):
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        idx = rng.choice(len(X), size=sample, replace=False)
        X, y = X[idx], y[idx]
    try:
        model = LogisticRegression(
            penalty="l1", solver="saga", C=0.01, max_iter=200,
            multi_class="multinomial", random_state=seed, n_jobs=-1, tol=1e-3,
        )
        model.fit(X, y)
        return np.flatnonzero(np.abs(model.coef_).sum(axis=0) > 0), None
    except Exception as exc:
        return np.array([], dtype=int), str(exc)


def rfe_steps(X, y, names, full=True):
    def est():
        return RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))

    rfe = RFE(est(), n_features_to_select=18, step=0.1 if full else 0.2)
    rfe.fit(X, y)
    rfe_features = [names[i] for i, s in enumerate(rfe.support_) if s]

    def score(features, cv):
        idx = [names.index(f) for f in features]
        return float(np.mean(cross_val_score(
            est(), X[:, idx], y, cv=cv, scoring="f1_macro", n_jobs=1,
        )))

    out = {
        "rfe18_features": rfe_features,
        "rfe18_cv_score": score(rfe_features, 5 if full else 3),
    }
    if full:
        rfecv = RFECV(est(), step=0.1, min_features_to_select=8, cv=5, scoring="f1_macro", n_jobs=1)
        rfecv.fit(X, y)
        rfecv_features = [names[i] for i, s in enumerate(rfecv.support_) if s]
        out.update({
            "rfecv_n_features": int(rfecv.n_features_),
            "rfecv_features": rfecv_features,
            "rfecv_cv_score": score(rfecv_features, 5),
            "rfecv_curve": {
                int(k): float(v) for k, v in zip(rfecv.cv_results_["n_features"], rfecv.cv_results_["mean_test_score"])
            },
        })
    else:
        out.update({
            "rfecv_n_features": 18, "rfecv_features": rfe_features,
            "rfecv_cv_score": out["rfe18_cv_score"], "rfecv_curve": {},
        })
    out["rfe_ranking"] = {n: int(r) for n, r in zip(names, rfe.ranking_)}
    return out


def selection_run(X, y, names, cfg, full=True):
    t0 = time.time()
    Xv, names_v = variance_step(X, names, cfg["features"]["variance_threshold"])
    t1 = time.time()
    Xk, names_k, corr, fval = correlation_step(Xv, y, names_v, cfg["features"]["pearson_cutoff"])
    del Xv, corr, fval
    t2 = time.time()
    Xs, coef, nz, names_l = lasso_step(Xk, y, names_k, cfg["features"]["lasso_alpha"])
    del Xk, coef
    t3 = time.time()
    X4 = np.ascontiguousarray(Xs[:, nz])
    del Xs
    if full:
        algo_l, err = logistic_l1(X4, y)
    else:
        algo_l, err = np.array([], dtype=int), "skipped in stability runs"
    names_l1 = [names_l[i] for i in algo_l] if len(algo_l) else []
    rfe = rfe_steps(X4, y, names_l, full=full)
    del X4
    t4 = time.time()
    return {
        "names_v": names_v, "names_k": names_k,
        "nz": nz, "names_l": names_l, "rfe": rfe,
        "counts": {
            "start": len(names),
            "after_variance": len(names_v),
            "after_correlation": len(names_k),
            "after_lasso": len(names_l),
            "after_rfecv": rfe["rfecv_n_features"],
            "rfe18": 18,
        },
        "lasso_nonzero": int(len(nz)),
        "l1_logistic_nonzero": int(len(algo_l)),
        "l1_overlap_with_lasso": int(len(set(names_l1) & set(names_l))),
        "l1_error": err,
        "timings": {
            "variance": round(t1 - t0, 1), "correlation": round(t2 - t1, 1),
            "lasso": round(t3 - t2, 1), "rfe": round(t4 - t3, 1),
        },
    }


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["P", "S"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    reports = Path(cfg["paths"]["reports"])
    models = Path(cfg["paths"]["models"])
    interim = Path(cfg["paths"]["interim"])

    n = int(cfg["features"]["selection_sample"])
    X, y, train, drug_moe, food_moe = build_sample(cfg, args.mode, n, cfg["seed"])
    print(f"selection sample {X.shape}, class counts {pd.Series(y).value_counts().sort_index().to_dict()}")

    base = selection_run(X, y, ALL_PAIR_NAMES, cfg)
    print("counts", base["counts"], "timings", base["timings"], flush=True)

    # ---- Stability: repeat the selection on three subsamples. The expensive
    # RFECV and L1-logistic diagnostics are skipped; the four selection steps
    # still run end to end, fixed at the 18-feature target.
    rng = np.random.default_rng(cfg["seed"])
    runs = []
    for r in range(int(cfg["features"]["stability_runs"])):
        idx = np.sort(rng.choice(len(X), size=int(0.6 * len(X)), replace=False))
        Xs_sub = X[idx]
        ys_sub = y[idx]
        res = selection_run(Xs_sub, ys_sub, ALL_PAIR_NAMES, cfg, full=False)
        del Xs_sub, ys_sub
        runs.append({"seed_rows": int(len(idx)), "features": res["rfe"]["rfe18_features"],
                     "counts": res["counts"]})
        print(f"stability run {r + 1}: {len(res['rfe']['rfe18_features'])} features", flush=True)

    freq: dict[str, int] = {}
    for run in runs:
        for f in set(run["features"]):
            freq[f] = freq.get(f, 0) + 1
    stable = {f for f, c in freq.items() if c >= 2}
    jaccard = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = set(runs[i]["features"]), set(runs[j]["features"])
            jaccard.append({"run_a": i + 1, "run_b": j + 1, "jaccard": round(len(a & b) / len(a | b), 3)})

    # Final 18: stable features first, then the full-sample RFE18 as filler.
    rfe18 = base["rfe"]["rfe18_features"]
    final = [f for f in rfe18 if f in stable]
    if len(final) < 18:
        final += [f for f in rfe18 if f not in final]
    if len(final) < 18:
        final += [f for f in base["rfe"]["rfecv_features"] if f not in final]
    final = final[:18]

    idx_final = [ALL_PAIR_NAMES.index(f) for f in final]
    scaler = StandardScaler()
    scaler.fit(X[:, idx_final])
    joblib.dump(scaler, models / f"scaler_{args.mode}.pkl")

    feature_info = []
    for f in final:
        p = parse_pair_name(f)
        feature_info.append({
            "name": f, **p,
            "drug_descriptor_meaning": descriptor_meaning(p["drug_descriptor"]),
            "food_descriptor_meaning": descriptor_meaning(p["food_descriptor"]),
            "selected_in_full_run": f in rfe18,
            "stability_hits": int(freq.get(f, 0)),
        })
    out = {
        "mode": args.mode,
        "selection_split": "Split G training partition",
        "n_selected": len(feature_info),
        "features": feature_info,
        "counts": base["counts"],
        "paper_counts": {"start": 3780, "after_variance": 2845, "after_correlation": 1568, "after_lasso": 1516, "final": 18},
        "rfe_comparison": {
            "rfecv_n_features": base["rfe"]["rfecv_n_features"],
            "rfecv_cv_macro_f1": base["rfe"]["rfecv_cv_score"],
            "rfe18_cv_macro_f1": base["rfe"]["rfe18_cv_score"],
            "rfecv_curve": base["rfe"]["rfecv_curve"],
        },
        "l1_logistic_comparison": {
            "lasso_nonzero": base["lasso_nonzero"],
            "l1_logistic_nonzero": base["l1_logistic_nonzero"],
            "overlap": base["l1_overlap_with_lasso"],
            "error": base["l1_error"],
        },
        "stability": {
            "runs": runs,
            "frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
            "stable_features": sorted(stable),
            "jaccard_pairwise": jaccard,
            "stable_kept_in_final": int(sum(1 for f in final if f in stable)),
        },
        "timings": base["timings"],
        "sample_rows": int(len(X)),
        "class_counts": {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
    }
    save_json(out, models / f"selected_features_{args.mode}.json")
    save_json({"counts": base["counts"], "timings": base["timings"], "sample_rows": int(len(X))},
              reports / f"brick7_selection_log_{args.mode}.json")

    lines = [
        f"# Brick 7 - feature selection (Mode {args.mode})",
        "",
        f"Selection sample: {len(X):,} training pairs; class counts {out['class_counts']}.",
        "",
        "| Step | Remaining features |",
        "|---|---|",
        f"| start | {base['counts']['start']} |",
        f"| variance > {cfg['features']['variance_threshold']} | {base['counts']['after_variance']} |",
        f"| Pearson |r| < {cfg['features']['pearson_cutoff']} | {base['counts']['after_correlation']} |",
        f"| LASSO alpha={cfg['features']['lasso_alpha']} nonzero | {base['counts']['after_lasso']} |",
        f"| RFECV optimum | {base['counts']['after_rfecv']} |",
        f"| RFE fixed 18 | 18 |",
        "",
        f"RFECV macro-F1 {out['rfe_comparison']['rfecv_cv_macro_f1']:.4f}; "
        f"RFE18 macro-F1 {out['rfe_comparison']['rfe18_cv_macro_f1']:.4f}.",
        "",
        f"Stability: {out['stability']['stable_kept_in_final']} of 18 final features were chosen in at least 2 of 3 runs.",
        "",
        "## Selected features",
        "",
        pd.DataFrame(feature_info)[["name", "term", "stability_hits"]].to_markdown(index=False),
        "",
        f"LASSO vs L1-logistic: {base['lasso_nonzero']} vs {base['l1_logistic_nonzero']} features, "
        f"overlap {base['l1_overlap_with_lasso']}.",
    ]
    (reports / f"brick7_selection_{args.mode}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
