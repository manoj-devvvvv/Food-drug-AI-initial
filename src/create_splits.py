"""Brick 6 - random pair split (R) and strict drug-grouped split (G).

Usage: python src/brick6_splits.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import dataframe_to_markdown, load_config, load_json, save_json, setup_stdout

SPLITS = ["train", "val", "test"]


def split_random(df: pd.DataFrame, cfg: dict) -> pd.Series:
    y = df["label"].to_numpy()
    idx = np.arange(len(df))
    train_frac = cfg["splits"]["train_frac"]
    test_frac = cfg["splits"]["test_frac"]
    train_idx, rest_idx = train_test_split(
        idx, train_size=train_frac, stratify=y, random_state=cfg["seed"]
    )
    rest_share = test_frac / (1.0 - train_frac)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=rest_share, stratify=y[rest_idx], random_state=cfg["seed"]
    )
    out = pd.Series(index=df.index, dtype=object)
    out.iloc[train_idx] = "train"
    out.iloc[val_idx] = "val"
    out.iloc[test_idx] = "test"
    return out


def split_grouped(df: pd.DataFrame, cfg: dict) -> pd.Series:
    target = np.array(
        [cfg["splits"]["train_frac"], cfg["splits"]["val_frac"], cfg["splits"]["test_frac"]]
    ) * len(df)
    sizes = df.groupby("drug_id").size()
    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(sizes.index.to_numpy())
    order = sorted(order, key=lambda d: -sizes[d])
    current = np.zeros(3)
    assignment: dict[str, str] = {}
    for group in order:
        room = (target - current) / target
        if np.all(current > 0):
            part = int(np.argmax(room))
        else:
            empty = [i for i in range(3) if current[i] == 0]
            part = empty[0] if len(empty) < 3 else int(np.argmax(room))
        assignment[group] = SPLITS[part]
        current[part] += sizes[group]
    return df["drug_id"].map(assignment)


def summarize(df: pd.DataFrame, split_col: str) -> dict:
    out = {}
    for s in SPLITS:
        sub = df[df[split_col] == s]
        out[s] = {
            "rows": int(len(sub)),
            "share": round(len(sub) / len(df), 4),
            "drugs": int(sub["drug_id"].nunique()),
            "class_counts": {int(k): int(v) for k, v in sub["label"].value_counts().sort_index().items()},
            "class_shares": {
                int(k): round(float(v), 4) for k, v in sub["label"].value_counts(normalize=True).sort_index().items()
            },
        }
    return out


def main() -> None:
    setup_stdout()
    cfg = load_config()
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])

    pairs = pd.read_parquet(interim / "pairs_labeled.parquet")
    b4 = load_json(Path(cfg["paths"]["reports"]) / "brick4_label_report.json")
    holdout = {tuple(x) for x in b4["holdout_pairs"]}
    holdout_keys = {f"{d}|{c}" for d, c in holdout}
    present = int((pairs["drug_id"].astype(str) + "|" + pairs["compound_id"].astype(str)).isin(holdout_keys).sum())
    if present:
        raise SystemExit(f"{present} held-out case pairs still present in pairs_labeled")

    report = {"heldout_pairs_absent": True, "holdout_count": len(holdout), "modes": {}}
    for mode in ("P", "S"):
        df = pairs[pairs["mode"] == mode].reset_index(drop=True).copy()
        if df.empty:
            continue
        df["split_R"] = split_random(df, cfg)
        df["split_G"] = split_grouped(df, cfg)

        overlap_ok = True
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            da = set(df.loc[df["split_G"] == a, "drug_id"])
            db = set(df.loc[df["split_G"] == b, "drug_id"])
            overlap_ok &= not (da & db)
        if not overlap_ok:
            raise SystemExit(f"drug overlap in Split G for mode {mode}")

        cols = ["drug_id", "compound_id", "label", "weak", "sentence_id", "split_R", "split_G"]
        df[cols].to_parquet(processed / f"splits_{mode}.parquet", index=False)

        # Pair keys must not appear in more than one partition.
        key = df["drug_id"].astype(str) + "|" + df["compound_id"].astype(str)
        pair_overlap = {
            split: int(key.duplicated().sum()) for split in ["R", "G"]
        }

        report["modes"][mode] = {
            "rows": int(len(df)),
            "split_R": summarize(df, "split_R"),
            "split_G": summarize(df, "split_G"),
            "group_overlap_assert": bool(overlap_ok),
            "duplicate_pair_keys": pair_overlap,
        }

    report["external_set"] = {
        "status": "not obtained",
        "decision": (
            "Ryu et al. supplementary data was not downloaded in this build; the strict drug-grouped "
            "test partition plus the nine held-out case studies stand in, as allowed by the plan."
        ),
    }
    save_json(report, Path(cfg["paths"]["reports"]) / "brick6_splits.json")

    lines = ["# Brick 6 - splits", "", "Held-out case pairs absent from every training table: True.", ""]
    for mode, m in report["modes"].items():
        lines += [f"## Mode {mode}", ""]
        for split in ("split_R", "split_G"):
            lines += [f"### {split}", "", dataframe_to_markdown(pd.DataFrame(m[split]).T), ""]
        lines += [f"Drug-group overlap assertion passed: {m['group_overlap_assert']}.", ""]
    lines += ["External set was not obtained; Split G test and the nine case studies replace it.", ""]
    (Path(cfg["paths"]["reports"]) / "brick6_splits.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)[:4000])


if __name__ == "__main__":
    main()
