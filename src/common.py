"""Shared paths, config and small helpers for the DFI project."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("interim", "processed", "models", "reports", "figures"):
        p = ROOT / cfg["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(p)
    for key in ("drugs_raw", "interactions_raw", "foods_raw"):
        cfg["paths"][key] = str((ROOT / cfg["paths"][key]).resolve())
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("interim", "processed", "models", "reports", "figures"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def norm_name(value) -> str:
    """Lower-case, strip punctuation and collapse whitespace for lexicon joins."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def setup_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
