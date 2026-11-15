"""FastAPI service for the drug-food interaction recommender.

Run from dfi/:  uvicorn web.app:app --reload --port 8000
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
sys.path.insert(0, str(ROOT / "src"))

from common import norm_name  # noqa: E402
from recommend import DFIRecommender  # noqa: E402

LABELS = {0: "Avoid together", 1: "Beneficial / take with food", 2: "No significant interaction"}
SEVERITY = {0: "negative", 1: "positive", 2: "neutral"}

_engines: dict[str, DFIRecommender] = {}
_names: dict[str, list[str]] = {}
_lock = threading.Lock()


def engine(mode: str) -> DFIRecommender:
    if mode not in _engines:
        with _lock:
            if mode not in _engines:
                _engines[mode] = DFIRecommender(mode)
    return _engines[mode]


def names(kind: str) -> list[str]:
    if kind not in _names:
        e = engine("S")
        if kind == "drug":
            values = {n for n in e.drug_by_name if n}
            display = {norm_name(v): v for v in e.drug_name.values()}
            _names["drug"] = sorted((display.get(k, k) for k in values), key=str.lower)
        else:
            compounds = {norm_name(v): v for v in e.foods["name"]}
            foods = {norm_name(v): v for v in e.links["food_name"].unique()}
            merged = {**compounds, **foods}
            _names["food"] = sorted(merged.values(), key=str.lower)
    return _names[kind]


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine("S")
    yield
    _engines.clear()
    _names.clear()


app = FastAPI(title="Drug-Food Interaction Predictor", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")


class PredictRequest(BaseModel):
    drug: str = Field(min_length=1, max_length=300, description="Drug name or SMILES")
    food: str = Field(min_length=1, max_length=300, description="Food or compound name, or SMILES")
    mode: Literal["S", "P"] = "S"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "static" / "landing.html")


@app.get("/predict")
def predict_page() -> FileResponse:
    return FileResponse(WEB / "static" / "index.html")


@app.get("/about")
def about_page() -> FileResponse:
    return FileResponse(WEB / "static" / "about.html")


@app.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    safe = Path(name).name
    if safe != name or not safe.endswith(".png"):
        raise HTTPException(status_code=404, detail="Figure not found.")
    path = ROOT / "figures" / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(path)


@app.get("/api/health")
def health() -> dict:
    e = engine("S")
    return {
        "status": "ready",
        "mode": "S",
        "drugs": int(len(e.drugs)),
        "compounds": int(len(e.foods)),
        "confidence_threshold": float(e.cfg["models"]["confidence_threshold"]),
    }


@app.get("/api/examples")
def examples() -> dict:
    return {
        "pairs": [
            {"drug": "warfarin", "food": "grapefruit juice", "note": "avoid · monitor INR"},
            {"drug": "simvastatin", "food": "grapefruit", "note": "CYP3A4 inhibition"},
            {"drug": "metronidazole", "food": "alcohol", "note": "disulfiram-like"},
            {"drug": "warfarin", "food": "dimethyl disulfide", "note": "paper case"},
            {"drug": "tetracycline", "food": "calcium", "note": "chelation"},
            {"drug": "methotrexate", "food": "ethanol", "note": "abstains"},
            {"drug": "warfarin", "food": "glucose", "note": "control"},
            {"drug": "warfarin", "food": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "note": "SMILES"},
        ]
    }


@app.get("/api/suggest")
def suggest(
    kind: Literal["drug", "food"] = "drug",
    q: str = Query("", max_length=100),
    limit: int = Query(8, ge=1, le=20),
) -> dict:
    key = norm_name(q)
    pool = names(kind)
    if not key:
        picks = pool[:limit]
    else:
        starts = [n for n in pool if norm_name(n).startswith(key)]
        contains = [n for n in pool if key in norm_name(n) and n not in starts]
        picks = (starts + contains)[:limit]
    return {"suggestions": picks}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    drug, food = req.drug.strip(), req.food.strip()
    if not drug or not food:
        raise HTTPException(status_code=422, detail="Both fields are required.")
    t0 = time.perf_counter()
    try:
        result = dict(engine(req.mode).recommend(drug, food))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifacts are missing; run scripts/run_all.sh.") from exc
    result["mode"] = req.mode
    result["inputs"] = {"drug": drug, "food": food}
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    cls = result.get("predicted_class")
    if cls is not None:
        cls = int(cls)
        if result.get("status") in ("ok", "rule", "evidence"):
            result["label"] = LABELS[cls]
            result["severity"] = SEVERITY[cls]
        else:
            result["label"] = "Low confidence - ask a pharmacist"
            result["severity"] = "warn"
    probs = result.get("probabilities")
    if probs:
        def getp(c: int) -> float:
            return float(probs[str(c)] if str(c) in probs else probs[c])

        result["probability_rows"] = [
            {"class": c, "label": LABELS[c], "probability": getp(c)} for c in (0, 1, 2)
        ]
        result["probabilities"] = None
    if "confidence" in result and result["confidence"] is not None:
        result["confidence"] = float(result["confidence"])
    if "domain_similarity" in result and result["domain_similarity"] is not None:
        result["domain_similarity"] = float(result["domain_similarity"])
    return result
