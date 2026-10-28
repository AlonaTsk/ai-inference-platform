"""Inference HTTP service.

GET  /healthz  liveness: the process is up
GET  /readyz   readiness: the model is loaded and can serve
POST /predict  {"texts": [...]} -> one prediction per text
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))
MAX_TEXT_LEN = 2000

log = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

state: dict = {"model": None, "metrics": None}


def load_model(model_dir: Path) -> None:
    state["model"] = joblib.load(model_dir / "model.joblib")
    metrics_path = model_dir / "metrics.json"
    state["metrics"] = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    log.info("model loaded from %s", model_dir)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model(MODEL_DIR)
    except FileNotFoundError:
        # Stay alive but report not-ready, so Kubernetes stops routing traffic
        # here instead of crash-looping and hiding the cause.
        log.error("model not found in %s; /readyz will fail", MODEL_DIR)
    yield


app = FastAPI(title="ticket-inference", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str | None
    inference_ms: float


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/metadata")
def metadata() -> dict:
    if state["metrics"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return state["metrics"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    for t in req.texts:
        if not t.strip() or len(t) > MAX_TEXT_LEN:
            raise HTTPException(status_code=422, detail=f"each text must be 1..{MAX_TEXT_LEN} non-blank characters")

    start = time.perf_counter()
    probs = model.predict_proba(req.texts)
    elapsed_ms = (time.perf_counter() - start) * 1000

    classes = list(model.classes_)
    predictions = []
    for row in probs:
        best = int(row.argmax())
        predictions.append(Prediction(
            label=classes[best],
            confidence=round(float(row[best]), 4),
            scores={c: round(float(p), 4) for c, p in zip(classes, row)},
        ))
    version = state["metrics"]["trained_at"] if state["metrics"] else None
    return PredictResponse(predictions=predictions, model_version=version, inference_ms=round(elapsed_ms, 3))
