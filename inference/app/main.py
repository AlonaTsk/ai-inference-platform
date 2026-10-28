"""Inference HTTP service.

GET  /healthz   liveness: the process is up
GET  /readyz    readiness: the model is loaded and can serve
GET  /metadata  training metrics of the loaded model
POST /predict   {"texts": [...]} -> one prediction per text
"""

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints, field_validator

MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))
MAX_TEXT_LEN = 2000

# Uvicorn configures this logger, so messages share its handler and format.
log = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class LoadedModel:
    model: Any  # fitted sklearn Pipeline
    metrics: dict[str, Any] | None

    @property
    def version(self) -> str | None:
        return self.metrics["trained_at"] if self.metrics else None


def load_model(model_dir: Path) -> LoadedModel:
    model = joblib.load(model_dir / "model.joblib")
    metrics_path = model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return LoadedModel(model=model, metrics=metrics)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_dir = Path(os.getenv("MODEL_DIR", "model"))
    try:
        app.state.loaded = load_model(model_dir)
        log.info("model loaded from %s", model_dir)
    except FileNotFoundError:
        # Stay alive but report not-ready, so Kubernetes stops routing traffic
        # here instead of crash-looping and hiding the cause.
        app.state.loaded = None
        log.error("model not found in %s; /readyz will fail", model_dir)
    yield


app = FastAPI(title="ticket-inference", lifespan=lifespan)


# `async` because it does no I/O: a plain `def` dependency would be run in the
# thread pool, adding a thread hand-off to every request (measured ~0.6 ms).
async def loaded_model(request: Request) -> LoadedModel:
    loaded: LoadedModel | None = request.app.state.loaded
    if loaded is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return loaded


Model = Annotated[LoadedModel, Depends(loaded_model)]
TicketText = Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEXT_LEN)]


class PredictRequest(BaseModel):
    texts: list[TicketText] = Field(min_length=1, max_length=MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def reject_blank(cls, texts: list[str]) -> list[str]:
        if any(not t.strip() for t in texts):
            raise ValueError("texts must not be blank")
        return texts


class Prediction(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str | None
    inference_ms: float


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(_: Model) -> dict[str, str]:
    return {"status": "ready"}


@app.get("/metadata")
def metadata(loaded: Model) -> dict[str, Any]:
    if loaded.metrics is None:
        raise HTTPException(status_code=404, detail="no metrics stored with this model")
    return loaded.metrics


# A plain `def` endpoint runs in FastAPI's thread pool, so CPU-bound
# inference does not block the event loop.
@app.post("/predict")
def predict(req: PredictRequest, loaded: Model) -> PredictResponse:
    start = time.perf_counter()
    probs = loaded.model.predict_proba(req.texts)
    elapsed_ms = (time.perf_counter() - start) * 1000

    classes = list(loaded.model.classes_)
    predictions = []
    for row in probs:
        best = int(row.argmax())
        predictions.append(
            Prediction(
                label=classes[best],
                confidence=round(float(row[best]), 4),
                scores={c: round(float(p), 4) for c, p in zip(classes, row, strict=True)},
            )
        )
    return PredictResponse(predictions=predictions, model_version=loaded.version, inference_ms=round(elapsed_ms, 3))
