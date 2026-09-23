"""Сервис инференса: uvicorn mlproject.api:app --reload"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from mlproject import __version__
from mlproject.predict import Predictor

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
state: dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor(MODEL_PATH)  # модель грузится один раз при старте
    yield
    state.clear()


app = FastAPI(title="ML Service", version=__version__, lifespan=lifespan)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    scores: list[float]
    model_version: str


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Latency-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str("predictor" in state)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        scores = state["predictor"].predict_proba(request.records)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictResponse(scores=scores, model_version=__version__)
