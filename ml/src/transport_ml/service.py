"""HTTP inference service: one loaded artifact, strict feature contract, no training.

The service never builds features and never reads labels. It receives the feature row that
``transport_ml.features.FeatureBuilder`` produced in the caller, so offline batch scoring and
the streaming Backend use one implementation.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from transport_ml import __version__
from transport_ml.model import DelayModel

MAX_ITEMS = int(os.environ.get("ML_MAX_ITEMS", "256"))


class PredictItem(BaseModel):
    """One prediction request. Target keys are metadata only and never reach the model."""

    model_config = {"extra": "forbid"}

    request_id: str = Field(max_length=128)
    tr_id: str | None = Field(default=None, max_length=64)
    cutoff_t: str | None = Field(default=None, max_length=64)
    target_stop_id: str | None = Field(default=None, max_length=64)
    target_planned_at: str | None = Field(default=None, max_length=64)
    features: dict[str, float | None]


class PredictRequest(BaseModel):
    model_config = {"extra": "forbid"}

    feature_schema_version: str
    items: Annotated[list[PredictItem], Field(min_length=1, max_length=MAX_ITEMS)]
    no_hint: bool = False


class PredictResult(BaseModel):
    request_id: str
    delay_s: float
    model_used: str
    used_hint: bool
    late_probability: float | None
    late_threshold_s: float | None
    calibration: dict[str, Any]


class PredictResponse(BaseModel):
    model_version: str
    feature_schema_version: str
    inference_ms: float
    results: list[PredictResult]


class ModelInfo(BaseModel):
    model_version: str
    feature_schema_version: str
    service_version: str
    features: list[str]
    feature_config: dict[str, Any]
    models: dict[str, Any]
    classifiers: dict[str, Any]
    late_threshold_s: float | None
    calibration: dict[str, Any]
    train_rows: int | None
    training_group: str | None
    max_items: int
    hint_policy: str = "gps_estimated"


def load_model(directory: Path) -> DelayModel:
    return DelayModel(directory)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the artifact once at startup; readiness stays false if it cannot be loaded."""
    directory = Path(os.environ.get("ML_MODEL_DIR", "ml/pretrained/v3"))
    app.state.model_dir = directory
    app.state.model = None
    app.state.load_error = None
    try:
        app.state.model = load_model(directory)
    except Exception as error:  # Readiness must report the reason instead of crash-looping.
        app.state.load_error = f"{type(error).__name__}: {error}"
    yield


app = FastAPI(
    title="Transport delay ML inference",
    version=__version__,
    summary="Signed delay in seconds and calibrated late probability for prepared features",
    lifespan=lifespan,
)


def require_model(request: Request) -> DelayModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "detail": request.app.state.load_error},
        )
    return model


@app.get("/health/live", tags=["health"])
def live() -> dict:
    return {"status": "live", "service_version": __version__}


@app.get("/health/ready", tags=["health"])
def ready(request: Request) -> JSONResponse:
    model = request.app.state.model
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "code": "model_unavailable",
                "detail": request.app.state.load_error,
            },
        )
    return JSONResponse(
        content={
            "status": "ready",
            "model_version": model.version,
            "feature_schema_version": model.manifest["feature_schema_version"],
            "model_dir": str(request.app.state.model_dir),
        }
    )


@app.get("/v1/model", response_model=ModelInfo, tags=["model"])
def model_info(request: Request) -> ModelInfo:
    model = require_model(request)
    manifest = model.manifest
    return ModelInfo(
        model_version=model.version,
        feature_schema_version=manifest["feature_schema_version"],
        service_version=__version__,
        features=model.features,
        feature_config=manifest["feature_config"],
        models={
            name: {
                **{key: spec[key] for key in ("depth", "iterations", "residual") if key in spec},
                "member_count": len(spec.get("members", [spec])),
            }
            for name, spec in manifest["models"].items()
        },
        classifiers={
            name: {
                "depth": spec["depth"],
                "iterations": spec["iterations"],
                "threshold_s": spec["threshold_s"],
                "calibration": spec["calibration"],
            }
            for name, spec in manifest.get("classifiers", {}).items()
        },
        late_threshold_s=model.late_threshold_s,
        calibration=model.calibration_status(),
        train_rows=manifest.get("train_rows"),
        training_group=manifest.get("training_group"),
        max_items=MAX_ITEMS,
        hint_policy=manifest.get("hint_policy", "gps_estimated"),
    )


def build_frame(items: list[PredictItem], expected: list[str]) -> pd.DataFrame:
    """Reject unknown, missing or non-finite features instead of silently reordering them."""
    rows = []
    for item in items:
        supplied = item.features
        unknown = sorted(set(supplied) - set(expected))
        missing = [name for name in expected if name not in supplied]
        if unknown or missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "schema_mismatch",
                    "detail": "Feature set does not match the model manifest",
                    "request_id": item.request_id,
                    "unknown": unknown,
                    "missing": missing,
                },
            )
        values = {}
        for name in expected:
            value = supplied[name]
            if value is not None and not np.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "bad_request",
                        "detail": f"Feature {name} must be finite or null",
                        "request_id": item.request_id,
                    },
                )
            values[name] = np.nan if value is None else float(value)
        rows.append(values)
    return pd.DataFrame(rows, columns=expected, dtype=float)


@app.post("/v1/predict", response_model=PredictResponse, tags=["model"])
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    model = require_model(request)
    if payload.feature_schema_version != model.manifest["feature_schema_version"]:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "schema_mismatch",
                "detail": "Feature schema version differs from the loaded model",
                "expected": model.manifest["feature_schema_version"],
                "received": payload.feature_schema_version,
            },
        )
    frame = build_frame(payload.items, model.features)
    started = perf_counter()
    try:
        inference = model.infer(frame, no_hint=payload.no_hint)
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail={"code": "bad_request", "detail": str(error)}
        ) from error
    elapsed_ms = (perf_counter() - started) * 1000
    probability = inference["late_probability"]
    results = [
        PredictResult(
            request_id=item.request_id,
            delay_s=float(inference["delay_s"][index]),
            model_used=str(inference["model_used"][index]),
            used_hint=bool(inference["used_hint"][index]),
            late_probability=None if probability is None else float(probability[index]),
            late_threshold_s=inference["late_threshold_s"],
            calibration=inference["calibration"],
        )
        for index, item in enumerate(payload.items)
    ]
    return PredictResponse(
        model_version=inference["model_version"],
        feature_schema_version=model.manifest["feature_schema_version"],
        inference_ms=elapsed_ms,
        results=results,
    )


def main() -> None:
    """Console entry point ``transport-ml-serve``."""
    import uvicorn

    uvicorn.run(
        "transport_ml.service:app",
        host=os.environ.get("ML_HOST", "0.0.0.0"),  # noqa: S104 - container service
        port=int(os.environ.get("ML_PORT", "8001")),
        log_level=os.environ.get("ML_LOG_LEVEL", "info"),
        access_log=False,
    )
