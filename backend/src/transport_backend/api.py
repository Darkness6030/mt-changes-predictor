"""Dispatcher API. Schemas are published through OpenAPI at /openapi.json and /docs.

The API only reads the engine's already computed snapshot: an HTTP request never triggers
inference, so refreshing the browser cannot change the measured latency of the stream.
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from transport_backend import SCHEMA_VERSION, __version__
from transport_backend.config import Settings
from transport_backend.engine import Engine


class ReplayCommand(BaseModel):
    model_config = {"extra": "forbid"}

    action: Literal["start", "pause", "reset", "speed"]
    speed: float | None = Field(default=None, gt=0, le=3600)
    start_at: str | None = Field(default=None, max_length=64)


def error(status_code: int, code: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "detail": detail, "schema_version": SCHEMA_VERSION},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings if hasattr(app.state, "settings") else Settings.from_env()
    app.state.settings = settings
    engine = Engine(settings)
    app.state.engine = engine
    await engine.start()
    try:
        yield
    finally:
        await engine.stop()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Transport dispatcher backend",
        version=__version__,
        summary="NDTP ingestion, causal features, delay predictions and dispatcher snapshots",
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings

    def get_engine(request: Request) -> Engine:
        engine: Engine | None = getattr(request.app.state, "engine", None)
        if engine is None:
            raise error(503, "not_ready", "Engine is still starting")
        return engine

    EngineDep = Annotated[Engine, Depends(get_engine)]

    @app.get("/health/live", tags=["health"])
    def live() -> dict:
        return {"status": "live", "backend_version": __version__}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request) -> JSONResponse:
        engine: Engine | None = getattr(request.app.state, "engine", None)
        if engine is None or not engine.ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "code": "not_ready"},
            )
        return JSONResponse(
            content={
                "status": "ready",
                "run_id": engine.run_id,
                "mode": engine.settings.mode,
                "ml_ready": engine.ml.ready,
            }
        )

    @app.get("/api/v1/status", tags=["status"])
    def status(engine: EngineDep) -> dict:
        return engine.status()

    @app.get("/api/v1/snapshot", tags=["dispatcher"])
    def snapshot(
        engine: EngineDep,
        risk: Literal["green", "yellow", "red"] | None = None,
        only_attention: bool = False,
        stale: bool | None = None,
        limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
    ) -> dict:
        return engine.snapshot(risk=risk, only_attention=only_attention, stale=stale, limit=limit)

    @app.get("/api/v1/vehicles/{tr_id}", tags=["dispatcher"])
    def vehicle(tr_id: str, engine: EngineDep) -> dict:
        try:
            return engine.vehicle_detail(tr_id)
        except KeyError as missing:
            raise error(404, "unknown_vehicle", f"No state or plan for {tr_id}") from missing

    @app.get("/api/v1/predictions", tags=["dispatcher"])
    def predictions(
        engine: EngineDep,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        tr_id: str | None = None,
    ) -> dict:
        items = list(engine.prediction_log)
        if tr_id is not None:
            items = [item for item in items if item["tr_id"] == tr_id]
        items = [
            {key: value for key, value in item.items() if key != "cutoff_ns"} for item in items
        ]
        return {"run_id": engine.run_id, "rows": len(items[-limit:]), "predictions": items[-limit:]}

    @app.get("/api/v1/alerts", tags=["dispatcher"])
    def alerts(engine: EngineDep, state: Literal["active", "resolved", "expired"] | None = None):
        items = engine.alert_list(state=state)
        return {"run_id": engine.run_id, "rows": len(items), "alerts": items}

    @app.post("/api/v1/alerts/{alert_id}/ack", tags=["dispatcher"])
    def acknowledge(alert_id: str, engine: EngineDep) -> dict:
        try:
            alert = engine.acknowledge(alert_id)
        except KeyError as missing:
            raise error(404, "unknown_alert", f"No alert {alert_id}") from missing
        return {
            "alert": alert.to_dict(),
            "note": "Отметка диспетчера об ознакомлении, не команда транспортному средству",
        }

    @app.get("/api/v1/metrics/quality", tags=["metrics"])
    def quality(engine: EngineDep) -> dict:
        return engine.quality()

    @app.get("/api/v1/metrics", tags=["metrics"])
    def metrics(engine: EngineDep) -> dict:
        status_data = engine.status()
        return {
            "run_id": engine.run_id,
            "performance": status_data["performance"],
            "state": status_data["state"],
            "ndtp": status_data["ndtp"],
            "ml": status_data["ml"],
            "errors": status_data["errors"],
        }

    @app.post("/api/v1/replay/control", tags=["replay"])
    async def replay_control(command: ReplayCommand, engine: EngineDep) -> dict:
        settings = engine.settings
        if settings.mode != "replay" or not settings.replay_control_enabled:
            raise error(409, "replay_disabled", "Replay control is available in demo replay mode")
        if command.action == "start":
            engine.clock.resume()
        elif command.action == "pause":
            engine.clock.pause()
        elif command.action == "speed":
            if command.speed is None:
                raise error(400, "bad_request", "Speed is required for the speed action")
            engine.clock.set_speed(command.speed)
        else:
            await engine.reset(command.start_at)
        return {"clock": engine.clock_view(), "run_id": engine.run_id}

    return app


app = create_app()


def main() -> None:
    """Console entry point ``transport-backend serve`` runs this via uvicorn."""
    import uvicorn

    uvicorn.run(
        "transport_backend.api:app",
        host=os.environ.get("BACKEND_HOST", "0.0.0.0"),  # noqa: S104 - container service
        port=int(os.environ.get("BACKEND_PORT", "8000")),
        log_level=os.environ.get("BACKEND_LOG_LEVEL", "info"),
        access_log=False,
    )
