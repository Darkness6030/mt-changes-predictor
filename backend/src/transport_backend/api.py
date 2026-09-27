"""Dispatcher API. Schemas are published through OpenAPI at /openapi.json and /docs.

The API only reads the engine's already computed snapshot: an HTTP request never creates a
prediction, so refreshing the browser cannot change the measured latency of the stream. The
explanation endpoint only splits an already published prediction, on demand and cached.
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from transport_backend import SCHEMA_VERSION, __version__, schemas
from transport_backend.clock import parse_source
from transport_backend.config import Settings
from transport_backend.demo import DemoController, DemoError, DemoSource
from transport_backend.engine import Engine, PredictionChanged
from transport_backend.view_history import HistoryUnavailable


class ReplayCommand(BaseModel):
    model_config = {"extra": "forbid"}

    action: Literal["start", "pause", "reset", "speed", "seek"]
    speed: float | None = Field(default=None, gt=0, le=3600)
    start_at: str | None = Field(default=None, max_length=64)


class SourceCommand(BaseModel):
    model_config = {"extra": "forbid"}

    source: DemoSource
    speed: float = Field(default=60, gt=0, le=3600)
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
    demo = DemoController(settings, engine)
    app.state.demo = demo
    try:
        yield
    finally:
        await demo.close()


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
        demo = getattr(request.app.state, "demo", None)
        engine: Engine | None = demo.engine if demo else getattr(request.app.state, "engine", None)
        if engine is None:
            raise error(503, "not_ready", "Engine is still starting")
        return engine

    EngineDep = Annotated[Engine, Depends(get_engine)]

    @app.get("/api/v1/demo/sources", tags=["demo"])
    async def demo_sources(request: Request) -> dict:
        return await request.app.state.demo.view()

    @app.post("/api/v1/demo/source", tags=["demo"])
    async def demo_source(command: SourceCommand, request: Request) -> dict:
        demo = request.app.state.demo
        try:
            await demo.switch(command.source, command.speed, command.start_at)
        except (DemoError, ValueError, OverflowError) as exc:
            raise error(409, "source_switch_failed", str(exc)) from exc
        finally:
            request.app.state.engine = demo.engine
        return await demo.view()

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

    @app.get("/api/v1/snapshot", tags=["dispatcher"], response_model=schemas.Snapshot)
    def snapshot(
        engine: EngineDep,
        risk: Literal["green", "yellow", "red"] | None = None,
        only_attention: bool = False,
        stale: bool | None = None,
        limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
    ) -> dict:
        return engine.snapshot(risk=risk, only_attention=only_attention, stale=stale, limit=limit)

    @app.get("/api/v1/vehicles/{tr_id}", tags=["dispatcher"], response_model=schemas.VehicleDetail)
    def vehicle(tr_id: str, engine: EngineDep) -> dict:
        try:
            return engine.vehicle_detail(tr_id)
        except KeyError as missing:
            raise error(404, "unknown_vehicle", f"No state or plan for {tr_id}") from missing

    @app.get(
        "/api/v1/vehicles/{tr_id}/explanation",
        tags=["dispatcher"],
        response_model=schemas.ExplanationAnswer,
    )
    async def vehicle_explanation(
        tr_id: str,
        engine: EngineDep,
        prediction_id: Annotated[str, Query(min_length=1, max_length=64)],
    ) -> dict:
        """Why the current prediction has this value; computed on demand and cached."""
        try:
            return await engine.explanation(tr_id, prediction_id)
        except PredictionChanged as changed:
            raise error(
                409, "prediction_changed", "Прогноз обновился: объяснение относится к новому"
            ) from changed
        except Exception as failure:  # noqa: BLE001 - ML transport/contract errors alike
            raise error(
                503, "ml_unavailable", f"Объяснение недоступно: {type(failure).__name__}"
            ) from failure

    @app.get("/api/v1/history", tags=["dispatcher"])
    def history(
        engine: EngineDep,
        run_id: Annotated[str, Query(min_length=1, max_length=80)],
        at: Annotated[str, Query(min_length=1, max_length=64)],
    ) -> dict:
        if run_id != engine.run_id:
            raise error(
                409, "run_changed", "Начался новый прогон. История прошлого прогона очищена"
            )
        if not engine.view_history.enabled:
            raise error(409, "history_disabled", "Журнал доступен только для NDTP-потока")
        try:
            at_ns = parse_source(at)
        except (ValueError, OverflowError) as exc:
            raise error(
                400, "bad_request", "Укажите корректное время источника без часового пояса"
            ) from exc
        try:
            return engine.view_history.view(at_ns)
        except HistoryUnavailable as exc:
            raise error(404, "history_unavailable", str(exc)) from exc

    @app.get("/api/v1/predictions", tags=["dispatcher"], response_model=schemas.PredictionList)
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

    @app.get("/api/v1/alerts", tags=["dispatcher"], response_model=schemas.AlertList)
    def alerts(engine: EngineDep, state: Literal["active", "resolved", "expired"] | None = None):
        items = engine.alert_list(state=state)
        return {"run_id": engine.run_id, "rows": len(items), "alerts": items}

    @app.post("/api/v1/alerts/{alert_id}/ack", tags=["dispatcher"], response_model=schemas.AlertAck)
    async def acknowledge(alert_id: str, engine: EngineDep) -> dict:
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
    async def replay_control(command: ReplayCommand, request: Request) -> dict:
        demo = request.app.state.demo
        if demo.lock.locked():
            raise error(409, "source_busy", "Дождитесь завершения команды источника")
        async with demo.lock:
            return await apply_replay_command(command, demo.engine)

    async def apply_replay_command(command: ReplayCommand, engine: Engine) -> dict:
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
        elif command.action == "seek":
            if not command.start_at:
                raise error(400, "bad_request", "Для перемотки укажите start_at")
            try:
                await engine.seek(command.start_at)
            except (ValueError, OverflowError) as exc:
                raise error(400, "bad_request", str(exc)) from exc
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
