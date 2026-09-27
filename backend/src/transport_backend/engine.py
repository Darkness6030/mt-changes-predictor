"""Orchestration: clocks, ingestion, causal features, ML calls, alerts and snapshots.

Invariants this module is responsible for:

* a prediction for T uses only events already received with ``event_time <= T``;
* the target visit is chosen from the plan in ``(T+600, T+900]``, never from a forecast;
* a missing prediction is published with a status, never as ``delay_s = 0``;
* one incident per ``(run_id, tr_id, target_stop_id)``;
* labels, when configured, are used strictly after the event by the evaluation sidecar.
"""

import asyncio
import contextlib
import json
import secrets
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
from transport_ml.features import FeatureBuilder, FeatureConfig

from transport_backend import __version__
from transport_backend.clock import (
    SECOND_NS,
    SourceClock,
    format_source,
    parse_source,
    wall_iso,
    wall_now,
)
from transport_backend.config import Settings
from transport_backend.current_deviation import CurrentDeviationMonitor
from transport_backend.events import TelemetryEvent
from transport_backend.explain import evidence, recommendation
from transport_backend.mlclient import Latency, MlClient
from transport_backend.ndtp_server import NdtpServer
from transport_backend.replay import PointSchedule, ReplaySource, load_mapping
from transport_backend.state import FleetState, PlanStore
from transport_backend.view_history import ViewHistory

STATUS_NO_TARGET = "no_target_in_horizon"


@dataclass
class Alert:
    alert_id: str
    tr_id: str
    target_stop_id: str
    target_planned_at: str
    first_alert_at: str
    first_alert_at_wall: str
    planned_lead_s: float
    risk_level: str
    delay_s: float
    late_probability: float | None
    latest_prediction_at: str
    updates: int = 1
    state: str = "active"
    acknowledged_at: str | None = None
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(vars(self))


class Sidecar:
    """Measures the streaming path against labels only after the event has happened."""

    def __init__(self, path: Path):
        import pandas as pd

        frame = pd.read_csv(path, dtype={"sample_id": str})
        required = {"sample_id", "target_delay_s"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Labels file must contain {sorted(required)}")
        self.labels = dict(zip(frame.sample_id, frame.target_delay_s.astype(float), strict=True))
        self.pending: list[dict] = []
        self.measured: list[dict] = []
        self.source = str(path)

    def record(self, view: dict) -> None:
        sample_id = view.get("sample_id")
        if sample_id is None or sample_id not in self.labels or view.get("delay_s") is None:
            return
        fact = self.labels[sample_id]
        planned_ns = parse_source(view["target_planned_at"])
        self.pending.append(
            {
                "sample_id": sample_id,
                "tr_id": view["tr_id"],
                "cutoff_t": view["cutoff_t"],
                "prediction": float(view["delay_s"]),
                "cur_dev_s": view.get("cur_dev_s"),
                "cur_dev_source": view.get("cur_dev_source"),
                "late_probability": view.get("late_probability"),
                "risk_level": view.get("risk_level"),
                "target_delay_s": fact,
                "arrival_ns": planned_ns + int(fact * SECOND_NS),
                "cutoff_ns": parse_source(view["cutoff_t"]),
            }
        )

    def mature(self, now_ns: int) -> None:
        """Move records whose real arrival has passed into the measured set."""
        still_pending = []
        for row in self.pending:
            if row["arrival_ns"] <= now_ns:
                row["lead_s"] = (row["arrival_ns"] - row["cutoff_ns"]) / SECOND_NS
                self.measured.append(row)
            else:
                still_pending.append(row)
        self.pending = still_pending

    def report(self) -> dict:
        rows = self.measured
        if not rows:
            return {
                "source": self.source,
                "measured_rows": 0,
                "pending_rows": len(self.pending),
                "note": "Метрика появится после наступления фактического времени цели",
            }
        prediction = np.array([row["prediction"] for row in rows])
        fact = np.array([row["target_delay_s"] for row in rows])
        hint = np.array([np.nan if row["cur_dev_s"] is None else row["cur_dev_s"] for row in rows])
        lead = np.array([row["lead_s"] for row in rows])
        late = (fact > 120).astype(float)
        probability = np.array(
            [np.nan if row["late_probability"] is None else row["late_probability"] for row in rows]
        )
        finite = np.isfinite(probability)
        report = {
            "source": self.source,
            "measured_rows": len(rows),
            "pending_rows": len(self.pending),
            "mae_s": float(np.mean(np.abs(fact - prediction))),
            "zero_mae_s": float(np.mean(np.abs(fact))),
            "cur_dev_mae_s": (
                float(np.mean(np.abs(fact[np.isfinite(hint)] - hint[np.isfinite(hint)])))
                if np.isfinite(hint).any()
                else None
            ),
            "absolute_error_p50_s": float(np.median(np.abs(fact - prediction))),
            "absolute_error_p90_s": float(np.quantile(np.abs(fact - prediction), 0.9)),
            "lead_time_s": {
                "min": float(lead.min()),
                "p50": float(np.median(lead)),
                "max": float(lead.max()),
                "note": "Фактический lead time = момент фактического прибытия минус cutoff T",
            },
            "late_rate": float(late.mean()),
            "hint_sources": {},
        }
        if finite.any():
            report["late_probability_brier"] = float(
                np.mean((probability[finite] - late[finite]) ** 2)
            )
            report["late_probability_rows"] = int(finite.sum())
        for row in rows:
            key = row["cur_dev_source"] or "unknown"
            report["hint_sources"][key] = report["hint_sources"].get(key, 0) + 1
        return report


class Engine:
    """One run: a fixed data source, one clock and one mutable fleet state."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.feature_config = FeatureConfig()
        self.revision = 0
        self.started_wall = wall_iso()
        self.run_id = self._new_run_id()
        self.view_history = ViewHistory(settings)
        self.history_latency = Latency()
        self.ml = MlClient(settings.ml_url, settings.ml_timeout_s, settings.ml_batch_size)
        self.cycle_latency = Latency()
        self.feature_latency = Latency()
        self.publish_latency = Latency()
        self.current_deviation = CurrentDeviationMonitor(
            settings.current_deviation_max_age_s, settings.risk
        )
        self.predictions: dict[str, dict] = {}
        self.prediction_log: deque[dict] = deque(maxlen=settings.max_predictions_kept)
        self.alerts: dict[str, Alert] = {}
        self.sidecar: Sidecar | None = None
        self.prediction_seq = 0
        self.cycles = 0
        self.predicted_points = 0
        self.errors: dict[str, int] = {}
        self.last_cycle_at: str | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._plan_shift_s = settings.plan_shift_s
        self._load_sources()

    # ------------------------------------------------------------------ setup

    def _new_run_id(self) -> str:
        stamp = wall_now().strftime("%Y%m%dT%H%M%SZ")
        return f"run-{stamp}-{secrets.token_hex(2)}"

    def _plan_path(self) -> Path:
        root = self.settings.data_root / self.settings.split
        name = "schedule_plan.csv" if self.settings.split == "validate" else "schedule.csv"
        return root / name

    def _load_sources(self) -> None:
        settings = self.settings
        if settings.plan_shift_auto:
            self._plan_shift_s = self._auto_shift()
        self.plan = PlanStore.from_csv(self._plan_path(), self._plan_shift_s)
        traffic_path = settings.data_root / settings.split / "traffic.csv"
        self.mapping = load_mapping(traffic_path)
        self.state = FleetState(
            plan=self.plan,
            history_window_s=settings.history_window_s,
            history_max_events=settings.history_max_events,
            stale_after_s=settings.stale_after_s,
            mapping=dict(self.mapping),
        )
        self.replay: ReplaySource | None = None
        self.points: PointSchedule | None = None
        self.ndtp: NdtpServer | None = None
        if settings.mode == "replay":
            units = {tr_id: unit for unit, tr_id in self.mapping.items()}
            self.replay = ReplaySource(traffic_path, self.feature_config, units)
            self.clock = SourceClock("driven", settings.replay_speed)
        else:
            self.clock = SourceClock("follow")
        if settings.use_points:
            with contextlib.suppress(FileNotFoundError):
                self.points = PointSchedule(settings.data_root, settings.split, self._plan_shift_s)
        if settings.labels is not None and settings.labels.exists():
            self.sidecar = Sidecar(settings.labels)

    def _auto_shift(self) -> float:
        """Whole-day shift so a live stream of today lands inside the historical plan window.

        Only used when explicitly requested; the applied shift is published in every
        snapshot so a demo alignment can never look like real schedule data.
        """
        import pandas as pd

        plan = pd.read_csv(self._plan_path(), usecols=["time_begin"])
        median = pd.to_datetime(plan.time_begin).median()
        today = pd.Timestamp(wall_now().replace(tzinfo=None).date())
        days = (today - pd.Timestamp(median.date())).days
        return float(days * 86400)

    def _replay_start_ns(self) -> int:
        settings = self.settings
        if settings.replay_start:
            return parse_source(settings.replay_start)
        first_event_ns = self.replay.first_ns if self.replay is not None else None
        if first_event_ns is None:
            raise ValueError("Replay needs telemetry to pick a start time")
        if self.points is None or self.points.first_ns is None:
            return first_event_ns
        # Start one warm-up window before the first official point: the feature windows are
        # filled from real events, and the demo does not begin with hours of empty night.
        return max(first_event_ns, self.points.first_ns - int(settings.replay_warmup_s * SECOND_NS))

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        settings = self.settings
        await self.ml.refresh_model()
        if settings.mode == "ndtp":
            self.ndtp = NdtpServer(
                settings.ndtp_host,
                settings.ndtp_port,
                self._ingest,
                max_connections=settings.ndtp_max_connections,
                time_offset_s=settings.ndtp_time_offset_s,
                read_timeout_s=settings.ndtp_read_timeout_s,
            )
            await self.ndtp.start()
        else:
            start_ns = self._replay_start_ns()
            self.replay.reset(start_ns)
            if self.points is not None:
                self.points.reset(start_ns)
            self.clock.start(start_ns, paused=not settings.replay_autostart)
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="engine-loop")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self.ndtp is not None:
            await self.ndtp.stop()
        await self.ml.aclose()

    @property
    def ready(self) -> bool:
        source_ready = self.ndtp.running if self.ndtp is not None else self.replay is not None
        return self._running and source_ready and bool(self.plan.visits)

    async def reset(self, start_at: str | None = None) -> None:
        """A new run: new run_id, empty state, cleared incidents and clocks."""
        self.run_id = self._new_run_id()
        self.view_history = ViewHistory(self.settings)
        self.history_latency = Latency()
        self.current_deviation.clear()
        self.predictions.clear()
        self.prediction_log.clear()
        self.alerts.clear()
        self.prediction_seq = 0
        self.predicted_points = 0
        self.errors.clear()
        self.state = FleetState(
            plan=self.plan,
            history_window_s=self.settings.history_window_s,
            history_max_events=self.settings.history_max_events,
            stale_after_s=self.settings.stale_after_s,
            mapping=dict(self.mapping),
        )
        if self.sidecar is not None and self.settings.labels is not None:
            self.sidecar = Sidecar(self.settings.labels)
        if self.replay is not None:
            start_ns = parse_source(start_at) if start_at else self._replay_start_ns()
            self.replay.reset(start_ns)
            if self.points is not None:
                self.points.reset(start_ns)
            self.clock.start(start_ns, paused=False)
        self.revision += 1

    async def seek(self, start_at: str) -> None:
        """Rebuild the bounded causal prefix and freeze at an exact source timestamp."""
        if self.replay is None:
            raise ValueError("Перемотка доступна только для исторического replay")
        target_ns = parse_source(start_at)
        # Validation precedes reset, so a rejected seek cannot destroy the current run.
        await self.reset(start_at)
        self.clock.start(target_ns, paused=True)
        if (
            self.replay.first_ns is None
            or self.replay.last_ns is None
            or not self.replay.first_ns <= target_ns <= self.replay.last_ns
        ):
            # Navigation outside the log is valid and intentionally displays an empty state.
            # Resuming before its first event will deliver data once the clock reaches it.
            self.replay.reset(target_ns)
            return
        self.replay.reset(target_ns - int(self.settings.history_window_s * SECOND_NS))
        while events := self.replay.due(target_ns):
            for event in events:
                self.state.add(event, target_ns)
        self.state.trim(target_ns)
        self.current_deviation.refresh(self.state, target_ns)
        # Periodic predictions use only this rebuilt prefix, with estimated/missing hints.
        # Official points before the seek are not evaluated or replayed retrospectively.
        self.revision += 1

    # ------------------------------------------------------------------ ingestion

    def _ingest(self, event: TelemetryEvent) -> None:
        """Called from the NDTP server task for every decoded navigation cell."""
        # Unknown units must not advance the clock used to trim every vehicle.
        if event.unit_id not in self.mapping:
            self.state.key(event)
            return
        now_ns = self.clock.now_ns()
        if now_ns is None:
            # Bootstrap is constrained by the explicitly loaded plan, not host UTC.
            times = self.plan.plan.time_begin.astype("int64")
            margin = int(self.settings.history_window_s * SECOND_NS)
            if times.empty or not (
                times.min() - margin <= event.event_time_ns <= times.max() + margin
            ):
                self.state.rejected_future += 1
                return
        if self.state.add(event, now_ns):
            self.clock.observe(event.event_time_ns)

    async def _loop(self) -> None:
        settings = self.settings
        while self._running:
            started = perf_counter()
            try:
                await self._cycle()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # A bad cycle must not stop ingestion or the API.
                key = f"{type(error).__name__}: {error}"
                self.errors[key] = self.errors.get(key, 0) + 1
            self.cycle_latency.add((perf_counter() - started) * 1000)
            await asyncio.sleep(settings.tick_interval_s)

    async def _cycle(self) -> None:
        run_id = self.run_id
        now_ns = self.clock.now_ns()
        if now_ns is None:
            return
        if self.replay is not None and not self.clock.paused:
            for event in self.replay.due(now_ns):
                self.state.add(event, now_ns)
        self.state.trim(now_ns)
        self.current_deviation.refresh(self.state, now_ns)
        self.cycles += 1
        self.last_cycle_at = wall_iso()
        if self.sidecar is not None:
            self.sidecar.mature(now_ns)
        if self.ml.model is None or not self.ml.ready:
            await self.ml.refresh_model()
        if run_id != self.run_id:
            return
        requests = self._collect(now_ns)
        if requests:
            await self._predict(requests, now_ns)
        if run_id == self.run_id:
            self._expire_alerts(now_ns)
            self._record_view(not_before_ns=now_ns)

    def _record_view(self, *, not_before_ns: int = 0) -> None:
        # No await: capture one consistent publication, after ML has returned. Ingestion
        # and alert acknowledgement also run on the event loop; no historical backfill.
        if not self.view_history.enabled:
            return
        now_ns = self.clock.now_ns()
        # A follow clock may step back when a delayed packet arrives during the ML
        # await. Never date that result before either its cutoff or this cycle's time.
        cutoff_floor = max(
            (item.get("cutoff_ns", 0) for item in self.predictions.values()), default=0
        )
        if now_ns is None or not self.view_history.due(
            now_ns, not_before_ns=max(not_before_ns, cutoff_floor)
        ):
            return
        started = perf_counter()
        snapshot = self.snapshot(now_ns=now_ns)
        snapshot.pop("history", None)
        details = {
            view["tr_id"]: self.vehicle_detail(view["tr_id"], now_ns=now_ns, view=view)
            for view in snapshot["vehicles"]
        }
        self.view_history.append(now_ns, snapshot, details)
        self.history_latency.add((perf_counter() - started) * 1000)

    # ------------------------------------------------------------------ predictions

    def _collect(self, now_ns: int) -> list[dict]:
        """Build the prediction queue: official points first, then periodic refresh."""
        requests: list[dict] = []
        claimed: set[str] = set()
        if self.points is not None and not self.clock.paused:
            # At high replay speed a whole tick covers minutes of source time. The grace
            # window follows the speed, but never exceeds half of the retained history, so a
            # point is only predicted while its causal events are still in the window.
            grace_s = min(
                self.settings.history_window_s / 2,
                max(60.0, 4 * self.settings.tick_interval_s * self.clock.speed),
            )
            for point in self.points.due(now_ns, grace_s=grace_s):
                requests.append(
                    {
                        "tr_id": point.tr_id,
                        "cutoff_ns": point.cutoff_ns,
                        "target_stop_id": point.target_stop_id,
                        "target_planned_ns": point.target_planned_ns,
                        "cur_dev_s": point.cur_dev_s,
                        "cur_dev_source": "supplied",
                        "sample_id": point.sample_id,
                        "trigger": "point",
                    }
                )
                claimed.add(point.tr_id)
        interval_ns = int(self.settings.predict_interval_s * SECOND_NS)
        for tr_id in sorted(self.state.tracks):
            if tr_id in claimed:
                continue
            previous = self.predictions.get(tr_id)
            if previous is not None and previous.get("cutoff_ns", 0) + interval_ns > now_ns:
                continue
            requests.append(
                {
                    "tr_id": tr_id,
                    "cutoff_ns": now_ns,
                    "target_stop_id": None,
                    "target_planned_ns": None,
                    "cur_dev_s": None,
                    "cur_dev_source": None,
                    "sample_id": None,
                    "trigger": "periodic",
                }
            )
        return requests

    def _status_before_features(self, request: dict, now_ns: int) -> str | None:
        tr_id = request["tr_id"]
        track = self.state.tracks.get(tr_id)
        if tr_id not in self.plan.visits:
            return "no_schedule"
        if track is None or track.last_valid_ns is None:
            return "warming_up"
        # Staleness is checked before history size: a vehicle silent for an hour is stale,
        # not warming up, even though trimming left only a couple of events in the window.
        age_s = (request["cutoff_ns"] - track.last_valid_ns) / SECOND_NS
        if age_s > self.settings.stale_after_s:
            return "stale"
        if len(track.events) < 3:
            return "warming_up"
        return None

    async def _predict(self, requests: list[dict], now_ns: int) -> None:
        run_id = self.run_id
        try:
            info = self.ml.model or {}
            config = FeatureConfig(**info.get("feature_config", {}))
            if info.get("feature_schema_version", config.schema_version) != config.schema_version:
                raise ValueError("ML feature schema and config disagree")
            if config.max_speed_kmh != FeatureConfig().max_speed_kmh:
                raise ValueError("ML speed cleaning differs from ingestion")
            if config.schedule_context and self.settings.history_window_s < 1800:
                raise ValueError("Schedule context requires 1800 seconds of retained history")
            if info.get("hint_policy", "gps_estimated") not in {"supplied_only", "gps_estimated"}:
                raise ValueError("Unknown ML hint policy")
            self.feature_config = config
        except (TypeError, ValueError) as error:
            self.ml.ready = False
            self.ml.last_error = f"Invalid ML contract: {error}"
            for request in requests:
                self._publish(request, status="ml_unavailable")
            self.revision += 1
            return
        prepared: list[dict] = []
        skipped: list[dict] = []
        for request in requests:
            status = self._status_before_features(request, now_ns)
            if status is not None:
                skipped.append({**request, "status": status})
                continue
            target = self.plan.target(request["tr_id"], request["cutoff_ns"])
            if target is None:
                skipped.append({**request, "status": STATUS_NO_TARGET})
                continue
            if request["target_stop_id"] is None:
                request["target_stop_id"] = str(target.tt_action_item_id)
                request["target_planned_ns"] = int(target.time_begin.value)
            elif request["target_stop_id"] != str(target.tt_action_item_id):
                # The supplied point and the plan must agree on the first visit in the window.
                if int(target.time_begin.value) != request["target_planned_ns"]:
                    skipped.append({**request, "status": "invalid_input"})
                    continue
            track = self.state.tracks[request["tr_id"]]
            request["_event_received_monotonic"] = (
                track.last_received_monotonic
                if track.last_event_ns is not None and track.last_event_ns <= request["cutoff_ns"]
                else None
            )
            prepared.append(request)
        started = perf_counter()
        features = self._build_features(prepared)
        self.feature_latency.add((perf_counter() - started) * 1000)
        items = []
        for request in prepared:
            row = features.get(id(request))
            if row is None:
                skipped.append({**request, "status": "invalid_input"})
                continue
            items.append(
                {
                    "request_id": str(len(items)),
                    "tr_id": request["tr_id"],
                    "cutoff_t": format_source(request["cutoff_ns"]),
                    "target_stop_id": request["target_stop_id"],
                    "target_planned_at": format_source(request["target_planned_ns"]),
                    "features": {
                        name: (None if value is None or np.isnan(value) else float(value))
                        for name, value in row.items()
                    },
                }
            )
            request["_features"] = row
            request["_request_id"] = str(len(items) - 1)
        results = await self.ml.predict(items, self.feature_config.schema_version) if items else {}
        if run_id != self.run_id:
            return  # A reset invalidates all in-flight results of the previous run.
        for request in prepared:
            if "_request_id" not in request:
                continue
            result = results.get(request["_request_id"])
            if result is None:
                self._publish(request, status="ml_unavailable")
                continue
            self._publish(request, status="ok", result=result)
        for request in skipped:
            self._publish(request, status=request["status"])
        self.revision += 1

    def _build_features(self, requests: list[dict]) -> dict[int, dict]:
        """One FeatureBuilder per cycle over the bounded history: the same code as training."""
        if not requests:
            return {}
        tr_ids = sorted({request["tr_id"] for request in requests})
        history = self.state.history_frame(tr_ids)
        if history.empty:
            return {}
        builder = FeatureBuilder(history, self.plan.plan, self.feature_config)
        rows: dict[int, dict] = {}
        for request in requests:
            hint = request["cur_dev_s"]
            source = request["cur_dev_source"]
            if hint is None and (self.ml.model or {}).get("hint_policy") == "supplied_only":
                # V3 learns noisy GPS/plan matches as separate features, never as the
                # supplied cur_dev_s input on which the main regressor was trained.
                source = "missing"
            elif hint is None:
                estimated = self.state.estimate_deviation(request["tr_id"], request["cutoff_ns"])
                if estimated is not None:
                    hint = estimated.seconds
                    source = "estimated"
                    request["cur_dev_age_s"] = (
                        request["cutoff_ns"] - estimated.arrival_ns
                    ) / SECOND_NS
                    request["cur_dev_visit_id"] = estimated.visit_id
                    request["cur_dev_distance_m"] = estimated.distance_m
                else:
                    source = "missing"
            request["cur_dev_s"] = hint
            request["cur_dev_source"] = source
            point = {
                "sample_id": request.get("sample_id") or "live",
                "tr_id": request["tr_id"],
                "T": format_source(request["cutoff_ns"]),
                "target_stop_id": request["target_stop_id"],
                "target_time_begin": format_source(request["target_planned_ns"]),
                "cur_dev_s": np.nan if hint is None else float(hint),
            }
            try:
                rows[id(request)] = builder.one(point)
            except ValueError as error:
                request["_feature_error"] = str(error)
        return rows

    def _publish(self, request: dict, *, status: str, result: dict | None = None) -> None:
        tr_id = request["tr_id"]
        previous = self.predictions.get(tr_id)
        cutoff_ns = request["cutoff_ns"]
        track = self.state.tracks.get(tr_id)
        data_age_s = None
        if track is not None and track.last_valid_ns is not None:
            data_age_s = (cutoff_ns - track.last_valid_ns) / SECOND_NS
        if status != "ok":
            # ml_unavailable keeps the previous number visible with its real age; other
            # statuses publish no number at all, so nothing can be read as "no delay".
            if (
                status == "ml_unavailable"
                and previous is not None
                and previous.get("delay_s") is not None
            ):
                view = {**previous, "status": status, "quality_flags": ["ml_unavailable"]}
                view["prediction_age_s"] = (cutoff_ns - previous["cutoff_ns"]) / SECOND_NS
                self.predictions[tr_id] = view
                return
            self.predictions[tr_id] = {
                "prediction_id": f"p-{self.prediction_seq:06d}",
                "run_id": self.run_id,
                "tr_id": tr_id,
                "sample_id": request.get("sample_id"),
                "status": status,
                "cutoff_t": format_source(cutoff_ns),
                "cutoff_ns": cutoff_ns,
                "computed_at": wall_iso(),
                "target_stop_id": request.get("target_stop_id"),
                "target_planned_at": format_source(request.get("target_planned_ns")),
                "horizon_s": (
                    None
                    if request.get("target_planned_ns") is None
                    else (request["target_planned_ns"] - cutoff_ns) / SECOND_NS
                ),
                "delay_s": None,
                "risk_level": None,
                "risk_basis": None,
                "late_probability": None,
                "model_version": self.ml.model.get("model_version") if self.ml.model else None,
                "data_age_s": data_age_s,
                "prediction_age_s": 0.0,
                "stale": status == "stale",
                "cur_dev_source": request.get("cur_dev_source"),
                "cur_dev_s": request.get("cur_dev_s"),
                "detail": request.get("_feature_error"),
                "evidence": [],
                "recommendation": None,
                "quality_flags": [],
            }
            return
        self.prediction_seq += 1
        received = request.get("_event_received_monotonic")
        if received is not None:
            self.publish_latency.add((perf_counter() - received) * 1000)
        features = request["_features"]
        delay_s = float(result["delay_s"])
        policy = self.settings.risk
        target_planned_ns = request["target_planned_ns"]
        items = evidence(features, stale_after_s=self.settings.stale_after_s)
        view = {
            "prediction_id": f"p-{self.prediction_seq:06d}",
            "run_id": self.run_id,
            "tr_id": tr_id,
            "sample_id": request.get("sample_id"),
            "status": "ok",
            "cutoff_t": format_source(cutoff_ns),
            "cutoff_ns": cutoff_ns,
            "computed_at": wall_iso(),
            "trigger": request["trigger"],
            "target_stop_id": request["target_stop_id"],
            "target_planned_at": format_source(target_planned_ns),
            "target_address": self.plan.address(request["target_stop_id"]),
            "target_lon": float(features["target_lon"]),
            "target_lat": float(features["target_lat"]),
            "horizon_s": (target_planned_ns - cutoff_ns) / SECOND_NS,
            "delay_s": delay_s,
            "expected_arrival_at": format_source(
                target_planned_ns + int(round(delay_s * SECOND_NS))
            ),
            "risk_level": policy.level(delay_s),
            "risk_basis": policy.basis(delay_s),
            "late_probability": result.get("late_probability"),
            "late_threshold_s": result.get("late_threshold_s"),
            "calibration": result.get("calibration"),
            "model_version": result.get("model_version"),
            "model_used": result.get("model_used"),
            "feature_schema_version": self.feature_config.schema_version,
            "cur_dev_s": request.get("cur_dev_s"),
            "cur_dev_source": request.get("cur_dev_source"),
            "cur_dev_age_s": request.get("cur_dev_age_s"),
            "data_age_s": data_age_s,
            "prediction_age_s": 0.0,
            "stale": False,
            "evidence": items,
            # Exact split of delay_s from the ML service (model arithmetic, not a cause).
            "explanation": result.get("explanation"),
            "recommendation": recommendation(delay_s, items, policy),
            "quality_flags": (
                track.quality_flags(cutoff_ns, self.settings.stale_after_s) if track else []
            ),
        }
        self.predictions[tr_id] = view
        self.prediction_log.append(view)
        if request["trigger"] == "point":
            self.predicted_points += 1
        if self.sidecar is not None:
            self.sidecar.record(view)
        self._update_alert(view)

    # ------------------------------------------------------------------ alerts

    def _update_alert(self, view: dict) -> None:
        alert_id = f"{self.run_id}:{view['tr_id']}:{view['target_stop_id']}"
        existing = self.alerts.get(alert_id)
        risky = view["risk_level"] in {"yellow", "red"}
        if existing is None:
            if not risky:
                return
            self.alerts[alert_id] = Alert(
                alert_id=alert_id,
                tr_id=view["tr_id"],
                target_stop_id=view["target_stop_id"],
                target_planned_at=view["target_planned_at"],
                first_alert_at=view["cutoff_t"],
                first_alert_at_wall=view["computed_at"],
                planned_lead_s=view["horizon_s"],
                risk_level=view["risk_level"],
                delay_s=view["delay_s"],
                late_probability=view.get("late_probability"),
                latest_prediction_at=view["cutoff_t"],
                evidence=view["evidence"],
            )
            return
        existing.updates += 1
        existing.latest_prediction_at = view["cutoff_t"]
        existing.delay_s = view["delay_s"]
        existing.late_probability = view.get("late_probability")
        existing.risk_level = view["risk_level"]
        existing.evidence = view["evidence"]
        existing.state = "active" if risky else "resolved"

    def _expire_alerts(self, now_ns: int) -> None:
        """Expire incidents whose target time has passed and keep the store bounded."""
        stale_before = now_ns - int(self.settings.alert_retention_s * SECOND_NS)
        for alert_id, alert in list(self.alerts.items()):
            planned_ns = parse_source(alert.target_planned_at)
            if alert.state == "active" and planned_ns < now_ns:
                alert.state = "expired"
            if alert.state != "active" and planned_ns < stale_before:
                del self.alerts[alert_id]
        overflow = len(self.alerts) - self.settings.max_alerts_kept
        if overflow > 0:
            closed = sorted(
                (item for item in self.alerts.items() if item[1].state != "active"),
                key=lambda item: parse_source(item[1].target_planned_at),
            )
            for alert_id, _ in closed[:overflow]:
                del self.alerts[alert_id]

    def acknowledge(self, alert_id: str) -> Alert:
        alert = self.alerts.get(alert_id)
        if alert is None:
            raise KeyError(alert_id)
        if alert.acknowledged_at is None:
            alert.acknowledged_at = wall_iso()
            self.revision += 1
        return alert

    # ------------------------------------------------------------------ views

    def _vehicle_view(self, tr_id: str, now_ns: int) -> dict:
        track = self.state.tracks.get(tr_id)
        prediction = self.predictions.get(tr_id)
        position = None
        telemetry_age_s = None
        position_age_s = None
        if track is not None:
            if track.last_event_ns is not None:
                telemetry_age_s = (now_ns - track.last_event_ns) / SECOND_NS
            if track.last_trusted is not None:
                # A spoofed or jumped fix is never drawn: show the last plausible one, aged.
                last = track.last_trusted
                position_age_s = (now_ns - last.event_time_ns) / SECOND_NS
                position = {
                    "lon": last.lon,
                    "lat": last.lat,
                    "speed_kmh": last.speed_kmh,
                    "heading_deg": last.heading_deg,
                    "event_at": format_source(last.event_time_ns),
                    "age_s": position_age_s,
                }
        next_visit = self.plan.next_visit(tr_id, now_ns)
        if prediction is not None:
            prediction = {
                **prediction,
                "prediction_age_s": (now_ns - prediction["cutoff_ns"]) / SECOND_NS,
            }
            prediction.pop("cutoff_ns", None)
        current_deviation = self.current_deviation.view(
            tr_id, now_ns, position_age_s, self.settings.stale_after_s
        )
        current_deviation["visit_address"] = self.plan.address(current_deviation["visit_id"])
        return {
            "tr_id": tr_id,
            "unit_id": track.unit_id if track else None,
            "mapped": True,
            "position": position,
            "last_event_at": format_source(track.last_event_ns) if track else None,
            "telemetry_age_s": telemetry_age_s,
            "position_age_s": position_age_s,
            "stale": bool(position_age_s is None or position_age_s > self.settings.stale_after_s),
            "events_in_window": len(track.events) if track else 0,
            "invalid_gps_fraction": track.invalid_fraction() if track else None,
            "quality_flags": (
                track.quality_flags(now_ns, self.settings.stale_after_s) if track else ["no_data"]
            ),
            "has_schedule": tr_id in self.plan.visits,
            "next_visit": (
                None
                if next_visit is None
                else {
                    "target_stop_id": str(next_visit.tt_action_item_id),
                    "planned_at": format_source(int(next_visit.time_begin.value)),
                    "address": self.plan.address(str(next_visit.tt_action_item_id)),
                    "lon": float(next_visit.lon),
                    "lat": float(next_visit.lat),
                }
            ),
            "prediction": prediction,
            "current_deviation": current_deviation,
            "segment": self.plan.segment(
                tr_id, prediction.get("target_stop_id") if prediction else None
            ),
        }

    def snapshot(
        self,
        *,
        risk: str | None = None,
        only_attention: bool = False,
        stale: bool | None = None,
        limit: int | None = None,
        now_ns: int | None = None,
    ) -> dict:
        now_ns = (self.clock.now_ns() or 0) if now_ns is None else now_ns
        vehicles = [
            self._vehicle_view(tr_id, now_ns)
            for tr_id in sorted(set(self.state.tracks) | set(self.predictions))
        ]
        summary = {
            "vehicles": len(vehicles),
            "with_prediction": sum(
                1 for v in vehicles if v["prediction"] and v["prediction"]["status"] == "ok"
            ),
            "attention": sum(
                1
                for v in vehicles
                if v["prediction"] and v["prediction"].get("risk_level") in {"yellow", "red"}
            ),
            "red": sum(
                1
                for v in vehicles
                if v["prediction"] and v["prediction"].get("risk_level") == "red"
            ),
            "no_prediction": sum(
                1 for v in vehicles if not v["prediction"] or v["prediction"]["status"] != "ok"
            ),
            "stale": sum(1 for v in vehicles if v["stale"]),
            "unmapped": len(self.state.unmapped),
            "alerts_active": sum(1 for a in self.alerts.values() if a.state == "active"),
        }
        filtered = vehicles
        if risk:
            filtered = [
                v for v in filtered if v["prediction"] and v["prediction"].get("risk_level") == risk
            ]
        if only_attention:
            filtered = [
                v
                for v in filtered
                if v["prediction"] and v["prediction"].get("risk_level") in {"yellow", "red"}
            ]
        if stale is not None:
            filtered = [v for v in filtered if v["stale"] is stale]
        if limit is not None:
            filtered = filtered[:limit]
        return {
            "schema_version": "1",
            "run_id": self.run_id,
            "revision": self.revision,
            "mode": self.settings.mode,
            "server_sent_at": wall_iso(),
            "clock": self.clock_view(now_ns),
            "history": self.view_history.metadata(),
            "summary": summary,
            "risk_policy": self.settings.risk.to_dict(),
            "vehicles": filtered,
            "alerts": self.alert_list(closed_limit=20),
        }

    def alert_list(self, *, state: str | None = None, closed_limit: int | None = None) -> list:
        """Active incidents first; closed ones are trimmed so the snapshot stays small."""
        items = [alert for alert in self.alerts.values() if state is None or alert.state == state]
        items.sort(key=lambda alert: (alert.state != "active", -abs(alert.delay_s)))
        if closed_limit is not None:
            active = [alert for alert in items if alert.state == "active"]
            closed = [alert for alert in items if alert.state != "active"][:closed_limit]
            items = active + closed
        return [alert.to_dict() for alert in items]

    def clock_view(self, now_ns: int | None = None) -> dict:
        view = self.clock.to_dict()
        if now_ns is not None and view["source_time"] is not None:
            view["source_time"] = format_source(now_ns)
        view["plan_shift_s"] = self._plan_shift_s
        if self._plan_shift_s:
            view["plan_shift_note"] = (
                "Демонстрационное выравнивание даты плана; исходные времена данных не изменены"
            )
        if self.replay is not None:
            view["source_window"] = {
                "first": format_source(self.replay.first_ns),
                "last": format_source(self.replay.last_ns),
            }
            view["source_progress"] = self.replay.progress()
            view["events_delivered"] = self.replay.cursor
            view["events_total"] = len(self.replay)
        return view

    def vehicle_detail(
        self, tr_id: str, *, now_ns: int | None = None, view: dict | None = None
    ) -> dict:
        if tr_id not in self.state.tracks and tr_id not in self.plan.visits:
            raise KeyError(tr_id)
        now_ns = (self.clock.now_ns() or 0) if now_ns is None else now_ns
        view = dict(view) if view is not None else self._vehicle_view(tr_id, now_ns)
        track = self.state.tracks.get(tr_id)
        track_points = []
        if track is not None:
            events = [event for event in track.events if track.trusted(event)][
                -self.settings.track_points :
            ]
            track_points = [
                {
                    "lon": event.lon,
                    "lat": event.lat,
                    "speed_kmh": event.speed_kmh,
                    "event_at": format_source(event.event_time_ns),
                }
                for event in events
            ]
        visits = self.plan.window(tr_id, now_ns, before_s=1800, after_s=3600)
        view["run_id"] = self.run_id
        view["revision"] = self.revision
        view["track"] = track_points
        view["plan"] = [
            {
                "target_stop_id": str(row.tt_action_item_id),
                "planned_at": format_source(int(row.time_begin.value)),
                "address": self.plan.address(str(row.tt_action_item_id)),
                "lon": float(row.lon),
                "lat": float(row.lat),
                "passed": int(row.time_begin.value) <= now_ns,
            }
            for row in visits.itertuples()
        ]
        history = [
            {
                key: item[key]
                for key in (
                    "prediction_id",
                    "cutoff_t",
                    "delay_s",
                    "risk_level",
                    "late_probability",
                    "target_stop_id",
                    "trigger",
                )
                if key in item
            }
            for item in self.prediction_log
            if item["tr_id"] == tr_id
        ][-20:]
        view["prediction_history"] = history
        return view

    def status(self) -> dict:
        now_ns = self.clock.now_ns()
        return {
            "schema_version": "1",
            "backend_version": __version__,
            "run_id": self.run_id,
            "mode": self.settings.mode,
            "ready": self.ready,
            "started_at": self.started_wall,
            "revision": self.revision,
            "clock": self.clock_view(),
            "history": self.view_history.metadata(),
            "split": self.settings.split,
            "plan": {
                "file": str(self._plan_path()),
                "vehicles": len(self.plan.visits),
                "visits": len(self.plan.plan),
                "shift_s": self._plan_shift_s,
            },
            "points": (
                None
                if self.points is None
                else {
                    "total": len(self.points),
                    "delivered": self.points.cursor,
                    "skipped": self.points.skipped,
                    "predicted": self.predicted_points,
                }
            ),
            "state": self.state.summary(now_ns),
            "ml": self.ml.to_dict(),
            "ndtp": (None if self.ndtp is None else self.ndtp.counters.to_dict()),
            "replay": (
                None
                if self.replay is None
                else {
                    "events_total": len(self.replay),
                    "events_delivered": self.replay.cursor,
                    "finished": self.replay.finished,
                    "control_enabled": self.settings.replay_control_enabled,
                }
            ),
            "performance": {
                "cycles": self.cycles,
                "last_cycle_at": self.last_cycle_at,
                "cycle_ms": self.cycle_latency.to_dict(),
                "history_capture_ms": self.history_latency.to_dict(),
                "feature_build_ms": self.feature_latency.to_dict(),
                "event_to_prediction_ms": self.publish_latency.to_dict(),
                "ml_round_trip_ms": self.ml.latency.to_dict(),
                "ml_inference_ms": self.ml.service_latency.to_dict(),
                "predictions": self.prediction_seq,
                "tick_interval_s": self.settings.tick_interval_s,
                "predict_interval_s": self.settings.predict_interval_s,
            },
            "errors": self.errors,
            "unmapped_units": dict(sorted(self.state.unmapped.items())[:20]),
        }

    def quality(self) -> dict:
        """Only measured numbers: the offline report of the loaded bundle and the replay sidecar."""
        offline = None
        path = self.settings.offline_metrics
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            test = data.get("test", {})
            offline = {
                "source": str(path),
                "rows": test.get("rows"),
                "model_mae_s": test.get("model_mae_s"),
                "no_hint_mae_s": test.get("no_hint_mae_s"),
                "cur_dev_mae_s": test.get("cur_dev_mae_s"),
                "zero_mae_s": test.get("zero_mae_s"),
                "late_probability": test.get("late_probability"),
                "note": test.get(
                    "note", "Offline benchmark на размеченном test одного дня, не score платформы"
                ),
            }
        sidecar = None if self.sidecar is None else self.sidecar.report()
        group = (self.ml.model or {}).get("training_group") or ""
        if sidecar is not None and "test" in group.split(";")[0] and "test" in sidecar["source"]:
            # A bundle refitted on test labels cannot be scored honestly on the same labels.
            sidecar["in_sample"] = True
            sidecar["in_sample_note"] = (
                "Модель обучена в том числе на этой разметке: MAE потока in-sample, оптимистична"
            )
        return {
            "schema_version": "1",
            "run_id": self.run_id,
            "offline": offline,
            "replay_sidecar": sidecar,
            "model": self.ml.model
            and {
                "model_version": self.ml.model.get("model_version"),
                "feature_schema_version": self.ml.model.get("feature_schema_version"),
                "calibration": self.ml.model.get("calibration"),
                "training_group": self.ml.model.get("training_group"),
                "train_rows": self.ml.model.get("train_rows"),
            },
        }
