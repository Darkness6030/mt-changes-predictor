"""Engine behaviour on the real dataset: parity with offline features, statuses, alerts."""

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from transport_backend.clock import SECOND_NS, format_source
from transport_backend.config import RiskPolicy, Settings
from transport_backend.engine import Engine
from transport_ml.data import load_inputs, load_points
from transport_ml.features import FeatureBuilder, FeatureConfig

DATA = Path("dataset")


class FakeMl:
    """Stands in for the HTTP service: records the features it was given."""

    def __init__(self, delay_s: float = 150.0, fail: bool = False):
        self.delay_s = delay_s
        self.fail = fail
        self.seen: list[dict] = []
        self.model = {
            "model_version": "fake",
            "feature_schema_version": "1",
            "late_threshold_s": 120.0,
            "calibration": {"method": "platt", "status": "fitted_on_development"},
        }
        self.ready = True
        self.last_error = None
        self.requests = 0
        self.errors = 0
        self.timeouts = 0
        from transport_backend.mlclient import Latency

        self.latency = Latency()
        self.service_latency = Latency()

    async def refresh_model(self):
        return self.model

    async def predict(self, items, feature_schema_version):
        self.requests += 1
        self.seen.extend(items)
        if self.fail:
            self.errors += 1
            self.ready = False
            self.last_error = "stub failure"
            return {}
        return {
            item["request_id"]: {
                "request_id": item["request_id"],
                "delay_s": self.delay_s,
                "model_used": "main" if item["features"]["cur_dev_s"] is not None else "fallback",
                "used_hint": item["features"]["cur_dev_s"] is not None,
                "late_probability": 0.7,
                "late_threshold_s": 120.0,
                "calibration": self.model["calibration"],
                "model_version": "fake",
            }
            for item in items
        }

    async def aclose(self):
        return None

    def to_dict(self):
        return {"ready": self.ready, "model_version": "fake"}


def make_engine(**overrides) -> Engine:
    options = {
        "mode": "replay",
        "data_root": DATA,
        "split": "test",
        "labels": Path("dataset/labels/labels_test.csv"),
        "offline_metrics": Path("ml/pretrained/v2/metrics.json"),
        "replay_autostart": False,
        "risk": RiskPolicy(),
    }
    settings = Settings(**{**options, **overrides})
    engine = Engine(settings)
    engine.ml = FakeMl()
    return engine


@pytest.fixture(scope="module")
def offline_features():
    points = load_points(DATA, "test")
    traffic, plan = load_inputs(DATA, "test")
    builder = FeatureBuilder(traffic, plan, FeatureConfig())
    return points, builder


async def test_streaming_features_equal_the_offline_batch(offline_features):
    """The same prefix of events must produce the same features as offline training."""
    points, builder = offline_features
    engine = make_engine()
    chosen = points.iloc[[10, 40, 120, 250]]
    expected = builder.transform(chosen)
    for position, row in enumerate(chosen.itertuples()):
        cutoff_ns = int(row.T.value)
        engine.clock.start(cutoff_ns - int(1800 * SECOND_NS))
        engine.replay.reset(cutoff_ns - int(1800 * SECOND_NS))
        engine.state.tracks.clear()
        for event in engine.replay.due(cutoff_ns):
            engine.state.add(event, cutoff_ns)
        request = {
            "tr_id": row.tr_id,
            "cutoff_ns": cutoff_ns,
            "target_stop_id": row.target_stop_id,
            "target_planned_ns": int(row.target_time_begin.value),
            "cur_dev_s": float(row.cur_dev_s),
            "cur_dev_source": "supplied",
            "sample_id": row.sample_id,
            "trigger": "point",
        }
        built = engine._build_features([request])[id(request)]
        streamed = pd.Series(built, name=position, dtype=float)
        assert_series_equal(
            streamed,
            expected.iloc[position].rename(position).astype(float),
            check_names=False,
        )


async def test_future_events_do_not_change_a_past_prediction(offline_features):
    points, builder = offline_features
    row = points.iloc[40]
    engine = make_engine()
    cutoff_ns = int(row["T"].value)
    engine.clock.start(cutoff_ns - int(1800 * SECOND_NS))
    engine.replay.reset(cutoff_ns - int(1800 * SECOND_NS))
    base_request = {
        "tr_id": row.tr_id,
        "cutoff_ns": cutoff_ns,
        "target_stop_id": row.target_stop_id,
        "target_planned_ns": int(row.target_time_begin.value),
        "cur_dev_s": float(row.cur_dev_s),
        "cur_dev_source": "supplied",
        "sample_id": row.sample_id,
        "trigger": "point",
    }
    for event in engine.replay.due(cutoff_ns):
        engine.state.add(event, cutoff_ns)
    before = engine._build_features([dict(base_request)])
    # Deliver 10 more minutes of telemetry, then rebuild features for the same cutoff.
    for event in engine.replay.due(cutoff_ns + int(600 * SECOND_NS)):
        engine.state.add(event, cutoff_ns + int(600 * SECOND_NS))
    after = engine._build_features([dict(base_request)])
    assert list(before.values())[0] == list(after.values())[0]


async def test_point_driven_cycle_produces_predictions_alerts_and_sidecar():
    engine = make_engine(replay_speed=600.0, predict_interval_s=60.0, replay_autostart=True)
    await engine.start()
    try:
        deadline = asyncio.get_running_loop().time() + 20
        while asyncio.get_running_loop().time() < deadline:
            if engine.predicted_points >= 5 and engine.alerts:
                break
            await asyncio.sleep(0.2)
        assert engine.predicted_points >= 5, engine.status()
        snapshot = engine.snapshot()
        assert snapshot["run_id"] == engine.run_id and snapshot["revision"] > 0
        ok = [
            vehicle["prediction"]
            for vehicle in snapshot["vehicles"]
            if vehicle["prediction"] and vehicle["prediction"]["status"] == "ok"
        ]
        assert ok, snapshot["summary"]
        for prediction in ok:
            assert 600 < prediction["horizon_s"] <= 900
            assert prediction["risk_level"] == "red"  # The stub always answers 150 s.
            assert prediction["late_probability"] == 0.7
            assert prediction["delay_s"] == 150.0
            assert prediction["recommendation"]
            assert prediction["cur_dev_source"] in {"supplied", "estimated", "missing"}
        assert engine.alerts, engine.status()
        alert = list(engine.alerts.values())[0]
        assert 600 < alert.planned_lead_s <= 900
        acknowledged = engine.acknowledge(alert.alert_id)
        assert acknowledged.acknowledged_at
        # Statuses for vehicles without a target must not be a zero delay.
        missing = [
            vehicle["prediction"]
            for vehicle in snapshot["vehicles"]
            if vehicle["prediction"] and vehicle["prediction"]["status"] != "ok"
        ]
        assert all(item["delay_s"] is None for item in missing)
        assert {item["status"] for item in missing} <= {
            "no_target_in_horizon",
            "no_schedule",
            "warming_up",
            "stale",
            "invalid_input",
            "ml_unavailable",
        }
        detail = engine.vehicle_detail(ok[0]["tr_id"])
        assert detail["plan"] and detail["prediction_history"]
        quality = engine.quality()
        assert quality["offline"]["model_mae_s"] == pytest.approx(79.0874, abs=1e-3)
        assert quality["replay_sidecar"]["measured_rows"] >= 0
    finally:
        await engine.stop()


async def test_ml_failure_keeps_the_previous_number_with_its_age():
    engine = make_engine(replay_speed=600.0, replay_autostart=True)
    await engine.start()
    try:
        deadline = asyncio.get_running_loop().time() + 20
        while asyncio.get_running_loop().time() < deadline:
            if engine.predicted_points >= 3:
                break
            await asyncio.sleep(0.2)
        tr_id = next(key for key, view in engine.predictions.items() if view["status"] == "ok")
        engine.ml.fail = True
        previous = engine.predictions[tr_id]
        request = {
            "tr_id": tr_id,
            "cutoff_ns": previous["cutoff_ns"] + int(120 * SECOND_NS),
            "target_stop_id": None,
            "target_planned_ns": None,
            "cur_dev_s": None,
            "cur_dev_source": None,
            "sample_id": None,
            "trigger": "periodic",
        }
        await engine._predict([request], request["cutoff_ns"])
        current = engine.predictions[tr_id]
        assert current["status"] in {"ml_unavailable", "no_target_in_horizon", "stale"}
        if current["status"] == "ml_unavailable":
            assert current["delay_s"] == previous["delay_s"]
            assert current["prediction_age_s"] > 0
    finally:
        await engine.stop()


async def test_replay_control_reset_starts_a_new_run():
    engine = make_engine()
    await engine.start()
    try:
        first_run = engine.run_id
        engine.clock.set_speed(120.0)
        await engine.reset(None)
        assert engine.run_id != first_run
        assert not engine.alerts and not engine.predictions
        assert engine.state.accepted_events == 0
        status = engine.status()
        assert status["clock"]["paused"] is False
        assert status["points"]["total"] == 353
        assert np.isfinite(status["performance"]["cycles"])
        assert format_source(engine.clock.now_ns()).startswith("2026-01-06")
    finally:
        await engine.stop()


async def test_seek_backwards_rebuilds_causal_history_and_batch_features(offline_features):
    points, builder = offline_features
    engine = make_engine()
    engine.clock.set_speed(120)
    try:
        # Seek forward first, then backwards: future state and alerts must disappear.
        for index in (250, 40):
            row = points.iloc[index]
            old_run = engine.run_id
            await engine.seek(str(row["T"]))
            cutoff_ns = int(row["T"].value)
            assert engine.run_id != old_run
            assert engine.clock.paused and engine.clock.now_ns() == cutoff_ns
            assert engine.clock.speed == 120
            assert not engine.predictions and not engine.alerts and not engine.prediction_log
            assert engine.state.accepted_events > 0
            assert all(
                cutoff_ns - int(engine.settings.history_window_s * SECOND_NS)
                <= event.event_time_ns
                <= cutoff_ns
                for track in engine.state.tracks.values()
                for event in track.events
            )
            request = {
                "tr_id": row.tr_id,
                "cutoff_ns": cutoff_ns,
                "target_stop_id": row.target_stop_id,
                "target_planned_ns": int(row.target_time_begin.value),
                "cur_dev_s": float(row.cur_dev_s),
                "cur_dev_source": "supplied",
                "sample_id": row.sample_id,
                "trigger": "point",
            }
            actual = engine._build_features([request])[id(request)]
            expected = builder.transform(points.iloc[[index]]).iloc[0]
            assert_series_equal(pd.Series(actual, dtype=float), expected, check_names=False)
            # A paused seek still produces periodic forecasts from the restored history.
            await engine._cycle()
            assert engine.predictions
            assert all(p["cutoff_ns"] == cutoff_ns for p in engine.predictions.values())
            assert engine.clock.now_ns() == cutoff_ns
            cursor = engine.replay.cursor
            engine.clock.start(cutoff_ns + 60 * SECOND_NS)
            await engine._cycle()
            assert engine.replay.cursor > cursor
    finally:
        await engine.stop()


async def test_seek_accepts_endpoints_and_rejects_invalid_time_without_reset():
    engine = make_engine()
    try:
        for target in (engine.replay.first_ns, engine.replay.last_ns):
            await engine.seek(format_source(target))
            assert engine.clock.now_ns() == target
        old_run = engine.run_id
        old_cursor = engine.replay.cursor
        for value in ("invalid", "NaT", "2026-01-05", "2026-01-08", "2026-01-06T12:00:00Z"):
            with pytest.raises(ValueError):
                await engine.seek(value)
            assert engine.run_id == old_run
            assert engine.replay.cursor == old_cursor
    finally:
        await engine.stop()
