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
        for value in ("invalid", "NaT", "2026-01-06T12:00:00Z"):
            with pytest.raises(ValueError):
                await engine.seek(value)
            assert engine.run_id == old_run
            assert engine.replay.cursor == old_cursor
    finally:
        await engine.stop()


@pytest.mark.parametrize("target", ["2026-01-05 12:00:15", "2026-01-08 12:00:15"])
async def test_seek_outside_data_is_empty_and_can_return(target):
    engine = make_engine()
    try:
        await engine.seek("2026-01-06 11:30:15")
        assert engine.snapshot()["vehicles"]
        await engine.seek(target)
        await engine._cycle()
        snapshot = engine.snapshot()
        assert snapshot["clock"]["source_time"] == target
        assert snapshot["clock"]["paused"] is True
        assert snapshot["vehicles"] == []
        assert snapshot["alerts"] == []
        await engine.seek("2026-01-06 11:30:15")
        assert engine.snapshot()["vehicles"]
    finally:
        await engine.stop()


async def test_current_deviation_survives_missing_ml_and_seek_but_not_reset():
    from unittest.mock import patch

    engine = make_engine()
    try:
        await engine.seek("2026-01-06 11:30:15")
        first = engine.snapshot()
        assert any(v["current_deviation"]["status"] == "ok" for v in first["vehicles"])
        assert all(v["prediction"] is None for v in first["vehicles"])
        with patch.object(
            engine.state, "estimate_deviation", side_effect=AssertionError("HTTP scan")
        ):
            for _ in range(5):
                assert engine.snapshot()["vehicles"] == first["vehicles"]
                detail = engine.vehicle_detail(first["vehicles"][0]["tr_id"])
                assert detail["current_deviation"] == first["vehicles"][0]["current_deviation"]
        engine.ml.fail = True
        await engine._cycle()
        assert any(v["current_deviation"]["status"] == "ok" for v in engine.snapshot()["vehicles"])
        await engine.seek("2026-01-05 12:00:00")
        assert not engine.current_deviation.estimates
        await engine.seek("2026-01-06 11:30:15")
        assert engine.current_deviation.estimates
        await engine.reset()
        assert not engine.current_deviation.estimates
    finally:
        await engine.stop()


async def test_context_contract_and_autonomous_prefix_match_offline():
    points = load_points(DATA, "test")
    traffic, plan = load_inputs(DATA, "test")
    config = FeatureConfig(schedule_context=True)
    builder = FeatureBuilder(traffic, plan, config)
    engine = make_engine()
    engine.ml.model.update(
        feature_schema_version="2", feature_config=config.to_dict(), hint_policy="supplied_only"
    )
    chosen = points.iloc[[10, 40, 120, 250]].copy()
    chosen["cur_dev_s"] = np.nan
    expected = builder.transform(chosen)
    for position, row in enumerate(chosen.itertuples()):
        at = int(row.T.value)
        engine.clock.start(at)
        engine.clock.paused = True
        engine.replay.reset(at - 1800 * SECOND_NS)
        engine.state.tracks.clear()
        for event in engine.replay.due(at):
            engine.state.add(event, at)
        request = {
            "tr_id": row.tr_id,
            "cutoff_ns": at,
            "target_stop_id": row.target_stop_id,
            "target_planned_ns": int(row.target_time_begin.value),
            "cur_dev_s": None,
            "cur_dev_source": "missing",
            "sample_id": None,
            "trigger": "periodic",
        }
        await engine._predict([request], at)
        # No supplied point/hint enters this periodic request even when GPS can be matched.
        actual = engine.ml.seen[-1]["features"]
        assert actual["cur_dev_s"] is None
        assert_series_equal(
            pd.Series(actual, dtype=float), expected.iloc[position], check_names=False
        )
        prediction = engine.predictions[row.tr_id]
        assert prediction["feature_schema_version"] == "2"
        assert prediction["model_used"] == "fallback"
        assert prediction["cur_dev_source"] == "missing"


async def test_incompatible_model_contract_cannot_send_wrong_features():
    engine = make_engine()
    engine.ml.model.update(feature_schema_version="2", feature_config={"schedule_context": False})
    await engine._predict([], 0)
    assert not engine.ml.ready
    assert "disagree" in engine.ml.last_error
    assert engine.ml.seen == []


class ExplainingMl(FakeMl):
    """Adds the on-demand explanation call and counts how often it is used."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.explain_calls = 0
        self.delay_override: float | None = None

    async def explain(self, item, feature_schema_version):
        self.explain_calls += 1
        delay = self.delay_s if self.delay_override is None else self.delay_override
        groups = [{"group": "movement", "label": "Скорость и движение", "seconds": delay - 10}]
        return {
            "request_id": item["request_id"],
            "delay_s": delay,
            "model_version": "fake",
            "explanation": {
                "base_s": 10.0,
                "groups": groups,
                "other_s": 0.0,
                "total_s": delay,
                "model_used": "main",
            },
        }


async def test_explanation_is_on_demand_cached_and_bound_to_the_prediction():
    from transport_backend.engine import PredictionChanged

    engine = make_engine(replay_speed=600.0, replay_autostart=True)
    engine.ml = ExplainingMl()
    await engine.start()
    try:
        deadline = asyncio.get_running_loop().time() + 20
        while asyncio.get_running_loop().time() < deadline and engine.predicted_points < 3:
            await asyncio.sleep(0.2)
    finally:
        await engine.stop()  # Freeze the current predictions for the checks below.
    ready = [tr for tr, view in engine.predictions.items() if view["status"] == "ok"]
    assert ready and engine.ml.explain_calls == 0  # The cycle never explains.
    tr_id = ready[0]
    prediction_id = engine.predictions[tr_id]["prediction_id"]
    first = await engine.explanation(tr_id, prediction_id)
    again = await engine.explanation(tr_id, prediction_id)
    assert first is again and engine.ml.explain_calls == 1
    assert first["explanation"]["total_s"] == engine.predictions[tr_id]["delay_s"]
    with pytest.raises(PredictionChanged):
        await engine.explanation(tr_id, "p-999999")
    # A different number from ML (e.g. a swapped model) is refused, not shown as the reason.
    engine.explanations.clear()
    engine.ml.delay_override = 1.0
    with pytest.raises(PredictionChanged):
        await engine.explanation(tr_id, prediction_id)
    await engine.reset()
    assert not engine.feature_rows and not engine.explanations


def test_attention_uses_delay_first_then_calibrated_probability():
    from transport_backend.engine import needs_attention

    policy = RiskPolicy()
    assert policy.attention(150.0, 0.1) == "delay"
    assert policy.attention(-90.0, None) == "delay"  # Early running is also a risk.
    assert policy.attention(30.0, 0.7) == "probability"
    assert policy.attention(30.0, 0.2) is None
    assert policy.attention(30.0, None) is None
    ok = {"status": "ok", "risk_level": "green", "attention": "probability"}
    assert needs_attention({"prediction": ok})
    assert not needs_attention({"prediction": {**ok, "status": "stale"}})
    assert not needs_attention({"prediction": None})


def test_recommendation_escalates_a_likely_late_arrival():
    from transport_backend.explain import recommendation

    policy = RiskPolicy()
    calm = recommendation(80.0, [], policy, 0.2)
    likely = recommendation(80.0, [], policy, 0.6)
    assert "наблюдением" in calm and "60%" in likely
    assert recommendation(80.0, [], policy) == calm  # Without a classifier: delay only.
    assert "Опережение" in recommendation(-90.0, [], policy, 0.9)


def test_acknowledgement_carries_to_the_next_stop_of_the_same_episode():
    engine = make_engine()

    def view(stop, cutoff, delay=150.0):
        return {
            "tr_id": "bus",
            "target_stop_id": stop,
            "target_planned_at": cutoff,
            "cutoff_t": cutoff,
            "computed_at": "wall",
            "horizon_s": 700.0,
            "risk_level": "red",
            "delay_s": delay,
            "late_probability": 0.8,
            "evidence": [],
            "attention": "delay",
        }

    engine._update_alert(view("s1", "2026-01-06 10:00:00"))
    first = next(iter(engine.alerts.values()))
    engine.acknowledge(first.alert_id)
    engine._update_alert(view("s2", "2026-01-06 10:05:00"))
    second = engine.alerts[f"{engine.run_id}:bus:s2"]
    assert second.acknowledged_at == first.acknowledged_at
    assert second.acknowledged_from == first.alert_id
    # Alerts already open for the vehicle's later stops are covered by the click as well.
    engine._update_alert(view("s3", "2026-01-06 10:06:00"))
    engine.alerts[f"{engine.run_id}:bus:s3"].acknowledged_at = None
    engine.alerts[f"{engine.run_id}:bus:s3"].acknowledged_from = None
    engine.alerts[f"{engine.run_id}:bus:s1"].acknowledged_at = None
    engine.acknowledge(f"{engine.run_id}:bus:s1")
    assert engine.alerts[f"{engine.run_id}:bus:s3"].acknowledged_from == first.alert_id
    # A new episode long after the last acknowledged prediction must be seen again.
    engine._update_alert(view("s9", "2026-01-06 11:00:00"))
    assert engine.alerts[f"{engine.run_id}:bus:s9"].acknowledged_at is None


@pytest.mark.parametrize(("date", "days"), [("2026-09-28", 265), ("2026-01-06", 0)])
def test_live_stream_date_aligns_the_timetable_by_whole_days(date, days):
    from transport_backend.events import TelemetryEvent

    engine = make_engine(mode="ndtp", plan_shift_auto=True, use_points=False)
    unit, tr_id = next(iter(engine.mapping.items()))
    event = TelemetryEvent(
        source="ndtp_live",
        unit_id=unit,
        event_time_ns=pd.Timestamp(f"{date} 08:00:00").value,
        gps_valid=True,
        lon=37.6,
        lat=55.7,
        speed_kmh=20.0,
        heading_deg=90.0,
    )
    engine._ingest(event)
    assert engine._plan_shift_s == days * 86_400
    assert engine.state.plan is engine.plan
    assert tr_id in engine.state.tracks  # Accepted, not rejected as a far-future packet.
    # The next plan visit is on the stream's day at the same time of day as in the dataset.
    first = engine.plan.plan.time_begin.min()
    assert first.date() == (pd.Timestamp("2026-01-06") + pd.Timedelta(days=days)).date()
    assert first.time() == pd.Timestamp("2026-01-06 02:18:00").time()


def test_env_defaults_align_live_streams_but_not_replay(monkeypatch):
    monkeypatch.delenv("BACKEND_PLAN_SHIFT_S", raising=False)
    monkeypatch.setenv("BACKEND_MODE", "ndtp")
    assert Settings.from_env().plan_shift_auto
    monkeypatch.setenv("BACKEND_MODE", "replay")
    assert not Settings.from_env().plan_shift_auto
    monkeypatch.setenv("BACKEND_MODE", "ndtp")
    monkeypatch.setenv("BACKEND_PLAN_SHIFT_S", "0")
    assert not Settings.from_env().plan_shift_auto


def test_recommendation_prioritises_first_and_last_trips():
    from transport_backend.explain import recommendation

    policy = RiskPolicy()
    last = {"number": 14, "total": 14, "first": False, "last": True}
    middle = {"number": 5, "total": 14, "first": False, "last": False}
    assert "последнего рейса дня (рейс 14 из 14)" in recommendation(150.0, [], policy, 0.9, last)
    assert "рейса дня" not in recommendation(150.0, [], policy, 0.9, middle)
    # An on-time first trip needs no escalation.
    assert "рейса дня" not in recommendation(20.0, [], policy, 0.1, {**last, "first": True})
