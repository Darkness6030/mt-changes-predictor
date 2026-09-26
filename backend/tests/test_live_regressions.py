"""Regression scenarios discovered in the realtime compliance audit."""

import asyncio
from dataclasses import replace
from time import perf_counter

import pandas as pd
import pytest
from transport_backend.clock import SECOND_NS
from transport_backend.config import Settings
from transport_backend.events import from_nav00
from transport_backend.ndtp import encode_nav00, parse_nav00
from transport_backend.state import FleetState, PlanStore, VehicleTrack

from .test_engine import FakeMl, make_engine
from .test_state import T, event, plan_frame


def test_late_packets_and_overflow_preserve_latest_position_and_dedup():
    track = VehicleTrack("bus", "unit-1", max_events=3)
    for seconds in range(100):
        track.add(event(seconds))
    assert not track.add(event(98))
    track.add(event(1))
    assert track.last_valid_ns == event(99).event_time_ns
    assert [item.event_time_ns for item in track.events] == [
        event(i).event_time_ns for i in (97, 98, 99)
    ]
    assert track.fingerprints == {item.fingerprint() for item in track.events}
    track.trim(event(98).event_time_ns)
    assert not track.add(event(99))


def test_prefix_order_does_not_change_hint_or_position():
    states = [FleetState(plan=PlanStore(plan_frame())) for _ in range(2)]
    events = [event(i, speed=0) for i in range(-300, -180, 12)]
    events += [event(-10, lon=37.62), event(-10, lon=37.63)]
    for state, rows in zip(states, (events, list(reversed(events))), strict=True):
        for row in rows:
            state.add(row)
    assert states[0].tracks["bus"].last_valid == states[1].tracks["bus"].last_valid
    assert states[0].estimate_deviation("bus", T.value) == states[1].estimate_deviation(
        "bus", T.value
    )


def test_unmapped_and_future_packets_cannot_move_fleet_clock():
    engine = make_engine(mode="ndtp")
    unit, vehicle = next(iter(engine.mapping.items()))
    base = replace(event(0), source="ndtp_live", unit_id=unit, tr_id=None)
    engine._ingest(replace(base, event_time_ns=base.event_time_ns + 365 * 86400 * SECOND_NS))
    assert engine.clock.now_ns() is None
    engine._ingest(base)
    before = engine.clock.now_ns()
    assert before is not None
    for unit_id in ("unknown", unit):
        engine._ingest(replace(base, unit_id=unit_id, event_time_ns=before + 86400 * SECOND_NS))
    after = engine.clock.now_ns()
    assert after - before < SECOND_NS
    engine.state.trim(after)
    assert len(engine.state.tracks[vehicle].events) == 1
    assert engine.state.rejected_future == 2
    assert engine.state.unmapped == {"unknown": 1}


def test_ndtp_defaults_to_autonomous_predictions(monkeypatch):
    engine = make_engine(mode="ndtp")
    assert engine.points is None
    assert Settings(mode="replay").use_points
    assert not Settings(mode="ndtp").use_points
    monkeypatch.setenv("BACKEND_MODE", "ndtp")
    monkeypatch.setenv("BACKEND_USE_POINTS", "")
    assert not Settings.from_env().use_points
    monkeypatch.setenv("BACKEND_USE_POINTS", "true")
    assert Settings.from_env().use_points  # Explicit historical benchmark remains available.


def test_invalid_coordinates_do_not_refresh_valid_gps():
    raw = encode_nav00(
        timestamp=1767700800, lon=200, lat=95, gps_valid=True, speed_kmh=368, heading_deg=400
    )
    normalized = from_nav00(parse_nav00(raw[2:]), 1, source="ndtp_live")
    assert not normalized.gps_valid
    assert normalized.lon is None and normalized.lat is None
    assert normalized.speed_kmh is None and normalized.heading_deg is None
    assert "coordinate_out_of_range" in normalized.quality_flags


def test_segment_uses_visit_keys_and_refuses_long_gaps():
    plan = PlanStore(plan_frame())
    segment = plan.segment("bus", "target")
    assert segment["from"]["target_stop_id"] == "excluded"
    assert segment["to"]["target_stop_id"] == "target"
    assert plan.segment("bus", "passed") is None
    frame = plan_frame()
    frame.loc[3, "time_begin"] += pd.Timedelta(hours=2)
    assert PlanStore(frame).segment("bus", "later") is None
    assert plan.segment("missing", "target") is None


@pytest.mark.parametrize("reset_during_request", [True, False])
async def test_inflight_result_is_bound_to_run_and_ingestion_timestamp(reset_during_request):
    engine = make_engine(use_points=False)
    cutoff = int(T.value)
    engine.replay.reset(cutoff - 1800 * SECOND_NS)
    for row in engine.replay.due(cutoff):
        engine.state.add(row, cutoff)
    engine.clock.start(cutoff, paused=True)
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedMl(FakeMl):
        async def predict(self, items, schema):
            entered.set()
            await release.wait()
            # Simulate a new event received while inference was in flight.
            for track in engine.state.tracks.values():
                track.last_received_monotonic = perf_counter()
            return await super().predict(items, schema)

    engine.ml = DelayedMl()
    for track in engine.state.tracks.values():
        track.last_received_monotonic = perf_counter() - 1
    task = asyncio.create_task(engine._predict(engine._collect(cutoff), cutoff))
    await asyncio.wait_for(entered.wait(), timeout=3)
    try:
        if reset_during_request:
            await engine.reset(None)
        release.set()
        await task
        if reset_during_request:
            assert not engine.predictions and not engine.alerts and not engine.prediction_log
        else:
            assert engine.publish_latency.samples
            assert min(engine.publish_latency.samples) >= 1000
            view = next(
                v for v in engine.snapshot()["vehicles"] if v["prediction"]["status"] == "ok"
            )
            detail = engine.vehicle_detail(view["tr_id"])
            assert detail["run_id"] == engine.run_id
            assert detail["segment"] == view["segment"]
    finally:
        release.set()
        await task
