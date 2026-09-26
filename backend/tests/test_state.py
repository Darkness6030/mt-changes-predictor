"""Bounded state, plan windows and the causal deviation estimate."""

import pandas as pd
import pytest
from transport_backend.clock import SECOND_NS, SourceClock
from transport_backend.events import TelemetryEvent
from transport_backend.state import FleetState, PlanStore

T = pd.Timestamp("2026-01-06 12:00:00")


def plan_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tr_id": ["bus"] * 4,
            "tt_action_item_id": ["passed", "excluded", "target", "later"],
            "time_begin": [T + pd.Timedelta(seconds=s) for s in (-300, 600, 780, 1200)],
            "geom": [
                "POINT (37.600 55.700)",
                "POINT (37.610 55.700)",
                "POINT (37.620 55.700)",
                "POINT (37.630 55.700)",
            ],
            "building_address": ["Адрес 1", "Адрес 2", "Адрес 3", None],
        }
    )


def event(seconds: float, *, lon=37.6, lat=55.7, valid=True, speed=20.0) -> TelemetryEvent:
    return TelemetryEvent(
        source="csv_replay",
        unit_id="unit-1",
        tr_id="bus",
        event_time_ns=int(T.value + seconds * SECOND_NS),
        gps_valid=valid,
        lon=lon if valid else None,
        lat=lat if valid else None,
        speed_kmh=speed,
        heading_deg=90.0,
    )


@pytest.fixture
def state() -> FleetState:
    return FleetState(
        plan=PlanStore(plan_frame()),
        history_window_s=1800,
        history_max_events=10,
        stale_after_s=120,
    )


def test_target_window_excludes_600_and_includes_900(state):
    assert state.plan.target("bus", T.value).tt_action_item_id == "target"
    exactly_600 = int(T.value + 0)
    shifted = PlanStore(plan_frame().assign(time_begin=lambda f: f.time_begin))
    assert shifted.target("bus", exactly_600 + 0) is not None
    # A plan 600 s away is outside the window, 900 s away is inside it.
    edge = PlanStore(
        pd.DataFrame(
            {
                "tr_id": ["bus", "bus"],
                "tt_action_item_id": ["at_600", "at_900"],
                "time_begin": [T + pd.Timedelta(seconds=600), T + pd.Timedelta(seconds=900)],
                "geom": ["POINT (37.6 55.7)"] * 2,
            }
        )
    )
    assert edge.target("bus", T.value).tt_action_item_id == "at_900"
    assert edge.target("bus", T.value + 1) is not None
    assert edge.target("unknown", T.value) is None
    assert state.plan.address("passed") == "Адрес 1"


def test_duplicates_conflicts_and_window_are_bounded(state):
    assert state.add(event(-10))
    assert not state.add(event(-10))  # Identical normalized event.
    assert state.add(event(-10, speed=40.0))  # Same time, different field: kept.
    track = state.tracks["bus"]
    assert track.conflicting_times == 1 and state.duplicate_events == 1
    for index in range(20):
        state.add(event(-index - 20, speed=float(index)))
    assert len(track.events) == 10 and track.truncated
    state.trim(int(T.value))
    assert "history_truncated" in track.quality_flags(int(T.value), 120)


def test_unknown_unit_is_not_assigned_a_vehicle():
    state = FleetState(plan=PlanStore(plan_frame()))
    unmapped = TelemetryEvent(
        source="ndtp_live",
        unit_id="999",
        event_time_ns=int(T.value),
        gps_valid=True,
        lon=37.6,
        lat=55.7,
        speed_kmh=10.0,
    )
    assert not state.add(unmapped)
    assert state.unmapped == {"999": 1} and not state.tracks
    state.mapping["999"] = "bus"
    assert state.add(unmapped) and "bus" in state.tracks


def test_future_events_are_quarantined(state):
    assert not state.add(event(600), now_ns=int(T.value))
    assert state.rejected_future == 1


def test_deviation_uses_the_last_visit_actually_reached():
    state = FleetState(plan=PlanStore(plan_frame()), history_max_events=200)
    # The vehicle stands at the "passed" stop around its planned time, about 80 s late.
    for offset in range(-240, -180, 12):
        state.add(event(offset, lon=37.600, lat=55.700, speed=0.0))
    for offset in range(-120, 0, 12):
        state.add(event(offset, lon=37.615, lat=55.700, speed=25.0))
    deviation = state.estimate_deviation("bus", int(T.value))
    assert deviation is not None
    assert deviation.visit_id == "passed"
    assert deviation.seconds == pytest.approx(84.0, abs=13.0)
    assert deviation.distance_m < 60
    assert (int(T.value) - deviation.arrival_ns) / SECOND_NS > 0


def test_deviation_is_missing_without_a_nearby_visit():
    state = FleetState(plan=PlanStore(plan_frame()), history_max_events=200)
    for offset in range(-240, 0, 12):
        state.add(event(offset, lon=38.0, lat=56.0))
    assert state.estimate_deviation("bus", int(T.value)) is None


def test_clock_modes():
    driven = SourceClock("driven", speed=10.0)
    driven.start(int(T.value), paused=True)
    assert driven.now_ns() == int(T.value)
    driven.resume()
    assert driven.now_ns() >= int(T.value)
    driven.pause()
    paused = driven.now_ns()
    assert driven.now_ns() == paused
    with pytest.raises(ValueError, match="Replay speed"):
        driven.set_speed(0)
    follow = SourceClock("follow")
    assert follow.now_ns() is None
    follow.observe(int(T.value))
    assert follow.now_ns() >= int(T.value)
    follow.observe(int(T.value) - 10**9)  # Older event does not move the clock backwards.
    assert follow.now_ns() >= int(T.value)
