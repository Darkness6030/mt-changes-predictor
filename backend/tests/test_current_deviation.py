"""Current colour is causal GPS evidence, independent from predictions and supplied hints."""

from unittest.mock import patch

import pandas as pd
import pytest
from transport_backend.config import RiskPolicy, Settings
from transport_backend.current_deviation import CurrentDeviationMonitor
from transport_backend.state import FleetState, PlanStore

from .test_state import T, event, plan_frame


@pytest.mark.parametrize(
    ("offsets", "level", "seconds"),
    [
        ((-310, -300), "green", 0),
        ((-240, -228), "yellow", 72),
        ((-168, -156), "red", 144),
        ((-400, -380), "yellow", -80),
    ],
)
def test_current_colour_uses_signed_gps_deviation(offsets, level, seconds):
    state = FleetState(plan=PlanStore(plan_frame()))
    for offset in offsets:
        state.add(event(offset))
    monitor = CurrentDeviationMonitor(600, RiskPolicy())
    monitor.refresh(state, T.value)
    view = monitor.view("bus", T.value, 0, 120)
    assert view["status"] == "ok" and view["risk_level"] == level
    assert view["delay_s"] == seconds
    assert view["source"] == "gps_plan" and view["visit_id"] == "passed"


def test_current_estimate_stales_without_losing_historical_value():
    state = FleetState(plan=PlanStore(plan_frame()))
    for offset in (-240, -228):
        state.add(event(offset))
    monitor = CurrentDeviationMonitor(300, RiskPolicy())
    monitor.refresh(state, T.value)
    assert monitor.view("bus", T.value, 120, 120)["status"] == "ok"
    stale_gps = monitor.view("bus", T.value, 121, 120)
    assert stale_gps["status"] == "stale" and stale_gps["reason"] == "stale_gps"
    assert stale_gps["delay_s"] == 72 and stale_gps["risk_level"] is None
    assert monitor.view("bus", T.value, None, 120)["reason"] == "no_valid_position"
    later = T.value + 73 * 10**9
    assert monitor.view("bus", later, 0, 120)["reason"] == "estimate_too_old"
    assert monitor.view("bus", T.value + 72 * 10**9, 0, 120)["status"] == "ok"
    monitor.clear()
    assert monitor.view("bus", T.value, 0, 120)["status"] == "unavailable"
    assert not monitor.estimates and not monitor._keys


def test_current_estimate_excludes_future_and_schedule_facts():
    plan = plan_frame().assign(time_fact_begin=T + pd.Timedelta(days=10))
    state = FleetState(plan=PlanStore(plan))
    for offset in (-240, -228, 20, 40):
        state.add(event(offset))
    monitor = CurrentDeviationMonitor(300, RiskPolicy())
    monitor.refresh(state, T.value)
    original = monitor.view("bus", T.value, 0, 120)
    # Arbitrary future GPS and future facts cannot change a current observation.
    state.add(event(50, lon=38))
    plan["time_fact_begin"] = T - pd.Timedelta(days=20)
    state.plan = PlanStore(plan)
    monitor.refresh(state, T.value)
    assert monitor.view("bus", T.value, 0, 120) == original
    assert original["delay_s"] == 72
    assert monitor.view("bus", T.value - 300 * 10**9, 0, 120)["delay_s"] is None


def test_paused_refresh_reuses_estimate_but_late_packets_and_trim_invalidate():
    state = FleetState(plan=PlanStore(plan_frame()))
    for offset in (-240, -228):
        state.add(event(offset))
    monitor = CurrentDeviationMonitor(300, RiskPolicy())
    with patch.object(state, "estimate_deviation", wraps=state.estimate_deviation) as estimate:
        monitor.refresh(state, T.value)
        for _ in range(20):
            monitor.refresh(state, T.value)
            monitor.view("bus", T.value, 0, 120)
        assert estimate.call_count == 1
        state.add(event(-216))
        monitor.refresh(state, T.value)
        assert estimate.call_count == 2
        state.tracks["bus"].trim(T.value)
        monitor.refresh(state, T.value)
        assert estimate.call_count == 3
        assert monitor.view("bus", T.value, 0, 120)["status"] == "unavailable"
    state.tracks.clear()
    monitor.refresh(state, T.value)
    assert not monitor.estimates and not monitor._keys


def test_no_valid_gps_has_no_current_colour():
    state = FleetState(plan=PlanStore(plan_frame()))
    for offset in (-240, -228):
        state.add(event(offset, valid=False))
    monitor = CurrentDeviationMonitor(300, RiskPolicy())
    monitor.refresh(state, T.value)
    view = monitor.view("bus", T.value, None, 120)
    assert view["delay_s"] is None and view["risk_level"] is None
    assert view["status"] == "unavailable"
    with pytest.raises(ValueError, match="max age"):
        Settings(current_deviation_max_age_s=0)
