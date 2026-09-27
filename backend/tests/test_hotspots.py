"""Delay hotspots: gains between consecutive observed passages, attributed to segments."""

import pandas as pd
from transport_backend.clock import SECOND_NS
from transport_backend.hotspots import DelayHotspots
from transport_backend.state import Deviation, PlanStore

T = pd.Timestamp("2026-01-06 12:00:00")


def plan() -> PlanStore:
    ids = ["a", "b", "c", "d", "e", "f"]
    return PlanStore(
        pd.DataFrame(
            {
                "tr_id": ["bus"] * 6,
                "tt_action_item_id": ids,
                "time_begin": [T + pd.Timedelta(minutes=3 * i) for i in range(6)],
                "geom": [f"POINT (37.6{i} 55.7)" for i in range(6)],
                "building_address": [f"Остановка {x}" for x in ids],
            }
        )
    )


def seen(visit: str, minute: float, deviation: float) -> Deviation:
    return Deviation(deviation, int(T.value + minute * 60 * SECOND_NS), visit, 10.0, 1)


def test_gains_accumulate_on_the_segment_between_observed_stops():
    hotspots, store = DelayHotspots(), plan()
    hotspots.observe("bus", seen("a", 0.5, 30), store)
    hotspots.observe("bus", seen("a", 0.5, 30), store)  # Same visit: nothing new.
    hotspots.observe("bus", seen("b", 4.5, 90), store)  # +60 s on a → b.
    hotspots.observe("bus", seen("c", 7.0, 60), store)  # −30 s: recovered on b → c.
    report = hotspots.report()
    assert [row["segment_id"] for row in report] == ["bus:a:b"]
    assert report[0]["gain_total_s"] == 60 and report[0]["passes"] == 1
    assert report[0]["from"]["address"] == "Остановка a"


def test_jumps_over_unobserved_stops_and_resets_are_not_attributed():
    hotspots, store = DelayHotspots(), plan()
    hotspots.observe("bus", seen("a", 0.5, 0), store)
    hotspots.observe("bus", seen("f", 16, 200), store)  # Five stops later: skipped.
    hotspots.observe("bus", seen("b", 4, 50), store)  # Backwards (seek/reset): skipped.
    assert hotspots.report() == []
    hotspots.clear()
    assert not hotspots.last and not hotspots.segments
