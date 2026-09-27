"""What-if reserve calculation: delay carry-over, layover slack and the reserve's takeover."""

from transport_backend.clock import SECOND_NS, parse_source
from transport_backend.whatif import TripWindow, reserve_whatif

MIN = 60 * SECOND_NS


def trips():
    base = parse_source("2026-01-06 10:00:00")
    # Trips of 50 min with an 8 min planned layover between them.
    return [
        TripWindow(n, base + (n - 1) * 58 * MIN, base + (n - 1) * 58 * MIN + 50 * MIN)
        for n in range(1, 6)
    ]


def test_without_reserve_the_delay_carries_minus_layover_slack():
    now = parse_source("2026-01-06 10:40:00")  # Late on trip 1.
    result = reserve_whatif(trips(), 1, 600.0, now, reserve_in_s=3600)
    rows = result["trips"]
    # 8 min layover − 3 min minimum = 5 min slack per terminal: 600 → 300 → 0 s.
    assert [row["delay_without_s"] for row in rows[:3]] == [300.0, 0.0, 0.0]
    assert result["late_trips_without"] == 1


def test_reserve_takes_the_first_trip_it_can_start_and_saves_the_late_trips():
    now = parse_source("2026-01-06 10:40:00")
    result = reserve_whatif(trips(), 1, 900.0, now, reserve_in_s=10 * 60)
    rows = result["trips"]
    # Trip 2 is planned at 10:58; the reserve is there at 10:50 and starts on time.
    assert rows[0]["served_by"] == "reserve" and rows[0]["delay_with_s"] == 0.0
    assert rows[0]["delay_without_s"] == 600.0
    assert result["late_trips_without"] == 2 and result["late_trips_with"] == 0
    assert result["reserve_takes_trip"] == 2 and result["delay_saved_s"] > 0


def test_a_late_reserve_only_helps_when_it_beats_the_vehicle():
    now = parse_source("2026-01-06 10:40:00")
    # Reserve in 40 min (11:20): later than the late vehicle would start trip 2 (11:08).
    result = reserve_whatif(trips(), 1, 600.0, now, reserve_in_s=40 * 60)
    assert result["trips"][0]["served_by"] == "vehicle"
    assert result["reserve_takes_trip"] == 3


def test_no_following_trip_means_no_answer():
    now = parse_source("2026-01-06 14:00:00")
    assert reserve_whatif(trips(), 5, 300.0, now, 600)["available"] is False
