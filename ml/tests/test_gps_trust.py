from transport_ml.features import SECOND
from transport_ml.gps_trust import (
    IMPLAUSIBLE_SPEED,
    JUMP,
    SHEREMETYEVO_SPOOF_ZONE,
    SPOOF_ZONE,
    GpsTrustFilter,
)

CITY = (37.506, 55.811)  # Real route area ~19 km south of the spoofed circle.
AIRFIELD = SHEREMETYEVO_SPOOF_ZONE[:2]


def feed(fixes):
    """fixes: (seconds, lon, lat, speed) -> list of verdicts in the same order."""
    trust = GpsTrustFilter()
    return [trust.update(int(t * SECOND), lon, lat, speed) for t, lon, lat, speed in fixes]


def test_airfield_circle_is_rejected_regardless_of_speed():
    fixes = [(0, *CITY, 20.0), (15, *AIRFIELD, 99.0), (30, AIRFIELD[0], 55.9636, 12.0)]
    assert feed(fixes) == [None, SPOOF_ZONE, SPOOF_ZONE]


def test_jump_away_and_back_keeps_real_track():
    # Spoofed fixes outside the zone: 18 km away within 15 s, then back near the route.
    far = (37.30, 55.95)
    fixes = [(0, *CITY, 20.0), (15, *far, 97.0), (30, far[0] + 0.001, far[1], 97.0)]
    fixes += [(45, CITY[0] + 0.001, CITY[1], 25.0)]
    assert feed(fixes) == [None, JUMP, JUMP, None]


def test_slow_consistent_new_track_relocks_after_five_fixes():
    far = (37.30, 55.95)
    fixes = [(0, *CITY, 20.0)] + [(15 * i, far[0] + 1e-4 * i, far[1], 20.0) for i in range(1, 7)]
    assert feed(fixes) == [None, JUMP, JUMP, JUMP, JUMP, None, None]


def test_spoofed_start_does_not_lock_out_the_real_track():
    # First fixes are fast and spoofed: no anchor is taken from them.
    fixes = [(0, 37.30, 55.95, 99.0), (15, 37.301, 55.95, 98.0), (30, *CITY, 15.0)]
    assert feed(fixes) == [IMPLAUSIBLE_SPEED, IMPLAUSIBLE_SPEED, None]


def test_normal_driving_and_missing_speed_are_trusted():
    fixes = [(15 * i, CITY[0] + 2e-3 * i, CITY[1], None if i == 2 else 40.0) for i in range(5)]
    assert feed(fixes) == [None] * 5
