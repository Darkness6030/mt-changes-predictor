"""Causal plausibility filter for GPS fixes that the device itself reports as valid.

The labelled day contains GNSS spoofing: fixes with ``location_valid=True`` circle about
1 km around one point of Sheremetyevo airfield at a nearly constant 97-99 km/h, entered and
left by 15-20 km jumps within seconds. No planned stop lies within ~13 km of that point.
The filter marks such fixes as suspect; it never edits coordinates and never looks ahead.

It is used for display and GPS/plan matching only. Model features keep the device flag,
because the published models were trained on it; changing that requires a new schema.
"""

from dataclasses import dataclass

from transport_ml.features import SECOND, distance_m

SPOOF_ZONE = "gps_spoof_zone"
IMPLAUSIBLE_SPEED = "gps_implausible_speed"
JUMP = "gps_jump"

# (lon, lat, radius_m): centre of the spoofed circles observed in the labelled traffic.
SHEREMETYEVO_SPOOF_ZONE = (37.4146, 55.9726, 2500.0)


@dataclass(frozen=True)
class GpsTrustConfig:
    spoof_zones: tuple[tuple[float, float, float], ...] = (SHEREMETYEVO_SPOOF_ZONE,)
    jump_min_m: float = 1000.0
    jump_speed_kmh: float = 150.0
    # A city bus is not trusted above this speed when there is no earlier anchor to compare
    # with. On its own the speed never rejects a fix: emulator speeds are not specified.
    anchor_max_speed_kmh: float = 90.0
    # Consecutive, mutually consistent fixes needed to accept a new track after a jump, so a
    # spoofed first anchor cannot lock out the real track forever.
    relock_fixes: int = 5


class GpsTrustFilter:
    """Streaming state for one vehicle; feed valid fixes in event-time order."""

    def __init__(self, config: GpsTrustConfig | None = None):
        self.config = config or GpsTrustConfig()
        self.anchor: tuple[int, float, float] | None = None
        self.candidates: list[tuple[int, float, float]] = []

    def _speed_kmh(self, first: tuple[int, float, float], time_ns, lon, lat) -> tuple:
        meters = float(distance_m(first[1], first[2], lon, lat))
        seconds = max((time_ns - first[0]) / SECOND, 1.0)
        return meters, meters / seconds * 3.6

    def update(self, time_ns: int, lon: float, lat: float, speed_kmh: float | None) -> str | None:
        """Return None for a trusted fix, otherwise the reason it is suspect."""
        config = self.config
        for zone_lon, zone_lat, radius_m in config.spoof_zones:
            if distance_m(zone_lon, zone_lat, lon, lat) <= radius_m:
                self.candidates = []
                return SPOOF_ZONE
        fast = speed_kmh is not None and speed_kmh >= config.anchor_max_speed_kmh
        if self.anchor is None:
            if fast:
                return IMPLAUSIBLE_SPEED
            self.anchor = (time_ns, lon, lat)
            return None
        meters, implied = self._speed_kmh(self.anchor, time_ns, lon, lat)
        if meters <= config.jump_min_m or implied <= config.jump_speed_kmh:
            self.anchor = (time_ns, lon, lat)
            self.candidates = []
            return None
        if fast:
            self.candidates = []
            return JUMP
        if self.candidates:
            _, step = self._speed_kmh(self.candidates[-1], time_ns, lon, lat)
            if step > config.jump_speed_kmh:
                self.candidates = []
        self.candidates.append((time_ns, lon, lat))
        if len(self.candidates) < config.relock_fixes:
            return JUMP
        self.anchor = (time_ns, lon, lat)
        self.candidates = []
        return None
