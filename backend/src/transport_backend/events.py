"""One normalized telemetry event for both CSV replay and live NDTP."""

from dataclasses import dataclass, field

from transport_ml.features import FeatureConfig

from transport_backend.clock import SECOND_NS, TIME_BASIS
from transport_backend.ndtp import Nav00

SOURCES = ("csv_replay", "ndtp_live", "ndtp_replay")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Immutable, already cleaned of transport details but not of data-quality facts."""

    source: str
    unit_id: str
    event_time_ns: int
    gps_valid: bool
    lon: float | None = None
    lat: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = None
    altitude_m: float | None = None
    tr_id: str | None = None
    packet_id: str | None = None
    time_basis: str = TIME_BASIS
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def fingerprint(self) -> tuple:
        """Identity for deduplication: equal (tr_id, time) with different fields is kept."""
        return (
            self.unit_id,
            self.event_time_ns,
            self.lon,
            self.lat,
            self.speed_kmh,
            self.heading_deg,
            self.gps_valid,
        )


def from_nav00(
    nav: Nav00, unit_id: int, *, source: str, time_offset_s: float = 0.0
) -> TelemetryEvent:
    """Map a decoded Nav00 cell onto the dataset's time basis.

    ``time_offset_s`` is the declared, reversible mapping between NDTP Unix seconds and the
    dataset's naive axis. The default of 0 treats the naive axis as UTC; the assumption is
    documented in docs/API_CONTRACT.md and is not an organiser-confirmed timezone.
    """
    flags: list[str] = []
    if not nav.gps_valid:
        flags.append("invalid_gps")
    # A synthetic zero coordinate can still arrive with valid=true, so position is checked too.
    zero_position = nav.lon == 0.0 and nav.lat == 0.0
    if zero_position:
        flags.append("zero_position")
    in_range = -180 <= nav.lon <= 180 and -90 <= nav.lat <= 90
    if not in_range:
        flags.append("coordinate_out_of_range")
    usable = nav.gps_valid and not zero_position and in_range
    speed_valid = 0 <= nav.speed_kmh <= FeatureConfig().max_speed_kmh
    heading_valid = 0 <= nav.heading_deg <= 360
    if not speed_valid:
        flags.append("invalid_speed")
    if not heading_valid:
        flags.append("invalid_heading")
    return TelemetryEvent(
        source=source,
        unit_id=str(unit_id),
        event_time_ns=int(round((nav.timestamp + time_offset_s) * SECOND_NS)),
        gps_valid=usable,
        lon=nav.lon if usable else None,
        lat=nav.lat if usable else None,
        speed_kmh=nav.speed_kmh if usable and speed_valid else None,
        heading_deg=nav.heading_deg if usable and heading_valid else None,
        altitude_m=nav.altitude_m if usable else None,
        quality_flags=tuple(flags),
    )
