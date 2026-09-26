"""Three independent time axes: source (data), virtual replay and wall clock.

The source axis keeps the dataset's naive nanoseconds, so features and plan joins use the
same basis as offline training. Wall clock is only used for latency, staleness of the
connection and log timestamps, never for choosing a target visit.
"""

import time
from datetime import UTC, datetime

import pandas as pd

SECOND_NS = 1_000_000_000
TIME_BASIS = "dataset_naive_ns"


def wall_now() -> datetime:
    return datetime.now(tz=UTC)


def wall_iso() -> str:
    return wall_now().isoformat(timespec="milliseconds").replace("+00:00", "Z")


def format_source(nanoseconds: int | None) -> str | None:
    """Dataset-naive timestamp string; never gets a Z suffix or an invented offset."""
    if nanoseconds is None:
        return None
    return str(pd.Timestamp(int(nanoseconds)))


def parse_source(value: str) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Expected a valid, non-null source timestamp")
    if stamp.tzinfo is not None:
        raise ValueError("Source timestamps are timezone-naive in this dataset")
    return int(stamp.value)


class SourceClock:
    """Virtual clock over the source axis.

    ``driven`` mode advances by ``speed`` per real second and is used for CSV replay.
    ``follow`` mode tracks the newest received event and then advances in real time, so a
    silent stream still ages into ``stale`` instead of freezing.
    """

    def __init__(self, mode: str = "driven", speed: float = 1.0):
        if mode not in {"driven", "follow"}:
            raise ValueError("Clock mode must be driven or follow")
        self.mode = mode
        self.speed = float(speed)
        self.paused = mode == "driven"
        self._base_source_ns: int | None = None
        self._base_monotonic = time.monotonic()
        self._observed_ns: int | None = None

    def start(self, source_ns: int, *, paused: bool = False) -> None:
        self._base_source_ns = int(source_ns)
        self._base_monotonic = time.monotonic()
        self.paused = paused

    def observe(self, source_ns: int) -> None:
        """Follow mode: the newest already-received event defines the current source time."""
        if self._observed_ns is None or source_ns > self._observed_ns:
            self._observed_ns = int(source_ns)
            if self.mode == "follow":
                self._base_source_ns = self._observed_ns
                self._base_monotonic = time.monotonic()
                self.paused = False

    def now_ns(self) -> int | None:
        if self._base_source_ns is None:
            return None
        if self.paused:
            return self._base_source_ns
        elapsed = time.monotonic() - self._base_monotonic
        speed = self.speed if self.mode == "driven" else 1.0
        return self._base_source_ns + int(elapsed * speed * SECOND_NS)

    def pause(self) -> None:
        current = self.now_ns()
        if current is not None:
            self._base_source_ns = current
        self._base_monotonic = time.monotonic()
        self.paused = True

    def resume(self) -> None:
        if self._base_source_ns is None:
            return
        self._base_monotonic = time.monotonic()
        self.paused = False

    def set_speed(self, speed: float) -> None:
        if speed <= 0 or speed > 3600:
            raise ValueError("Replay speed must be in (0, 3600]")
        current = self.now_ns()
        if current is not None:
            self._base_source_ns = current
        self._base_monotonic = time.monotonic()
        self.speed = float(speed)

    def to_dict(self) -> dict:
        return {
            "time_basis": TIME_BASIS,
            "mode": self.mode,
            "source_time": format_source(self.now_ns()),
            "wall_time": wall_iso(),
            "speed": self.speed if self.mode == "driven" else 1.0,
            "paused": self.paused,
        }
