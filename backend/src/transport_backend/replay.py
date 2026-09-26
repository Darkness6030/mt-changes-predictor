"""Event-time CSV replay: feeds only the prefix of events whose time is at or before T.

This is the official contract reproduced as a stream. Delivery anomalies present in
``receive_time`` are deliberately not simulated here; that would be a separate robustness
experiment and must not be called the official event-time evaluation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transport_ml.data import ID_COLUMNS, TRAFFIC_COLUMNS, load_points
from transport_ml.features import FeatureConfig, prepare_traffic

from transport_backend.clock import SECOND_NS
from transport_backend.events import TelemetryEvent


def load_mapping(traffic_path: Path) -> dict[str, str]:
    """unit_id -> tr_id from the only available source: the distributed telemetry file."""
    frame = pd.read_csv(traffic_path, usecols=["tr_id", "unit_id"], dtype=ID_COLUMNS)
    frame = frame.dropna().astype(str).drop_duplicates()
    conflicts = frame.groupby("unit_id").tr_id.nunique()
    if (conflicts > 1).any():
        raise ValueError("A unit_id maps to more than one tr_id; explicit mapping required")
    return dict(zip(frame.unit_id, frame.tr_id, strict=True))


@dataclass
class PredictionPoint:
    """Official prediction point: exact T, target keys and the allowed hint."""

    sample_id: str
    tr_id: str
    cutoff_ns: int
    target_stop_id: str
    target_planned_ns: int
    cur_dev_s: float | None


class ReplaySource:
    """Sorted, immutable event log plus a cursor. The bounded state lives in FleetState."""

    def __init__(
        self,
        traffic_path: Path,
        config: FeatureConfig | None = None,
        units: dict[str, str] | None = None,
    ):
        raw = pd.read_csv(traffic_path, usecols=TRAFFIC_COLUMNS, dtype=ID_COLUMNS)
        clean = prepare_traffic(raw, config or FeatureConfig())
        # Stable global event-time order; ties keep the order-independent tie-break above.
        clean = clean.sort_values(["event_time", "tr_id"], kind="stable").reset_index(drop=True)
        self.times = clean.event_time.astype("int64").to_numpy()
        self.tr_id = clean.tr_id.to_numpy(dtype=object)
        self.lon = clean.lon.to_numpy(dtype=float)
        self.lat = clean.lat.to_numpy(dtype=float)
        self.speed = clean.speed.to_numpy(dtype=float)
        self.heading = clean.heading.to_numpy(dtype=float)
        self.valid = clean.gps_valid.to_numpy(dtype=bool)
        # The CSV allowlist has no unit_id column, so the real terminal id comes from the
        # same mapping the live path uses; otherwise the UI would show tr_id twice.
        self.units = units or {}
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.times)

    @property
    def first_ns(self) -> int | None:
        return int(self.times[0]) if len(self.times) else None

    @property
    def last_ns(self) -> int | None:
        return int(self.times[-1]) if len(self.times) else None

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self.times)

    def reset(self, from_ns: int) -> None:
        self.cursor = int(np.searchsorted(self.times, from_ns, side="left"))

    def due(self, now_ns: int, limit: int = 20_000) -> list[TelemetryEvent]:
        """Every not-yet-delivered event with ``event_time <= now``, in source order."""
        end = int(np.searchsorted(self.times, now_ns, side="right"))
        end = min(end, self.cursor + limit)
        events = []
        for index in range(self.cursor, end):
            valid = bool(self.valid[index])
            lon = float(self.lon[index]) if valid and np.isfinite(self.lon[index]) else None
            lat = float(self.lat[index]) if valid and np.isfinite(self.lat[index]) else None
            speed = float(self.speed[index]) if np.isfinite(self.speed[index]) else None
            heading = float(self.heading[index]) if np.isfinite(self.heading[index]) else None
            events.append(
                TelemetryEvent(
                    source="csv_replay",
                    unit_id=self.units.get(str(self.tr_id[index]), str(self.tr_id[index])),
                    tr_id=str(self.tr_id[index]),
                    event_time_ns=int(self.times[index]),
                    gps_valid=valid and lon is not None and lat is not None,
                    lon=lon,
                    lat=lat,
                    speed_kmh=speed,
                    heading_deg=heading,
                    quality_flags=() if valid else ("invalid_gps",),
                )
            )
        self.cursor = end
        return events

    def progress(self) -> float:
        return 0.0 if not len(self.times) else self.cursor / len(self.times)


class PointSchedule:
    """Official prediction points used as exact prediction ticks in replay."""

    def __init__(self, root: Path, split: str, shift_s: float = 0.0):
        frame = load_points(root, split).sort_values("T", kind="stable").reset_index(drop=True)
        shift = int(shift_s * SECOND_NS)
        self.points = [
            PredictionPoint(
                sample_id=str(row.sample_id),
                tr_id=str(row.tr_id),
                cutoff_ns=int(row.T.value) + shift,
                target_stop_id=str(row.target_stop_id),
                target_planned_ns=int(row.target_time_begin.value) + shift,
                cur_dev_s=None if pd.isna(row.cur_dev_s) else float(row.cur_dev_s),
            )
            for row in frame.itertuples()
        ]
        self.cursor = 0
        self.skipped = 0

    def __len__(self) -> int:
        return len(self.points)

    @property
    def first_ns(self) -> int | None:
        return self.points[0].cutoff_ns if self.points else None

    def reset(self, from_ns: int) -> None:
        self.cursor = 0
        self.skipped = 0
        while self.cursor < len(self.points) and self.points[self.cursor].cutoff_ns < from_ns:
            self.cursor += 1

    def due(self, now_ns: int, grace_s: float = 60.0, limit: int = 64) -> list[PredictionPoint]:
        """Points whose T has just passed; a point older than the grace window is skipped.

        A skipped point is counted, not silently predicted later: its causal history may
        already have left the bounded window.
        """
        ready = []
        while self.cursor < len(self.points) and self.points[self.cursor].cutoff_ns <= now_ns:
            point = self.points[self.cursor]
            self.cursor += 1
            if (now_ns - point.cutoff_ns) / SECOND_NS > grace_s:
                self.skipped += 1
                continue
            ready.append(point)
            if len(ready) >= limit:
                break
        return ready
