"""Bounded vehicle state, plan store and causal estimation of the current deviation.

Memory is bounded twice: by a time window and by a per-vehicle event cap. Dropping events
sets a quality flag instead of silently shortening feature windows.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from transport_ml.data import PLAN_COLUMNS, TRAFFIC_COLUMNS
from transport_ml.features import distance_m, prepare_plan
from transport_ml.gps_trust import GpsTrustFilter

from transport_backend.clock import SECOND_NS, format_source
from transport_backend.events import TelemetryEvent


@dataclass(frozen=True)
class Deviation:
    """Causally estimated deviation at the last planned visit proven to be reached."""

    seconds: float
    arrival_ns: int
    visit_id: str
    distance_m: float
    matched_visits: int

    def to_dict(self, at_ns: int) -> dict:
        return {
            "cur_dev_s": self.seconds,
            "cur_dev_source": "estimated",
            "cur_dev_age_s": (at_ns - self.arrival_ns) / SECOND_NS,
            "cur_dev_visit_id": self.visit_id,
            "cur_dev_match_distance_m": self.distance_m,
            "cur_dev_matched_visits": self.matched_visits,
        }


TRIP_BREAK_S = 1800.0
MIN_TRIP_VISITS = 3
TURNAROUND_M = 60.0
TURNAROUND_LOOKBACK = 3


def plan_trips(visits: pd.DataFrame) -> list[dict | None]:
    """Split one vehicle's ordered plan into trips; one entry per visit.

    The dataset has no trip or route ids. A terminal shows up as a return to the same point
    (within 60 m) one to three visits later: arrival and departure after a layover, possibly
    via a turning loop. A new trip starts at that departure; a break longer than 30 minutes
    (lunch, depot) also ends a trip. Fragments shorter than three visits are not
    counted as trips. In this dataset every real vehicle serves its own line, so a vehicle's
    first and last trip are also the line's first and last trip of the day.
    """
    if visits.empty:
        return []
    times = visits.time_begin.astype("int64").to_numpy()
    lon, lat = visits.lon.to_numpy(), visits.lat.to_numpy()
    starts = [0]
    for index in range(1, len(visits)):
        gap_s = (times[index] - times[index - 1]) / SECOND_NS
        back = range(max(starts[-1], index - TURNAROUND_LOOKBACK), index)
        returned = any(
            distance_m(lon[prior], lat[prior], lon[index], lat[index]) <= TURNAROUND_M
            for prior in back
        )
        if returned or gap_s > TRIP_BREAK_S:
            starts.append(index)
    bounds = [
        (start, end)
        for start, end in zip(starts, [*starts[1:], len(visits)], strict=True)
        if end - start >= MIN_TRIP_VISITS
    ]
    result: list[dict | None] = [None] * len(visits)
    for number, (start, end) in enumerate(bounds, start=1):
        info = {
            "number": number,
            "total": len(bounds),
            "first": number == 1,
            "last": number == len(bounds),
            "start_at": format_source(int(times[start])),
            "end_at": format_source(int(times[end - 1])),
        }
        for index in range(start, end):
            result[index] = info
    return result


class PlanStore:
    """Plan-only schedule: the fact columns are never read, not even to be ignored later."""

    def __init__(self, plan: pd.DataFrame, shift_s: float = 0.0):
        self.shift_s = float(shift_s)
        prepared = prepare_plan(plan)
        if self.shift_s:
            prepared = prepared.assign(
                time_begin=prepared.time_begin + pd.Timedelta(seconds=self.shift_s)
            )
        self.plan = prepared
        self.addresses = {}
        if "building_address" in plan.columns:
            frame = plan.dropna(subset=["building_address"])
            self.addresses = dict(
                zip(frame.tt_action_item_id.astype(str), frame.building_address, strict=True)
            )
        self.visits = {
            str(key): rows.reset_index(drop=True)
            for key, rows in self.plan.groupby("tr_id", sort=False)
        }
        self.trips: dict[str, dict] = {}
        for rows in self.visits.values():
            for visit_id, info in zip(
                rows.tt_action_item_id.astype(str), plan_trips(rows), strict=True
            ):
                if info is not None:
                    self.trips[visit_id] = info

    @classmethod
    def from_csv(cls, path: Path, shift_s: float = 0.0) -> "PlanStore":
        columns = [*PLAN_COLUMNS, "building_address"]
        frame = pd.read_csv(path, dtype={"tr_id": str, "tt_action_item_id": str})
        missing = set(PLAN_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Plan file lacks required columns: {sorted(missing)}")
        return cls(frame[[name for name in columns if name in frame.columns]], shift_s)

    @property
    def vehicles(self) -> list[str]:
        return sorted(self.visits)

    def address(self, visit_id: str) -> str | None:
        return self.addresses.get(str(visit_id))

    def trip(self, visit_id: str | None) -> dict | None:
        """Trip of a planned visit: number, total and whether it is the day's first or last."""
        return None if visit_id is None else self.trips.get(str(visit_id))

    def target(self, tr_id: str, at_ns: int) -> pd.Series | None:
        """First planned visit in ``(T+600, T+900]``; the window is a plan rule, not a guess."""
        visits = self.visits.get(str(tr_id))
        if visits is None:
            return None
        horizon = (visits.time_begin.astype("int64").to_numpy() - at_ns) / SECOND_NS
        candidates = np.flatnonzero((horizon > 600) & (horizon <= 900))
        return None if not len(candidates) else visits.iloc[candidates[0]]

    def segment(self, tr_id: str, target_stop_id: str | None) -> dict | None:
        """Schematic approach to a target visit, never a claimed road geometry."""
        visits = self.visits.get(str(tr_id))
        if visits is None or target_stop_id is None:
            return None
        matches = np.flatnonzero(visits.tt_action_item_id.eq(target_stop_id).to_numpy())
        if not len(matches) or matches[0] == 0:
            return None
        index = int(matches[0])
        previous, target = visits.iloc[index - 1], visits.iloc[index]
        gap = (target.time_begin - previous.time_begin).total_seconds()
        # Long breaks and simultaneous visits do not define an unambiguous segment.
        if not 0 < gap <= 1800:
            return None

        def endpoint(visit):
            return {
                "target_stop_id": str(visit.tt_action_item_id),
                "address": self.address(str(visit.tt_action_item_id)),
                "planned_at": format_source(int(visit.time_begin.value)),
                "lon": float(visit.lon),
                "lat": float(visit.lat),
            }

        return {
            "segment_id": f"{tr_id}:{previous.tt_action_item_id}:{target.tt_action_item_id}",
            "kind": "planned_visit_schematic",
            "from": endpoint(previous),
            "to": endpoint(target),
        }

    def next_visit(self, tr_id: str, at_ns: int) -> pd.Series | None:
        visits = self.visits.get(str(tr_id))
        if visits is None:
            return None
        future = np.flatnonzero(visits.time_begin.astype("int64").to_numpy() > at_ns)
        return None if not len(future) else visits.iloc[future[0]]

    def window(self, tr_id: str, at_ns: int, before_s: float, after_s: float) -> pd.DataFrame:
        visits = self.visits.get(str(tr_id))
        if visits is None:
            return pd.DataFrame(columns=self.plan.columns)
        times = visits.time_begin.astype("int64").to_numpy()
        mask = (times >= at_ns - before_s * SECOND_NS) & (times <= at_ns + after_s * SECOND_NS)
        return visits.loc[mask]


class VehicleTrack:
    """Bounded per-vehicle history with deduplication by full normalized event identity."""

    def __init__(self, tr_id: str | None, unit_id: str, max_events: int):
        self.tr_id = tr_id
        self.unit_id = unit_id
        self.events: deque[TelemetryEvent] = deque(maxlen=max_events)
        self.fingerprints: set[tuple] = set()
        self.duplicates = 0
        self.conflicting_times = 0
        self.out_of_order = 0
        self.dropped_by_window = 0
        self.truncated = False
        self.last_event_ns: int | None = None
        self.last_valid_ns: int | None = None
        self.last_valid: TelemetryEvent | None = None
        # Device-valid fixes that failed the plausibility filter (spoofing, jumps), keyed by
        # fingerprint. They stay in ``events`` for the model and are hidden from display.
        self.gps_filter = GpsTrustFilter()
        self.suspect: dict[tuple, str] = {}
        self.suspect_fixes = 0
        self.last_trusted: TelemetryEvent | None = None
        self.last_suspect_reason: str | None = None
        self.received_events = 0
        # Monotonic wall clock of the newest accepted event: used only to measure the real
        # delay between ingestion and a published prediction.
        self.last_received_monotonic: float | None = None

    def add(self, event: TelemetryEvent) -> bool:
        fingerprint = event.fingerprint()
        if fingerprint in self.fingerprints:
            self.duplicates += 1
            return False
        if len(self.events) == self.events.maxlen:
            self.truncated = True
        if self.last_event_ns is not None:
            if event.event_time_ns < self.last_event_ns:
                self.out_of_order += 1
            elif event.event_time_ns == self.last_event_ns:
                self.conflicting_times += 1
        # The normal ordered stream stays O(1); only late/conflicting packets need sorting.
        key = (event.event_time_ns, repr(fingerprint))
        last_key = (
            (self.events[-1].event_time_ns, repr(self.events[-1].fingerprint()))
            if self.events
            else None
        )
        if last_key is None or key >= last_key:
            if len(self.events) == self.events.maxlen:
                evicted = self.events.popleft().fingerprint()
                self.fingerprints.discard(evicted)
                self.suspect.pop(evicted, None)
            self.events.append(event)
            self.fingerprints.add(fingerprint)
        else:
            ordered = sorted(
                [*self.events, event],
                key=lambda item: (item.event_time_ns, repr(item.fingerprint())),
            )
            self.events = deque(ordered[-self.events.maxlen :], maxlen=self.events.maxlen)
            self.fingerprints = {item.fingerprint() for item in self.events}
            self.suspect = {
                key: value for key, value in self.suspect.items() if key in self.fingerprints
            }
        self.received_events += 1
        self.last_received_monotonic = perf_counter()
        self.last_event_ns = max(self.last_event_ns or event.event_time_ns, event.event_time_ns)
        if event.gps_valid and (
            self.last_valid is None
            or key >= (self.last_valid.event_time_ns, repr(self.last_valid.fingerprint()))
        ):
            self.last_valid_ns = event.event_time_ns
            self.last_valid = event
            # Only the newest fix advances the filter; a late packet is not re-judged and counts
            # as device-valid. The model input does not depend on this verdict.
            reason = self.gps_filter.update(
                event.event_time_ns, event.lon, event.lat, event.speed_kmh
            )
            self.last_suspect_reason = reason
            if reason is None:
                self.last_trusted = event
            else:
                self.suspect[fingerprint] = reason
                self.suspect_fixes += 1
        return True

    def trusted(self, event: TelemetryEvent) -> bool:
        """A device-valid fix that also passed the plausibility filter."""
        return event.gps_valid and event.fingerprint() not in self.suspect

    def trim(self, before_ns: int) -> None:
        while self.events and self.events[0].event_time_ns < before_ns:
            removed = self.events.popleft()
            self.fingerprints.discard(removed.fingerprint())
            self.suspect.pop(removed.fingerprint(), None)
            self.dropped_by_window += 1
        if self.last_valid is not None and self.last_valid.event_time_ns < before_ns:
            # The last known position is kept for display, but its real age is reported.
            self.truncated = True

    def frame(self) -> pd.DataFrame:
        """Rows in the dataset's telemetry schema, so the shared FeatureBuilder can read them."""
        return pd.DataFrame(
            {
                "tr_id": [self.tr_id] * len(self.events),
                "event_time": pd.to_datetime(
                    [event.event_time_ns for event in self.events], unit="ns"
                ),
                "location_valid": [event.gps_valid for event in self.events],
                "lon": [event.lon for event in self.events],
                "lat": [event.lat for event in self.events],
                "speed": [event.speed_kmh for event in self.events],
                "heading": [event.heading_deg for event in self.events],
            },
            columns=TRAFFIC_COLUMNS,
        )

    def valid_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times, lon, lat = [], [], []
        for event in self.events:
            if self.trusted(event) and event.lon is not None and event.lat is not None:
                times.append(event.event_time_ns)
                lon.append(event.lon)
                lat.append(event.lat)
        return np.array(times, dtype="int64"), np.array(lon, float), np.array(lat, float)

    def quality_flags(self, at_ns: int, stale_after_s: float) -> list[str]:
        flags = []
        if self.last_valid_ns is None:
            flags.append("no_valid_position")
        elif (at_ns - self.last_valid_ns) / SECOND_NS > stale_after_s:
            flags.append("stale_gps")
        if self.events and not all(event.gps_valid for event in self.events):
            flags.append("invalid_gps")
        if self.last_suspect_reason is not None:
            flags.append("gps_spoofing_suspected")
        if len(self.events) < 5:
            flags.append("sparse_history")
        if self.truncated:
            flags.append("history_truncated")
        if self.conflicting_times:
            flags.append("conflicting_event_times")
        return flags

    def invalid_fraction(self) -> float | None:
        if not self.events:
            return None
        return float(np.mean([not event.gps_valid for event in self.events]))


@dataclass
class FleetState:
    """Owner of ingest state: one instance per run, mutated from the ingest task only."""

    plan: PlanStore
    history_window_s: float = 1800.0
    history_max_events: int = 900
    stale_after_s: float = 120.0
    mapping: dict[str, str] = field(default_factory=dict)
    tracks: dict[str, VehicleTrack] = field(default_factory=dict)
    unmapped: dict[str, int] = field(default_factory=dict)
    accepted_events: int = 0
    rejected_future: int = 0
    duplicate_events: int = 0

    def key(self, event: TelemetryEvent) -> str | None:
        tr_id = event.tr_id or self.mapping.get(event.unit_id)
        if tr_id is None:
            if event.unit_id not in self.unmapped and len(self.unmapped) >= 256:
                self.unmapped.pop(next(iter(self.unmapped)))
            self.unmapped[event.unit_id] = self.unmapped.get(event.unit_id, 0) + 1
            return None
        return tr_id

    def add(self, event: TelemetryEvent, now_ns: int | None = None) -> bool:
        """Accept an already received event; a far-future timestamp is quarantined."""
        tr_id = self.key(event)
        if tr_id is None:
            return False
        if now_ns is not None and event.event_time_ns > now_ns + 60 * SECOND_NS:
            self.rejected_future += 1
            return False
        track = self.tracks.get(tr_id)
        if track is None:
            track = VehicleTrack(tr_id, event.unit_id, self.history_max_events)
            self.tracks[tr_id] = track
        if not track.add(event):
            self.duplicate_events += 1
            return False
        self.accepted_events += 1
        return True

    def trim(self, now_ns: int) -> None:
        before = now_ns - int(self.history_window_s * SECOND_NS)
        for track in self.tracks.values():
            track.trim(before)

    def history_frame(self, tr_ids: list[str]) -> pd.DataFrame:
        frames = [self.tracks[tr_id].frame() for tr_id in tr_ids if tr_id in self.tracks]
        frames = [frame for frame in frames if len(frame)]
        if not frames:
            return pd.DataFrame(columns=TRAFFIC_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def estimate_deviation(
        self,
        tr_id: str,
        at_ns: int,
        radius_m: float = 60.0,
        early_s: float = 300.0,
        late_s: float = 900.0,
        lookback_s: float = 1800.0,
    ) -> Deviation | None:
        """Causally match valid GPS to planned visits and return the last proven arrival.

        Each visit is matched inside its own time window around the planned time, so a later
        pass near the same stop cannot be attributed to an earlier visit. The reported moment
        is the middle of the dwell inside the radius: on the labelled data that is unbiased
        against the supplied ``cur_dev_s``, while the first point inside the radius is early.
        Doors are not in the dataset, so this is geometry plus plan order, never a fact read.
        Parameters were chosen on train only (see ``transport-backend check-hint``).
        """
        track = self.tracks.get(tr_id)
        visits = self.plan.visits.get(tr_id)
        if track is None or visits is None:
            return None
        times, lon, lat = track.valid_arrays()
        usable = times <= at_ns
        times, lon, lat = times[usable], lon[usable], lat[usable]
        if len(times) < 2:
            return None
        planned = visits.time_begin.astype("int64").to_numpy()
        candidates = np.flatnonzero(
            (planned <= at_ns) & (planned >= at_ns - lookback_s * SECOND_NS)
        )
        best: Deviation | None = None
        matched = 0
        for index in candidates:
            low = planned[index] - int(early_s * SECOND_NS)
            high = min(at_ns, planned[index] + int(late_s * SECOND_NS))
            window = np.flatnonzero((times >= low) & (times <= high))
            if not len(window):
                continue
            distances = distance_m(
                lon[window],
                lat[window],
                float(visits.lon.iloc[index]),
                float(visits.lat.iloc[index]),
            )
            near = np.flatnonzero(distances <= radius_m)
            if not len(near):
                continue
            matched += 1
            chosen = window[near[len(near) // 2]]
            best = Deviation(
                seconds=(times[chosen] - planned[index]) / SECOND_NS,
                arrival_ns=int(times[chosen]),
                visit_id=str(visits.tt_action_item_id.iloc[index]),
                distance_m=float(distances[near[len(near) // 2]]),
                matched_visits=matched,
            )
        return best

    def summary(self, at_ns: int | None) -> dict:
        return {
            "vehicles": len(self.tracks),
            "events_in_state": sum(len(track.events) for track in self.tracks.values()),
            "accepted_events": self.accepted_events,
            "duplicate_events": self.duplicate_events,
            "suspect_gps_fixes": sum(track.suspect_fixes for track in self.tracks.values()),
            "rejected_future_events": self.rejected_future,
            "unmapped_units": len(self.unmapped),
            "last_event_at": format_source(
                max(
                    (t.last_event_ns for t in self.tracks.values() if t.last_event_ns is not None),
                    default=None,
                )
            ),
            "source_time": format_source(at_ns),
        }
