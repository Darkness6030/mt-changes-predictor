"""One feature implementation for batch and already-received telemetry prefixes."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from transport_ml.data import PLAN_COLUMNS, POINT_COLUMNS, TRAFFIC_COLUMNS, timestamps

SCHEMA_VERSION = "1"
SECOND = 1_000_000_000


@dataclass(frozen=True)
class FeatureConfig:
    windows_s: tuple[int, ...] = (60, 180, 300, 600)
    max_speed_kmh: float = 130.0
    stopped_speed_kmh: float = 3.0
    max_gap_s: float = 60.0
    stale_after_s: float = 120.0

    def to_dict(self) -> dict:
        return asdict(self)


def distance_m(lon1, lat1, lon2, lat2):
    """Great-circle distance, not road distance; accepts scalars or arrays."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    value = np.sin((lat2 - lat1) / 2) ** 2
    value += np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 12_742_000 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def prepare_plan(plan: pd.DataFrame) -> pd.DataFrame:
    # An already prepared plan is reused unchanged: streaming callers rebuild the builder on
    # every prediction cycle and must not pay for parsing 5 500 rows of WKT each time.
    prepared = {"lon", "lat"}.issubset(plan.columns) and plan.time_begin.dtype == "datetime64[ns]"
    if prepared:
        return plan
    plan = plan[PLAN_COLUMNS].copy()
    for key in ("tr_id", "tt_action_item_id"):
        if plan[key].isna().any():
            raise ValueError(f"Null plan key: {key}")
        plan[key] = plan[key].astype(str)
    plan["time_begin"] = timestamps(plan.time_begin)
    if plan.duplicated(["tr_id", "tt_action_item_id"]).any():
        raise ValueError("Duplicate planned visit")
    coordinates = plan.geom.str.extract(r"^POINT\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)$")
    plan[["lon", "lat"]] = coordinates.astype(float).to_numpy()
    if not (plan.lon.between(-180, 180) & plan.lat.between(-90, 90)).all():
        raise ValueError("Invalid stop geometry")
    return plan.sort_values(["tr_id", "time_begin", "tt_action_item_id"]).reset_index(drop=True)


def prepare_traffic(traffic: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    frame = traffic[TRAFFIC_COLUMNS].copy()
    if frame.tr_id.isna().any():
        raise ValueError("Null telemetry vehicle")
    frame["tr_id"] = frame.tr_id.astype(str)
    frame["event_time"] = timestamps(frame.event_time)
    frame["location_valid"] = frame.location_valid.astype(str).str.lower().eq("true")
    for key in ("lon", "lat", "speed", "heading"):
        frame[key] = pd.to_numeric(frame[key], errors="raise")
    frame["gps_valid"] = (
        frame.location_valid
        & frame.lon.between(-180, 180)
        & frame.lat.between(-90, 90)
        & ~(frame.lon.eq(0) & frame.lat.eq(0))
    )
    frame["speed"] = frame.speed.where(
        frame.gps_valid & frame.speed.between(0, config.max_speed_kmh)
    )
    frame["heading"] = frame.heading.where(frame.gps_valid & frame.heading.between(0, 360))
    # Preserve conflicting equal-time events with an arrival-order-independent tie-break.
    return (
        frame.drop_duplicates().sort_values(TRAFFIC_COLUMNS, kind="stable").reset_index(drop=True)
    )


class FeatureBuilder:
    """Never reads labels or facts. A caller may pass all events or a causal prefix."""

    def __init__(
        self, traffic: pd.DataFrame, plan: pd.DataFrame, config: FeatureConfig | None = None
    ):
        self.config = config or FeatureConfig()
        self.plan = prepare_plan(plan)
        self.visit_keys = self.plan.set_index(["tr_id", "tt_action_item_id"])
        self.visits = {
            str(key): rows.reset_index(drop=True)
            for key, rows in self.plan.groupby("tr_id", sort=False)
        }
        clean = prepare_traffic(traffic, self.config)
        self.history = {}
        for key, rows in clean.groupby("tr_id", sort=False):
            self.history[str(key)] = {
                "time": rows.event_time.astype("int64").to_numpy(),
                **{
                    name: rows[name].to_numpy()
                    for name in ("lon", "lat", "speed", "heading", "gps_valid")
                },
            }

    def target(self, tr_id: str, at: pd.Timestamp) -> pd.Series | None:
        visits = self.visits.get(str(tr_id))
        if visits is None:
            return None
        horizon = (visits.time_begin - at).dt.total_seconds()
        candidates = visits[horizon.between(600, 900, inclusive="right")]
        return None if candidates.empty else candidates.iloc[0]

    def one(self, point: dict) -> dict[str, float]:
        at = pd.Timestamp(point["T"])
        if at.tzinfo is not None or pd.isna(at):
            raise ValueError("Expected naive, non-null source time")
        first = self.target(str(point["tr_id"]), at)
        if first is None:
            raise ValueError("No planned visit in (T+600, T+900]")
        key = (str(point["tr_id"]), str(point["target_stop_id"]))
        if key not in self.visit_keys.index:
            raise ValueError("Unknown planned visit")
        visit = self.visit_keys.loc[key]
        # Several visits can share the earliest plan time: preserve the supplied target ID.
        if visit.time_begin != first.time_begin:
            raise ValueError("Target is not the first planned visit in the horizon")
        if visit.time_begin != pd.Timestamp(point["target_time_begin"]):
            raise ValueError("Target time does not match the plan")
        hint = float(point.get("cur_dev_s", np.nan))
        if np.isinf(hint):
            raise ValueError("Infinite current deviation")
        horizon = (visit.time_begin.value - at.value) / SECOND
        angle = 2 * np.pi * (at.hour * 3600 + at.minute * 60 + at.second) / 86400
        features = {
            "cur_dev_s": hint,
            "horizon_s": horizon,
            "time_sin": np.sin(angle),
            "time_cos": np.cos(angle),
            "target_lon": visit.lon,
            "target_lat": visit.lat,
            "event_age_s": np.nan,
            "gps_age_s": np.nan,
            "gps_stale": 1.0,
            "last_speed": np.nan,
            "target_distance_m": np.nan,
            "observed_stop_s": np.nan,
        }
        history = self.history.get(str(point["tr_id"]))
        # Search the exact nanosecond boundary before any aggregation or latest-value lookup.
        stop = 0 if history is None else np.searchsorted(history["time"], at.value, side="right")
        if stop:
            age = (at.value - history["time"][:stop]) / SECOND
            features["event_age_s"] = age[-1]
            valid_indices = np.flatnonzero(history["gps_valid"][:stop])
            if len(valid_indices):
                last = valid_indices[-1]
                features.update(
                    gps_age_s=age[last],
                    gps_stale=float(age[last] > self.config.stale_after_s),
                    last_speed=history["speed"][last],
                    target_distance_m=distance_m(
                        history["lon"][last], history["lat"][last], visit.lon, visit.lat
                    ),
                )
            speed = history["speed"][:stop]
            if np.isfinite(speed[-1]) and age[-1] <= self.config.max_gap_s:
                start = stop - 1
                while start > 0 and speed[start] <= self.config.stopped_speed_kmh:
                    gap = (history["time"][start] - history["time"][start - 1]) / SECOND
                    if (
                        gap > self.config.max_gap_s
                        or not np.isfinite(speed[start - 1])
                        or speed[start - 1] > self.config.stopped_speed_kmh
                    ):
                        break
                    if age[start - 1] > max(self.config.windows_s):
                        break
                    start -= 1
                features["observed_stop_s"] = (
                    age[start] - age[-1] if speed[-1] <= self.config.stopped_speed_kmh else 0.0
                )
        for window in self.config.windows_s:
            values = {
                "count": 0.0,
                "valid_fraction": np.nan,
                "speed_mean": np.nan,
                "speed_std": np.nan,
                "speed_last": np.nan,
                "stopped_fraction": np.nan,
                "span_s": 0.0,
                "distance_delta_m": np.nan,
            }
            if stop:
                start = np.searchsorted(history["time"], at.value - window * SECOND, side="right")
                speeds = history["speed"][start:stop]
                finite = speeds[np.isfinite(speeds)]
                count = stop - start
                values["count"] = float(count)
                if count:
                    valid = history["gps_valid"][start:stop]
                    values["valid_fraction"] = valid.mean()
                    values["span_s"] = (history["time"][stop - 1] - history["time"][start]) / SECOND
                    indices = np.flatnonzero(valid) + start
                    if len(indices) >= 2:
                        distances = distance_m(
                            history["lon"][indices[[0, -1]]],
                            history["lat"][indices[[0, -1]]],
                            visit.lon,
                            visit.lat,
                        )
                        values["distance_delta_m"] = distances[0] - distances[1]
                if len(finite):
                    values.update(
                        speed_mean=finite.mean(),
                        speed_std=finite.std(),
                        speed_last=finite[-1],
                        stopped_fraction=(finite <= self.config.stopped_speed_kmh).mean(),
                    )
            features.update({f"{key}_{window}s": value for key, value in values.items()})
        return features

    def transform(self, points: pd.DataFrame) -> pd.DataFrame:
        # Explicit allowlist makes target/fact columns inert, even for in-memory callers.
        rows = [self.one(row) for row in points[POINT_COLUMNS].to_dict("records")]
        return pd.DataFrame(rows, index=points.index, dtype=float)
