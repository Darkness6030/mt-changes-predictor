"""Causal geometry relative to the timetable, without actual arrival times or IDs.

Straight segments are schematic, not road geometry. Their matching error is a feature,
and an ambiguous/far match is never treated as an observed arrival. All history is
bounded to 30 minutes, matching the backend's default retention window.
"""

import numpy as np

from transport_ml.features import SECOND, distance_m

CONTEXT_COLUMNS = (
    "plan_distance_now_m",
    "plan_remaining_m",
    "plan_stops_remaining",
    "plan_required_speed_kmh",
    "target_approach_m",
    "target_approach_s",
    "match_distance_m",
    "match_deviation_s",
    "match_heading_cos",
    "match_segment_s",
    "arrival_deviation_s",
    "arrival_age_s",
    "arrival_distance_m",
    "arrival_matches",
    "arrival_previous_deviation_s",
    "arrival_change_s",
    "arrival_median_deviation_s",
    "gps_path_300s_m",
    "gps_path_600s_m",
    "gps_speed_300s_kmh",
    "gps_speed_600s_kmh",
    "weighted_speed_300s",
    "weighted_speed_600s",
    "weighted_stopped_300s",
    "weighted_stopped_600s",
    "time_coverage_300s",
    "time_coverage_600s",
    "target_required_speed_kmh",
    "target_heading_cos",
)


class ScheduleContext:
    def __init__(self, visits, history, config):
        self.history = history
        self.config = config
        self.plans = {}
        for key, rows in visits.items():
            self.plans[key] = {
                "time": rows.time_begin.astype("int64").to_numpy(),
                "lon": rows.lon.to_numpy(),
                "lat": rows.lat.to_numpy(),
            }

    def one(self, tr_id, at_ns, visit, basic):
        result = dict.fromkeys(CONTEXT_COLUMNS, np.nan)
        result["arrival_matches"] = 0.0
        plan = self.plans[tr_id]
        pt = plan["time"]
        remaining = np.flatnonzero((pt >= at_ns) & (pt <= visit.time_begin.value))
        result["plan_stops_remaining"] = float(len(remaining))
        target_index = np.searchsorted(pt, visit.time_begin.value, side="left")
        if target_index > 0:
            gap = (visit.time_begin.value - pt[target_index - 1]) / SECOND
            if 0 < gap <= 1800:
                result["target_approach_s"] = gap
                result["target_approach_m"] = float(
                    distance_m(
                        plan["lon"][target_index - 1],
                        plan["lat"][target_index - 1],
                        visit.lon,
                        visit.lat,
                    )
                )
        history = self.history.get(tr_id)
        if history is None:
            return result
        stop = np.searchsorted(history["time"], at_ns, side="right")
        start = np.searchsorted(history["time"], at_ns - 1800 * SECOND, side="left")
        valid = np.flatnonzero(history["gps_valid"][start:stop]) + start
        if not len(valid):
            return result
        last = valid[-1]
        if (at_ns - history["time"][last]) / SECOND > self.config.stale_after_s:
            return result
        lon, lat = history["lon"][last], history["lat"][last]
        self._arrivals(result, plan, history, valid, at_ns)
        self._segments(
            result, plan, lon, lat, history["heading"][last], history["time"][last], at_ns
        )
        # Plan position at T is an interpolation of future PLAN, never future telemetry.
        right = np.searchsorted(pt, at_ns, side="right")
        if 0 < right < len(pt) and 0 < pt[right] - pt[right - 1] <= 1800 * SECOND:
            fraction = (at_ns - pt[right - 1]) / (pt[right] - pt[right - 1])
            x = plan["lon"][right - 1] * (1 - fraction) + plan["lon"][right] * fraction
            y = plan["lat"][right - 1] * (1 - fraction) + plan["lat"][right] * fraction
            result["plan_distance_now_m"] = float(distance_m(lon, lat, x, y))
            if len(remaining):
                xs = np.r_[x, plan["lon"][remaining], visit.lon]
                ys = np.r_[y, plan["lat"][remaining], visit.lat]
                length = float(distance_m(xs[:-1], ys[:-1], xs[1:], ys[1:]).sum())
                result["plan_remaining_m"] = length
                result["plan_required_speed_kmh"] = length / basic["horizon_s"] * 3.6
        result["target_required_speed_kmh"] = basic["target_distance_m"] / basic["horizon_s"] * 3.6
        dx = (visit.lon - lon) * np.cos(np.radians(lat))
        dy = visit.lat - lat
        heading = history["heading"][last]
        if np.isfinite(heading) and dx * dx + dy * dy > 1e-14:
            bearing = np.arctan2(dx, dy)
            result["target_heading_cos"] = np.cos(np.radians(heading) - bearing)
        for window in (300, 600):
            self._movement(result, history, stop, at_ns, window)
        return result

    @staticmethod
    def _arrivals(result, plan, history, valid, at_ns):
        # Same physical radius/time admissibility as the existing backend estimator;
        # unlike supplied cur_dev, these noisy estimates stay separately identifiable.
        times = history["time"][valid]
        candidates = np.flatnonzero(
            (plan["time"] <= at_ns) & (plan["time"] >= at_ns - 1500 * SECOND)
        )
        arrivals = []
        for index in candidates:
            planned = plan["time"][index]
            window = np.flatnonzero(
                (times >= planned - 300 * SECOND) & (times <= planned + 900 * SECOND)
            )
            distances = distance_m(
                history["lon"][valid[window]],
                history["lat"][valid[window]],
                plan["lon"][index],
                plan["lat"][index],
            )
            near = np.flatnonzero(distances <= 60)
            if len(near):
                mid = near[len(near) // 2]
                arrival = times[window[mid]]
                arrivals.append(
                    (
                        (arrival - planned) / SECOND,
                        (at_ns - arrival) / SECOND,
                        float(distances[mid]),
                    )
                )
        result["arrival_matches"] = float(len(arrivals))
        if arrivals:
            result["arrival_deviation_s"], result["arrival_age_s"], result["arrival_distance_m"] = (
                arrivals[-1]
            )
            result["arrival_median_deviation_s"] = float(np.median([a[0] for a in arrivals[-3:]]))
        if len(arrivals) > 1:
            result["arrival_previous_deviation_s"] = arrivals[-2][0]
            result["arrival_change_s"] = arrivals[-1][0] - arrivals[-2][0]

    @staticmethod
    def _segments(result, plan, lon, lat, heading, gps_ns, at_ns):
        dt = np.diff(plan["time"]) / SECOND
        indices = np.flatnonzero(
            (dt > 0)
            & (dt <= 1800)
            & (plan["time"][:-1] >= at_ns - 1800 * SECOND)
            & (plan["time"][1:] <= at_ns + 600 * SECOND)
        )
        if not len(indices):
            return
        scale_x = 111_195 * np.cos(np.radians(lat))
        ax = (plan["lon"][indices] - lon) * scale_x
        ay = (plan["lat"][indices] - lat) * 111_195
        bx = (plan["lon"][indices + 1] - lon) * scale_x
        by = (plan["lat"][indices + 1] - lat) * 111_195
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        fraction = np.clip(-(ax * dx + ay * dy) / np.maximum(length2, 1), 0, 1)
        distances = np.hypot(ax + fraction * dx, ay + fraction * dy)
        implied = plan["time"][indices] + fraction * dt[indices] * SECOND
        # Time only breaks geometric ties; no supplied deviation participates in this match.
        best = np.lexsort((np.abs(implied - gps_ns), distances))[0]
        result["match_distance_m"] = float(distances[best])
        result["match_segment_s"] = float(dt[indices[best]])
        if distances[best] <= 200 and length2[best] >= 25:
            result["match_deviation_s"] = (gps_ns - implied[best]) / SECOND
            if np.isfinite(heading):
                result["match_heading_cos"] = float(
                    np.cos(np.radians(heading) - np.arctan2(dx[best], dy[best]))
                )

    def _movement(self, result, history, stop, at_ns, window):
        times = history["time"]
        start = max(0, np.searchsorted(times, at_ns - window * SECOND, side="right") - 1)
        indices = np.arange(start, stop - 1)
        if not len(indices):
            return
        durations = (
            times[indices + 1] - np.maximum(times[indices], at_ns - window * SECOND)
        ) / SECOND
        gaps = (times[indices + 1] - times[indices]) / SECOND
        usable = (durations > 0) & (gaps <= self.config.max_gap_s)
        gps = usable & history["gps_valid"][indices] & history["gps_valid"][indices + 1]
        if gps.any():
            chosen = indices[gps]
            distances = distance_m(
                history["lon"][chosen],
                history["lat"][chosen],
                history["lon"][chosen + 1],
                history["lat"][chosen + 1],
            )
            good = distances / gaps[gps] * 3.6 <= self.config.max_speed_kmh
            duration = durations[gps][good].sum()
            if duration > 0:
                length = (distances[good] * durations[gps][good] / gaps[gps][good]).sum()
                result[f"gps_path_{window}s_m"] = float(length)
                result[f"gps_speed_{window}s_kmh"] = float(length / duration * 3.6)
        speeds = history["speed"][indices]
        available = usable & np.isfinite(speeds)
        duration = durations[available].sum()
        result[f"time_coverage_{window}s"] = float(duration / window)
        if duration > 0:
            weights = durations[available] / duration
            result[f"weighted_speed_{window}s"] = float(weights @ speeds[available])
            result[f"weighted_stopped_{window}s"] = float(
                weights @ (speeds[available] <= self.config.stopped_speed_kmh)
            )
