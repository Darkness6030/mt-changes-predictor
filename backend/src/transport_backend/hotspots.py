"""Where delay accumulates: gains between consecutive GPS-observed stop passages.

Each time a vehicle's causal GPS/plan estimate moves to a later planned visit, the change in
deviation is attributed to the segment between the two visits. Summed over the run, this
shows the stretches where time is lost, from received telemetry only (no labels), so it
works on a live NDTP stream. Segments are schematic (planned visit to planned visit).
"""

from dataclasses import dataclass, field

from transport_backend.clock import SECOND_NS, format_source
from transport_backend.state import Deviation, PlanStore

MAX_VISIT_STEP = 3  # Farther jumps skip unobserved stops; the gain is not attributable.
MAX_ARRIVAL_GAP_S = 1800.0
MAX_SEGMENTS = 500


@dataclass
class DelayHotspots:
    last: dict[str, tuple[int, Deviation]] = field(default_factory=dict)
    segments: dict[str, dict] = field(default_factory=dict)

    def clear(self) -> None:
        self.last.clear()
        self.segments.clear()

    def observe(self, tr_id: str, estimate: Deviation | None, plan: PlanStore) -> None:
        if estimate is None:
            return
        visits = plan.visits.get(tr_id)
        if visits is None:
            return
        ids = visits.tt_action_item_id.astype(str)
        matches = ids.index[ids.eq(estimate.visit_id)]
        if not len(matches):
            return
        index = int(matches[0])
        previous = self.last.get(tr_id)
        self.last[tr_id] = (index, estimate)
        if previous is None:
            return
        before_index, before = previous
        step = index - before_index
        gap_s = (estimate.arrival_ns - before.arrival_ns) / SECOND_NS
        if not 0 < step <= MAX_VISIT_STEP or not 0 < gap_s <= MAX_ARRIVAL_GAP_S:
            return
        gain_s = estimate.seconds - before.seconds
        start, end = visits.iloc[before_index], visits.iloc[index]
        key = f"{tr_id}:{before.visit_id}:{estimate.visit_id}"
        item = self.segments.get(key)
        if item is None:
            item = {
                "segment_id": key,
                "tr_id": tr_id,
                "from": {
                    "target_stop_id": before.visit_id,
                    "address": plan.address(before.visit_id),
                    "lon": float(start.lon),
                    "lat": float(start.lat),
                },
                "to": {
                    "target_stop_id": estimate.visit_id,
                    "address": plan.address(estimate.visit_id),
                    "lon": float(end.lon),
                    "lat": float(end.lat),
                },
                "passes": 0,
                "gain_total_s": 0.0,
                "gain_max_s": float("-inf"),
            }
            self.segments[key] = item
        item["passes"] += 1
        item["gain_total_s"] += gain_s
        item["gain_max_s"] = max(item["gain_max_s"], gain_s)
        item["last_at"] = format_source(estimate.arrival_ns)
        if len(self.segments) > MAX_SEGMENTS:
            smallest = min(self.segments.values(), key=lambda row: row["gain_total_s"])
            del self.segments[smallest["segment_id"]]

    def report(self, limit: int = 5) -> list[dict]:
        """Segments where delay grew the most over the run, largest total first."""
        growing = [row for row in self.segments.values() if row["gain_total_s"] > 0]
        growing.sort(key=lambda row: row["gain_total_s"], reverse=True)
        return [
            {**row, "gain_mean_s": row["gain_total_s"] / row["passes"]} for row in growing[:limit]
        ]
