"""Published GPS/plan deviation, independent of ML predictions and HTTP polling."""

from dataclasses import dataclass, field

from transport_backend.clock import SECOND_NS, format_source
from transport_backend.config import RiskPolicy
from transport_backend.state import Deviation, FleetState


@dataclass
class CurrentDeviationMonitor:
    """Keep one causal estimate per vehicle; reads only format the last published estimate."""

    max_age_s: float
    risk: RiskPolicy
    estimates: dict[str, Deviation | None] = field(default_factory=dict)
    _keys: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    def clear(self) -> None:
        self.estimates.clear()
        self._keys.clear()

    def refresh(self, state: FleetState, at_ns: int) -> None:
        for tr_id, track in state.tracks.items():
            # A paused clock is not enough: late events or trimming can change the prefix.
            key = (at_ns, track.received_events, track.dropped_by_window)
            if self._keys.get(tr_id) != key:
                self.estimates[tr_id] = state.estimate_deviation(tr_id, at_ns)
                self._keys[tr_id] = key
        for tr_id in self._keys.keys() - state.tracks.keys():
            del self._keys[tr_id]
            self.estimates.pop(tr_id, None)

    def view(
        self, tr_id: str, at_ns: int, position_age_s: float | None, gps_limit_s: float
    ) -> dict:
        estimate = self.estimates.get(tr_id)
        age_s = (at_ns - estimate.arrival_ns) / SECOND_NS if estimate else None
        if estimate is None or age_s < 0:
            # Backward navigation must never expose a cached observation from the future.
            estimate, age_s = None, None
            status, reason = "unavailable", "no_match"
        elif position_age_s is None:
            status, reason = "stale", "no_valid_position"
        elif position_age_s > gps_limit_s:
            status, reason = "stale", "stale_gps"
        elif age_s > self.max_age_s:
            status, reason = "stale", "estimate_too_old"
        else:
            status, reason = "ok", None
        return {
            "status": status,
            "delay_s": estimate.seconds if estimate else None,
            "risk_level": self.risk.level(estimate.seconds) if status == "ok" else None,
            "source": "gps_plan" if estimate else None,
            "observed_at": format_source(estimate.arrival_ns) if estimate else None,
            "age_s": age_s,
            "max_age_s": self.max_age_s,
            "visit_id": estimate.visit_id if estimate else None,
            "match_distance_m": estimate.distance_m if estimate else None,
            "reason": reason,
        }
