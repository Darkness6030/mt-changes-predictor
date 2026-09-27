"""Trip structure derived from the PLANNED timetable only (schema 3).

A planned gap longer than ``LAYOVER_S`` between neighbouring visits is treated as a
terminal layover. After a layover the supplied/observed deviation of the previous trip
loses most of its value: on train+test the correlation of ``cur_dev_s`` with the target
falls from ~0.68 to ~0 when such a gap lies between T and the target. These features let
the model see that structure. They read planned times only; actual arrival times are
never available here, and future *plan* is explicitly allowed by the task rules.
"""

import numpy as np

from transport_ml.features import SECOND

LAYOVER_S = 240.0
CAP_S = 7200.0
TRIP_COLUMNS = (
    "plan_max_gap_s",
    "layover_between",
    "layover_sum_s",
    "target_after_layover_s",
    "target_trip_pos_s",
    "trip_pos_T_s",
    "target_to_trip_end_s",
    "last_visit_age_s",
    "hint_after_slack",
)
# Columns derived from the supplied hint; hint-free models must exclude them.
HINT_COLUMNS = frozenset({"cur_dev_s", "hint_after_slack"})


def trip_features(plan_ns: np.ndarray, at_ns: int, target_ns: int, hint: float) -> dict:
    """``plan_ns`` is the vehicle's sorted planned visit times in nanoseconds."""
    target = np.searchsorted(plan_ns, target_ns, side="left")
    last = np.searchsorted(plan_ns, at_ns, side="right") - 1
    gaps = np.diff(plan_ns) / SECOND  # gaps[k] = plan[k + 1] - plan[k]
    ahead = gaps[max(last, 0) : target]
    layovers = np.flatnonzero(gaps > LAYOVER_S)
    before_target = layovers[layovers < target]
    trip_start = plan_ns[before_target[-1] + 1] if len(before_target) else plan_ns[0]
    between = layovers[(layovers >= max(last, 0)) & (layovers < target)]
    started = layovers[plan_ns[layovers + 1] <= at_ns]
    current_start = plan_ns[started[-1] + 1] if len(started) else plan_ns[0]
    following = layovers[layovers >= target]
    slack = float(gaps[between].sum()) if len(between) else 0.0
    return {
        "plan_max_gap_s": min(float(ahead.max()), 3600.0) if len(ahead) else np.nan,
        "layover_between": float(len(between) > 0),
        "layover_sum_s": min(slack, 3600.0),
        "target_after_layover_s": (
            (target_ns - plan_ns[between[-1] + 1]) / SECOND if len(between) else np.nan
        ),
        "target_trip_pos_s": min((target_ns - trip_start) / SECOND, CAP_S),
        "trip_pos_T_s": float(np.clip((at_ns - current_start) / SECOND, -900.0, CAP_S)),
        "target_to_trip_end_s": (
            min((plan_ns[following[0]] - target_ns) / SECOND, CAP_S) if len(following) else np.nan
        ),
        "last_visit_age_s": (at_ns - plan_ns[last]) / SECOND if last >= 0 else np.nan,
        # A late vehicle can absorb delay during a planned layover (NaN stays NaN).
        "hint_after_slack": max(hint - max(slack - 60.0, 0.0), -600.0) if len(between) else hint,
    }
