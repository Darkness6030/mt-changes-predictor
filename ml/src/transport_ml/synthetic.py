"""Synthetic train vehicles are time-shifted noisy copies of real vehicles of the same day.

Measured on the train plan: every synthetic vehicle (tr_id >= 9 000 000) repeats the stop
geometry of one real vehicle with a constant shift, and its delays correlate ~0.99 with
that vehicle's delays. Since validate points are other moments of the same real vehicles,
an unfiltered synthetic copy is a noisy copy of hidden validate answers. We therefore map
families from PLANNED geometry/time only and drop every synthetic point whose real-time
counterpart lies near a protected (held-out or validate) point of the same family.
"""

import numpy as np
import pandas as pd

SYNTHETIC_MIN_ID = 9_000_000


def is_synthetic(tr_id: pd.Series) -> np.ndarray:
    return (tr_id.astype("int64") >= SYNTHETIC_MIN_ID).to_numpy()


def synthetic_families(plan: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Map synthetic tr_id -> (real tr_id, shift seconds) by the most frequent exact
    (stop geometry, planned-time shift) match. Reads planned columns only."""
    frame = plan[["tr_id", "time_begin", "geom"]].copy()
    frame["tr_id"] = frame.tr_id.astype(str)
    frame["time_begin"] = pd.to_datetime(frame.time_begin, format="mixed")
    synthetic = is_synthetic(frame.tr_id)
    real, copies = frame[~synthetic], frame[synthetic]
    result = {}
    for tr_id, rows in copies.groupby("tr_id"):
        matched = real.merge(rows[["geom", "time_begin"]], on="geom", suffixes=("", "_copy"))
        if matched.empty:
            raise ValueError(f"Synthetic vehicle {tr_id} has no real family")
        shift = (matched.time_begin_copy - matched.time_begin).dt.total_seconds().round()
        counts = matched.assign(shift=shift).groupby(["tr_id", "shift"]).size()
        parent, best = counts.idxmax()
        result[str(tr_id)] = (str(parent), float(best))
    return result


def keep_synthetic(
    points: pd.DataFrame,
    families: dict[str, tuple[str, float]],
    protected: pd.DataFrame,
    margin_s: float,
) -> np.ndarray:
    """Boolean mask over synthetic ``points``: False near any protected (tr_id, T)."""
    keep = np.ones(len(points), dtype=bool)
    parent = points.tr_id.map(lambda key: families[str(key)][0]).to_numpy()
    shift = points.tr_id.map(lambda key: families[str(key)][1]).to_numpy()
    real_time = points["T"].astype("int64").to_numpy() - (shift * 1e9).astype("int64")
    guarded = protected.assign(tr_id=protected.tr_id.astype(str))
    for vehicle, rows in guarded.groupby("tr_id"):
        selected = np.flatnonzero(parent == vehicle)
        if not len(selected):
            continue
        times = rows["T"].astype("int64").to_numpy()
        distance = np.abs(real_time[selected, None] - times[None, :]).min(axis=1)
        keep[selected[distance <= margin_s * 1e9]] = False
    return keep
