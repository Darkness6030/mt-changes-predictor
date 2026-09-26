"""Offline check of the online deviation estimator against the supplied ``cur_dev_s``.

``cur_dev_s`` is input data available at T in every split, not a label, so comparing with it
is not leakage. The parameters of the estimator were chosen on train; test is only reported.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from transport_ml.data import ID_COLUMNS, PLAN_COLUMNS, load_points
from transport_ml.features import FeatureConfig, prepare_traffic

from transport_backend.events import TelemetryEvent
from transport_backend.state import FleetState, PlanStore


def check_hint(data_root: Path, split: str, radius_m: float = 60.0) -> dict:
    points = load_points(data_root, split)
    if split == "train":
        # Synthetic vehicles have no real geometry to match against.
        points = points[points.tr_id.astype("int64") < 9_000_000].reset_index(drop=True)
    plan_name = "schedule_plan.csv" if split == "validate" else "schedule.csv"
    plan = pd.read_csv(data_root / split / plan_name, usecols=PLAN_COLUMNS, dtype=ID_COLUMNS)
    traffic = pd.read_csv(
        data_root / split / "traffic.csv",
        usecols=["tr_id", "event_time", "location_valid", "lon", "lat", "speed", "heading"],
        dtype=ID_COLUMNS,
    )
    clean = prepare_traffic(traffic, FeatureConfig())
    state = FleetState(plan=PlanStore(plan), history_window_s=10**9, history_max_events=10**7)
    for row in clean.itertuples():
        state.add(
            TelemetryEvent(
                source="csv_replay",
                unit_id=row.tr_id,
                tr_id=row.tr_id,
                event_time_ns=int(row.event_time.value),
                gps_valid=bool(row.gps_valid),
                lon=None if not row.gps_valid else float(row.lon),
                lat=None if not row.gps_valid else float(row.lat),
                speed_kmh=None if pd.isna(row.speed) else float(row.speed),
                heading_deg=None if pd.isna(row.heading) else float(row.heading),
            )
        )
    estimated, supplied = [], []
    for row in points.itertuples():
        deviation = state.estimate_deviation(row.tr_id, int(row.T.value), radius_m=radius_m)
        if deviation is None:
            continue
        estimated.append(deviation.seconds)
        supplied.append(float(row.cur_dev_s))
    estimated = np.array(estimated)
    supplied = np.array(supplied)
    if not len(estimated):
        return {"split": split, "rows": len(points), "covered": 0}
    difference = estimated - supplied
    return {
        "split": split,
        "radius_m": radius_m,
        "rows": int(len(points)),
        "covered": int(len(estimated)),
        "coverage": float(len(estimated) / len(points)),
        "mae_vs_supplied_s": float(np.mean(np.abs(difference))),
        "median_abs_s": float(np.median(np.abs(difference))),
        "p90_abs_s": float(np.quantile(np.abs(difference), 0.9)),
        "bias_s": float(np.mean(difference)),
        "note": (
            "Оценка по геометрии и порядку плана; параметры подобраны на train. "
            "Не замена подсказке из прогнозных точек в офлайн-контуре."
        ),
    }
