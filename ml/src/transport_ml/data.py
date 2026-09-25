"""Read only explicitly permitted inputs; labels never enter the feature builder."""

from pathlib import Path

import numpy as np
import pandas as pd

POINT_COLUMNS = ["sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s"]
PLAN_COLUMNS = ["tr_id", "tt_action_item_id", "time_begin", "geom"]
TRAFFIC_COLUMNS = ["tr_id", "event_time", "location_valid", "lon", "lat", "speed", "heading"]
ID_COLUMNS = {key: str for key in ("sample_id", "tr_id", "target_stop_id", "tt_action_item_id")}


def timestamps(values: pd.Series) -> pd.Series:
    """Keep the dataset's naive time basis and nanoseconds, without inventing a timezone."""
    result = pd.to_datetime(values, format="mixed", errors="raise")
    if result.dt.tz is not None or result.isna().any():
        raise ValueError("Expected non-null, timezone-naive dataset timestamps")
    return result.astype("datetime64[ns]")


def point_path(root: Path, split: str) -> Path:
    if split not in {"train", "test", "validate"}:
        raise ValueError(f"Unknown split: {split}")
    return root / ("validate/points.csv" if split == "validate" else f"labels/labels_{split}.csv")


def load_points(root: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(point_path(root, split), usecols=POINT_COLUMNS, dtype=ID_COLUMNS)
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Prediction points need unique non-null sample_id")
    if frame[["tr_id", "target_stop_id"]].isna().any().any():
        raise ValueError("Vehicle and visit identifiers cannot be null")
    for column in ("T", "target_time_begin"):
        frame[column] = timestamps(frame[column])
    horizon = frame.target_time_begin - frame["T"]
    if not horizon.dt.total_seconds().between(600, 900, inclusive="right").all():
        raise ValueError("Target horizon must be in (600, 900] seconds")
    if np.isinf(frame.cur_dev_s.to_numpy(dtype=float)).any():
        raise ValueError("cur_dev_s must be finite or missing")
    return frame


def load_labels(root: Path, split: str, points: pd.DataFrame) -> np.ndarray:
    if split == "validate":
        raise ValueError("Validate has no labels")
    labels = pd.read_csv(
        point_path(root, split), usecols=["sample_id", "target_delay_s"], dtype=ID_COLUMNS
    ).set_index("sample_id")
    values = labels.loc[points.sample_id, "target_delay_s"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Targets must be finite")
    return values


def load_inputs(root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_path(root, split)  # Validate the split before constructing file paths.
    traffic = pd.read_csv(root / split / "traffic.csv", usecols=TRAFFIC_COLUMNS, dtype=ID_COLUMNS)
    filename = "schedule_plan.csv" if split == "validate" else "schedule.csv"
    plan = pd.read_csv(root / split / filename, usecols=PLAN_COLUMNS, dtype=ID_COLUMNS)
    return traffic, plan
