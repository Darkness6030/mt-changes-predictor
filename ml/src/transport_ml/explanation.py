"""Exact additive explanation of one delay prediction, grouped for a dispatcher.

For a residual member ``prediction = cur_dev_s + expected + sum(SHAP)``; for a direct member
``prediction = expected + sum(SHAP)``. Ensemble weights are applied to both parts, so the
groups below always add up to the published delay. This explains the model's arithmetic,
not the physical cause of a delay: there is no incident, door or passenger data.
"""

import numpy as np
import pandas as pd

# Ordered: the first matching rule wins. Every feature of schema 1 and 2 is covered.
GROUPS = (
    ("hint", "Текущее отклонение (подсказка cur_dev)", ("cur_dev_s",)),
    (
        "schedule_match",
        "Положение относительно расписания по GPS",
        ("arrival_", "match_", "plan_distance_now_m"),
    ),
    (
        "dwell",
        "Стоянки и простой",
        ("observed_stop_s", "stopped_fraction_", "weighted_stopped_"),
    ),
    (
        "movement",
        "Скорость и движение",
        ("speed_", "last_speed", "gps_speed_", "gps_path_", "weighted_speed_", "distance_delta_"),
    ),
    (
        "route_ahead",
        "Путь до целевой остановки",
        (
            "target_distance_m",
            "plan_remaining_m",
            "plan_stops_remaining",
            "plan_required_speed_kmh",
            "target_required_speed_kmh",
            "target_approach_",
            "target_heading_cos",
            "horizon_s",
        ),
    ),
    ("place_time", "Время суток и место", ("time_sin", "time_cos", "target_lon", "target_lat")),
    (
        "data_quality",
        "Свежесть и полнота телеметрии",
        (
            "event_age_s",
            "gps_age_s",
            "gps_stale",
            "count_",
            "valid_fraction_",
            "span_s_",
            "time_coverage_",
        ),
    ),
)
LABELS = {key: label for key, label, _ in GROUPS}


def group_of(feature: str) -> str:
    for key, _, patterns in GROUPS:
        if any(feature == p or (p.endswith("_") and feature.startswith(p)) for p in patterns):
            return key
    raise ValueError(f"Feature has no explanation group: {feature}")


def summarise(
    feature_values: dict[str, float], base_s: float, offset_s: float, top: int = 4
) -> dict:
    """Group per-feature contributions (seconds); the remainder keeps the sum exact."""
    grouped: dict[str, float] = {}
    for feature, value in feature_values.items():
        key = group_of(feature)
        grouped[key] = grouped.get(key, 0.0) + float(value)
    if offset_s:
        grouped["hint"] = grouped.get("hint", 0.0) + float(offset_s)
    ordered = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
    shown = [
        {"group": key, "label": LABELS[key], "seconds": round(value, 3)}
        for key, value in ordered[:top]
    ]
    rest = sum(value for _, value in ordered[top:])
    return {
        "base_s": round(float(base_s), 3),
        "groups": shown,
        "other_s": round(rest, 3),
        "total_s": round(float(base_s) + sum(grouped.values()), 3),
    }


def member_shap(model, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row SHAP values and expected value of one CatBoost regressor."""
    from catboost import Pool

    values = np.asarray(model.get_feature_importance(Pool(frame), type="ShapValues"))
    return values[:, :-1], values[:, -1]
