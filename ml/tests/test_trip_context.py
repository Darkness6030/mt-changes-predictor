import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.research import feature_columns
from transport_ml.synthetic import keep_synthetic, synthetic_families
from transport_ml.trip_context import HINT_COLUMNS, TRIP_COLUMNS, trip_features

AT = pd.Timestamp("2026-01-06 12:00")


def ns(*offsets):
    return np.array([(AT + pd.Timedelta(seconds=s)).value for s in offsets], dtype="int64")


def test_layover_between_now_and_target_is_detected_and_absorbs_hint():
    # visits: -120 (passed), +60, then a 600 s planned layover, +660, +780 (target)
    plan = ns(-120, 60, 660, 780)
    values = trip_features(plan, AT.value, plan[3], 300.0)
    assert values["layover_between"] == 1.0
    assert values["plan_max_gap_s"] == 600
    assert values["layover_sum_s"] == 600
    assert values["target_after_layover_s"] == 120
    assert values["target_trip_pos_s"] == 120
    assert values["last_visit_age_s"] == 120
    assert values["hint_after_slack"] == 300 - (600 - 60)
    assert np.isnan(trip_features(plan, AT.value, plan[3], np.nan)["hint_after_slack"])


def test_no_layover_keeps_supplied_hint_and_trip_position():
    plan = ns(-600, -480, -360, -240, -120, 60, 240, 420, 600, 780, 1500)
    values = trip_features(plan, AT.value, plan[9], 42.0)
    assert values["layover_between"] == 0.0
    assert np.isnan(values["target_after_layover_s"])
    assert values["hint_after_slack"] == 42.0
    assert values["target_trip_pos_s"] == 1380
    assert values["trip_pos_T_s"] == 600
    assert values["target_to_trip_end_s"] == 0  # the next gap (720 s) starts at the target


def scenario():
    plan = pd.DataFrame(
        {
            "tr_id": ["bus"] * 4,
            "tt_action_item_id": ["passed", "next", "after", "target"],
            "time_begin": [AT + pd.Timedelta(seconds=s) for s in (-120, 60, 500, 780)],
            "geom": [f"POINT (37.6{i} 55.7)" for i in range(4)],
        }
    )
    traffic = pd.DataFrame(
        {
            "tr_id": ["bus"] * 4,
            "event_time": [AT + pd.Timedelta(seconds=s) for s in (-60, -30, 0, 1)],
            "location_valid": [True] * 4,
            "lon": [37.6, 37.6, 37.6, 37.63],
            "lat": [55.7] * 4,
            "speed": [0, 0, 20, 40],
            "heading": [90] * 4,
        }
    )
    points = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "tr_id": "bus",
                "T": AT,
                "target_stop_id": "target",
                "target_time_begin": AT + pd.Timedelta(seconds=780),
                "cur_dev_s": 120.0,
            }
        ]
    )
    return traffic, plan, points


def build(traffic, plan, points):
    config = FeatureConfig(schedule_context=True, trip_context=True)
    return FeatureBuilder(traffic, plan, config).transform(points)


def test_schema_three_is_causal_and_ignores_facts_and_labels():
    traffic, plan, points = scenario()
    full = build(traffic, plan, points)
    assert set(TRIP_COLUMNS) <= set(full)
    assert_frame_equal(full, build(traffic[traffic.event_time <= AT], plan, points))
    plan["time_fact_begin"] = "forbidden"
    points["target_delay_s"] = -9999
    assert_frame_equal(full, build(traffic, plan, points))


def test_schema_three_requires_schedule_context_and_fallback_drops_hint_columns():
    assert FeatureConfig(schedule_context=True, trip_context=True).schema_version == "3"
    with pytest.raises(ValueError):
        FeatureConfig(trip_context=True)
    traffic, plan, points = scenario()
    features = build(traffic, plan, points)
    assert not HINT_COLUMNS & set(feature_columns(features, "full", "fallback"))
    assert HINT_COLUMNS <= set(feature_columns(features, "full", "main"))


def test_synthetic_copies_near_protected_moments_are_purged():
    geoms = [f"POINT (37.{i} 55.7)" for i in range(6)]
    times = [AT + pd.Timedelta(minutes=5 * i) for i in range(6)]
    shift = pd.Timedelta(seconds=1315)
    plan = pd.DataFrame(
        {
            "tr_id": ["122048"] * 6 + ["9000000"] * 6,
            "time_begin": times + [t + shift for t in times],
            "geom": geoms * 2,
        }
    )
    families = synthetic_families(plan)
    assert families == {"9000000": ("122048", 1315.0)}
    copies = pd.DataFrame(
        {"tr_id": ["9000000"] * 3, "T": [AT + shift, AT + shift + pd.Timedelta(hours=1), AT]}
    )
    protected = pd.DataFrame({"tr_id": ["122048"], "T": [AT + pd.Timedelta(minutes=10)]})
    keep = keep_synthetic(copies, families, protected, margin_s=1200)
    # real-time counterparts: AT (10 min away), AT+1h (kept), AT-1315 s (~32 min, kept)
    assert keep.tolist() == [False, True, True]
