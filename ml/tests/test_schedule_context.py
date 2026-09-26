import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from transport_ml.features import FeatureBuilder, FeatureConfig, prepare_plan


def scenario():
    at = pd.Timestamp("2026-01-06 12:00")
    plan = pd.DataFrame(
        {
            "tr_id": ["bus"] * 3,
            "tt_action_item_id": ["passed", "next", "target"],
            "time_begin": [at + pd.Timedelta(seconds=s) for s in (-120, 300, 780)],
            "geom": ["POINT (37.6 55.7)", "POINT (37.61 55.7)", "POINT (37.62 55.7)"],
        }
    )
    traffic = pd.DataFrame(
        {
            "tr_id": ["bus"] * 5,
            "event_time": [at + pd.Timedelta(seconds=s) for s in (-90, -60, -30, 0, 1)],
            "location_valid": [True] * 5,
            "lon": [37.6, 37.6, 37.6, 37.604, 37.62],
            "lat": [55.7] * 5,
            "speed": [0, 0, 0, 30, 30],
            "heading": [90] * 5,
        }
    )
    points = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "tr_id": "bus",
                "T": at,
                "target_stop_id": "target",
                "target_time_begin": at + pd.Timedelta(seconds=780),
                "cur_dev_s": 999.0,
            }
        ]
    )
    return at, traffic, plan, points


def features(traffic, plan, points):
    return FeatureBuilder(traffic, plan, FeatureConfig(schedule_context=True)).transform(points)


def test_context_is_causal_and_independent_of_supplied_hint_and_facts():
    at, traffic, plan, points = scenario()
    full = features(traffic, plan, points)
    prefix = features(traffic[traffic.event_time <= at], plan, points)
    assert_frame_equal(full, prefix)
    assert full.arrival_deviation_s.iloc[0] == 60
    assert full.arrival_age_s.iloc[0] == 60
    assert full.target_heading_cos.iloc[0] == 1
    plan["time_fact_begin"] = "forbidden"
    points["cur_dev_s"] = np.nan
    points["target_delay_s"] = -9999
    changed = features(traffic, prepare_plan(plan), points)
    assert_frame_equal(full.drop(columns="cur_dev_s"), changed.drop(columns="cur_dev_s"))


def test_context_handles_missing_stale_and_off_plan_positions():
    at, traffic, plan, points = scenario()
    empty = features(traffic.iloc[:0], plan, points).iloc[0]
    assert empty.arrival_matches == 0
    assert np.isnan(empty.match_deviation_s)
    old = traffic.copy()
    old.event_time -= pd.Timedelta(hours=1)
    stale = features(old, plan, points).iloc[0]
    assert np.isnan(stale.arrival_deviation_s)
    traffic[["lon", "lat"]] = [38, 56]
    far = features(traffic, plan, points).iloc[0]
    assert far.match_distance_m > 1000
    assert np.isnan(far.match_deviation_s)
    assert far.arrival_matches == 0


def test_context_uses_only_retained_history_and_does_not_bridge_gaps():
    at, traffic, plan, points = scenario()
    past = traffic.iloc[[0]].copy()
    past.event_time = at - pd.Timedelta(hours=1)
    complete = pd.concat([past, traffic], ignore_index=True)
    assert_frame_equal(features(complete, plan, points), features(traffic, plan, points))
    traffic = traffic.iloc[[0, 3]].copy()  # 90 second gap: no invented continuous motion.
    result = features(traffic, plan, points).iloc[0]
    assert result.time_coverage_300s == 0
    assert np.isnan(result.weighted_speed_300s)


def test_context_nanosecond_cutoff_and_negative_delay():
    at, traffic, plan, points = scenario()
    traffic.loc[4, "event_time"] = at + pd.Timedelta(nanoseconds=1)
    expected = features(traffic, plan, points)
    assert_frame_equal(expected, features(traffic.iloc[:4], plan, points))
    # The observed stop happened 60 seconds before its plan: preserve signed early arrival.
    plan.loc[0, "time_begin"] = at
    result = features(traffic, plan, points).iloc[0]
    assert result.arrival_deviation_s == -60
