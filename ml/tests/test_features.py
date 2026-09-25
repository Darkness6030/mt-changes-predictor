import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from transport_ml.features import FeatureBuilder

T = pd.Timestamp("2026-01-06 12:00:00")


@pytest.fixture
def inputs():
    plan = pd.DataFrame(
        {
            "tr_id": ["bus"] * 3,
            "tt_action_item_id": ["excluded", "target", "last"],
            "time_begin": [T + pd.Timedelta(seconds=s) for s in (600, 780, 900)],
            "geom": ["POINT (37.6 55.7)"] * 3,
        }
    )
    traffic = pd.DataFrame(
        {
            "tr_id": ["bus"] * 4,
            "event_time": [T - pd.Timedelta(seconds=s) for s in (120, 60, 0, -1)],
            "location_valid": [True] * 4,
            "lon": [37.59, 37.591, 37.592, 80],
            "lat": [55.7] * 4,
            "speed": [20, 0, 0, 100],
            "heading": [90] * 4,
        }
    )
    points = pd.DataFrame(
        [
            {
                "sample_id": "point",
                "tr_id": "bus",
                "T": T,
                "target_stop_id": "target",
                "target_time_begin": T + pd.Timedelta(seconds=780),
                "cur_dev_s": 42.0,
            }
        ]
    )
    return traffic, plan, points


def test_future_changes_cannot_change_features(inputs):
    traffic, plan, points = inputs
    original = FeatureBuilder(traffic, plan).transform(points)
    future = traffic.event_time > T
    traffic.loc[future, ["lon", "lat", "speed"]] = [0, 0, 999]
    traffic = pd.concat([traffic, traffic[future]] * 2, ignore_index=True)
    assert_frame_equal(original, FeatureBuilder(traffic, plan).transform(points))
    assert original.event_age_s.iloc[0] == 0
    assert original.observed_stop_s.iloc[0] == 60


def test_exact_nanosecond_cutoff_and_prefix_parity(inputs):
    traffic, plan, points = inputs
    traffic.loc[3, "event_time"] = T + pd.Timedelta(nanoseconds=1)
    whole = FeatureBuilder(traffic, plan).transform(points)
    prefix = FeatureBuilder(traffic[traffic.event_time <= T], plan).transform(points)
    assert_frame_equal(whole, prefix)
    assert whole.last_speed.iloc[0] == 0


def test_fact_and_label_columns_are_inert(inputs):
    traffic, plan, points = inputs
    expected = FeatureBuilder(traffic, plan).transform(points)
    plan["time_fact_begin"] = "not even a valid time"
    points["target_delay_s"] = -123456
    points["target_class"] = "late"
    assert_frame_equal(expected, FeatureBuilder(traffic, plan).transform(points))
    assert not {"tr_id", "target_stop_id", "target_delay_s"} & set(expected)


def test_target_boundaries_and_wrong_target(inputs):
    traffic, plan, points = inputs
    builder = FeatureBuilder(traffic, plan)
    assert builder.target("bus", T).tt_action_item_id == "target"
    plan = plan[plan.tt_action_item_id != "target"]
    assert FeatureBuilder(traffic, plan).target("bus", T).tt_action_item_id == "last"
    assert builder.target("unknown", T) is None
    assert builder.target("bus", T + pd.Timedelta(hours=1)) is None
    points["target_stop_id"] = "excluded"
    with pytest.raises(ValueError, match="first planned"):
        builder.transform(points)


def test_empty_invalid_outliers_and_stale(inputs):
    traffic, plan, points = inputs
    empty = FeatureBuilder(traffic.iloc[:0], plan).transform(points)
    assert empty.gps_stale.iloc[0] == 1
    assert empty["count_600s"].iloc[0] == 0
    assert np.isnan(empty.last_speed.iloc[0])
    traffic = traffic[traffic.event_time <= T].copy()
    traffic["speed"] = 368
    traffic["location_valid"] = False
    invalid = FeatureBuilder(traffic, plan).transform(points)
    assert np.isnan(invalid.target_distance_m.iloc[0])
    assert np.isnan(invalid.speed_mean_600s.iloc[0])
    traffic["location_valid"] = True
    outliers = FeatureBuilder(traffic, plan).transform(points)
    assert np.isnan(outliers.last_speed.iloc[0])
    traffic["speed"] = 0
    traffic["event_time"] -= pd.Timedelta(minutes=30)
    stale = FeatureBuilder(traffic, plan).transform(points)
    assert stale.gps_stale.iloc[0] == 1
    assert np.isnan(stale.observed_stop_s.iloc[0])


def test_large_gap_is_not_observed_standing_time(inputs):
    traffic, plan, points = inputs
    traffic = traffic.iloc[:3].copy()
    traffic.event_time = [T - pd.Timedelta(minutes=10), T - pd.Timedelta(minutes=5), T]
    traffic.speed = 0
    result = FeatureBuilder(traffic, plan).transform(points)
    assert result.observed_stop_s.iloc[0] == 0


def test_conflicting_timestamps_are_order_independent(inputs):
    traffic, plan, points = inputs
    traffic.loc[1, "event_time"] = T
    assert_frame_equal(
        FeatureBuilder(traffic, plan).transform(points),
        FeatureBuilder(traffic.iloc[::-1], plan).transform(points),
    )


def test_supplied_target_wins_ties_at_the_earliest_time(inputs):
    traffic, plan, points = inputs
    tied = plan.iloc[[1]].copy()
    tied["tt_action_item_id"] = "a-first-by-id"
    tied["geom"] = "POINT (38 56)"
    with_tie = pd.concat([plan, tied], ignore_index=True)
    assert FeatureBuilder(traffic, with_tie).target("bus", T).tt_action_item_id == "a-first-by-id"
    assert_frame_equal(
        FeatureBuilder(traffic, plan).transform(points),
        FeatureBuilder(traffic, with_tie).transform(points),
    )


def test_target_one_nanosecond_after_left_boundary(inputs):
    traffic, plan, points = inputs
    target_time = T + pd.Timedelta(seconds=600, nanoseconds=1)
    plan.loc[plan.tt_action_item_id == "target", "time_begin"] = target_time
    points["target_time_begin"] = target_time
    result = FeatureBuilder(traffic, plan).transform(points)
    assert result.horizon_s.iloc[0] > 600
