import numpy as np
import pandas as pd
import pytest
from transport_ml.group_validation import GroupConfig, fit_outer_train, group_mask
from transport_ml.training import TrainConfig


def test_whole_vehicle_split_is_stable_and_label_independent():
    points = pd.DataFrame({"tr_id": np.repeat([str(i) for i in range(13)], 10)})
    mask = group_mask(points, GroupConfig())
    assert points.loc[mask, "tr_id"].nunique() == 4
    assert set(points.loc[mask, "tr_id"]).isdisjoint(points.loc[~mask, "tr_id"])
    shuffled = points.sample(frac=1, random_state=7).assign(target_delay_s=99999)
    shuffled_mask = group_mask(shuffled, GroupConfig())
    assert set(shuffled.loc[shuffled_mask, "tr_id"]) == set(points.loc[mask, "tr_id"])


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_invalid_holdout_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        group_mask(pd.DataFrame({"tr_id": ["1", "2", "3"]}), GroupConfig(fraction))


def test_unknown_synthetic_families_and_too_few_groups_rejected():
    with pytest.raises(ValueError, match="real vehicles"):
        group_mask(pd.DataFrame({"tr_id": ["1", "2", "9000000"]}), GroupConfig())
    with pytest.raises(ValueError, match="training vehicles"):
        group_mask(pd.DataFrame({"tr_id": ["1", "2"]}), GroupConfig())


def test_holdout_labels_features_and_times_cannot_change_fitted_models():
    time = pd.date_range("2026-01-06", periods=100, freq="5min")
    points = pd.DataFrame(
        {
            "tr_id": np.repeat(["1", "2", "3", "4"], 100),
            "T": np.tile(time, 4),
            "target_time_begin": np.tile(time + pd.Timedelta(minutes=12), 4),
        }
    )
    features = pd.DataFrame(
        {
            "cur_dev_s": np.sin(np.arange(400)) * 80,
            "speed": np.arange(400) % 41,
        }
    )
    target = features.cur_dev_s.to_numpy() + np.arange(400) % 17
    holdout = group_mask(points, GroupConfig())
    config = TrainConfig(iterations=3, threads=1)
    models, specs, selection = fit_outer_train(points, features, target, holdout, config)
    poisoned_points, poisoned_features, poisoned_target = (
        points.copy(),
        features.copy(),
        target.copy(),
    )
    poisoned_points.loc[holdout, ["T", "target_time_begin"]] += pd.Timedelta(days=100)
    poisoned_features.loc[holdout] = 999999
    poisoned_target[holdout] = -999999
    other_models, other_specs, other_selection = fit_outer_train(
        poisoned_points, poisoned_features, poisoned_target, holdout, config
    )
    assert specs == other_specs
    assert selection == other_selection
    for name, model in models.items():
        columns = specs[name]["features"]
        np.testing.assert_array_equal(
            model.predict(features[columns]), other_models[name].predict(features[columns])
        )
    assert "cur_dev_s" not in specs["fallback"]["features"]
