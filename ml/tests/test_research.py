import numpy as np
import pandas as pd
import pytest
from transport_ml import research
from transport_ml.data import TRAFFIC_COLUMNS
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.research import Candidate, build_augmented, inner_folds


def test_group_and_forward_splits_keep_unavailable_outcomes_out():
    points = pd.DataFrame(
        [
            {"tr_id": str(vehicle), "T": pd.Timestamp(f"2026-01-06 {hour}:00")}
            for vehicle in range(10)
            for hour in (8, 10, 13, 15, 18, 20)
        ]
    )
    points["target_time_begin"] = points["T"] + pd.Timedelta(minutes=12)
    y = np.zeros(len(points))
    y[0] = 86400  # An old forecast whose actual outcome is still in the future.
    for name, fit, valid, cutoff in inner_folds(points, y, {"seed": 20260927}):
        assert not (fit & valid).any()
        if name.startswith("group"):
            assert not set(points.tr_id[fit]) & set(points.tr_id[valid])
        else:
            assert not fit[0]
            assert (points.loc[fit, "T"] < cutoff).all()
            outcomes = points.target_time_begin + pd.to_timedelta(y, unit="s")
            assert (outcomes[fit] < cutoff + pd.Timedelta(minutes=30)).all()


def test_augmented_points_do_not_create_new_targets_or_backfill_hints():
    at = pd.Timestamp("2026-01-06 12:00")
    plan = pd.DataFrame(
        {
            "tr_id": ["b", "b"],
            "tt_action_item_id": ["previous", "target"],
            "time_begin": [at + pd.Timedelta(seconds=s) for s in (500, 780)],
            "geom": ["POINT (37.6 55.7)"] * 2,
        }
    )
    points = pd.DataFrame(
        [
            {
                "sample_id": "p",
                "tr_id": "b",
                "T": at,
                "target_stop_id": "target",
                "target_time_begin": at + pd.Timedelta(seconds=780),
                "cur_dev_s": 900,
            }
        ]
    )
    builder = FeatureBuilder(pd.DataFrame(columns=TRAFFIC_COLUMNS), plan)
    augmented, features = build_augmented(points, builder)
    assert len(augmented) == 5
    assert set(augmented.target_stop_id) == {"target"}
    assert set(augmented.parent) == {0}
    assert augmented.cur_dev_s.isna().all()
    assert features.cur_dev_s.isna().all()
    assert features.horizon_s.between(600, 900, inclusive="right").all()
    assert (augmented["T"] < at).any()


def test_augmentation_never_crosses_parent_fold_or_time_purge(monkeypatch):
    captured = {}

    class RecordingModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, target, sample_weight):
            captured.update(features=features, target=target, weights=sample_weight)

    monkeypatch.setattr(research, "CatBoostRegressor", RecordingModel)
    x = pd.DataFrame({"cur_dev_s": [0.0, 0.0], "speed": [1.0, 2.0]})
    cutoff = pd.Timestamp("2026-01-06 12:00")
    augmented = pd.DataFrame(
        {
            "parent": [0, 1, 0],
            "T": [cutoff - pd.Timedelta(seconds=1), cutoff, cutoff + pd.Timedelta(seconds=1)],
        }
    )
    extra = pd.DataFrame({"cur_dev_s": [np.nan] * 3, "speed": [3.0, 4.0, 5.0]})
    candidate = Candidate("augmentation", "fallback", "base", 2, 10, augmentation=True)
    research.fit_candidate(
        candidate,
        x,
        np.array([10.0, 9999.0]),
        np.array([True, False]),
        augmented=(augmented, extra),
        cutoff=cutoff,
    )
    assert captured["features"].speed.tolist() == [1.0, 3.0]
    np.testing.assert_array_equal(captured["target"], [10.0, 10.0])
    np.testing.assert_array_equal(captured["weights"], [0.5, 0.5])


def test_feature_schema_and_invalid_recipes():
    assert FeatureConfig().schema_version == "1"
    assert FeatureConfig(schedule_context=True).schema_version == "2"
    with pytest.raises(ValueError, match="supplied hints"):
        Candidate("bad", "fallback", "base", 2, 10, residual=True)
    with pytest.raises(ValueError, match="no supplied hints"):
        Candidate("bad", "main", "base", 2, 10, augmentation=True)


@pytest.mark.parametrize("splits", [[], ["validate"], ["train", "train"], ["train", "validate"]])
def test_recipe_refit_accepts_only_labelled_splits(tmp_path, splits):
    recipe = {
        "models": {"main": [], "fallback": []},
        "hint_policy": "supplied_only",
        "feature_config": {"schedule_context": True},
        "classifier_source": "ml/pretrained/v3",
        "training_splits": splits,
    }
    with pytest.raises(ValueError, match="training_splits"):
        research.train_recipe(tmp_path / "missing-data", tmp_path / "model", recipe)
