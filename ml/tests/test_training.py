import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from transport_ml.model import DelayModel, sha256, write_json
from transport_ml.submission import validate_submission
from transport_ml.training import temporal_masks


def test_split_purges_unknown_outcomes_and_history():
    time = pd.date_range("2026-01-06", periods=100, freq="5min")
    points = pd.DataFrame({"T": time, "target_time_begin": time + pd.Timedelta(minutes=12)})
    target = np.zeros(100)
    target[0] = 86400  # An outcome unavailable even though its forecast is in the past.
    fit, development, boundary = temporal_masks(points, target, 0.25)
    assert not fit[0]
    assert not np.any(fit & development)
    assert (points.loc[fit, "T"] < boundary - pd.Timedelta(minutes=10)).all()
    outcome = points.target_time_begin + pd.to_timedelta(target, unit="s")
    assert (outcome[fit] < boundary).all()


def test_model_roundtrip_missing_hint_and_schema(tmp_path):
    features = pd.DataFrame({"cur_dev_s": np.arange(20, dtype=float), "speed": np.arange(20)})
    target = np.arange(20, dtype=float) - 10
    manifest = {"feature_schema_version": "1", "features": list(features), "models": {}}
    for name, columns in (("main", list(features)), ("fallback", ["speed"])):
        model = CatBoostRegressor(iterations=5, depth=2, verbose=False, allow_writing_files=False)
        model.fit(features[columns], target)
        path = tmp_path / f"{name}.cbm"
        model.save_model(str(path))
        manifest["models"][name] = {
            "file": path.name,
            "features": columns,
            "residual": False,
            "sha256": sha256(path),
        }
    write_json(tmp_path / "manifest.json", manifest)
    restored = DelayModel(tmp_path)
    missing = features.copy()
    missing.loc[[0, 4], "cur_dev_s"] = np.nan
    prediction = restored.predict(missing)
    np.testing.assert_allclose(prediction[[0, 4]], restored.predict(features, no_hint=True)[[0, 4]])
    np.testing.assert_allclose(prediction[1:4], restored.predict(features)[1:4])
    with pytest.raises(ValueError, match="names/order"):
        restored.predict(features[["speed", "cur_dev_s"]])
    assert (prediction < 0).any()
    (tmp_path / "main.cbm").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        DelayModel(tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "sample_id;prediction\na;1\na;2\n",
        "sample_id;prediction\na;nan\nb;1\n",
        "sample_id;prediction\na;inf\nb;1\n",
        "sample_id,prediction\na,1\nb,2\n",
        "sample_id;prediction\na;1\nc;2\n",
        "sample_id;prediction;extra\na;1;0\nb;2;0\n",
    ],
)
def test_bad_submissions_are_rejected(tmp_path, content):
    path = tmp_path / "submission.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        validate_submission(path, pd.Series(["a", "b"]))


def test_signed_submission(tmp_path):
    path = tmp_path / "submission.csv"
    path.write_text("sample_id;prediction\na;-12.5\nb;0.0\n")
    assert validate_submission(path, pd.Series(["a", "b"])).prediction.iloc[0] == -12.5


def test_ensemble_weights_residuals_and_fallback_roundtrip(tmp_path):
    features = pd.DataFrame({"cur_dev_s": np.arange(20, dtype=float), "speed": np.arange(20)})
    manifest = {"feature_schema_version": "2", "features": list(features), "models": {}}
    expected = np.zeros(20)
    members = []
    for index, weight in enumerate((0.25, 0.75)):
        model = CatBoostRegressor(iterations=5, depth=2, verbose=False, allow_writing_files=False)
        model.fit(features[["speed"]], np.arange(20, dtype=float) - (index + 1) * 20)
        path = tmp_path / f"member-{index}.cbm"
        model.save_model(str(path))
        expected += weight * (model.predict(features[["speed"]]) + features.cur_dev_s)
        members.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "features": ["speed"],
                "weight": weight,
                "residual": True,
            }
        )
    manifest["models"]["main"] = {"members": members}
    manifest["models"]["fallback"] = {
        "members": [{**member, "residual": False} for member in members]
    }
    write_json(tmp_path / "manifest.json", manifest)
    restored = DelayModel(tmp_path)
    np.testing.assert_allclose(restored.predict(features), expected)
    missing = features.assign(cur_dev_s=np.nan)
    np.testing.assert_allclose(restored.predict(missing), expected - features.cur_dev_s)
    assert (restored.predict(missing) < 0).all()
    manifest["models"]["main"]["members"][0]["weight"] = 2
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="sum to one"):
        DelayModel(tmp_path)
