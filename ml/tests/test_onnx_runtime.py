"""ONNX Runtime backend: parity with CatBoost, service switch and honest fallback."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from transport_ml.data import load_points
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.model import DelayModel
from transport_ml.onnx_runtime import OnnxDelayModel, probe_features

pytest.importorskip("onnxruntime")


def validate_features(model: DelayModel) -> pd.DataFrame:
    points = load_points(Path("dataset"), "validate").head(40)
    traffic = pd.read_csv("dataset/validate/traffic.csv", dtype={"tr_id": str})
    plan = pd.read_csv("dataset/validate/schedule_plan.csv", dtype={"tr_id": str})
    config = FeatureConfig(**model.manifest["feature_config"])
    return FeatureBuilder(traffic, plan, config).transform(points)


@pytest.mark.parametrize("bundle", ["ml/pretrained/v3", "ml/pretrained/v5"])
def test_onnx_reproduces_catboost_on_real_and_probe_rows(bundle, tmp_path):
    reference = DelayModel(Path(bundle))
    onnx = OnnxDelayModel(Path(bundle), onnx_dir=tmp_path)
    assert onnx.runtime == "onnx" and len(onnx.onnx_files) == 8
    for frame in (validate_features(reference), probe_features(reference)):
        report = onnx.parity(frame, reference=reference)
        assert report["delay_main_s"] < 1e-3 and report["delay_no_hint_s"] < 1e-3
        assert report["probability_main"] < 1e-5 and report["probability_no_hint"] < 1e-5
    # The published bundle is untouched: no ONNX file is written next to the models.
    assert not list(Path(bundle).glob("*.onnx"))


def test_service_switch_serves_onnx_with_the_same_numbers():
    reference = DelayModel(Path("ml/pretrained/v5"))
    features = validate_features(reference).head(8)
    items = [
        {
            "request_id": str(index),
            "features": {k: (None if pd.isna(v) else float(v)) for k, v in row.items()},
        }
        for index, (_, row) in enumerate(features.iterrows())
    ]
    from transport_ml.service import app

    environment = {"ML_MODEL_DIR": "ml/pretrained/v5", "ML_RUNTIME": "onnx"}
    with patch.dict("os.environ", environment), TestClient(app) as client:
        info = client.get("/v1/model").json()
        assert info["runtime"] == "onnx" and "CatBoost" in info["runtime_note"]
        body = client.post(
            "/v1/predict", json={"feature_schema_version": "2", "items": items}
        ).json()
    delays = np.array([row["delay_s"] for row in body["results"]])
    np.testing.assert_allclose(delays, reference.predict(features), atol=1e-3)


def test_failed_export_falls_back_to_catboost_and_says_so():
    from transport_ml import service

    with (
        patch.dict("os.environ", {"ML_RUNTIME": "onnx"}),
        patch("transport_ml.onnx_runtime.OnnxDelayModel", side_effect=RuntimeError("boom")),
    ):
        model = service.load_model(Path("ml/pretrained/v5"))
    assert model.runtime == "catboost" and "ONNX недоступен" in model.runtime_note
