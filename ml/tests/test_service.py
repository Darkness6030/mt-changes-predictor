"""Contract tests for the ML HTTP service against the published v2 bundle."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from transport_ml.data import load_points
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.model import DelayModel
from transport_ml.service import app

BUNDLE = "ml/pretrained/v2"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as started:  # Lifespan loads the artifact exactly once.
        yield started


@pytest.fixture(scope="module")
def model_features():
    model = DelayModel(BUNDLE)
    points = load_points(Path("dataset"), "validate").head(5)
    traffic = pd.read_csv("dataset/validate/traffic.csv", dtype={"tr_id": str})
    plan = pd.read_csv("dataset/validate/schedule_plan.csv", dtype={"tr_id": str})
    config = FeatureConfig(**model.manifest["feature_config"])
    return model, points, FeatureBuilder(traffic, plan, config).transform(points)


def payload(features: pd.DataFrame, points: pd.DataFrame) -> dict:
    items = []
    for index, (_, row) in enumerate(features.iterrows()):
        values = {name: (None if pd.isna(value) else float(value)) for name, value in row.items()}
        items.append(
            {
                "request_id": str(index),
                "tr_id": points.tr_id.iloc[index],
                "cutoff_t": str(points["T"].iloc[index]),
                "features": values,
            }
        )
    return {"feature_schema_version": "1", "items": items}


def test_health_and_model_metadata(client, model_features):
    model, _, _ = model_features
    assert client.get("/health/live").json()["status"] == "live"
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["model_version"] == model.version
    info = client.get("/v1/model").json()
    assert info["features"] == model.features
    assert info["late_threshold_s"] == 120.0
    assert info["calibration"]["status"] == "fitted_on_development"


def test_http_matches_offline_batch_inference(client, model_features):
    """The service must not become a second, slightly different inference path."""
    model, points, features = model_features
    response = client.post("/v1/predict", json=payload(features, points))
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == model.version
    delays = np.array([row["delay_s"] for row in body["results"]])
    np.testing.assert_allclose(delays, model.predict(features), rtol=0, atol=1e-9)
    probabilities = np.array([row["late_probability"] for row in body["results"]])
    np.testing.assert_allclose(probabilities, model.predict_late_probability(features))
    assert all(row["model_used"] == "main" and row["used_hint"] for row in body["results"])
    assert all(0.0 <= row["late_probability"] <= 1.0 for row in body["results"])


def test_missing_hint_routes_to_fallback(client, model_features):
    model, points, features = model_features
    body = payload(features, points)
    for item in body["items"]:
        item["features"]["cur_dev_s"] = None
    results = client.post("/v1/predict", json=body).json()["results"]
    assert {row["model_used"] for row in results} == {"fallback"}
    expected = model.predict(features.assign(cur_dev_s=np.nan))
    np.testing.assert_allclose([row["delay_s"] for row in results], expected, atol=1e-9)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda body: body["items"][0]["features"].pop("horizon_s"), 422),
        (lambda body: body["items"][0]["features"].update(unexpected=1.0), 422),
        # A client cannot send a bare JSON Infinity, but "Infinity" parses to inf in lax mode.
        (lambda body: body["items"][0]["features"].update(horizon_s="Infinity"), 422),
        (lambda body: body.update(feature_schema_version="999"), 422),
        (lambda body: body.update(items=[]), 422),
    ],
)
def test_invalid_requests_are_rejected(client, model_features, mutate, code):
    _, points, features = model_features
    body = payload(features, points)
    mutate(body)
    response = client.post("/v1/predict", json=body)
    assert response.status_code == code
    assert response.json()["detail"]


def test_service_reports_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_MODEL_DIR", str(tmp_path / "absent"))
    with TestClient(app, raise_server_exceptions=False) as started:
        assert started.get("/health/live").status_code == 200
        assert started.get("/health/ready").status_code == 503
        assert started.get("/v1/model").status_code == 503
