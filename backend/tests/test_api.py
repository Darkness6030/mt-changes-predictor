"""HTTP contract of the dispatcher API, including error shapes and replay control."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from transport_backend.api import create_app
from transport_backend.config import Settings

from .test_engine import FakeMl

DATA = Path("dataset")


@pytest.fixture(scope="module")
def client():
    settings = Settings(
        mode="replay",
        data_root=DATA,
        split="test",
        replay_speed=600.0,
        replay_autostart=True,
        labels=Path("dataset/labels/labels_test.csv"),
    )
    app = create_app(settings)
    with TestClient(app) as started:
        started.app.state.engine.ml = FakeMl()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if started.get("/api/v1/snapshot").json()["vehicles"]:
                break
            time.sleep(0.2)
        yield started


def test_health_and_openapi(client):
    assert client.get("/health/live").json()["status"] == "live"
    assert client.get("/health/ready").status_code == 200
    schema = client.get("/openapi.json").json()
    for path in ("/api/v1/snapshot", "/api/v1/vehicles/{tr_id}", "/api/v1/status"):
        assert path in schema["paths"]


def test_snapshot_shape_and_filters(client):
    snapshot = client.get("/api/v1/snapshot").json()
    assert snapshot["schema_version"] == "1"
    assert snapshot["mode"] == "replay"
    assert snapshot["clock"]["time_basis"] == "dataset_naive_ns"
    assert snapshot["clock"]["source_time"].startswith("2026-01-06")
    assert "note" in snapshot["risk_policy"]
    assert set(snapshot["summary"]) >= {"vehicles", "with_prediction", "stale", "no_prediction"}
    filtered = client.get("/api/v1/snapshot", params={"only_attention": True}).json()
    assert len(filtered["vehicles"]) <= len(snapshot["vehicles"])
    assert client.get("/api/v1/snapshot", params={"risk": "nonsense"}).status_code == 422


def test_vehicle_detail_and_unknown_vehicle(client):
    snapshot = client.get("/api/v1/snapshot").json()
    tr_id = snapshot["vehicles"][0]["tr_id"]
    detail = client.get(f"/api/v1/vehicles/{tr_id}").json()
    assert detail["tr_id"] == tr_id
    assert "track" in detail and "plan" in detail and "prediction_history" in detail
    missing = client.get("/api/v1/vehicles/no-such-vehicle")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "unknown_vehicle"


def test_status_metrics_and_quality(client):
    status = client.get("/api/v1/status").json()
    assert status["mode"] == "replay" and status["plan"]["vehicles"] == 13
    assert status["points"]["total"] == 353
    assert status["replay"]["events_total"] > 0
    assert "cycle_ms" in status["performance"]
    quality = client.get("/api/v1/metrics/quality").json()
    assert quality["offline"]["rows"] == 353
    assert quality["replay_sidecar"]["source"].endswith("labels_test.csv")
    assert "performance" in client.get("/api/v1/metrics").json()


def test_replay_control_and_unknown_alert(client):
    paused = client.post("/api/v1/replay/control", json={"action": "pause"})
    assert paused.status_code == 200 and paused.json()["clock"]["paused"] is True
    speed = client.post("/api/v1/replay/control", json={"action": "speed", "speed": 60})
    assert speed.json()["clock"]["speed"] == 60
    assert client.post("/api/v1/replay/control", json={"action": "speed"}).status_code == 400
    assert client.post("/api/v1/replay/control", json={"action": "fly"}).status_code == 422
    run_id = client.get("/api/v1/status").json()["run_id"]
    reset = client.post("/api/v1/replay/control", json={"action": "reset"})
    assert reset.json()["run_id"] != run_id
    assert client.post("/api/v1/alerts/none/ack").status_code == 404


def test_replay_control_is_refused_outside_replay_mode():
    settings = Settings(mode="ndtp", data_root=DATA, split="test", ndtp_port=0)
    with TestClient(create_app(settings)) as started:
        started.app.state.engine.ml = FakeMl()
        response = started.post("/api/v1/replay/control", json={"action": "pause"})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "replay_disabled"
        status = started.get("/api/v1/status").json()
        assert status["mode"] == "ndtp" and status["ndtp"]["connections_total"] == 0
        assert status["replay"] is None
