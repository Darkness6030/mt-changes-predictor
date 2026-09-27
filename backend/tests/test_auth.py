"""Optional sign-in: off by default, roles gate actions, tokens never leak."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from transport_backend.api import create_app
from transport_backend.auth import ANONYMOUS, AuthConfig
from transport_backend.config import Settings

from .test_engine import FakeMl

DISPATCHER = "dispatcher-secret-1"
VIEWER = "viewer-secret-2"
USERS = f"dispatcher:Иванова И.:{DISPATCHER}; viewer:Табло: зал:{VIEWER}"


def test_parse_roles_names_and_rejections():
    auth = AuthConfig.parse(USERS)
    assert auth.enabled
    assert auth.principal(f"Bearer {DISPATCHER}").name == "Иванова И."
    viewer = auth.principal(f"bearer {VIEWER}")
    assert viewer.name == "Табло: зал" and viewer.role == "viewer" and not viewer.can_act
    assert auth.principal("Bearer wrong-token-000") is None
    assert auth.principal(DISPATCHER) is None  # The scheme is required.
    assert auth.principal(None) is None
    assert AuthConfig.parse("").principal(None) is ANONYMOUS
    for bad in ("admin:X:long-enough-token", "viewer::long-enough-token", "viewer:X:short"):
        with pytest.raises(ValueError):
            AuthConfig.parse(bad)
    with pytest.raises(ValueError):
        AuthConfig.parse(f"viewer:A:{VIEWER};dispatcher:B:{VIEWER}")
    assert DISPATCHER not in repr(auth)  # Only digests are kept.


@pytest.fixture(scope="module")
def client():
    settings = Settings(
        mode="replay", data_root=Path("dataset"), split="test", replay_autostart=False
    )
    with TestClient(create_app(settings, AuthConfig.parse(USERS))) as started:
        started.app.state.engine.ml = FakeMl()
        yield started


def headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_reads_require_a_token_and_health_stays_open(client):
    assert client.get("/health/live").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    response = client.get("/api/v1/snapshot")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"]["code"] == "unauthorized"
    assert client.get("/api/v1/snapshot", headers=headers(VIEWER)).status_code == 200
    assert client.get("/api/v1/auth").json() == {
        "enabled": True,
        "authenticated": False,
        "name": None,
        "role": None,
        "can_act": False,
    }
    me = client.get("/api/v1/auth", headers=headers(DISPATCHER)).json()
    assert me["name"] == "Иванова И." and me["can_act"]


def test_viewer_cannot_act_and_dispatcher_can(client):
    command = {"action": "pause"}
    viewer = client.post("/api/v1/replay/control", json=command, headers=headers(VIEWER))
    assert viewer.status_code == 403
    assert client.post("/api/v1/alerts/none/ack", headers=headers(VIEWER)).status_code == 403
    # The dispatcher passes the gate: an unknown alert is the engine's 404, not a 401/403.
    assert client.post("/api/v1/alerts/none/ack", headers=headers(DISPATCHER)).status_code == 404
    assert (
        client.post("/api/v1/replay/control", json=command, headers=headers(DISPATCHER)).status_code
        == 200
    )


def test_status_never_exposes_tokens(client):
    body = client.get("/api/v1/status", headers=headers(VIEWER)).text
    assert DISPATCHER not in body and VIEWER not in body
