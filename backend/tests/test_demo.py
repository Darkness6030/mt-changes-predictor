"""Source ownership, real TCP replay and recovery contracts of the local demo panel."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from transport_backend.api import create_app
from transport_backend.config import Settings
from transport_backend.demo import DemoController, DemoError
from transport_backend.engine import Engine
from transport_backend.ndtp import encode_handshake
from transport_backend.ndtp_server import NdtpServer

from .test_engine import FakeMl


@pytest.fixture
def demo_engine_factory(monkeypatch):
    def build(settings):
        engine = Engine(settings)
        engine.ml = FakeMl()
        return engine

    monkeypatch.setattr("transport_backend.demo.Engine", build)
    return build


async def wait_until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.02)


def settings():
    return Settings(
        data_root=Path("dataset"),
        split="test",
        ndtp_port=0,
        demo_enabled=True,
        replay_start="2026-01-06 08:00:00",
        replay_speed=60,
        replay_autostart=False,
    )


async def test_csv_to_real_tcp_and_back_retires_previous_run(demo_engine_factory):
    original = demo_engine_factory(settings())
    await original.start()
    control = DemoController(settings(), original)
    try:
        live = await control.switch("ndtp_replay", 60, "2026-01-06 08:00:00")
        await wait_until(lambda: live.ndtp.counters.nav_events > 0)
        assert live.settings.mode == "ndtp" and live.points is None
        assert live.ndtp.counters.handshakes > 0
        assert live.state.tracks
        assert original._task is None
        sender = control.sender
        previous_run = live.run_id
        replay = await control.switch("replay", 30, None)
        assert replay.run_id != previous_run and replay.points is not None
        assert not live.ndtp.running and live.ndtp.counters.connections_open == 0
        assert sender.done() and control.sender is None
        assert replay.predictions == {} and replay.alerts == {}
    finally:
        await control.close()


async def test_unavailable_emulator_and_invalid_time_preserve_current_run(demo_engine_factory):
    original = demo_engine_factory(settings())
    await original.start()
    control = DemoController(settings(), original)

    async def unavailable(request):
        raise httpx.ConnectError("not running", request=request)

    await control.client.aclose()
    control.client = httpx.AsyncClient(
        transport=httpx.MockTransport(unavailable), base_url="http://emu"
    )
    try:
        run_id = original.run_id
        with pytest.raises(DemoError, match="недоступен"):
            await control.switch("emulator", 60, None)
        with pytest.raises(ValueError):
            await control.switch("ndtp_replay", 60, "nonsense")
        assert control.engine is original and original.ready and original.run_id == run_id
        view = await control.view()
        assert not view["sources"][2]["available"]
    finally:
        await control.close()


async def test_emulator_config_stopped_before_new_source_and_failure_recovers(demo_engine_factory):
    base = settings()
    original = demo_engine_factory(base)
    await original.start()
    control = DemoController(base, original)
    configs = []
    fail_start = False

    async def emulator(request):
        if request.method == "POST":
            config = json.loads(request.content)
            configs.append(config)
            if fail_start and config["units"]:
                return httpx.Response(500)
        return httpx.Response(200, json={})

    await control.client.aclose()
    control.client = httpx.AsyncClient(
        transport=httpx.MockTransport(emulator), base_url="http://emu"
    )
    try:
        official = await control.switch("emulator", 60, None)
        assert official.settings.plan_shift_auto and official.points is None
        assert len(configs[-1]["units"]) == 2
        assert all(str(unit["unitId"]) in official.mapping for unit in configs[-1]["units"])
        await control.switch("replay", 60, None)
        assert configs[-1]["units"] == [] and not official.ndtp.running
        assert control.engine._plan_shift_s == 0
        fail_start = True
        with pytest.raises(DemoError, match="восстановлен CSV"):
            await control.switch("emulator", 60, None)
        assert control.engine.ready and control.active == "replay"
        assert control.phase == "error" and configs[-1]["units"] == []
    finally:
        await control.close()


async def test_concurrent_switch_rejected(demo_engine_factory):
    engine = demo_engine_factory(settings())
    await engine.start()
    control = DemoController(settings(), engine)
    try:
        async with control.lock:
            with pytest.raises(DemoError, match="Уже выполняется"):
                await control.switch("replay", 60, None)
        assert control.engine is engine
    finally:
        await control.close()


async def test_server_stop_closes_accepted_clients_and_releases_port():
    server = NdtpServer("127.0.0.1", 0, lambda _: None)
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(encode_handshake(664030))
    await writer.drain()
    await wait_until(lambda: server.counters.handshakes == 1)
    await asyncio.wait_for(server.stop(), timeout=2)
    assert await asyncio.wait_for(reader.read(), 2) == b""
    assert server.counters.connections_open == 0 and not server._clients
    replacement = NdtpServer("127.0.0.1", port, lambda _: None)
    await replacement.start()
    await replacement.stop()
    writer.close()
    await writer.wait_closed()


def test_demo_http_opt_in_validation_and_replay_lock():
    with TestClient(create_app(replace(settings(), demo_enabled=False))) as client:
        assert not client.get("/api/v1/demo/sources").json()["enabled"]
        assert client.post("/api/v1/demo/source", json={"source": "replay"}).status_code == 409
        for body in (
            {"source": "unknown"},
            {"source": "replay", "speed": 0},
            {"source": "replay", "url": "http://arbitrary"},
        ):
            assert client.post("/api/v1/demo/source", json=body).status_code == 422
