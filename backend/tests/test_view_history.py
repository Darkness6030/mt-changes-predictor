"""The UI journal must preserve what was known, without changing the live run."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from transport_backend.api import create_app
from transport_backend.clock import SECOND_NS, format_source, parse_source
from transport_backend.config import Settings
from transport_backend.replay import ReplaySource
from transport_backend.view_history import HistoryUnavailable, ViewHistory

from .test_engine import FakeMl, make_engine
from .test_state import T


def journal(**overrides):
    return ViewHistory(Settings(mode="ndtp", **overrides))


def test_journal_floor_lookup_and_immutable_roundtrip():
    archive = journal()
    snapshot = {"vehicles": [{"prediction": {"delay_s": 150}, "lon": 37.6}]}
    details = {"bus": {"track": [{"lat": 55.7}], "acknowledged_at": None}}
    archive.append(10 * SECOND_NS, snapshot, details)
    snapshot["vehicles"][0]["prediction"]["delay_s"] = -50
    details["bus"]["track"][0]["lat"] = 60
    details["bus"]["acknowledged_at"] = "later"
    archive.append(20 * SECOND_NS, snapshot, details)
    old = archive.view(19 * SECOND_NS)
    assert old["snapshot"]["vehicles"][0]["prediction"]["delay_s"] == 150
    assert old["details"]["bus"] == {"track": [{"lat": 55.7}], "acknowledged_at": None}
    assert old["recorded_at"] == format_source(10 * SECOND_NS)
    assert old["lag_s"] == 9
    # Mutating an HTTP response also cannot rewrite the archived bytes.
    old["snapshot"].clear()
    assert archive.view(10 * SECOND_NS)["snapshot"]["vehicles"]
    for outside in (9, 21):
        with pytest.raises(HistoryUnavailable):
            archive.view(outside * SECOND_NS)


def test_journal_caps_expiry_oversize_and_clock_regression(monkeypatch):
    archive = journal(view_history_window_s=30, view_history_max_frames=2)
    for second in (0, 1, 2):
        archive.append(second * SECOND_NS, {"time": second}, {})
    assert archive.metadata()["frames"] == 2
    with pytest.raises(HistoryUnavailable):
        archive.view(0)
    archive.append(35 * SECOND_NS, {"time": 35}, {})
    assert archive.metadata()["frames"] == 1  # Source-time horizon, not wall time.
    before = archive.view(35 * SECOND_NS)
    archive.append(34 * SECOND_NS, {"time": "late"}, {})
    assert archive.view(35 * SECOND_NS) == before
    assert not archive.due(34 * SECOND_NS)
    assert archive.metadata()["skipped_clock_samples"] == 1

    sample = journal(view_history_max_bytes=90)
    sample.append(0, {"payload": "a" * 200}, {})
    sample.append(SECOND_NS, {"payload": "b" * 200}, {})
    assert sample.metadata()["bytes"] <= 90 and sample.metadata()["frames"] == 1
    sample.append(2 * SECOND_NS, {"payload": list(range(1000))}, {})
    assert sample.metadata()["oversized_frames"] == 1
    assert sample.metadata()["bytes"] <= 90

    wall = [1.0]
    monkeypatch.setattr("transport_backend.view_history.monotonic", lambda: wall[0])
    sampled = journal()
    assert sampled.due(0)
    sampled.append(0, {}, {})
    assert not sampled.due(100 * SECOND_NS)  # Accelerated source cannot flood the journal.
    wall[0] += 1.1
    assert sampled.due(101 * SECOND_NS)
    assert not ViewHistory(Settings(mode="replay")).due(0)

    regressed = journal()
    assert not regressed.due(10 * SECOND_NS, not_before_ns=11 * SECOND_NS)
    assert regressed.metadata()["skipped_clock_samples"] == 1


@pytest.fixture
def engine():
    live = make_engine(mode="ndtp", labels=None, view_history_interval_s=1e-9)
    source = ReplaySource(
        live.settings.data_root / "test" / "traffic.csv",
        units={tr_id: unit for unit, tr_id in live.mapping.items()},
    )
    source.reset(int(T.value) - 1800 * SECOND_NS)
    for event in source.due(int(T.value) - SECOND_NS):
        live.state.add(event, int(T.value))
    live.clock.start(int(T.value), paused=True)
    return live


@pytest.mark.parametrize("clock_regresses", [False, True])
async def test_capture_is_publication_time_and_late_data_never_rewrites_history(
    engine, clock_regresses
):
    cutoff = int(T.value)
    engine.clock.start(cutoff - SECOND_NS, paused=True)
    engine._record_view()
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowMl(FakeMl):
        async def predict(self, items, schema):
            entered.set()
            await release.wait()
            return await super().predict(items, schema)

    engine.ml = SlowMl()
    engine.clock.start(cutoff, paused=True)
    cycle = asyncio.create_task(engine._cycle())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        track = next(iter(engine.state.tracks.values()))
        before = engine.view_history.view(cutoff - SECOND_NS)
        later = replace(track.last_valid, event_time_ns=cutoff + 30 * SECOND_NS, lon=37.8)
        engine._ingest(later)
        engine.clock.start(
            cutoff - SECOND_NS // 2 if clock_regresses else cutoff + 45 * SECOND_NS, paused=True
        )
        release.set()
        await cycle
        if clock_regresses:
            assert engine.view_history.metadata()["frames"] == 1
            assert engine.view_history.metadata()["skipped_clock_samples"] == 1
            engine.clock.start(cutoff + 45 * SECOND_NS, paused=True)
            engine._record_view()
        # The prediction belongs to cutoff T, but was unavailable at T+20.
        past = engine.view_history.view(cutoff + 20 * SECOND_NS)
        assert all(v["prediction"] is None for v in past["snapshot"]["vehicles"])
        published = engine.view_history.view(cutoff + 45 * SECOND_NS)
        assert any(v["prediction"]["status"] == "ok" for v in published["snapshot"]["vehicles"])
        assert published["snapshot"]["clock"]["source_time"] == published["recorded_at"]
        for vehicle in published["snapshot"]["vehicles"]:
            detail = published["details"][vehicle["tr_id"]]
            assert all(detail[key] == value for key, value in vehicle.items())
            assert all(
                parse_source(row["event_at"]) <= cutoff + 45 * SECOND_NS for row in detail["track"]
            )

        # Subsequent ack, late arrival and a new cycle may only change later frames.
        for alert_id in list(engine.alerts):
            engine.acknowledge(alert_id)
        engine._ingest(replace(later, event_time_ns=cutoff - 5 * SECOND_NS, lon=37.9))
        engine.clock.start(cutoff + 90 * SECOND_NS, paused=True)
        await engine._cycle()
        assert engine.view_history.view(cutoff - SECOND_NS) == before
        assert engine.view_history.view(cutoff + 45 * SECOND_NS) == published
        assert engine.snapshot()["clock"]["source_time"] != published["recorded_at"]
    finally:
        release.set()
        await cycle


async def test_history_http_is_read_only_run_scoped_and_errors_are_explicit(engine):
    await engine._cycle()
    at = engine.view_history.metadata()["last"]
    app = create_app()
    app.state.engine = engine
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        params = {"run_id": engine.run_id, "at": at}
        before = (engine.revision, engine.prediction_seq, engine.ml.requests, engine.clock.now_ns())
        results = await asyncio.gather(
            *[client.get("/api/v1/history", params=params) for _ in range(5)]
        )
        assert all(r.status_code == 200 and r.json() == results[0].json() for r in results)
        assert before == (
            engine.revision,
            engine.prediction_seq,
            engine.ml.requests,
            engine.clock.now_ns(),
        )
        for query, status, code in (
            ({**params, "run_id": "old"}, 409, "run_changed"),
            ({**params, "at": "NaT"}, 400, "bad_request"),
            ({**params, "at": "2026-01-06T12:00:00Z"}, 400, "bad_request"),
            ({**params, "at": "2026-01-01"}, 404, "history_unavailable"),
            ({**params, "at": "2027-01-01"}, 404, "history_unavailable"),
        ):
            response = await client.get("/api/v1/history", params=query)
            assert response.status_code == status and response.json()["detail"]["code"] == code
        old_run = engine.run_id
        await engine.reset()
        assert engine.view_history.metadata()["frames"] == 0
        assert (
            await client.get("/api/v1/history", params={"run_id": old_run, "at": at})
        ).status_code == 409
        empty = await client.get("/api/v1/history", params={"run_id": engine.run_id, "at": at})
        assert empty.status_code == 404
        # Response contains no non-JSON values; fixtures and consumers can persist it.
        json.dumps(results[0].json(), allow_nan=False)
