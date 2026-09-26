"""Our own historical packet replayer: dataset telemetry encoded as real NDTP frames.

This is explicitly **not** the organisers' emulator. It exists to prove the receiver on
real routes: the official emulator always stamps Nav00 with its own current time and moves
randomly, so it cannot follow a January schedule.

Nav00 carries whole seconds and integer speed, so the wire form is a quantisation of the
CSV. The official submission keeps using the exact CSV; equality after quantisation is not
claimed anywhere.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from transport_ml.data import ID_COLUMNS
from transport_ml.features import FeatureConfig, prepare_traffic

from transport_backend.clock import SECOND_NS, parse_source
from transport_backend.ndtp import encode_handshake, encode_nav00, encode_realtime

TRAFFIC_WITH_UNIT = [
    "tr_id",
    "unit_id",
    "event_time",
    "location_valid",
    "lon",
    "lat",
    "speed",
    "heading",
]


@dataclass
class ReplayStats:
    units: int = 0
    frames: int = 0
    bytes_sent: int = 0
    reconnects: int = 0
    errors: int = 0


async def _send_unit(
    host: str,
    port: int,
    unit_id: str,
    rows: pd.DataFrame,
    speed: float,
    offset_s: float,
    stats: ReplayStats,
    origin_ns: int,
    started: float,
) -> None:
    reader = writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(encode_handshake(int(unit_id)))
        await writer.drain()
        await asyncio.sleep(0.2)  # The specification pauses 200 ms after the handshake.
        request_id = 2
        loop = asyncio.get_running_loop()
        for row in rows.itertuples():
            event_ns = int(row.event_time.value)
            # All units share one virtual timeline. Pacing each unit by its own gaps would let
            # a sparse device race ahead in source time and age the others out of the window.
            due = started + (event_ns - origin_ns) / SECOND_NS / speed
            await asyncio.sleep(max(0.0, due - loop.time()))
            cell = encode_nav00(
                timestamp=int(round(event_ns / SECOND_NS - offset_s)),
                lon=None if pd.isna(row.lon) else float(row.lon),
                lat=None if pd.isna(row.lat) else float(row.lat),
                gps_valid=bool(row.gps_valid),
                speed_kmh=0.0 if pd.isna(row.speed) else float(row.speed),
                heading_deg=0.0 if pd.isna(row.heading) else float(row.heading),
            )
            frame = encode_realtime(int(unit_id), request_id, cell)
            request_id += 1
            writer.write(frame)
            await writer.drain()
            stats.frames += 1
            stats.bytes_sent += len(frame)
    except Exception:
        stats.errors += 1
        raise
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                stats.errors += 1


async def replay_to_ndtp(
    *,
    data_root: Path,
    split: str,
    host: str,
    port: int,
    speed: float = 30.0,
    start_at: str | None = None,
    duration_s: float | None = None,
    vehicles: list[str] | None = None,
    time_offset_s: float = 0.0,
) -> ReplayStats:
    """Stream a chosen slice of the dataset to a NDTP server, one connection per unit."""
    path = data_root / split / "traffic.csv"
    raw = pd.read_csv(path, usecols=TRAFFIC_WITH_UNIT, dtype=ID_COLUMNS)
    units = raw[["tr_id", "unit_id"]].dropna().astype(str).drop_duplicates()
    unit_by_vehicle = dict(zip(units.tr_id, units.unit_id, strict=True))
    clean = prepare_traffic(raw.drop(columns=["unit_id"]), FeatureConfig())
    if vehicles:
        clean = clean[clean.tr_id.isin(vehicles)]
    if start_at:
        start_ns = parse_source(start_at)
        clean = clean[clean.event_time.astype("int64") >= start_ns]
        if duration_s is not None:
            end_ns = start_ns + int(duration_s * SECOND_NS)
            clean = clean[clean.event_time.astype("int64") <= end_ns]
    clean = clean.sort_values(["event_time", "tr_id"], kind="stable")
    stats = ReplayStats()
    if clean.empty:
        return stats
    origin_ns = int(clean.event_time.iloc[0].value)
    # The 200 ms handshake pause of the specification happens before the first packet.
    started = asyncio.get_running_loop().time() + 0.5
    tasks = []
    for tr_id, rows in clean.groupby("tr_id", sort=True):
        unit_id = unit_by_vehicle.get(str(tr_id))
        if unit_id is None:
            continue
        stats.units += 1
        tasks.append(
            asyncio.create_task(
                _send_unit(
                    host, port, unit_id, rows, speed, time_offset_s, stats, origin_ns, started
                )
            )
        )
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return stats
