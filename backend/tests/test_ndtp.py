"""Binary protocol tests: golden bytes, CRC, TCP framing, reconnect and bad input."""

import asyncio

import pytest
from transport_backend.events import from_nav00
from transport_backend.ndtp import (
    CELL_PAYLOAD_SIZES,
    NPL_SIZE,
    NdtpError,
    UnknownCell,
    crc16_modbus,
    decode_stream,
    encode_handshake,
    encode_nav00,
    encode_realtime,
    first_nav00,
    parse_cells,
    parse_frame,
    parse_nav00,
    swap_bytes,
)
from transport_backend.ndtp_server import NdtpServer

UNIT = 1166336


def nav_cell(**overrides) -> bytes:
    defaults = {
        "timestamp": 1767700800,
        "lon": 37.6173210,
        "lat": 55.7551234,
        "gps_valid": True,
        "speed_kmh": 21.0,
        "heading_deg": 94.0,
    }
    return encode_nav00(**{**defaults, **overrides})


def test_crc16_modbus_reference_value():
    # Standard CRC-16/Modbus check value for "123456789".
    assert crc16_modbus(b"123456789") == 0x4B37
    assert swap_bytes(0x4B37) == 0x374B


def test_handshake_and_realtime_round_trip():
    stream = encode_handshake(UNIT) + encode_realtime(UNIT, 2, nav_cell())
    frames, consumed = decode_stream(bytearray(stream))
    assert consumed == len(stream)
    assert [frame.is_handshake for frame in frames] == [True, False]
    assert all(frame.crc_ok and frame.peer_address == UNIT for frame in frames)
    assert len(frames[0].body) == 18
    nav = first_nav00(frames[1].body)
    assert nav.lon == pytest.approx(37.6173210)
    assert nav.lat == pytest.approx(55.7551234)
    assert nav.gps_valid and nav.speed_kmh == 21.0 and nav.heading_deg == 94.0


def test_signs_and_validity_come_from_the_flag_bits():
    frames, _ = decode_stream(bytearray(encode_realtime(UNIT, 3, nav_cell(lon=-37.5, lat=-55.5))))
    nav = first_nav00(frames[0].body)
    assert nav.lon < 0 and nav.lat < 0
    invalid = parse_nav00(nav_cell(gps_valid=False)[2:])
    assert not invalid.gps_valid
    event = from_nav00(invalid, UNIT, source="ndtp_live")
    assert event.lon is None and "invalid_gps" in event.quality_flags
    zero = from_nav00(parse_nav00(nav_cell(lon=0.0, lat=0.0)[2:]), UNIT, source="ndtp_live")
    assert not zero.gps_valid and "zero_position" in zero.quality_flags


def test_partial_buffer_is_not_decoded_as_a_frame():
    stream = encode_realtime(UNIT, 4, nav_cell())
    for cut in (1, NPL_SIZE, NPL_SIZE + 5, len(stream) - 1):
        frames, consumed = decode_stream(bytearray(stream[:cut]))
        assert frames == [] and consumed == 0
    frames, consumed = decode_stream(bytearray(stream + stream[: len(stream) // 2]))
    assert len(frames) == 1 and consumed == len(stream)


def test_bad_crc_signature_and_size_are_rejected():
    stream = bytearray(encode_realtime(UNIT, 5, nav_cell()))
    broken = bytearray(stream)
    broken[6] ^= 0xFF  # Corrupt the CRC field only.
    assert not parse_frame(bytes(broken[:NPL_SIZE]), bytes(broken[NPL_SIZE:])).crc_ok
    signature = bytearray(stream)
    signature[0] = 0x00
    with pytest.raises(NdtpError, match="signature"):
        decode_stream(signature)
    size = bytearray(stream)
    size[2:4] = (5).to_bytes(2, "little")
    with pytest.raises(NdtpError, match="dataSize"):
        decode_stream(size)


def test_unknown_and_truncated_cells_are_refused():
    assert CELL_PAYLOAD_SIZES[0] == 26
    with pytest.raises(UnknownCell, match="no documented payload size"):
        parse_cells(bytes([99, 0, 1, 2, 3]))
    with pytest.raises(NdtpError, match="Truncated payload"):
        parse_cells(nav_cell()[:-3])
    with pytest.raises(NdtpError, match="Truncated cell header"):
        parse_cells(b"\x00")


def test_several_documented_cells_in_one_packet():
    body = nav_cell() + bytes([8, 0]) + bytes(CELL_PAYLOAD_SIZES[8]) + bytes([16, 0]) + bytes(8)
    cells = parse_cells(body)
    assert [cell[0] for cell in cells] == [0, 8, 16]
    assert first_nav00(body) is not None


async def start_server(events, **kwargs):
    server = NdtpServer("127.0.0.1", 0, events.append, **kwargs)
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    return server, port


async def test_server_handles_split_writes_coalesced_frames_and_reconnect():
    events = []
    server, port = await start_server(events)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        frame = encode_realtime(UNIT, 2, nav_cell())
        writer.write(encode_handshake(UNIT))
        await writer.drain()
        # One frame split across three writes, then two frames in a single write.
        for chunk in (frame[:4], frame[4:NPL_SIZE], frame[NPL_SIZE:]):
            writer.write(chunk)
            await writer.drain()
            await asyncio.sleep(0.01)
        writer.write(frame + encode_realtime(UNIT, 3, nav_cell(timestamp=1767700812)))
        await writer.drain()
        await asyncio.sleep(0.2)
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.1)
        assert len(events) == 3
        assert server.counters.handshakes == 1
        assert server.counters.connections_open == 0
        # A reconnect repeats the handshake and keeps the listener alive.
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(encode_handshake(UNIT, request_id=9) + frame)
        await writer.drain()
        await asyncio.sleep(0.2)
        writer.close()
        await writer.wait_closed()
        assert server.counters.handshakes == 2
        assert server.counters.connections_total == 2
        assert len(events) == 4
    finally:
        await server.stop()


async def test_bad_crc_closes_one_connection_without_stopping_the_listener():
    events = []
    server, port = await start_server(events)
    try:
        broken = bytearray(encode_realtime(UNIT, 2, nav_cell()))
        broken[6] ^= 0xFF
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(bytes(broken))
        await writer.drain()
        await asyncio.sleep(0.2)
        assert server.counters.crc_errors == 1
        assert server.running and not events
        _, writer2 = await asyncio.open_connection("127.0.0.1", port)
        writer2.write(encode_realtime(UNIT, 2, nav_cell()))
        await writer2.drain()
        await asyncio.sleep(0.2)
        assert len(events) == 1
        for opened in (writer, writer2):
            opened.close()
    finally:
        await server.stop()


async def test_connection_cap_rejects_extra_clients():
    events = []
    server, port = await start_server(events, max_connections=1)
    try:
        _, first = await asyncio.open_connection("127.0.0.1", port)
        first.write(encode_handshake(UNIT))
        await first.drain()
        await asyncio.sleep(0.1)
        _, second = await asyncio.open_connection("127.0.0.1", port)
        second.write(encode_realtime(UNIT, 2, nav_cell()))
        await asyncio.sleep(0.2)
        assert server.counters.rejected_connections == 1
        assert not events
        first.close()
        second.close()
    finally:
        await server.stop()


def _replay_rows(count: int):
    import pandas as pd

    start = pd.Timestamp("2026-01-06 12:00:00")
    return pd.DataFrame(
        {
            "event_time": [start + pd.Timedelta(seconds=15 * i) for i in range(count)],
            "lon": [37.6 + 0.001 * i for i in range(count)],
            "lat": [55.7] * count,
            "gps_valid": [True] * count,
            "speed": [20.0] * count,
            "heading": [90.0] * count,
        }
    )


async def test_sender_reconnects_and_resumes_after_a_reset():
    from transport_backend.ndtp_client import ReplayStats, _send_unit

    received: list[int] = []
    connections = 0

    async def handle(reader, writer):
        nonlocal connections
        connections += 1
        buffer = bytearray()
        limit = 3 if connections == 1 else 10_000  # The first connection drops early.
        seen = 0
        while seen <= limit:
            data = await reader.read(4096)
            if not data:
                break
            buffer.extend(data)
            frames, used = decode_stream(buffer)
            del buffer[:used]
            for frame in frames:
                if frame.is_realtime:
                    received.append(parse_nav00(parse_cells(frame.body)[0][2]).timestamp)
                    seen += 1
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    rows = _replay_rows(40)
    stats = ReplayStats()
    loop = asyncio.get_running_loop()
    origin = int(rows.event_time.iloc[0].value)
    try:
        await _send_unit(
            "127.0.0.1",
            port,
            str(UNIT),
            rows,
            1500.0,
            0.0,
            stats,
            origin,
            loop.time(),
            retry_delays_s=(0.05, 0.05, 0.05),
        )
        await asyncio.sleep(0.2)
    finally:
        server.close()
        await server.wait_closed()
    assert connections >= 2 and stats.reconnects >= 1 and stats.errors == 0
    last = int(rows.event_time.iloc[-1].value // 1_000_000_000)
    assert received[-1] == last  # The replay reached the end after the reset.
    # At most the frames in flight at the reset are lost; nothing is invented.
    assert len(set(received)) >= len(rows) - 3
    assert set(received) <= {int(t.value // 1_000_000_000) for t in rows.event_time}


async def test_sender_gives_up_after_the_retry_budget():
    from transport_backend.ndtp_client import ReplayStats, _send_unit

    probe = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()  # Nothing listens on this port any more.
    stats = ReplayStats()
    loop = asyncio.get_running_loop()
    rows = _replay_rows(3)
    with pytest.raises(OSError):
        await _send_unit(
            "127.0.0.1",
            port,
            str(UNIT),
            rows,
            1000.0,
            0.0,
            stats,
            int(rows.event_time.iloc[0].value),
            loop.time(),
            retry_delays_s=(0.01, 0.01),
        )
    assert stats.reconnects == 2 and stats.errors == 1 and stats.frames == 0
