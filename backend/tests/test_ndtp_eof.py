"""Distinguish a clean TCP EOF from a partially delivered NDTP frame."""

import asyncio

import pytest
from transport_backend.ndtp import NPL_SIZE, encode_handshake
from transport_backend.ndtp_server import NdtpServer


@pytest.mark.parametrize("cut", [0, 1, NPL_SIZE - 1, NPL_SIZE, NPL_SIZE + 1, None])
async def test_eof_counts_only_incomplete_frames(cut):
    server = NdtpServer("127.0.0.1", 0, lambda event: None)
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        frame = encode_handshake(1166336)
        writer.write(frame if cut is None else frame[:cut])
        await writer.drain()
        writer.write_eof()
        assert await asyncio.wait_for(reader.read(), timeout=2) == b""
        writer.close()
        await writer.wait_closed()
        assert server.counters.disconnects == 1
        assert server.counters.connections_open == 0
        assert server.counters.truncated_frames == int(cut not in (0, None))
        assert server.counters.handshakes == int(cut is None)
    finally:
        await server.stop()
