"""Asyncio TCP server for NDTP. The emulator and our replayer connect to it as clients.

Frames are read with ``readexactly``, so a split or coalesced TCP read cannot be mistaken
for a frame boundary. A malformed frame closes that one connection and lets the client
reconnect; the listener and every other connection keep running.
"""

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from struct import Struct

from transport_backend.clock import wall_iso
from transport_backend.events import TelemetryEvent, from_nav00
from transport_backend.ndtp import (
    NPL_SIZE,
    NdtpError,
    UnknownCell,
    first_nav00,
    parse_frame,
    parse_npl,
)

_HANDSHAKE_BODY = Struct("<HHHIII")


@dataclass
class NdtpCounters:
    connections_total: int = 0
    connections_open: int = 0
    rejected_connections: int = 0
    handshakes: int = 0
    realtime_frames: int = 0
    nav_events: int = 0
    crc_errors: int = 0
    invalid_frames: int = 0
    unknown_cells: int = 0
    truncated_frames: int = 0
    peer_mismatch: int = 0
    read_timeouts: int = 0
    disconnects: int = 0
    bytes_received: int = 0
    units: dict[str, int] = field(default_factory=dict)
    last_frame_at: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict:
        data = {key: value for key, value in vars(self).items() if key != "units"}
        data["units"] = dict(sorted(self.units.items()))
        return data


class NdtpServer:
    def __init__(
        self,
        host: str,
        port: int,
        on_event: Callable[[TelemetryEvent], None],
        *,
        max_connections: int = 64,
        time_offset_s: float = 0.0,
        read_timeout_s: float = 300.0,
        source: str = "ndtp_live",
    ):
        self.host = host
        self.port = port
        self.on_event = on_event
        self.max_connections = max_connections
        self.time_offset_s = time_offset_s
        self.read_timeout_s = read_timeout_s
        self.source = source
        self.counters = NdtpCounters()
        self._server: asyncio.AbstractServer | None = None
        self._clients: dict[asyncio.Task, asyncio.StreamWriter] = {}

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
        # Closing the listener alone leaves accepted connections alive. A source switch
        # must retire their callbacks before a new fleet/clock can receive any packets.
        clients = list(self._clients.items())
        for task, writer in clients:
            writer.close()
            task.cancel()
        if clients:
            await asyncio.gather(*(task for task, _ in clients), return_exceptions=True)
        if server is not None:
            with contextlib.suppress(Exception):
                await server.wait_closed()

    @property
    def running(self) -> bool:
        return self._server is not None and self._server.is_serving()

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("NDTP listener is not started")
        return self._server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        counters = self.counters
        if not self.running or counters.connections_open >= self.max_connections:
            counters.rejected_connections += 1
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return
        counters.connections_total += 1
        counters.connections_open += 1
        task = asyncio.current_task()
        self._clients[task] = writer
        reading_payload = False
        try:
            while True:
                reading_payload = False
                header = await asyncio.wait_for(
                    reader.readexactly(NPL_SIZE), timeout=self.read_timeout_s
                )
                data_size, *_ = parse_npl(header)
                reading_payload = True
                payload = await asyncio.wait_for(
                    reader.readexactly(data_size), timeout=self.read_timeout_s
                )
                counters.bytes_received += NPL_SIZE + data_size
                self._consume(parse_frame(header, payload))
        except asyncio.IncompleteReadError as error:
            # EOF at a frame boundary is a normal disconnect. A complete header
            # followed by no payload is still a truncated frame.
            if reading_payload or error.partial:
                counters.truncated_frames += 1
            counters.disconnects += 1
        except TimeoutError:
            counters.read_timeouts += 1
        except UnknownCell as error:
            counters.unknown_cells += 1
            counters.last_error = str(error)
        except NdtpError as error:
            counters.invalid_frames += 1
            counters.last_error = str(error)
        except (ConnectionResetError, BrokenPipeError):
            counters.disconnects += 1
        finally:
            self._clients.pop(task, None)
            counters.connections_open -= 1
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _consume(self, frame) -> None:
        counters = self.counters
        counters.last_frame_at = wall_iso()
        if not frame.crc_ok:
            counters.crc_errors += 1
            raise NdtpError("CRC mismatch over NPH and body")
        unit = str(frame.peer_address)
        if unit not in counters.units and len(counters.units) >= 1024:
            counters.units.pop(next(iter(counters.units)))
        counters.units[unit] = counters.units.get(unit, 0) + 1
        if frame.is_handshake:
            if len(frame.body) != _HANDSHAKE_BODY.size:
                raise NdtpError("Handshake body must be 18 bytes")
            _high, _low, _flags, peer, _max_size, _reserved = _HANDSHAKE_BODY.unpack(frame.body)
            if peer != frame.peer_address:
                counters.peer_mismatch += 1
                raise NdtpError("Handshake peerAddress differs from the NPL header")
            counters.handshakes += 1
            return
        if not frame.is_realtime:
            raise NdtpError(f"Unsupported NPH service/type {frame.service_id}/{frame.nph_type}")
        counters.realtime_frames += 1
        nav = first_nav00(frame.body)
        if nav is None:
            return
        counters.nav_events += 1
        self.on_event(
            from_nav00(
                nav, frame.peer_address, source=self.source, time_offset_s=self.time_offset_s
            )
        )
