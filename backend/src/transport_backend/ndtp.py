"""NDTP decoding and encoding exactly as specified in the emulator specification.

TCP is a byte stream: a frame is read as 15 header bytes and then ``dataSize`` bytes, never
as "one recv". Unknown cell types are rejected with a diagnostic instead of guessing a
payload length, because the cell header carries no length field.

Reference: ``dataset/docs/Emulator-and-Telematic-Packets-Specification.md`` sections 4-6.
"""

import struct
from dataclasses import dataclass

SIGNATURE = 0x7E7E
NPL_SIZE = 15
NPH_SIZE = 10
NPL_TYPE_NPH = 0x02
MAX_DATA_SIZE = 65535
HANDSHAKE = (0, 100)
REALTIME = (1, 101)
HANDSHAKE_BODY_SIZE = 18
PROTO_VERSION = (6, 2)
NAV00_TYPE = 0
NAV00_SIZE = 26

# Documented payload sizes; a type that is absent here cannot be skipped safely.
CELL_PAYLOAD_SIZES = {0: 26, 2: 26, 8: 6, 10: 37, 15: 50, 16: 8}

_NPL = struct.Struct("<HHHHBIH")
_NPH = struct.Struct("<HHHI")
_NAV00 = struct.Struct("<IIIBBHHHHHBB")
_HANDSHAKE = struct.Struct("<HHHIII")


class NdtpError(ValueError):
    """Malformed frame: the connection is closed and the counter is incremented."""


class UnknownCell(NdtpError):
    pass


def crc16_modbus(data: bytes) -> int:
    """CRC-16/Modbus over NPH header and body, polynomial 0xA001, init 0xFFFF."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def swap_bytes(value: int) -> int:
    """The NPL header stores the CRC with swapped bytes."""
    return ((value & 0xFF) << 8) | (value >> 8)


@dataclass(frozen=True)
class Nav00:
    timestamp: int
    lon: float | None
    lat: float | None
    gps_valid: bool
    speed_kmh: float
    speed_max_kmh: float
    heading_deg: float
    altitude_m: float
    satellites: int
    pdop: int
    flags: int

    @property
    def alarm(self) -> bool:
        return bool(self.flags & 0b10)


@dataclass(frozen=True)
class Frame:
    peer_address: int
    service_id: int
    nph_type: int
    request_id: int
    body: bytes
    crc_ok: bool

    @property
    def is_handshake(self) -> bool:
        return (self.service_id, self.nph_type) == HANDSHAKE

    @property
    def is_realtime(self) -> bool:
        return (self.service_id, self.nph_type) == REALTIME


def parse_nav00(payload: bytes) -> Nav00:
    """Signs come from the flag bits; magnitudes are unsigned and scaled by 1e7."""
    if len(payload) != NAV00_SIZE:
        raise NdtpError(f"Nav00 payload must be {NAV00_SIZE} bytes, got {len(payload)}")
    (
        timestamp,
        longitude,
        latitude,
        flags,
        _battery,
        speed,
        speed_max,
        course,
        _track,
        altitude,
        satellites,
        pdop,
    ) = _NAV00.unpack(payload)
    north = bool(flags & (1 << 5))
    east = bool(flags & (1 << 6))
    valid = bool(flags & (1 << 7))
    lon = (longitude / 1e7) * (1 if east else -1)
    lat = (latitude / 1e7) * (1 if north else -1)
    return Nav00(
        timestamp=timestamp,
        lon=lon,
        lat=lat,
        gps_valid=valid,
        speed_kmh=float(speed),
        speed_max_kmh=float(speed_max),
        heading_deg=float(course),
        altitude_m=float(altitude),
        satellites=satellites,
        pdop=pdop,
        flags=flags,
    )


def parse_cells(body: bytes) -> list[tuple[int, int, bytes]]:
    """Split a realtime body into ``(type, number, payload)`` using documented sizes only."""
    cells: list[tuple[int, int, bytes]] = []
    offset = 0
    while offset < len(body):
        if len(body) - offset < 2:
            raise NdtpError("Truncated cell header")
        cell_type, number = body[offset], body[offset + 1]
        size = CELL_PAYLOAD_SIZES.get(cell_type)
        if size is None:
            raise UnknownCell(
                f"Cell type {cell_type} has no documented payload size; refusing to guess"
            )
        start = offset + 2
        if len(body) < start + size:
            raise NdtpError(f"Truncated payload for cell type {cell_type}")
        cells.append((cell_type, number, body[start : start + size]))
        offset = start + size
    return cells


def first_nav00(body: bytes) -> Nav00 | None:
    for cell_type, _number, payload in parse_cells(body):
        if cell_type == NAV00_TYPE:
            return parse_nav00(payload)
    return None


def parse_npl(header: bytes) -> tuple[int, int, int, int]:
    """Validate the 15-byte NPL header and return ``(data_size, crc, type, peer)``."""
    if len(header) != NPL_SIZE:
        raise NdtpError("NPL header must be exactly 15 bytes")
    signature, data_size, _flags, crc, npl_type, peer, _request = _NPL.unpack(header)
    if signature != SIGNATURE:
        raise NdtpError(f"Bad NPL signature 0x{signature:04X}")
    if npl_type != NPL_TYPE_NPH:
        raise NdtpError(f"Unsupported NPL type {npl_type}")
    if not NPH_SIZE <= data_size <= MAX_DATA_SIZE:
        raise NdtpError(f"Invalid dataSize {data_size}")
    return data_size, crc, npl_type, peer


def parse_frame(header: bytes, payload: bytes) -> Frame:
    """``payload`` is exactly ``dataSize`` bytes: the NPH header plus the body."""
    data_size, crc, _type, peer = parse_npl(header)
    if len(payload) != data_size:
        raise NdtpError("Payload length does not match dataSize")
    service_id, nph_type, _flags, request_id = _NPH.unpack(payload[:NPH_SIZE])
    body = payload[NPH_SIZE:]
    # The emulator sets flags crc=0 but still fills the field, so it is always verified.
    crc_ok = crc == swap_bytes(crc16_modbus(payload))
    return Frame(peer, service_id, nph_type, request_id, body, crc_ok)


def decode_stream(buffer: bytearray) -> tuple[list[Frame], int]:
    """Decode every whole frame in a buffer; returns frames and consumed bytes.

    Used by tests and the historical replayer. The server reads with ``readexactly``
    instead, which removes any need to rescan a partial buffer.
    """
    frames: list[Frame] = []
    offset = 0
    while len(buffer) - offset >= NPL_SIZE:
        header = bytes(buffer[offset : offset + NPL_SIZE])
        data_size, *_ = parse_npl(header)
        end = offset + NPL_SIZE + data_size
        if len(buffer) < end:
            break
        frames.append(parse_frame(header, bytes(buffer[offset + NPL_SIZE : end])))
        offset = end
    return frames, offset


def encode_frame(peer_address: int, service_id: int, nph_type: int, request_id: int, body: bytes):
    """Build a frame the emulator would send; used by the historical replayer and tests."""
    payload = _NPH.pack(service_id, nph_type, 0x0001, request_id) + body
    crc = swap_bytes(crc16_modbus(payload))
    header = _NPL.pack(SIGNATURE, len(payload), 0, crc, NPL_TYPE_NPH, peer_address, 0)
    return header + payload


def encode_handshake(peer_address: int, request_id: int = 1) -> bytes:
    body = _HANDSHAKE.pack(PROTO_VERSION[0], PROTO_VERSION[1], 0, peer_address, MAX_DATA_SIZE, 0)
    if len(body) != HANDSHAKE_BODY_SIZE:
        raise NdtpError("Handshake body must be 18 bytes")
    return encode_frame(peer_address, *HANDSHAKE, request_id, body)


def encode_nav00(
    *,
    timestamp: int,
    lon: float | None,
    lat: float | None,
    gps_valid: bool,
    speed_kmh: float = 0.0,
    heading_deg: float = 0.0,
    altitude_m: float = 0.0,
    satellites: int = 10,
    pdop: int = 1,
) -> bytes:
    """Encode one Nav00 cell, quantising to the wire types of the specification."""
    longitude = 0 if lon is None else int(round(abs(lon) * 1e7))
    latitude = 0 if lat is None else int(round(abs(lat) * 1e7))
    flags = 0
    if lat is None or lat >= 0:
        flags |= 1 << 5
    if lon is None or lon >= 0:
        flags |= 1 << 6
    if gps_valid:
        flags |= 1 << 7
    payload = _NAV00.pack(
        int(timestamp),
        longitude,
        latitude,
        flags,
        0,
        int(round(max(0.0, min(65535.0, speed_kmh)))),
        int(round(max(0.0, min(65535.0, speed_kmh)))),
        int(round(max(0.0, min(65535.0, heading_deg)))) % 65536,
        0,
        int(round(max(0.0, min(65535.0, altitude_m)))),
        satellites,
        pdop,
    )
    return bytes([NAV00_TYPE, 0]) + payload


def encode_realtime(peer_address: int, request_id: int, cells: bytes) -> bytes:
    return encode_frame(peer_address, *REALTIME, request_id, cells)
