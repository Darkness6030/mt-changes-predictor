"""Bounded, immutable journal of published NDTP views, independent of ingestion.

Frames are sampled after a processing cycle, using publication source time (not an
earlier prediction cutoff). Reads never run ML, interpolate, or backfill late events.
Compressed bytes sever every reference to mutable engine state and bound storage.
"""

import json
import zlib
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from transport_backend.clock import SECOND_NS, format_source
from transport_backend.config import Settings


class HistoryUnavailable(ValueError):
    """The requested point is outside this run's retained publication journal."""


@dataclass(frozen=True)
class Frame:
    source_ns: int
    payload: bytes


class ViewHistory:
    def __init__(self, settings: Settings):
        self.enabled = settings.mode == "ndtp"
        self.window_s = settings.view_history_window_s
        self.interval_s = settings.view_history_interval_s
        self.max_frames = settings.view_history_max_frames
        self.max_bytes = settings.view_history_max_bytes
        self._frames: deque[Frame] = deque()
        self._bytes = 0
        self._last_sample_wall = -float("inf")
        self._last_source_ns: int | None = None
        self._lock = Lock()
        self.evicted = 0
        self.skipped_clock = 0
        self.oversized = 0

    def due(self, source_ns: int, *, not_before_ns: int = 0) -> bool:
        # Only the engine loop calls due/append. Readers only access immutable bytes.
        if not self.enabled or monotonic() - self._last_sample_wall < self.interval_s:
            return False
        self._last_sample_wall = monotonic()
        if source_ns < not_before_ns or (
            self._last_source_ns is not None and source_ns <= self._last_source_ns
        ):
            self.skipped_clock += 1
            return False
        return True

    def append(self, source_ns: int, snapshot: dict, details: dict) -> None:
        payload = zlib.compress(
            json.dumps(
                {"snapshot": snapshot, "details": details},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            level=1,
        )
        with self._lock:
            if self._last_source_ns is not None and source_ns <= self._last_source_ns:
                return
            self._last_source_ns = source_ns
            cutoff = source_ns - int(self.window_s * SECOND_NS)
            while self._frames and self._frames[0].source_ns < cutoff:
                self._evict()
            if len(payload) > self.max_bytes:
                self.oversized += 1
                return
            while self._frames and (
                len(self._frames) >= self.max_frames or self._bytes + len(payload) > self.max_bytes
            ):
                self._evict()
            self._frames.append(Frame(source_ns, payload))
            self._bytes += len(payload)

    def _evict(self) -> None:
        self._bytes -= len(self._frames.popleft().payload)
        self.evicted += 1

    def view(self, at_ns: int) -> dict:
        with self._lock:
            if not self._frames:
                raise HistoryUnavailable("История ещё не накоплена: ожидается первый снимок потока")
            if at_ns < self._frames[0].source_ns or at_ns > self._frames[-1].source_ns:
                raise HistoryUnavailable("Нет записанных данных в выбранный момент")
            frames = tuple(self._frames)
            frame = frames[bisect_right(frames, at_ns, key=lambda item: item.source_ns) - 1]
        # Decompression runs in the HTTP worker, outside the lock and ingestion loop.
        return {
            **json.loads(zlib.decompress(frame.payload)),
            "requested_at": format_source(at_ns),
            "recorded_at": format_source(frame.source_ns),
            "lag_s": (at_ns - frame.source_ns) / SECOND_NS,
        }

    def metadata(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "first": format_source(self._frames[0].source_ns) if self._frames else None,
                "last": format_source(self._frames[-1].source_ns) if self._frames else None,
                "frames": len(self._frames),
                "bytes": self._bytes,
                "window_s": self.window_s,
                "sample_interval_s": self.interval_s,
                "max_frames": self.max_frames,
                "max_bytes": self.max_bytes,
                "evicted_frames": self.evicted,
                "skipped_clock_samples": self.skipped_clock,
                "oversized_frames": self.oversized,
            }
