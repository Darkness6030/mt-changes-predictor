"""Explicit local demo sources; a switch owns and retires its producers and engine.

The official container is provisioned by Compose, controlled through its documented API.
The backend never needs a Docker socket or accepts executable commands/URLs from the UI.
"""

import asyncio
import contextlib
from dataclasses import asdict, replace
from typing import Literal

import httpx

from transport_backend.clock import parse_source
from transport_backend.config import Settings
from transport_backend.engine import Engine
from transport_backend.ndtp_client import ReplayStats, replay_to_ndtp

DemoSource = Literal["replay", "ndtp_replay", "emulator"]
SOURCE_NOTES = {
    "replay": "Исторические CSV: точное время, пауза, ускорение и перемотка.",
    "ndtp_replay": "Исторические CSV через настоящий NDTP/TCP; время квантовано до секунд.",
    "emulator": "Случайная телеметрия организаторов, текущее время. Не оценка качества ML.",
}


class DemoError(ValueError):
    pass


class DemoController:
    def __init__(self, settings: Settings, engine: Engine):
        self.base = settings
        self.engine = engine
        self.active = "replay" if settings.mode == "replay" else "external_ndtp"
        self.lock = asyncio.Lock()
        self.busy = False
        self.sender: asyncio.Task | None = None
        self.sender_stats: ReplayStats | None = None
        self.phase = "running"
        self.last_error: str | None = None
        self.speed = settings.replay_speed
        self.client = httpx.AsyncClient(base_url=settings.emulator_url, timeout=3.0)

    async def availability(self) -> tuple[bool, str | None]:
        if not self.base.demo_enabled:
            return False, "Панель демо отключена в конфигурации Backend"
        try:
            response = await self.client.get("/api/config")
            response.raise_for_status()
            if not isinstance(response.json(), dict):
                raise ValueError("Некорректный ответ эмулятора")
            return True, None
        except (httpx.HTTPError, ValueError):
            return False, (
                "Эмулятор недоступен. Импортируйте образ и запустите "
                "профиль emulator (docs/DEMO.md)."
            )

    async def view(self) -> dict:
        available, reason = await self.availability()
        return {
            "enabled": self.base.demo_enabled,
            "active": self.active,
            "run_id": self.engine.run_id,
            "busy": self.busy,
            "phase": self.phase,
            "last_error": self.last_error,
            "speed": self.speed,
            "sender": None if self.sender_stats is None else asdict(self.sender_stats),
            "sources": [
                {
                    "id": source,
                    "available": available if source == "emulator" else self.base.demo_enabled,
                    "reason": reason if source == "emulator" else None,
                    "note": note,
                }
                for source, note in SOURCE_NOTES.items()
            ],
        }

    async def _configure_emulator(self, units: list[dict]) -> None:
        response = await self.client.post(
            "/api/config",
            json={
                "targetHost": self.base.emulator_target_host,
                "targetPort": self.base.ndtp_port,
                "units": units,
            },
        )
        response.raise_for_status()

    async def _stop_producer(self) -> None:
        if self.sender is not None:
            self.sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.sender
            self.sender = None
        if self.active == "emulator":
            # If unreachable, do not start another source that could mix with a reconnect.
            await self._configure_emulator([])

    async def switch(self, source: DemoSource, speed: float, start_at: str | None) -> Engine:
        if not self.base.demo_enabled:
            raise DemoError("Панель демо отключена в конфигурации Backend")
        if self.lock.locked():
            raise DemoError("Уже выполняется команда источника, дождитесь завершения")
        async with self.lock:
            if start_at is not None:
                parse_source(start_at)  # Validate before retiring the current run.
            if source == "emulator":
                available, reason = await self.availability()
                if not available:
                    raise DemoError(reason)
            settings = replace(
                self.base,
                mode="replay" if source == "replay" else "ndtp",
                use_points=source == "replay",
                plan_shift_s=0.0,
                plan_shift_auto=source == "emulator",
                ndtp_time_offset_s=0.0,
                replay_start=start_at or self.base.replay_start,
                replay_speed=speed,
                replay_autostart=True,
            )
            # Loading/validating data happens while the previous run is still intact.
            candidate = await asyncio.to_thread(Engine, settings)
            previous = self.engine.settings
            self.busy = True
            self.last_error = None
            retired = False
            try:
                await self._stop_producer()
                await self.engine.stop()
                retired = True
                await candidate.start()
                self.engine = candidate
                self.sender_stats = None
                self.speed = speed
                self.phase = "running"
                if source == "emulator":
                    units = [
                        {"unitId": int(unit), "intervalMs": 3000, "autoGenerate": True, "cells": []}
                        for unit in sorted(candidate.mapping)[:2]
                    ]
                    await self._configure_emulator(units)
                elif source == "ndtp_replay":
                    self.phase = "connecting"
                    self.sender = asyncio.create_task(
                        self._send(candidate, start_at), name="demo-ndtp-sender"
                    )
                self.active = source
                return candidate
            except Exception as exc:
                await candidate.stop()
                if not retired:
                    self.last_error = (
                        "Не удалось остановить предыдущий источник; переключение отменено"
                    )
                    raise DemoError(self.last_error) from exc
                if source == "emulator":
                    with contextlib.suppress(httpx.HTTPError):
                        await self._configure_emulator([])
                # Recover to an empty CSV run, never resume a partially switched source.
                fallback = Engine(
                    replace(
                        previous,
                        mode="replay",
                        use_points=True,
                        plan_shift_auto=False,
                        plan_shift_s=0.0,
                    )
                )
                await fallback.start()
                self.engine = fallback
                self.active = "replay"
                self.phase = "error"
                self.last_error = "Не удалось запустить источник; восстановлен CSV replay"
                raise DemoError(self.last_error) from exc
            finally:
                self.busy = False

    async def _send(self, engine: Engine, start_at: str | None) -> None:
        try:
            self.phase = "running"
            self.sender_stats = await replay_to_ndtp(
                data_root=self.base.data_root,
                split=self.base.split,
                host="127.0.0.1",
                port=engine.ndtp.bound_port,
                speed=self.speed,
                start_at=start_at or self.base.replay_start,
            )
            self.phase = "error" if self.sender_stats.errors else "completed"
            if self.sender_stats.errors:
                self.last_error = "NDTP-отправитель завершился с ошибками подключения"
        except asyncio.CancelledError:
            raise
        except Exception:
            self.phase = "error"
            self.last_error = "Не удалось прочитать или отправить историческую телеметрию"

    async def close(self) -> None:
        try:
            with contextlib.suppress(httpx.HTTPError):
                await self._stop_producer()
        finally:
            await self.engine.stop()
            await self.client.aclose()
