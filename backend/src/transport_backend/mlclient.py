"""Async HTTP client for the ML service with timeouts, counters and honest degradation.

Backend never falls back to a formula that pretends to be the model: if the service is
unavailable the prediction status becomes ``ml_unavailable`` and the previous prediction
keeps its real age.
"""

import math
from dataclasses import dataclass, field
from time import perf_counter

import httpx


@dataclass
class Latency:
    """Bounded latency samples: keeps memory flat and still reports percentiles."""

    limit: int = 512
    samples: list[float] = field(default_factory=list)

    def add(self, milliseconds: float) -> None:
        self.samples.append(float(milliseconds))
        if len(self.samples) > self.limit:
            del self.samples[: len(self.samples) - self.limit]

    def quantile(self, fraction: float) -> float | None:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]

    def to_dict(self) -> dict:
        return {
            "samples": len(self.samples),
            "p50_ms": self.quantile(0.5),
            "p95_ms": self.quantile(0.95),
            "max_ms": max(self.samples) if self.samples else None,
        }


class MlClient:
    def __init__(self, base_url: str, timeout_s: float = 3.0, batch_size: int = 64):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.batch_size = max(1, batch_size)
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)
        self.model: dict | None = None
        self.ready = False
        self.last_error: str | None = None
        self.requests = 0
        self.errors = 0
        self.timeouts = 0
        self.latency = Latency()
        self.service_latency = Latency()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def refresh_model(self) -> dict | None:
        """Fetch the feature contract once the service is ready; used for schema checks."""
        try:
            response = await self.client.get("/v1/model")
            response.raise_for_status()
            self.model = response.json()
            self.ready = True
            self.last_error = None
        except Exception as error:
            self.ready = False
            self.last_error = f"{type(error).__name__}: {error}"
        return self.model

    async def predict(self, items: list[dict], feature_schema_version: str) -> dict[str, dict]:
        """Return results keyed by ``request_id``; an empty dict means the call failed."""
        results: dict[str, dict] = {}
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            payload = {
                "feature_schema_version": feature_schema_version,
                "items": chunk,
                "explain": False,
            }
            self.requests += 1
            try:
                started = perf_counter()
                response = await self.client.post("/v1/predict", json=payload)
                elapsed_ms = (perf_counter() - started) * 1000
                response.raise_for_status()
                body = response.json()
                self.latency.add(elapsed_ms)
                self.service_latency.add(float(body.get("inference_ms", 0.0)))
                self.ready = True
                self.last_error = None
                for result in body["results"]:
                    results[result["request_id"]] = {
                        **result,
                        "model_version": body["model_version"],
                    }
            except httpx.TimeoutException as error:
                self.timeouts += 1
                self.errors += 1
                self.ready = False
                self.last_error = f"timeout after {self.timeout_s:g} s: {error}"
            except Exception as error:
                self.errors += 1
                self.ready = False
                self.last_error = f"{type(error).__name__}: {error}"
        return results

    async def explain(self, item: dict, feature_schema_version: str) -> dict:
        """One on-demand explanation; raises on any transport or contract error."""
        payload = {
            "feature_schema_version": feature_schema_version,
            "items": [item],
            "explain": True,
        }
        response = await self.client.post("/v1/predict", json=payload)
        response.raise_for_status()
        body = response.json()
        return {**body["results"][0], "model_version": body["model_version"]}

    def to_dict(self) -> dict:
        model = self.model or {}
        return {
            "url": self.base_url,
            "ready": self.ready,
            "model_version": model.get("model_version"),
            "feature_schema_version": model.get("feature_schema_version"),
            "late_threshold_s": model.get("late_threshold_s"),
            "calibration": model.get("calibration"),
            "requests": self.requests,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "last_error": self.last_error,
            "round_trip": self.latency.to_dict(),
            "inference": self.service_latency.to_dict(),
        }
