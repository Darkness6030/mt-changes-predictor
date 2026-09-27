"""Environment-driven settings. Every threshold is configuration, not a carrier norm."""

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODES = ("replay", "ndtp")


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str | None) -> Path | None:
    value = os.environ.get(name, default)
    return None if value in (None, "") else Path(value)


@dataclass(frozen=True)
class RiskPolicy:
    """Dispatcher colour thresholds. A product decision, not the official target_class."""

    green_max_delay_s: float = 60.0
    red_min_delay_s: float = 120.0
    early_yellow_s: float = -60.0
    late_probability_red: float = 0.5

    def level(self, delay_s: float) -> str:
        if delay_s > self.red_min_delay_s:
            return "red"
        if delay_s > self.green_max_delay_s or delay_s < self.early_yellow_s:
            return "yellow"
        return "green"

    def basis(self, delay_s: float) -> str:
        if delay_s > self.red_min_delay_s:
            return f"delay_s > {self.red_min_delay_s:g}"
        if delay_s > self.green_max_delay_s:
            return f"delay_s > {self.green_max_delay_s:g}"
        if delay_s < self.early_yellow_s:
            return f"delay_s < {self.early_yellow_s:g} (опережение)"
        return "в пределах допустимого отклонения"

    def attention(self, delay_s: float, late_probability: float | None) -> str | None:
        """Why a dispatcher should look: a risky delay first, else a likely late arrival."""
        if self.level(delay_s) != "green":
            return "delay"
        if late_probability is not None and late_probability >= self.late_probability_red:
            return "probability"
        return None

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "note": "Продуктовые пороги UI, не официальная разметка target_class",
        }


@dataclass(frozen=True)
class Settings:
    mode: str = "replay"
    data_root: Path = Path("dataset")
    split: str = "test"
    use_points: bool | None = None
    labels: Path | None = None
    offline_metrics: Path | None = Path("ml/pretrained/v6/metrics.json")

    ml_url: str = "http://ml:8001"
    ml_timeout_s: float = 3.0
    ml_batch_size: int = 64

    ndtp_host: str = "0.0.0.0"  # noqa: S104 - container service listens on all interfaces
    ndtp_port: int = 9201
    ndtp_max_connections: int = 64
    ndtp_time_offset_s: float = 0.0
    ndtp_read_timeout_s: float = 300.0

    replay_speed: float = 30.0
    replay_start: str | None = None
    replay_autostart: bool = True
    replay_control_enabled: bool = True
    replay_warmup_s: float = 900.0

    history_window_s: float = 1800.0
    history_max_events: int = 900
    stale_after_s: float = 120.0
    current_deviation_max_age_s: float = 300.0
    predict_interval_s: float = 30.0
    tick_interval_s: float = 0.5
    max_predictions_kept: int = 4000
    max_alerts_kept: int = 400
    alert_retention_s: float = 1800.0
    track_points: int = 60

    # UI publication journal, separate from the telemetry window used by ML.
    view_history_window_s: float = 7200.0
    view_history_interval_s: float = 1.0
    view_history_max_frames: int = 7200
    view_history_max_bytes: int = 64 * 1024 * 1024

    plan_shift_s: float = 0.0
    plan_shift_auto: bool = False

    # Local demo control; no Docker socket or arbitrary target from HTTP requests.
    demo_enabled: bool = False
    emulator_url: str = "http://emulator:18080"
    emulator_target_host: str = "backend"

    risk: RiskPolicy = field(default_factory=RiskPolicy)

    def __post_init__(self) -> None:
        if self.use_points is None:
            object.__setattr__(self, "use_points", self.mode == "replay")
        if self.mode not in MODES:
            raise ValueError(f"BACKEND_MODE must be one of {MODES}")
        if self.split not in {"train", "test", "validate"}:
            raise ValueError("BACKEND_SPLIT must be train, test or validate")
        if self.replay_speed <= 0 or self.tick_interval_s <= 0:
            raise ValueError("Replay speed and tick interval must be positive")
        if self.current_deviation_max_age_s <= 0:
            raise ValueError("Current deviation max age must be positive")
        if self.history_window_s < 600:
            raise ValueError("History window must cover the 600 s feature windows")
        for value in (
            self.view_history_window_s,
            self.view_history_interval_s,
            self.view_history_max_frames,
            self.view_history_max_bytes,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("View history limits must be finite and positive")

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.environ.get("BACKEND_MODE", "replay").strip().lower()
        shift = os.environ.get("BACKEND_PLAN_SHIFT_S", "").strip().lower()
        if shift == "":
            # A live NDTP stream may carry today's date for the same timetable: align by
            # whole days to the stream. Replay reads the dataset's own dates.
            shift = "auto" if mode == "ndtp" else "0"
        return cls(
            mode=mode,
            data_root=Path(os.environ.get("BACKEND_DATA_ROOT", "dataset")),
            split=os.environ.get("BACKEND_SPLIT", "test").strip().lower(),
            use_points=(
                None
                if not os.environ.get("BACKEND_USE_POINTS")
                else _bool("BACKEND_USE_POINTS", False)
            ),
            labels=_path("BACKEND_LABELS", None),
            offline_metrics=_path("BACKEND_OFFLINE_METRICS", "ml/pretrained/v6/metrics.json"),
            ml_url=os.environ.get("BACKEND_ML_URL", "http://ml:8001").rstrip("/"),
            ml_timeout_s=_float("BACKEND_ML_TIMEOUT_S", 3.0),
            ml_batch_size=_int("BACKEND_ML_BATCH_SIZE", 64),
            ndtp_host=os.environ.get("BACKEND_NDTP_HOST", "0.0.0.0"),  # noqa: S104
            ndtp_port=_int("BACKEND_NDTP_PORT", 9201),
            ndtp_max_connections=_int("BACKEND_NDTP_MAX_CONNECTIONS", 64),
            ndtp_time_offset_s=_float("BACKEND_NDTP_TIME_OFFSET_S", 0.0),
            replay_speed=_float("BACKEND_REPLAY_SPEED", 30.0),
            replay_start=os.environ.get("BACKEND_REPLAY_START") or None,
            replay_autostart=_bool("BACKEND_REPLAY_AUTOSTART", True),
            replay_control_enabled=_bool("BACKEND_REPLAY_CONTROL", True),
            history_window_s=_float("BACKEND_HISTORY_WINDOW_S", 1800.0),
            history_max_events=_int("BACKEND_HISTORY_MAX_EVENTS", 900),
            view_history_window_s=_float("BACKEND_VIEW_HISTORY_WINDOW_S", 7200.0),
            view_history_interval_s=_float("BACKEND_VIEW_HISTORY_INTERVAL_S", 1.0),
            view_history_max_frames=_int("BACKEND_VIEW_HISTORY_MAX_FRAMES", 7200),
            view_history_max_bytes=_int("BACKEND_VIEW_HISTORY_MAX_BYTES", 64 * 1024 * 1024),
            stale_after_s=_float("BACKEND_STALE_AFTER_S", 120.0),
            current_deviation_max_age_s=_float("BACKEND_CURRENT_DEVIATION_MAX_AGE_S", 300.0),
            predict_interval_s=_float("BACKEND_PREDICT_INTERVAL_S", 30.0),
            tick_interval_s=_float("BACKEND_TICK_INTERVAL_S", 0.5),
            plan_shift_s=0.0 if shift in {"auto", ""} else float(shift),
            plan_shift_auto=shift == "auto",
            demo_enabled=_bool("BACKEND_DEMO_ENABLED", False),
            emulator_url=os.environ.get("BACKEND_EMULATOR_URL", "http://emulator:18080").rstrip(
                "/"
            ),
            emulator_target_host=os.environ.get("BACKEND_EMULATOR_TARGET_HOST", "backend"),
            risk=RiskPolicy(
                green_max_delay_s=_float("BACKEND_GREEN_MAX_DELAY_S", 60.0),
                red_min_delay_s=_float("BACKEND_RED_MIN_DELAY_S", 120.0),
                early_yellow_s=_float("BACKEND_EARLY_YELLOW_S", -60.0),
                late_probability_red=_float("BACKEND_LATE_PROBABILITY_RED", 0.5),
            ),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["risk"] = self.risk.to_dict()
        for key in ("data_root", "labels", "offline_metrics"):
            data[key] = None if data[key] is None else str(data[key])
        return data
