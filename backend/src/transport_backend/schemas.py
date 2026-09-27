"""Typed response models of the dispatcher API, published in OpenAPI (/docs).

They document and validate every field the UI relies on. ``extra="allow"`` keeps additive,
backward-compatible fields (docs/API_CONTRACT.md) from breaking a response; a missing or
wrongly typed documented field fails the request instead of reaching the UI silently.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Risk = Literal["green", "yellow", "red"]


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Evidence(Open):
    kind: str
    text: str
    value: float | None = None


class Calibration(Open):
    method: str | None = None
    status: str
    report: str | None = None
    fit_rows: int | None = None


class Trip(Open):
    """Trip of the target visit, reconstructed from the plan (no trip ids in the data)."""

    number: int
    total: int
    first: bool
    last: bool
    start_at: str
    end_at: str


class Prediction(Open):
    """Latest prediction of one vehicle. ``delay_s`` is null unless status allows a number."""

    prediction_id: str
    run_id: str
    tr_id: str
    sample_id: str | None = None
    status: str = Field(description="ok, warming_up, no_target_in_horizon, stale, …")
    cutoff_t: str | None = None
    computed_at: str | None = None
    trigger: str | None = None
    target_stop_id: str | None = None
    target_planned_at: str | None = None
    target_address: str | None = None
    trip: Trip | None = None
    target_lon: float | None = None
    target_lat: float | None = None
    horizon_s: float | None = None
    delay_s: float | None = Field(default=None, description="Signed: + late, − early")
    expected_arrival_at: str | None = None
    risk_level: Risk | None = None
    risk_basis: str | None = None
    attention: Literal["delay", "probability"] | None = None
    late_probability: float | None = Field(default=None, ge=0, le=1)
    late_threshold_s: float | None = None
    calibration: Calibration | None = None
    model_version: str | None = None
    model_used: Literal["main", "fallback"] | None = None
    feature_schema_version: str | None = None
    cur_dev_s: float | None = None
    cur_dev_source: str | None = None
    cur_dev_age_s: float | None = None
    data_age_s: float | None = None
    prediction_age_s: float | None = None
    stale: bool = False
    evidence: list[Evidence] = []
    recommendation: str | None = None
    quality_flags: list[str] = []


class Position(Open):
    lon: float
    lat: float
    speed_kmh: float | None = None
    heading_deg: float | None = None
    event_at: str
    age_s: float


class Visit(Open):
    target_stop_id: str
    planned_at: str
    address: str | None = None
    lon: float
    lat: float


class CurrentDeviation(Open):
    status: Literal["ok", "stale", "unavailable"]
    delay_s: float | None = None
    risk_level: Risk | None = None
    source: str | None = None
    observed_at: str | None = None
    age_s: float | None = None
    max_age_s: float
    visit_id: str | None = None
    match_distance_m: float | None = None
    reason: str | None = None
    visit_address: str | None = None


class VehicleView(Open):
    tr_id: str
    unit_id: str | None = None
    mapped: bool
    position: Position | None = None
    last_event_at: str | None = None
    telemetry_age_s: float | None = None
    position_age_s: float | None = None
    stale: bool
    events_in_window: int
    invalid_gps_fraction: float | None = None
    quality_flags: list[str]
    has_schedule: bool
    next_visit: Visit | None = None
    prediction: Prediction | None = None
    current_deviation: CurrentDeviation
    segment: dict[str, Any] | None = None


class PlanItem(Visit):
    passed: bool


class TrackPoint(Open):
    lon: float
    lat: float
    speed_kmh: float | None = None
    event_at: str


class HistoryItem(Open):
    prediction_id: str
    cutoff_t: str | None = None
    delay_s: float | None = None
    risk_level: Risk | None = None
    late_probability: float | None = None
    target_stop_id: str | None = None
    trigger: str | None = None


class VehicleDetail(VehicleView):
    run_id: str
    revision: int
    track: list[TrackPoint]
    plan: list[PlanItem]
    prediction_history: list[HistoryItem]


class Alert(Open):
    alert_id: str
    tr_id: str
    target_stop_id: str
    target_planned_at: str
    first_alert_at: str
    first_alert_at_wall: str
    planned_lead_s: float
    risk_level: Risk
    delay_s: float
    late_probability: float | None = None
    latest_prediction_at: str
    updates: int
    state: Literal["active", "resolved", "expired"]
    acknowledged_at: str | None = None
    evidence: list[Evidence] = []
    attention: Literal["delay", "probability"] | None = None
    trip_edge: Literal["first", "last"] | None = None
    observed_arrival_at: str | None = None
    observed_delay_s: float | None = None
    observed_distance_m: float | None = None
    warning_lead_s: float | None = None
    warning_outcome: Literal["confirmed", "within_norm", "opposite"] | None = None
    acknowledged_from: str | None = Field(
        default=None, description="Alert whose acknowledgement was carried over"
    )


class Summary(Open):
    vehicles: int
    with_prediction: int
    attention: int
    red: int
    no_prediction: int
    stale: int
    unmapped: int
    alerts_active: int
    edge_trips_at_risk: int = 0


class RiskPolicy(Open):
    green_max_delay_s: float
    red_min_delay_s: float
    early_yellow_s: float
    late_probability_red: float


class Snapshot(Open):
    schema_version: str
    run_id: str
    revision: int
    mode: str
    server_sent_at: str
    clock: dict[str, Any]
    history: dict[str, Any] | None = None
    summary: Summary
    risk_policy: RiskPolicy
    vehicles: list[VehicleView]
    alerts: list[Alert]
    hotspots: list[dict[str, Any]] = Field(
        default=[], description="Segments where delay grew the most (GPS-observed)"
    )


class AlertList(Open):
    run_id: str
    rows: int
    alerts: list[Alert]


class PredictionList(Open):
    run_id: str
    rows: int
    predictions: list[Prediction]


class ContributionGroup(Open):
    group: str
    label: str
    seconds: float


class Explanation(Open):
    base_s: float
    groups: list[ContributionGroup]
    other_s: float
    total_s: float
    model_used: Literal["main", "fallback"]


class ExplanationAnswer(Open):
    run_id: str
    tr_id: str
    prediction_id: str
    model_version: str
    delay_s: float
    explanation: Explanation


class AlertAck(Open):
    alert: Alert
    note: str
