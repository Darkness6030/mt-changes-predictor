/** Types mirroring the Backend OpenAPI contract (docs/API_CONTRACT.md, schema_version 1). */

export type RiskLevel = "green" | "yellow" | "red";

export type PredictionStatus =
  | "ok"
  | "warming_up"
  | "no_target_in_horizon"
  | "no_schedule"
  | "no_mapping"
  | "stale"
  | "ml_unavailable"
  | "invalid_input";

export interface Evidence {
  kind: string;
  text: string;
  value: number | null;
}

export interface Calibration {
  method: string | null;
  status: string;
  report?: string | null;
  fit_rows?: number;
}

export interface ContributionGroup {
  group: string;
  label: string;
  seconds: number;
}

/** Exact split of delay_s from the ML service: base + groups + other = total. */
export interface Explanation {
  base_s: number;
  groups: ContributionGroup[];
  other_s: number;
  total_s: number;
  model_used: string;
}

/** Trip of the target visit, reconstructed from the plan (no trip ids in the data). */
export interface Trip {
  number: number;
  total: number;
  first: boolean;
  last: boolean;
  start_at: string;
  end_at: string;
}

export interface WhatIfTrip {
  trip: number;
  planned_start_at: string;
  delay_without_s: number;
  delay_with_s: number;
  served_by: "vehicle" | "reserve";
}

export interface WhatIfAnswer {
  tr_id: string;
  available: boolean;
  reason?: string;
  prediction_id?: string;
  current_trip?: number;
  trips_total?: number;
  reserve_in_min?: number;
  reserve_ready_at?: string;
  reserve_takes_trip?: number | null;
  trips?: WhatIfTrip[];
  late_trips_without?: number;
  late_trips_with?: number;
  delay_saved_s?: number;
  assumptions?: string[];
}

export interface HotspotEnd {
  target_stop_id: string;
  address: string | null;
  lon: number;
  lat: number;
}

/** A segment where delay grew between consecutive GPS-observed stops, summed over the run. */
export interface Hotspot {
  segment_id: string;
  tr_id: string;
  from: HotspotEnd;
  to: HotspotEnd;
  passes: number;
  gain_total_s: number;
  gain_max_s: number;
  gain_mean_s: number;
  last_at?: string;
}

export interface ExplanationAnswer {
  run_id: string;
  tr_id: string;
  prediction_id: string;
  model_version: string;
  delay_s: number;
  explanation: Explanation;
}

export interface Prediction {
  prediction_id: string;
  run_id: string;
  tr_id: string;
  sample_id: string | null;
  status: PredictionStatus;
  cutoff_t: string;
  computed_at: string;
  trigger?: string;
  target_stop_id: string | null;
  target_planned_at: string | null;
  target_address?: string | null;
  target_lon?: number;
  target_lat?: number;
  horizon_s: number | null;
  delay_s: number | null;
  expected_arrival_at?: string | null;
  risk_level: RiskLevel | null;
  risk_basis: string | null;
  attention?: "delay" | "probability" | null;
  trip?: Trip | null;
  late_probability: number | null;
  late_threshold_s?: number | null;
  calibration?: Calibration | null;
  model_version: string | null;
  model_used?: string | null;
  feature_schema_version?: string;
  cur_dev_s: number | null;
  cur_dev_source: string | null;
  cur_dev_age_s?: number | null;
  data_age_s: number | null;
  prediction_age_s: number | null;
  stale: boolean;
  evidence: Evidence[];
  recommendation: string | null;
  quality_flags: string[];
  detail?: string | null;
}

export interface Position {
  lon: number | null;
  lat: number | null;
  speed_kmh: number | null;
  heading_deg: number | null;
  event_at: string | null;
  age_s: number | null;
}

export interface PlanVisit {
  target_stop_id: string;
  planned_at: string | null;
  address: string | null;
  lon: number;
  lat: number;
  passed?: boolean;
}

export interface Segment {
  segment_id: string;
  kind: "planned_visit_schematic";
  from: PlanVisit;
  to: PlanVisit;
}

export interface CurrentDeviation {
  status: "ok" | "stale" | "unavailable";
  delay_s: number | null;
  risk_level: RiskLevel | null;
  source: "gps_plan" | null;
  observed_at: string | null;
  age_s: number | null;
  max_age_s: number;
  visit_id: string | null;
  visit_address: string | null;
  match_distance_m: number | null;
  reason: "no_match" | "no_valid_position" | "stale_gps" | "estimate_too_old" | null;
}

export interface Vehicle {
  tr_id: string;
  unit_id: string | null;
  mapped: boolean;
  position: Position | null;
  last_event_at: string | null;
  telemetry_age_s: number | null;
  position_age_s: number | null;
  stale: boolean;
  events_in_window: number;
  invalid_gps_fraction: number | null;
  quality_flags: string[];
  has_schedule: boolean;
  next_visit: PlanVisit | null;
  segment?: Segment | null;
  prediction: Prediction | null;
  current_deviation?: CurrentDeviation;
}

export interface VehicleDetail extends Vehicle {
  run_id: string;
  revision: number;
  track: { lon: number | null; lat: number | null; speed_kmh: number | null; event_at: string }[];
  plan: PlanVisit[];
  prediction_history: {
    prediction_id: string;
    cutoff_t: string;
    delay_s: number | null;
    risk_level: RiskLevel | null;
    late_probability: number | null;
    target_stop_id: string | null;
    trigger?: string;
  }[];
}

export interface Alert {
  alert_id: string;
  tr_id: string;
  target_stop_id: string;
  target_planned_at: string;
  risk_level: RiskLevel;
  delay_s: number;
  late_probability: number | null;
  first_alert_at: string;
  first_alert_at_wall: string;
  latest_prediction_at: string;
  planned_lead_s: number;
  updates: number;
  state: "active" | "resolved" | "expired";
  acknowledged_at: string | null;
  acknowledged_from?: string | null;
  attention?: "delay" | "probability" | null;
  trip_edge?: "first" | "last" | null;
  observed_arrival_at?: string | null;
  observed_delay_s?: number | null;
  observed_distance_m?: number | null;
  warning_lead_s?: number | null;
  warning_outcome?: "confirmed" | "within_norm" | "opposite" | null;
  evidence: Evidence[];
}

export interface Clock {
  time_basis: string;
  mode: string;
  source_time: string | null;
  wall_time: string;
  speed: number;
  paused: boolean;
  plan_shift_s: number;
  plan_shift_note?: string;
  source_window?: { first: string | null; last: string | null };
  source_progress?: number;
  events_delivered?: number;
  events_total?: number;
}

export interface RiskPolicy {
  green_max_delay_s: number;
  red_min_delay_s: number;
  early_yellow_s: number;
  late_probability_red: number;
  note: string;
}

export interface Snapshot {
  history?: ViewHistory;
  schema_version: string;
  run_id: string;
  revision: number;
  mode: string;
  server_sent_at: string;
  clock: Clock;
  summary: {
    vehicles: number;
    with_prediction: number;
    attention: number;
    red: number;
    no_prediction: number;
    stale: number;
    unmapped: number;
    alerts_active: number;
    edge_trips_at_risk?: number;
  };
  risk_policy: RiskPolicy;
  vehicles: Vehicle[];
  alerts: Alert[];
  hotspots?: Hotspot[];
  fixture?: boolean;
}

export interface Status {
  history?: ViewHistory;
  backend_version: string;
  run_id: string;
  mode: string;
  ready: boolean;
  clock: Clock;
  split: string;
  plan: { file: string; vehicles: number; visits: number; shift_s: number };
  points: { total: number; delivered: number; skipped: number; predicted: number } | null;
  state: Record<string, unknown>;
  ml: {
    url: string;
    ready: boolean;
    model_version: string | null;
    calibration: Calibration | null;
    requests: number;
    errors: number;
    timeouts: number;
    last_error: string | null;
    round_trip: Percentiles;
    inference: Percentiles;
  };
  ndtp: Record<string, number | string | null> | null;
  replay: {
    events_total: number;
    events_delivered: number;
    finished: boolean;
    control_enabled: boolean;
  } | null;
  performance: {
    history_capture_ms?: Percentiles;
    cycles: number;
    cycle_ms: Percentiles;
    feature_build_ms: Percentiles;
    event_to_prediction_ms: Percentiles;
    ml_round_trip_ms: Percentiles;
    ml_inference_ms: Percentiles;
    predictions: number;
    tick_interval_s: number;
    predict_interval_s: number;
  };
  errors: Record<string, number>;
  unmapped_units: Record<string, number>;
  fixture?: boolean;
}

export interface ViewHistory {
  enabled: boolean;
  first: string | null;
  last: string | null;
  frames: number;
  bytes: number;
  window_s: number;
  sample_interval_s: number;
  max_frames: number;
  max_bytes: number;
  evicted_frames: number;
  skipped_clock_samples: number;
  oversized_frames: number;
}

export interface HistoryFrame {
  requested_at: string;
  recorded_at: string;
  lag_s: number;
  snapshot: Snapshot;
  details: Record<string, VehicleDetail>;
}

export type DemoSource = "replay" | "ndtp_replay" | "emulator";
export interface DemoState {
  enabled: boolean;
  active: DemoSource | "external_ndtp";
  run_id: string;
  busy: boolean;
  phase: "running" | "connecting" | "completed" | "error";
  last_error: string | null;
  speed: number;
  sender: { units: number; frames: number; errors: number } | null;
  sources: { id: DemoSource; available: boolean; reason: string | null; note: string }[];
}

export interface Percentiles {
  samples: number;
  p50_ms: number | null;
  p95_ms: number | null;
  max_ms: number | null;
}

export interface Quality {
  offline: {
    source: string;
    rows: number;
    model_mae_s: number;
    no_hint_mae_s: number;
    cur_dev_mae_s: number;
    zero_mae_s: number;
    late_probability?: Record<string, unknown>;
    note: string;
  } | null;
  early_warning?: {
    observed: number;
    confirmed?: number;
    confirmed_share?: number;
    lead_s?: { min: number; p50: number; max: number };
    lead_at_least_600s_share?: number;
    note: string;
  };
  replay_sidecar: {
    source: string;
    measured_rows: number;
    pending_rows: number;
    mae_s?: number;
    cur_dev_mae_s?: number | null;
    zero_mae_s?: number;
    absolute_error_p50_s?: number;
    lead_time_s?: { min: number; p50: number; max: number; note: string };
    late_probability_brier?: number;
    hint_sources?: Record<string, number>;
    note?: string;
    in_sample?: boolean;
    in_sample_note?: string;
  } | null;
  model: Record<string, unknown> | null;
}
