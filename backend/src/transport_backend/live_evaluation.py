"""Autonomous event-time evaluation on bounded telemetry, without supplied forecast points.

This deliberately does not simulate TCP delivery, wall-clock scheduling or outages. It
measures predictions at a fixed source-time cadence; labels are joined only after all
predictions have been saved. It is not a measured production publication-lead benchmark.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from transport_ml.calibration import reliability
from transport_ml.data import ID_COLUMNS, load_inputs
from transport_ml.features import FeatureBuilder, FeatureConfig, prepare_traffic
from transport_ml.model import DelayModel, sha256, write_json

from transport_backend.clock import SECOND_NS
from transport_backend.events import TelemetryEvent
from transport_backend.state import FleetState, PlanStore


def score_predictions(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Evaluation-only join, including target-level metrics for correlated repeated cutoffs."""
    keys = ["tr_id", "target_stop_id"]
    if labels.duplicated(keys).any():
        raise ValueError("Expected one evaluation outcome per visit")
    scored = predictions.merge(labels[keys + ["target_delay_s"]], on=keys, validate="many_to_one")
    if scored.empty:
        return {"scored_rows": 0, "scored_visits": 0}
    report = {
        "scored_rows": len(scored),
        "scored_visits": int(scored.groupby(keys).ngroups),
        "labelled_visits": len(labels),
        "visit_coverage": float(scored.groupby(keys).ngroups / len(labels)),
        "metrics": {},
    }
    for name in ("candidate", "baseline_estimated", "baseline_no_hint", "zero"):
        errors = np.abs(scored[name] - scored.target_delay_s)
        visit_error = scored.assign(error=errors).groupby(keys).error.mean()
        report["metrics"][name] = {
            "mae_s": float(errors.mean()),
            "visit_weighted_mae_s": float(visit_error.mean()),
            "p90_s": float(errors.quantile(0.9)),
        }
    report["by_vehicle"] = {
        str(key): {
            name: float(np.abs(group[name] - group.target_delay_s).mean())
            for name in report["metrics"]
        }
        for key, group in scored.groupby("tr_id")
    }
    late = (scored.target_delay_s > 120).to_numpy(dtype=float)
    report["late_probability"] = reliability(late, scored.probability.to_numpy())
    lead = scored.horizon_s + scored.target_delay_s
    report["lead_from_cutoff_s"] = {
        "min": float(lead.min()),
        "median": float(lead.median()),
        "p10": float(lead.quantile(0.1)),
        "fraction_at_least_600s": float((lead >= 600).mean()),
        "note": "Observed arrival minus source cutoff, not wall-clock publication time.",
    }
    return report


def evaluate_live(
    root: Path, directory: Path, model_dir: Path, baseline_dir: Path, *, split="test", interval_s=30
) -> dict:
    if interval_s <= 0 or split not in {"train", "test"}:
        raise ValueError("Use labelled train/test and a positive cadence")
    candidate, baseline = DelayModel(model_dir), DelayModel(baseline_dir)
    config = FeatureConfig(**candidate.manifest["feature_config"])
    if not config.schedule_context or candidate.manifest.get("hint_policy") != "supplied_only":
        raise ValueError("Candidate must use separate causal context and supplied-only hints")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "config.json",
        {
            "split": split,
            "interval_s": interval_s,
            "history_window_s": 1800,
            "history_max_events": 900,
            "stale_after_s": 120,
            "candidate": candidate.version,
            "baseline": baseline.version,
            "traffic_sha256": sha256(root / split / "traffic.csv"),
            "evaluator_sha256": sha256(Path(__file__)),
            "label_access": "After inference, for evaluation only; no supplied forecast points.",
        },
    )
    traffic, plan = load_inputs(root, split)
    clean = prepare_traffic(traffic, config).sort_values("event_time", kind="stable")
    state = FleetState(plan=PlanStore(plan), history_window_s=1800, history_max_events=900)
    records = list(clean.itertuples(index=False))
    step = interval_s * SECOND_NS
    first = (int(clean.event_time.min().value) // step) * step
    last = int(clean.event_time.max().value)
    cursor = 0
    rows, timings = [], []
    opportunities, stale, warming = 0, 0, 0
    started = perf_counter()
    for tick, at in enumerate(range(first, last + 1, step)):
        while cursor < len(records) and records[cursor].event_time.value <= at:
            row = records[cursor]
            valid = bool(row.gps_valid)
            state.add(
                TelemetryEvent(
                    source="csv_replay",
                    unit_id=row.tr_id,
                    tr_id=row.tr_id,
                    event_time_ns=int(row.event_time.value),
                    gps_valid=valid,
                    lon=float(row.lon) if valid else None,
                    lat=float(row.lat) if valid else None,
                    speed_kmh=None if pd.isna(row.speed) else float(row.speed),
                    heading_deg=None if pd.isna(row.heading) else float(row.heading),
                ),
                at,
            )
            cursor += 1
        state.trim(at)
        requests = []
        hints = []
        for tr_id in state.plan.visits:
            target = state.plan.target(tr_id, at)
            if target is None:
                continue
            opportunities += 1
            track = state.tracks.get(tr_id)
            if track is None or track.last_valid_ns is None or len(track.events) < 3:
                warming += 1
                continue
            if at - track.last_valid_ns > 120 * SECOND_NS:
                stale += 1
                continue
            requests.append(
                {
                    "sample_id": f"periodic:{tr_id}:{at}",
                    "tr_id": tr_id,
                    "T": pd.Timestamp(at),
                    "target_stop_id": str(target.tt_action_item_id),
                    "target_time_begin": target.time_begin,
                    "cur_dev_s": np.nan,
                }
            )
            hint = state.estimate_deviation(tr_id, at)
            hints.append(np.nan if hint is None else hint.seconds)
        if not requests:
            continue
        points = pd.DataFrame(requests)
        before = perf_counter()
        history = state.history_frame(points.tr_id.tolist())
        features = FeatureBuilder(history, state.plan.plan, config).transform(points)
        new = candidate.infer(features)
        old_features = features[baseline.features].copy()
        old_features["cur_dev_s"] = hints
        old = baseline.predict(old_features)
        old_no_hint = baseline.predict(old_features, no_hint=True)
        timings.append((perf_counter() - before) * 1000)
        horizon = (points.target_time_begin - points["T"]).dt.total_seconds()
        rows.append(
            points[["tr_id", "T", "target_stop_id", "target_time_begin"]].assign(
                horizon_s=horizon,
                candidate=new["delay_s"],
                probability=new["late_probability"],
                baseline_estimated=old,
                baseline_no_hint=old_no_hint,
                estimated_hint=hints,
                zero=0.0,
            )
        )
        if tick % 240 == 0:
            print(
                f"Source {pd.Timestamp(at)}: {sum(len(row) for row in rows)} predictions",
                flush=True,
            )
    if not rows:
        raise ValueError("No eligible autonomous predictions")
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    # First label access occurs after the complete causal inference run.
    labels = pd.read_csv(root / "labels" / f"labels_{split}.csv", dtype=ID_COLUMNS)
    report = {
        **score_predictions(predictions, labels),
        "predictions": len(predictions),
        "opportunities": opportunities,
        "stale": stale,
        "warming_up": warming,
        "runtime_s": perf_counter() - started,
        "feature_and_three_model_batch_ms": {
            "p50": float(np.median(timings)),
            "p95": float(np.quantile(timings, 0.95)),
        },
        "limitations": [
            "Event-time cadence, not a TCP/HTTP/delivery-latency benchmark.",
            "Repeated cutoffs for the same target are dependent; visit-weighted MAE is also given.",
            "Only labelled visits can be scored; unlabelled predictions remain in the archive.",
            "One day shared with training telemetry; not a new-day external benchmark.",
        ],
    }
    write_json(directory / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("ml/pretrained/v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_live(
                args.data,
                args.output,
                args.model,
                args.baseline,
                split=args.split,
                interval_s=args.interval,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
