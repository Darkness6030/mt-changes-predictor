"""ML-IMPROVE-18: pre-registered checks against the frozen v3 recipe.

Inner train folds of the improve12 protocol only; outer vehicles, test and validate are
never loaded. Run from the repository root:

    .venv/bin/python ml/experiments/improve18_experiments.py improve18-results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import transport_ml.research as research
from transport_ml.data import load_labels
from transport_ml.research import (
    Candidate,
    candidate_prediction,
    fit_candidate,
    inner_folds,
    load_research_data,
)

ROOT = Path("dataset")
PROTOCOL = json.loads(Path("ml/experiments/improve12-protocol.json").read_text())
RECIPE = json.loads(Path("ml/experiments/improve12-recipe.json").read_text())
EXTRA = ["eta_slack_s", "speed_ratio", "hint_minus_arrival_s", "hint_minus_match_s"]


def physics(features):
    """Derived only from causal features already computed at T."""
    result = features.copy()
    speed = result.gps_speed_600s_kmh.where(result.gps_speed_600s_kmh > 3)
    remaining = result.plan_remaining_m.fillna(result.target_distance_m)
    required = result.plan_required_speed_kmh.where(result.plan_required_speed_kmh > 1)
    result["eta_slack_s"] = remaining / (speed / 3.6) - result.horizon_s
    result["speed_ratio"] = result.gps_speed_600s_kmh / required
    result["hint_minus_arrival_s"] = result.cur_dev_s - result.arrival_deviation_s
    result["hint_minus_match_s"] = result.cur_dev_s - result.match_deviation_s
    return result.replace([np.inf, -np.inf], np.nan)


def out_of_fold(features, target, folds, members, seeds=(None,)):
    predictions = {}
    for name, fit, valid, cutoff in folds:
        total = np.zeros(int(valid.sum()))
        for member in members:
            for seed in seeds:
                config = dict(member["candidate"])
                if seed is not None:
                    config["seed"] = seed
                candidate = Candidate(**config)
                model, columns = fit_candidate(candidate, features, target, fit, cutoff=cutoff)
                values = candidate_prediction(model, columns, candidate, features.loc[valid])
                total += member["weight"] / len(seeds) * values
        predictions[name] = (np.flatnonzero(valid), total)
    return predictions


def score(predictions, target):
    def pooled(kind):
        errors = [
            np.abs(target[index] - values)
            for name, (index, values) in predictions.items()
            if name.startswith(kind)
        ]
        return float(np.concatenate(errors).mean())

    group, time = pooled("group"), pooled("time")
    return [round(group, 4), round(time, 4), round((group + time) / 2, 4)]


def main(output: Path) -> None:
    points, features, _ = load_research_data(ROOT)
    inner = ~points.tr_id.isin(PROTOCOL["outer_vehicles"])
    points = points[inner].reset_index(drop=True)
    features = features[inner].reset_index(drop=True)
    target = load_labels(ROOT, "train", points)
    folds = inner_folds(points, target, PROTOCOL)
    original = research.feature_columns

    def with_physics(frame, kind, mode):
        columns = original(frame, kind, mode)
        allowed = [
            name
            for name in EXTRA
            if name in frame
            and name not in columns
            and not (mode == "fallback" and name.startswith("hint"))
        ]
        return columns + allowed

    results, base = {}, {}
    for mode in ("main", "fallback"):
        members = RECIPE["models"][mode]
        base[mode] = out_of_fold(features, target, folds, members)
        results[f"{mode}:v3"] = score(base[mode], target)
        research.feature_columns = with_physics
        try:
            extended = out_of_fold(physics(features), target, folds, members)
        finally:
            research.feature_columns = original
        results[f"{mode}:v3+physics"] = score(extended, target)
        seeds = out_of_fold(features, target, folds, members, seeds=(42, 7, 101))
        results[f"{mode}:v3 x3 seeds"] = score(seeds, target)
    main_oof, fallback_oof = base["main"], base["fallback"]
    for weight in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        blend = {
            name: (index, weight * values + (1 - weight) * fallback_oof[name][1])
            for name, (index, values) in main_oof.items()
        }
        results[f"blend w={weight}"] = score(blend, target)
    output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
