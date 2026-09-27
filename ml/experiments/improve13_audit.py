"""ML-IMPROVE-13 audit: refit the frozen v4 recipe WITHOUT test labels, score test once.

Synthetic copies of test and validate moments (+-purge_margin_s) are removed from training,
so neither the model nor the synthetic families see test/validate outcomes. Writes the
``metrics.json`` consumed by the backend system panel (``test`` section = this audit).

    .venv/bin/python ml/experiments/improve13_audit.py \
        --recipe ml/experiments/improve13-recipe.json --bundle ml/pretrained/v4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transport_ml.data import load_inputs, load_labels, load_points
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.model import DelayModel, write_json
from transport_ml.research import Candidate, candidate_prediction, fit_candidate, load_training_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: <bundle>/metrics.json")
    parser.add_argument(
        "--classifier-metrics", type=Path, default=Path("ml/pretrained/v3/metrics.json")
    )
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    config = FeatureConfig(**recipe["feature_config"])
    spec = {**recipe["training_data"], "splits": ["train"], "protect": ["test", "validate"]}
    _, features, target, data_report = load_training_data(args.data, config, spec)
    fit = np.ones(len(target), dtype=bool)
    points = load_points(args.data, "test")
    traffic, plan = load_inputs(args.data, "test")
    test = FeatureBuilder(traffic, plan, config).transform(points)
    outcome = load_labels(args.data, "test", points)
    predictions = {}
    for mode, members in recipe["models"].items():
        values = np.zeros(len(points))
        for member in members:
            candidate = Candidate(**member["candidate"])
            model, columns = fit_candidate(candidate, features, target, fit)
            values += member["weight"] * candidate_prediction(model, columns, candidate, test)
        predictions[mode] = values

    def mae(values):
        return float(np.mean(np.abs(outcome - values)))

    main_error = np.abs(outcome - predictions["main"])
    DelayModel(args.bundle)  # checksum/schema validation of the audited bundle
    # Classifiers are byte-identical copies from v2/v3 with the same input columns, so their
    # measured test section is carried over verbatim rather than re-labelled as new.
    late_probability = json.loads(args.classifier_metrics.read_text())["test"]["late_probability"]
    frame = points[["tr_id"]].assign(error=main_error)
    report = {
        "protocol": (
            "Audit refit of the frozen recipe on train only (real + synthetic with copies of "
            "test/validate moments purged); the shipped bundle additionally uses test labels, "
            "so its own test error is NOT an out-of-sample number."
        ),
        "training_data": {k: v for k, v in data_report.items() if k != "synthetic_families"},
        "test": {
            "rows": len(points),
            "vehicles": int(points.tr_id.nunique()),
            "model_mae_s": mae(predictions["main"]),
            "no_hint_mae_s": mae(predictions["fallback"]),
            "zero_mae_s": mae(np.zeros(len(points))),
            "cur_dev_mae_s": mae(points.cur_dev_s.to_numpy()),
            "absolute_error_p50_s": float(np.median(main_error)),
            "absolute_error_p90_s": float(np.quantile(main_error, 0.9)),
            "late_probability": late_probability,
            "slices": {
                f"vehicle:{key}": {"rows": len(rows), "mae_s": float(rows.error.mean())}
                for key, rows in frame.groupby("tr_id")
            },
        },
    }
    output = args.output or args.bundle / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    pd.DataFrame({"sample_id": points.sample_id, "target_delay_s": outcome, **predictions}).to_csv(
        output.with_name("audit_test_predictions.csv"), index=False
    )
    print(json.dumps(report["test"] | {"slices": "..."}, indent=2))


if __name__ == "__main__":
    main()
