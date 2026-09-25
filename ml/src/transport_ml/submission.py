"""Generate and reread the exact platform CSV, with no access to validate outcomes."""

from pathlib import Path

import numpy as np
import pandas as pd

from transport_ml.data import load_points
from transport_ml.features import FeatureConfig
from transport_ml.model import DelayModel, sha256, write_json
from transport_ml.training import build_split


def validate_submission(path: Path, expected_ids: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", dtype={"sample_id": str})
    if list(frame) != ["sample_id", "prediction"]:
        raise ValueError("Expected exactly sample_id;prediction")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Missing or duplicate sample_id")
    if len(frame) != len(expected_ids) or set(frame.sample_id) != set(expected_ids):
        raise ValueError("Submission must cover exactly all prediction points")
    if not np.isfinite(pd.to_numeric(frame.prediction, errors="raise")).all():
        raise ValueError("Predictions must be finite")
    return frame


def predict(root: Path, directory: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    model = DelayModel(directory)
    points, features = build_split(
        root, "validate", FeatureConfig(**model.manifest["feature_config"])
    )
    template = validate_submission(root / "sample_submission.csv", points.sample_id)
    predictions = pd.Series(model.predict(features), index=points.sample_id)
    template["prediction"] = template.sample_id.map(predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, sep=";", index=False)
    validate_submission(output, load_points(root, "validate").sample_id)
    report = {
        "rows": len(template),
        "submission_sha256": sha256(output),
        "model_manifest_sha256": sha256(directory / "manifest.json"),
        "input_sha256": {
            name: sha256(root / "validate" / name)
            for name in ("traffic.csv", "schedule_plan.csv", "points.csv")
        },
    }
    write_json(output.with_suffix(".manifest.json"), report)
    return report
