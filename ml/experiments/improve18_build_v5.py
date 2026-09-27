"""Build ml/pretrained/v5: v4 regressors unchanged plus schema-2 late classifiers.

The classifier recipe is the one checked by ``improve18_late_classifier.py``; it is fitted
on all real labelled points (train + test) with Platt scaling on five-fold out-of-fold raw
scores. Regressor files are copied byte for byte, so predictions and the submission equal
v4. Run from the repository root:

    PYTHONPATH=ml/experiments .venv/bin/python ml/experiments/improve18_build_v5.py artifacts/v5
    .venv/bin/python -m transport_ml predict --model artifacts/v5 \
        --output artifacts/v5/submission.csv

The bundle README is written by hand.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from improve18_late_classifier import (
    PARAMS,
    THRESHOLD_S,
    columns_for,
    fit_calibrated,
    labelled,
)
from transport_ml.features import FeatureConfig
from transport_ml.model import DelayModel, sha256, write_json

SOURCE = Path("ml/pretrained/v4")
REPORT = "ml/experiments/improve18-late.json"


def main(directory: Path) -> None:
    source = DelayModel(SOURCE)
    config = FeatureConfig(**source.manifest["feature_config"])
    parts = [labelled(split, config) for split in ("train", "test")]
    features = pd.concat([part[1] for part in parts], ignore_index=True)
    if list(features) != source.features:
        raise ValueError("Classifier features differ from the regression manifest")
    late = (np.concatenate([part[2] for part in parts]) > THRESHOLD_S).astype(float)
    shutil.copytree(SOURCE, directory)
    # The v4 submission manifest names the v4 model hash: regenerate it with ``predict``.
    for name in ("submission.csv", "submission.manifest.json", "README.md"):
        (directory / name).unlink()
    metrics = json.loads((directory / "metrics.json").read_text())
    checked = json.loads(Path(REPORT).read_text())
    metrics["test"]["late_probability"] = {
        "threshold_s": THRESHOLD_S,
        "calibration": {"method": "platt", "status": "validated", "report": REPORT},
        "with_hint": checked["main"]["schema2_train_plus_test"],
        "no_hint": checked["fallback"]["schema2_train_plus_test"],
        "estimate": "out_of_fold on 5 random test folds, same as the regression estimate",
        "reference_v2_classifier": {
            "with_hint": checked["main"]["v2_classifier"],
            "no_hint": checked["fallback"]["v2_classifier"],
        },
    }
    write_json(directory / "metrics.json", metrics)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["classifiers"] = {}
    for name, mode in (("late", "main"), ("late_fallback", "fallback")):
        columns = columns_for(features, mode)
        model, platt = fit_calibrated(features[columns], late)
        path = directory / f"{name}.cbm"
        model.save_model(str(path))
        manifest["classifiers"][name] = {
            "features": columns,
            **PARAMS,
            "threshold_s": THRESHOLD_S,
            "trained_on": "real train + test; Platt on five-fold out-of-fold raw scores",
            "calibration": {
                **platt.to_dict(),
                "positive_rows": platt.positive_rows,
                "fold": "five-fold out-of-fold on the fitting rows",
                "status": "validated",
                "report": REPORT,
            },
            "file": path.name,
            "sha256": sha256(path),
        }
    manifest["classifier_provenance"] = {
        "source_manifest_sha256": source.version,
        "note": "Regressors are the v4 files unchanged; classifiers refitted on schema 2.",
    }
    write_json(directory / "manifest.json", manifest)
    restored = DelayModel(directory)
    probability = restored.predict_late_probability(features)
    if not np.isfinite(probability).all():
        raise ValueError("Non-finite probabilities after saving the bundle")
    np.testing.assert_array_equal(restored.predict(features), source.predict(features))
    print(json.dumps({"model_version": restored.version, "rows": len(late)}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
