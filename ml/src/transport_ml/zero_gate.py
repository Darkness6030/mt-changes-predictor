"""Hurdle gate for exactly-zero delays (ML-IMPROVE-15).

In the labelled data 72% of targets are exactly 0 s when the supplied ``cur_dev_s`` is exactly
0 (train and test alike), against 4% otherwise. GPS then typically matches no planned stop
(``arrival_matches`` = 0). For MAE the optimal forecast is the conditional median, which is
exactly zero once ``P(delay == 0) > 0.5``; boosted regressors average such rows with the
non-zero ones instead. A classifier on the same causal features estimates that probability.

Only causal inputs are used: the shared schema features (telemetry <= T, plan, supplied
hint). Schedule facts and fact metadata such as ``manual_fill`` are never read.
"""

from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from transport_ml.features import FeatureConfig
from transport_ml.model import sha256
from transport_ml.research import load_training_data

THRESHOLD = 0.5  # conditional median is zero above this probability; not tuned on a score


def fit_zero_gate(root: Path, config: FeatureConfig, training_data: dict, seed: int = 42):
    points, features, target, report = load_training_data(root, config, training_data)
    columns = list(features)
    model = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=5,
        random_seed=seed,
        thread_count=4,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(features[columns], (target == 0).astype(int))
    report = {k: v for k, v in report.items() if k != "synthetic_families"}
    report["zero_rate"] = float(np.mean(target == 0))
    return model, columns, report


def save_zero_gate(model, columns, report, directory: Path, name: str = "zero_gate.cbm") -> dict:
    path = directory / name
    model.save_model(str(path))
    return {
        "file": name,
        "sha256": sha256(path),
        "features": columns,
        "threshold": THRESHOLD,
        "mode": "main",
        "depth": 4,
        "iterations": 300,
        "training_data": report,
    }
