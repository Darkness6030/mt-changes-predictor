"""ML-IMPROVE-18: honest check of a schema-2 late classifier against the v2 classifiers.

The candidate is fixed before looking at any result: CatBoost Logloss on the 73 schema-2
features, depth 4, 300 trees, learning rate 0.05, L2 5, seed 42; Platt scaling fitted on
inner five-fold out-of-fold raw scores of the fitting rows only. Held-out test folds are
the same five random folds as ``improve18_test_refit.py``. The v2 classifiers never saw
test, so their test probabilities are a fair reference. Run from the repository root:

    .venv/bin/python ml/experiments/improve18_late_classifier.py improve18-late.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from transport_ml.calibration import fit_platt, reliability
from transport_ml.data import load_inputs, load_labels, load_points
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.model import DelayModel

ROOT = Path("dataset")
SEED = 20260927
THRESHOLD_S = 120.0
PARAMS = {"depth": 4, "iterations": 300, "learning_rate": 0.05, "l2_leaf_reg": 5}


def labelled(split: str, config: FeatureConfig):
    points = load_points(ROOT, split)
    points = points[points.tr_id.astype("int64") < 9_000_000].reset_index(drop=True)
    traffic, plan = load_inputs(ROOT, split)
    features = FeatureBuilder(traffic, plan, config).transform(points)
    return points, features, load_labels(ROOT, split, points)


def classifier() -> CatBoostClassifier:
    return CatBoostClassifier(
        **PARAMS,
        loss_function="Logloss",
        random_seed=42,
        thread_count=4,
        allow_writing_files=False,
        verbose=False,
    )


def raw(model, frame) -> np.ndarray:
    return np.asarray(model.predict(frame, prediction_type="RawFormulaVal"), dtype=float)


def fit_calibrated(x: pd.DataFrame, y: np.ndarray, seed: int = SEED):
    """Trees on all rows; Platt on inner out-of-fold scores of the same rows."""
    fold = np.random.default_rng(seed).integers(0, 5, len(y))
    scores = np.zeros(len(y))
    for index in range(5):
        inner = classifier().fit(x[fold != index], y[fold != index])
        scores[fold == index] = raw(inner, x[fold == index])
    return classifier().fit(x, y), fit_platt(scores, y)


def columns_for(features: pd.DataFrame, mode: str) -> list[str]:
    return [name for name in features if mode == "main" or name != "cur_dev_s"]


def main(output: Path) -> None:
    config = FeatureConfig(schedule_context=True)
    _, train_x, train_y = labelled("train", config)
    test_points, test_x, test_y = labelled("test", config)
    x = pd.concat([train_x, test_x], ignore_index=True)
    late = (np.r_[train_y, test_y] > THRESHOLD_S).astype(float)
    is_test = np.r_[np.zeros(len(train_y), bool), np.ones(len(test_y), bool)]
    test_rows = np.flatnonzero(is_test)
    fold = np.random.default_rng(SEED).integers(0, 5, len(test_y))

    reference = DelayModel(Path("ml/pretrained/v2"))
    traffic, plan = load_inputs(ROOT, "test")
    v2_x = FeatureBuilder(traffic, plan, FeatureConfig(**reference.manifest["feature_config"]))
    v2_x = v2_x.transform(test_points)

    results = {"candidate": PARAMS, "threshold_s": THRESHOLD_S, "rows": int(len(test_y))}
    for mode in ("main", "fallback"):
        columns = columns_for(x, mode)
        candidate = np.zeros(len(test_y))
        for index in range(5):
            held = np.zeros(len(late), bool)
            held[test_rows[fold == index]] = True
            model, platt = fit_calibrated(x.loc[~held, columns], late[~held])
            candidate[fold == index] = platt.apply(raw(model, x.loc[held, columns]))
        truth = late[is_test]
        v2 = reference.predict_late_probability(v2_x, no_hint=mode == "fallback")
        results[mode] = {
            "v2_classifier": reliability(truth, v2),
            "schema2_train_plus_test": reliability(truth, candidate),
        }
    output.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
