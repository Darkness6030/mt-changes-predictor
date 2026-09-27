"""ML-IMPROVE-18: does refitting the frozen v3 recipe on train + test labels help?

Held-out test points are predicted by (a) the recipe fitted on real train only (v3) and
(b) the same recipe fitted on real train plus the other test folds. "random" folds mimic
validate, whose points are interleaved with test points of the same vehicles; "vehicle"
folds hold out whole vehicles. Only plan columns and label files are read; schedule facts
and synthetic vehicles are never used. Run from the repository root:

    .venv/bin/python ml/experiments/improve18_test_refit.py improve18-test-refit.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transport_ml.data import load_inputs, load_labels, load_points
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.research import Candidate, candidate_prediction, fit_candidate

ROOT = Path("dataset")
RECIPE = json.loads(Path("ml/experiments/improve12-recipe.json").read_text())
SEED = 20260927


def labelled(split: str):
    points = load_points(ROOT, split)
    points = points[points.tr_id.astype("int64") < 9_000_000].reset_index(drop=True)
    traffic, plan = load_inputs(ROOT, split)
    builder = FeatureBuilder(traffic, plan, FeatureConfig(schedule_context=True))
    return points, builder.transform(points), load_labels(ROOT, split, points)


def main(output: Path) -> None:
    _, train_features, train_target = labelled("train")
    test_points, test_features, test_target = labelled("test")
    features = pd.concat([train_features, test_features], ignore_index=True)
    target = np.r_[train_target, test_target]
    is_test = np.r_[np.zeros(len(train_target), bool), np.ones(len(test_target), bool)]
    test_rows = np.flatnonzero(is_test)

    def fit_predict(fit, predict, mode):
        values = np.zeros(int(predict.sum()))
        for member in RECIPE["models"][mode]:
            candidate = Candidate(**member["candidate"])
            model, columns = fit_candidate(candidate, features, target, fit)
            values += member["weight"] * candidate_prediction(
                model, columns, candidate, features.loc[predict]
            )
        return values

    results = {}
    for scheme in ("random", "vehicle"):
        rng = np.random.default_rng(SEED)
        if scheme == "random":
            fold = rng.integers(0, 5, len(test_target))
        else:
            vehicles = np.array(sorted(test_points.tr_id.unique()))
            rng.shuffle(vehicles)
            fold = test_points.tr_id.map({v: i % 5 for i, v in enumerate(vehicles)}).to_numpy()
        for mode in ("main", "fallback"):
            errors = {"train_only": [], "train_plus_test": []}
            for index in range(5):
                held = np.zeros(len(target), bool)
                held[test_rows[fold == index]] = True
                truth = target[held]
                errors["train_only"].append(np.abs(truth - fit_predict(~is_test, held, mode)))
                errors["train_plus_test"].append(np.abs(truth - fit_predict(~held, held, mode)))
            results[f"{scheme}:{mode}"] = {
                name: round(float(np.concatenate(values).mean()), 4)
                for name, values in errors.items()
            }
    output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
