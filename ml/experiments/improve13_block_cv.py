"""ML-IMPROVE-13 selection protocol: same-day block cross-validation.

Validate points are 5-minute cutoffs of the SAME 11 real vehicles and day as the labelled
train/test points, grouped in contiguous blocks (median 6 points = 30 min) that sit between
labelled blocks. We reproduce that geometry: pool = real train + test labels, a block is a
maximal run of one split per vehicle on the 5-minute grid, and folds hold out whole blocks.

Synthetic train vehicles are shifted copies of real vehicles (see ``transport_ml.synthetic``);
for every fold we drop their copies of held-out AND validate moments (+-1200 s). Using them
unpurged leaks the held-out answers (``--show-leak``). No validate labels exist or are read.

    .venv/bin/python ml/experiments/improve13_block_cv.py --output ml/reports/ml-v4-block-cv.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transport_ml.data import load_inputs, load_labels, load_points
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.research import Candidate, candidate_prediction, fit_candidate
from transport_ml.synthetic import is_synthetic, keep_synthetic, synthetic_families
from transport_ml.trip_context import TRIP_COLUMNS

MARGIN_S = 1200


def load(root: Path):
    config = FeatureConfig(schedule_context=True, trip_context=True)
    frames, matrices, families = [], [], {}
    for split in ("train", "test", "validate"):
        points = load_points(root, split)
        traffic, plan = load_inputs(root, split)
        matrices.append(FeatureBuilder(traffic, plan, config).transform(points))
        target = np.full(len(points), np.nan)
        if split != "validate":
            target = load_labels(root, split, points)
        frames.append(points.assign(split=split, y=target))
        if split == "train":
            families = synthetic_families(plan)
    points = pd.concat(frames, ignore_index=True)
    points["synthetic"] = is_synthetic(points.tr_id)
    order = points.sort_values(["tr_id", "T"])
    previous = order.groupby("tr_id")[["split", "T"]].shift()
    start = (
        (order.split != previous.split)
        | (order["T"] - previous["T"] > pd.Timedelta(minutes=5))
        | previous["T"].isna()
    )
    points["block"] = start.cumsum().reindex(points.index)
    return points, pd.concat(matrices, ignore_index=True), families


def folds(points, k=10, repeats=2):
    pool = points.index[(points.split != "validate") & ~points.synthetic].to_numpy()
    blocks = np.array(sorted(points.loc[pool, "block"].unique()))
    for repeat in range(repeats):
        order = np.random.default_rng(100 + repeat).permutation(blocks)
        for chunk in np.array_split(order, k):
            held = pool[points.loc[pool, "block"].isin(chunk).to_numpy()]
            yield np.setdiff1d(pool, held), held


def members(kind):
    def c(mode, depth, iterations, residual, lr=0.05, features="full"):
        name = f"{mode}_{features}_{depth}_{iterations}"
        return Candidate(
            name, mode, features, depth, iterations, residual=residual, learning_rate=lr
        )

    return {
        "v3_main": [
            c("main", 6, 300, True),
            c("main", 4, 300, True),
            c("main", 4, 300, True, features="compact"),
        ],
        "v3_fallback": [
            c("fallback", 6, 300, False, features="compact"),
            c("fallback", 4, 700, False),
            c("fallback", 2, 300, False, features="compact"),
        ],
        "v4_main": [
            c("main", 6, 600, True),
            c("main", 6, 1000, False, 0.03),
            c("main", 8, 600, False),
        ],
        "v4_fallback": [
            c("fallback", 6, 600, False),
            c("fallback", 4, 1000, False, 0.03),
            c("fallback", 8, 600, False),
        ],
    }[kind]


def evaluate(points, features, families, kind, synthetic, trip):
    validate = points[points.split == "validate"][["tr_id", "T"]]
    copies = points.index[points.synthetic].to_numpy()
    matrix = features if trip else features.drop(columns=list(TRIP_COLUMNS))
    errors = []
    for fit_rows, held in folds(points):
        extra = np.array([], dtype=int)
        if synthetic == "all":
            extra = copies
        elif synthetic == "purged":
            protected = pd.concat([points.loc[held, ["tr_id", "T"]], validate])
            extra = copies[keep_synthetic(points.loc[copies], families, protected, MARGIN_S)]
        rows = np.r_[fit_rows, extra]
        x = matrix.loc[rows].reset_index(drop=True)
        y = points.loc[rows, "y"].to_numpy()
        chosen = members(kind)
        prediction = np.zeros(len(held))
        for candidate in chosen:
            model, columns = fit_candidate(candidate, x, y, np.ones(len(rows), dtype=bool))
            prediction += candidate_prediction(model, columns, candidate, matrix.loc[held])
        errors.append(np.abs(points.loc[held, "y"].to_numpy() - prediction / len(chosen)))
    return float(np.concatenate(errors).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show-leak", action="store_true")
    args = parser.parse_args()
    points, features, families = load(args.data)
    runs = [
        ("v3_main", "exclude", False),
        ("v3_fallback", "exclude", False),
        ("v4_main", "exclude", True),
        ("v4_main", "purged", False),
        ("v4_main", "purged", True),
        ("v4_fallback", "purged", True),
    ]
    if args.show_leak:
        runs.append(("v4_main", "all", True))
    result = {"folds": "10 block folds x 2 repeats, seeds 100/101", "mae_s": {}}
    for kind, synthetic, trip in runs:
        name = f"{kind}|synthetic={synthetic}|trip={trip}"
        result["mae_s"][name] = evaluate(points, features, families, kind, synthetic, trip)
        print(name, round(result["mae_s"][name], 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
