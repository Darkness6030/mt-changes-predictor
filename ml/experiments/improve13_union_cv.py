"""Block CV (same protocol as improve13_block_cv) for the v6 union and its seed bagging."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from improve13_block_cv import MARGIN_S, folds, load, members  # noqa: E402
from transport_ml.research import candidate_prediction, fit_candidate  # noqa: E402
from transport_ml.synthetic import keep_synthetic  # noqa: E402
from transport_ml.trip_context import TRIP_COLUMNS  # noqa: E402


def predictions(points, features, families, kind, synthetic, trip, seed):
    validate = points[points.split == "validate"][["tr_id", "T"]]
    copies = points.index[points.synthetic].to_numpy()
    matrix = features if trip else features.drop(columns=list(TRIP_COLUMNS))
    out, truth = [], []
    for fit_rows, held in folds(points):
        extra = np.array([], dtype=int)
        if synthetic == "purged":
            protected = pd.concat([points.loc[held, ["tr_id", "T"]], validate])
            extra = copies[keep_synthetic(points.loc[copies], families, protected, MARGIN_S)]
        rows = np.r_[fit_rows, extra]
        x = matrix.loc[rows].reset_index(drop=True)
        y = points.loc[rows, "y"].to_numpy()
        value = np.zeros(len(held))
        chosen = [m.__class__(**{**m.__dict__, "seed": seed}) for m in members(kind)]
        for candidate in chosen:
            model, columns = fit_candidate(candidate, x, y, np.ones(len(rows), dtype=bool))
            value += candidate_prediction(model, columns, candidate, matrix.loc[held])
        out.append(value / len(chosen))
        truth.append(points.loc[held, "y"].to_numpy())
    return np.concatenate(out), np.concatenate(truth)


def main():
    points, features, families = load(Path("dataset"))
    result = {}
    runs = {}
    for seed in (42, 7, 101):
        runs[("v4", seed)], y = predictions(
            points, features, families, "v3_main", "exclude", False, seed
        )
        runs[("trip", seed)], _ = predictions(
            points, features, families, "v4_main", "purged", True, seed
        )
        print("seed", seed, flush=True)

    def mae(p):
        return float(np.mean(np.abs(y - p)))

    result["v4_seed42"] = mae(runs[("v4", 42)])
    result["trip_seed42"] = mae(runs[("trip", 42)])
    result["union_seed42"] = mae((runs[("v4", 42)] + runs[("trip", 42)]) / 2)
    result["union_3seeds"] = mae(np.mean([runs[k] for k in runs], axis=0))
    result["v4_3seeds"] = mae(np.mean([runs[("v4", s)] for s in (42, 7, 101)], axis=0))
    result["trip_3seeds"] = mae(np.mean([runs[("trip", s)] for s in (42, 7, 101)], axis=0))
    print(json.dumps(result, indent=2))
    Path("ml/reports/ml-v6-union-cv.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
