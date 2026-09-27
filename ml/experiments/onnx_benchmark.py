"""Latency of the full inference call (delay + probability) on CatBoost vs ONNX Runtime.

Real validate feature rows, batch sizes of a typical cycle, single thread as in the service.
Run from the repository root:

    .venv/bin/python ml/experiments/onnx_benchmark.py ml/experiments/onnx-benchmark.json
"""

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from transport_ml.data import load_points
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.model import DelayModel
from transport_ml.onnx_runtime import OnnxDelayModel

BUNDLE = Path("ml/pretrained/v5")
REPEATS = 300


def features(model: DelayModel) -> pd.DataFrame:
    points = load_points(Path("dataset"), "validate")
    traffic = pd.read_csv("dataset/validate/traffic.csv", dtype={"tr_id": str})
    plan = pd.read_csv("dataset/validate/schedule_plan.csv", dtype={"tr_id": str})
    config = FeatureConfig(**model.manifest["feature_config"])
    return FeatureBuilder(traffic, plan, config).transform(points)


def timing(model: DelayModel, frame: pd.DataFrame) -> dict:
    model.infer(frame)  # Warm-up.
    samples = []
    for _ in range(REPEATS):
        started = perf_counter()
        model.infer(frame)
        samples.append((perf_counter() - started) * 1000)
    return {"p50_ms": float(np.median(samples)), "p95_ms": float(np.quantile(samples, 0.95))}


def main(output: Path, bundle: Path = BUNDLE) -> None:
    catboost = DelayModel(bundle)
    onnx = OnnxDelayModel(bundle, threads=1)
    rows = features(catboost)
    report = {
        "bundle": str(bundle),
        "platform": platform.platform(),
        "repeats": REPEATS,
        "parity": onnx.parity(rows, reference=catboost),
        "batches": {},
    }
    for size in (1, 8, 32, 128):
        frame = pd.concat([rows] * (size // len(rows) + 1), ignore_index=True).head(size)
        base, fast = timing(catboost, frame), timing(onnx, frame)
        report["batches"][str(size)] = {
            "catboost": base,
            "onnx": fast,
            "speedup_p50": base["p50_ms"] / fast["p50_ms"],
        }
    output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else BUNDLE)
