"""Small, versioned CatBoost artifact with a separately trained no-hint fallback."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from transport_ml.features import SCHEMA_VERSION


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class DelayModel:
    def __init__(self, directory: Path):
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["feature_schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported feature schema")
        self.models = {}
        for name, spec in self.manifest["models"].items():
            path = directory / spec["file"]
            if sha256(path) != spec["sha256"]:
                raise ValueError(f"Model checksum mismatch: {name}")
            model = CatBoostRegressor()
            model.load_model(str(path))
            self.models[name] = model

    def predict(self, features: pd.DataFrame, *, no_hint: bool = False) -> np.ndarray:
        expected = self.manifest["features"]
        if list(features.columns) != expected or not features.columns.is_unique:
            raise ValueError("Feature names/order do not match the model manifest")
        if np.isinf(features.to_numpy(dtype=float)).any():
            raise ValueError("Features contain infinity")
        result = np.empty(len(features))
        fallback = features.cur_dev_s.isna().to_numpy() | no_hint
        for name, mask in (("main", ~fallback), ("fallback", fallback)):
            if not mask.any():
                continue
            spec = self.manifest["models"][name]
            selected = features.loc[mask, spec["features"]]
            values = self.models[name].predict(selected)
            if spec["residual"]:
                values += features.loc[mask, "cur_dev_s"].to_numpy()
            result[mask] = values
        if not np.isfinite(result).all():
            raise ValueError("Model returned a non-finite prediction")
        return result
