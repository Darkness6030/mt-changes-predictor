"""Small, versioned CatBoost artifact: delay regressors plus a calibrated late classifier."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from transport_ml.calibration import Platt
from transport_ml.features import SCHEMA_VERSION


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class DelayModel:
    """Loads one artifact directory once and refuses mismatched schemas or checksums."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["feature_schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported feature schema")
        self.version = sha256(self.directory / "manifest.json")
        self.models = {}
        for name, spec in self.manifest["models"].items():
            self.models[name] = self._load(spec, CatBoostRegressor())
        # Classifiers are optional: the first published bundle has no probability model.
        self.classifiers = {}
        self.calibration = {}
        for name, spec in self.manifest.get("classifiers", {}).items():
            self.classifiers[name] = self._load(spec, CatBoostClassifier())
            parameters = spec["calibration"]
            self.calibration[name] = Platt(
                a=parameters["a"],
                b=parameters["b"],
                fit_rows=parameters["fit_rows"],
                positive_rows=parameters.get("positive_rows", 0),
            )

    def _load(self, spec: dict, model):
        path = self.directory / spec["file"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Model checksum mismatch: {spec['file']}")
        model.load_model(str(path))
        return model

    @property
    def features(self) -> list[str]:
        return list(self.manifest["features"])

    @property
    def late_threshold_s(self) -> float | None:
        spec = self.manifest.get("classifiers", {}).get("late")
        return None if spec is None else float(spec["threshold_s"])

    def calibration_status(self) -> dict:
        spec = self.manifest.get("classifiers", {}).get("late")
        if spec is None:
            return {"method": None, "status": "unavailable", "report": None}
        return {
            "method": spec["calibration"]["method"],
            "status": spec["calibration"]["status"],
            "report": spec["calibration"].get("report"),
            "fit_rows": spec["calibration"]["fit_rows"],
        }

    def _check(self, features: pd.DataFrame) -> None:
        if list(features.columns) != self.manifest["features"] or not features.columns.is_unique:
            raise ValueError("Feature names/order do not match the model manifest")
        if np.isinf(features.to_numpy(dtype=float)).any():
            raise ValueError("Features contain infinity")

    def _routing(self, features: pd.DataFrame, no_hint: bool) -> np.ndarray:
        """True where the no-hint model must be used, i.e. cur_dev_s is unknown."""
        return features.cur_dev_s.isna().to_numpy() | bool(no_hint)

    def predict(self, features: pd.DataFrame, *, no_hint: bool = False) -> np.ndarray:
        self._check(features)
        result = np.empty(len(features))
        fallback = self._routing(features, no_hint)
        for name, mask in (("main", ~fallback), ("fallback", fallback)):
            if not mask.any():
                continue
            spec = self.manifest["models"][name]
            values = self.models[name].predict(features.loc[mask, spec["features"]])
            if spec["residual"]:
                values += features.loc[mask, "cur_dev_s"].to_numpy()
            result[mask] = values
        if not np.isfinite(result).all():
            raise ValueError("Model returned a non-finite prediction")
        return result

    def predict_late_probability(
        self, features: pd.DataFrame, *, no_hint: bool = False
    ) -> np.ndarray | None:
        """Calibrated ``P(delay_s > threshold)``; None when the artifact has no classifier."""
        if not self.classifiers:
            return None
        self._check(features)
        result = np.full(len(features), np.nan)
        fallback = self._routing(features, no_hint)
        for name, mask in (("late", ~fallback), ("late_fallback", fallback)):
            if not mask.any():
                continue
            spec = self.manifest["classifiers"][name]
            scores = self.classifiers[name].predict(
                features.loc[mask, spec["features"]], prediction_type="RawFormulaVal"
            )
            result[mask] = self.calibration[name].apply(np.asarray(scores, dtype=float))
        if not np.isfinite(result).all():
            raise ValueError("Classifier returned a non-finite probability")
        return result

    def infer(self, features: pd.DataFrame, *, no_hint: bool = False) -> dict:
        """Single entry point used by the HTTP service: delay, routing and probability."""
        fallback = self._routing(features, no_hint)
        probability = self.predict_late_probability(features, no_hint=no_hint)
        return {
            "delay_s": self.predict(features, no_hint=no_hint),
            "model_used": np.where(fallback, "fallback", "main"),
            "used_hint": ~fallback,
            "late_probability": probability,
            "late_threshold_s": self.late_threshold_s,
            "calibration": self.calibration_status(),
            "model_version": self.version,
        }
