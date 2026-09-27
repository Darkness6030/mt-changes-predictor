"""Small, versioned CatBoost artifact: delay regressors plus a calibrated late classifier."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from transport_ml.calibration import Platt
from transport_ml.features import SUPPORTED_SCHEMAS, FeatureConfig


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class DelayModel:
    """Loads one artifact directory once and refuses mismatched schemas or checksums."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        schema = self.manifest["feature_schema_version"]
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError("Unsupported feature schema")
        if "feature_config" in self.manifest:
            if FeatureConfig(**self.manifest["feature_config"]).schema_version != schema:
                raise ValueError("Feature config does not match schema version")
        self.version = sha256(self.directory / "manifest.json")
        self.models = {}
        for name, spec in self.manifest["models"].items():
            members = spec.get("members", [spec])
            weights = [float(member.get("weight", 1.0)) for member in members]
            if not weights or not np.isfinite(weights).all() or min(weights) <= 0:
                raise ValueError("Ensemble weights must be positive and finite")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Ensemble weights must sum to one")
            self.models[name] = [
                (member, self._load(member, CatBoostRegressor()), weight)
                for member, weight in zip(members, weights, strict=True)
            ]
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
            values = np.zeros(int(mask.sum()))
            for spec, model, weight in self.models[name]:
                member = model.predict(features.loc[mask, spec["features"]])
                if spec["residual"]:
                    member += features.loc[mask, "cur_dev_s"].to_numpy()
                values += weight * member
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

    def explain(self, features: pd.DataFrame, *, no_hint: bool = False) -> list[dict]:
        """Grouped additive contributions per row; they sum to :meth:`predict` exactly."""
        from transport_ml.explanation import member_shap, summarise

        self._check(features)
        fallback = self._routing(features, no_hint)
        result: list[dict | None] = [None] * len(features)
        for name, mask in (("main", ~fallback), ("fallback", fallback)):
            if not mask.any():
                continue
            rows = features.loc[mask]
            contributions = pd.DataFrame(0.0, index=rows.index, columns=features.columns)
            base = np.zeros(len(rows))
            residual_weight = 0.0
            for spec, model, weight in self.models[name]:
                values, expected = member_shap(model, rows[spec["features"]])
                contributions[spec["features"]] += weight * values
                base += weight * expected
                residual_weight += weight if spec["residual"] else 0.0
            offset = residual_weight * rows.cur_dev_s.fillna(0.0).to_numpy()
            for position, index in enumerate(np.flatnonzero(mask)):
                summary = summarise(
                    contributions.iloc[position].to_dict(), base[position], offset[position]
                )
                summary["model_used"] = name
                result[index] = summary
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
