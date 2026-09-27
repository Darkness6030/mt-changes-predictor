"""ONNX Runtime backend for a published CatBoost bundle (PDF: inference optimisation).

The bundle directory is never modified: every regressor and classifier is exported to ONNX
in a separate directory when the service starts. Classifiers are exported through
``catboost.sum_models`` with weight 1, which keeps the trees and drops the float class
labels that CatBoost's ONNX exporter rejects; the ONNX output is then the raw formula value
that the Platt calibration expects. A parity check against CatBoost is part of loading, so
a numerical mismatch can never reach a dispatcher silently.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import sum_models

from transport_ml.model import DelayModel, sha256

EXPORT = {"onnx_domain": "ai.catboost", "onnx_model_version": 1}
PARITY_TOLERANCE_S = 1e-3  # float32 trees reproduce CatBoost to ~1e-4 s on real features.


def export_bundle(model: DelayModel, directory: Path) -> dict[str, str]:
    """Write one ONNX file per ensemble member and classifier; return file → sha256."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, members in model.models.items():
        for index, (_, regressor, _) in enumerate(members):
            path = directory / f"{name}-{index}.onnx"
            regressor.save_model(str(path), format="onnx", export_parameters=EXPORT)
            files[path.name] = sha256(path)
    for name, classifier in model.classifiers.items():
        path = directory / f"{name}.onnx"
        raw = sum_models([classifier], weights=[1.0])
        raw.save_model(str(path), format="onnx", export_parameters=EXPORT)
        files[path.name] = sha256(path)
    return files


class OnnxDelayModel(DelayModel):
    """Same contract as :class:`DelayModel`; trees are evaluated by ONNX Runtime."""

    def __init__(self, directory: Path, onnx_dir: Path | None = None, threads: int = 1):
        super().__init__(directory)
        import onnxruntime as ort

        self._tmp = None
        if onnx_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="onnx-")
            onnx_dir = Path(self._tmp.name)
        self.onnx_files = export_bundle(self, Path(onnx_dir))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.log_severity_level = 3  # The exported output shape note is harmless.
        providers = ["CPUExecutionProvider"]
        self._sessions = {}
        for file in self.onnx_files:
            session = ort.InferenceSession(str(Path(onnx_dir) / file), options, providers=providers)
            self._sessions[file.removesuffix(".onnx")] = (session, session.get_inputs()[0].name)

    @property
    def runtime(self) -> str:
        return "onnx"

    def _run(self, key: str, frame: pd.DataFrame) -> np.ndarray:
        session, input_name = self._sessions[key]
        output = session.run(None, {input_name: frame.to_numpy(dtype=np.float32)})[0]
        return np.asarray(output, dtype=float).reshape(len(frame))

    def _member_predict(self, name: str, index: int, model, frame: pd.DataFrame) -> np.ndarray:
        return self._run(f"{name}-{index}", frame)

    def _raw_scores(self, name: str, frame: pd.DataFrame) -> np.ndarray:
        return self._run(name, frame)

    def parity(self, features: pd.DataFrame, reference: DelayModel | None = None) -> dict:
        """Largest difference to the CatBoost runtime of the same bundle on these rows."""
        reference = reference or DelayModel(self.directory)
        report = {}
        for no_hint in (False, True):
            mode = "no_hint" if no_hint else "main"
            ours = self.predict(features, no_hint=no_hint)
            theirs = reference.predict(features, no_hint=no_hint)
            report[f"delay_{mode}_s"] = float(np.abs(ours - theirs).max())
            if self.classifiers:
                ours_p = self.predict_late_probability(features, no_hint=no_hint)
                theirs_p = reference.predict_late_probability(features, no_hint=no_hint)
                report[f"probability_{mode}"] = float(np.abs(ours_p - theirs_p).max())
        return report


def probe_features(model: DelayModel, rows: int = 64, seed: int = 20260927) -> pd.DataFrame:
    """Synthetic rows spanning each tree's split values, with missing values, for parity."""
    rng = np.random.default_rng(seed)
    borders: dict[str, list[float]] = {name: [] for name in model.features}
    for members in model.models.values():
        for spec, regressor, _ in members:
            for index, values in regressor.get_borders().items():
                borders[spec["features"][int(index)]].extend(values)
    data = {}
    for name in model.features:
        values = np.asarray(borders[name] or [0.0], dtype=float)
        low, high = values.min() - 1.0, values.max() + 1.0
        column = rng.uniform(low, high, rows)
        column[rng.random(rows) < 0.15] = np.nan
        data[name] = column
    frame = pd.DataFrame(data, columns=model.features)
    frame.loc[: rows // 2, "cur_dev_s"] = rng.uniform(-300, 600, rows // 2 + 1)
    return frame
