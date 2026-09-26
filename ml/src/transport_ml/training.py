"""Select on a real-only temporal development split; inspect test only after refit."""

import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from transport_ml.calibration import fit_platt, reliability
from transport_ml.data import load_inputs, load_labels, load_points, point_path
from transport_ml.features import SCHEMA_VERSION, FeatureBuilder, FeatureConfig
from transport_ml.model import DelayModel, sha256, write_json


@dataclass(frozen=True)
class TrainConfig:
    iterations: int = 600
    learning_rate: float = 0.05
    seed: int = 42
    threads: int = 4
    development_fraction: float = 0.25
    late_threshold_s: float = 120.0


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(target - prediction)))


def build_split(root: Path, split: str, config: FeatureConfig):
    points = load_points(root, split)
    traffic, plan = load_inputs(root, split)
    return points, FeatureBuilder(traffic, plan, config).transform(points)


def temporal_masks(points: pd.DataFrame, target: np.ndarray, fraction: float):
    if not 0 < fraction < 1:
        raise ValueError("Development fraction must be in (0, 1)")
    boundary = points["T"].sort_values().iloc[int(len(points) * (1 - fraction))]
    # Labels are usable only after their outcome; additionally separate 10-minute feature windows.
    outcome = points.target_time_begin + pd.to_timedelta(target, unit="s")
    fit = (points["T"] < boundary - pd.Timedelta(minutes=10)) & (outcome < boundary)
    development = points["T"] >= boundary
    if fit.sum() < 30 or development.sum() < 10:
        raise ValueError("Insufficient rows for a temporal split")
    return fit.to_numpy(), development.to_numpy(), boundary


def regressor(config: TrainConfig, depth: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=config.learning_rate,
        loss_function="MAE",
        random_seed=config.seed,
        thread_count=config.threads,
        l2_leaf_reg=5,
        allow_writing_files=False,
        verbose=False,
    )


def select_model(features, target, fit, development, config, *, no_hint):
    columns = [name for name in features if not (no_hint and name == "cur_dev_s")]
    candidates = []
    for depth in (4, 6):
        for residual in (False,) if no_hint else (False, True):
            offset = np.zeros(len(target)) if not residual else features.cur_dev_s.to_numpy()
            model = regressor(config, depth, config.iterations)
            model.fit(
                features.loc[fit, columns],
                (target - offset)[fit],
                eval_set=(features.loc[development, columns], (target - offset)[development]),
                early_stopping_rounds=70,
            )
            prediction = model.predict(features.loc[development, columns]) + offset[development]
            candidates.append(
                {
                    "depth": depth,
                    "residual": residual,
                    "iterations": model.tree_count_,
                    "development_mae_s": mae(target[development], prediction),
                }
            )
    best = min(candidates, key=lambda row: row["development_mae_s"])
    final = regressor(config, best["depth"], best["iterations"])
    offset = features.cur_dev_s.to_numpy() if best["residual"] else np.zeros(len(target))
    final.fit(features[columns], target - offset)
    return final, {**best, "features": columns}, candidates


def classifier(config: TrainConfig, depth: int, iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=config.learning_rate,
        loss_function="Logloss",
        random_seed=config.seed,
        thread_count=config.threads,
        l2_leaf_reg=5,
        allow_writing_files=False,
        verbose=False,
    )


def select_classifier(features, late, fit, development, config, *, no_hint):
    """Grow trees on the fit fold only, then calibrate two parameters on development.

    The classifier is deliberately not refitted on all rows: a Platt mapping is only
    meaningful for the exact scores it was fitted against.
    """
    columns = [name for name in features if not (no_hint and name == "cur_dev_s")]
    candidates = []
    for depth in (4, 6):
        model = classifier(config, depth, config.iterations)
        model.fit(
            features.loc[fit, columns],
            late[fit],
            eval_set=(features.loc[development, columns], late[development]),
            early_stopping_rounds=70,
        )
        scores = model.predict(features.loc[development, columns], prediction_type="RawFormulaVal")
        platt = fit_platt(np.asarray(scores, dtype=float), late[development])
        report = reliability(late[development], platt.apply(scores))
        candidates.append(
            {
                "depth": depth,
                "iterations": model.tree_count_,
                "development": {key: report[key] for key in ("brier", "log_loss", "roc_auc")},
                "model": model,
                "calibration": platt,
                "development_report": report,
            }
        )
    best = min(candidates, key=lambda row: row["development"]["brier"])
    spec = {
        "features": columns,
        "depth": best["depth"],
        "iterations": best["iterations"],
        "threshold_s": config.late_threshold_s,
        "trained_on": "fit fold only, so the calibration matches these scores",
        "calibration": {
            **best["calibration"].to_dict(),
            "positive_rows": best["calibration"].positive_rows,
            "fold": "development",
            "status": "fitted_on_development",
            "report": "ml/reports/ml-v2.md",
        },
    }
    keys = ("depth", "iterations", "development")
    trace = [{key: row[key] for key in keys} for row in candidates]
    return best["model"], spec, trace, best["development_report"]


def evaluation(points, target, prediction, fallback, train_median):
    summary = {
        "rows": len(points),
        "vehicles": points.tr_id.nunique(),
        "model_mae_s": mae(target, prediction),
        "no_hint_mae_s": mae(target, fallback),
        "zero_mae_s": mae(target, np.zeros(len(target))),
        "cur_dev_mae_s": mae(target, points.cur_dev_s.to_numpy()),
        "train_median_mae_s": mae(target, np.full(len(target), train_median)),
        "absolute_error_p50_s": float(np.median(np.abs(target - prediction))),
        "absolute_error_p90_s": float(np.quantile(np.abs(target - prediction), 0.9)),
    }
    slices = {
        f"vehicle:{key}": points.tr_id.eq(key).to_numpy() for key in sorted(points.tr_id.unique())
    }
    slices.update(early=target < -60, ontime=(target >= -60) & (target <= 120), late=target > 120)
    summary["slices"] = {
        key: {"rows": int(mask.sum()), "mae_s": mae(target[mask], prediction[mask])}
        for key, mask in slices.items()
        if mask.any()
    }
    return summary


def evaluate(root: Path, directory: Path, split: str = "test") -> dict:
    model = DelayModel(directory)
    config = FeatureConfig(**model.manifest["feature_config"])
    started = perf_counter()
    points, features = build_split(root, split, config)
    feature_seconds = perf_counter() - started
    target = load_labels(root, split, points)
    started = perf_counter()
    prediction = model.predict(features)
    predict_seconds = perf_counter() - started
    fallback = model.predict(features, no_hint=True)
    report = evaluation(points, target, prediction, fallback, model.manifest["train_median_s"])
    report.update(feature_build_seconds=feature_seconds, batch_predict_seconds=predict_seconds)
    columns = {
        "sample_id": points.sample_id,
        "target_delay_s": target,
        "prediction": prediction,
        "no_hint_prediction": fallback,
    }
    probability = model.predict_late_probability(features)
    if probability is not None:
        threshold = model.late_threshold_s
        late = (target > threshold).astype(float)
        report["late_probability"] = {
            "threshold_s": threshold,
            "calibration": model.calibration_status(),
            "with_hint": reliability(late, probability),
            "no_hint": reliability(late, model.predict_late_probability(features, no_hint=True)),
        }
        columns["late_probability"] = probability
    pd.DataFrame(columns).to_csv(directory / f"{split}_predictions.csv", index=False)
    return report


def train(root: Path, directory: Path, config: TrainConfig) -> dict:
    if config.iterations < 1 or config.threads < 1 or config.learning_rate <= 0:
        raise ValueError("Iterations, threads and learning rate must be positive")
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    feature_config = FeatureConfig()
    all_points = load_points(root, "train")
    # Synthetic families are unknown: use only the documented real-ID group for this first model.
    real = all_points.tr_id.astype("int64") < 9_000_000
    points = all_points.loc[real].reset_index(drop=True)
    target = load_labels(root, "train", points)
    traffic, plan = load_inputs(root, "train")
    features = FeatureBuilder(traffic, plan, feature_config).transform(points)
    fit, development, boundary = temporal_masks(points, target, config.development_fraction)
    roles = np.where(development, "development", np.where(fit, "fit", "purged"))
    points.assign(role=roles).to_csv(directory / "split.csv", index=False)
    manifest = {
        "feature_schema_version": SCHEMA_VERSION,
        "features": list(features),
        "feature_config": feature_config.to_dict(),
        "train_config": asdict(config),
        "time_basis": "dataset_naive_ns",
        "training_group": "tr_id < 9000000 (real-only)",
        "train_rows": len(points),
        "excluded_synthetic_rows": int((~real).sum()),
        "train_median_s": float(np.median(target)),
        "models": {},
        "classifiers": {},
    }
    report = {
        "selection": {
            "boundary": str(boundary),
            "fit_rows": int(fit.sum()),
            "development_rows": int(development.sum()),
            "purged_rows": int((roles == "purged").sum()),
            "development_cur_dev_mae_s": mae(
                target[development], points.cur_dev_s.to_numpy()[development]
            ),
            "development_train_median_mae_s": mae(
                target[development], np.full(development.sum(), np.median(target[fit]))
            ),
            "candidates": {},
        },
        "limitations": [
            "Test shares vehicles/day with train and telemetry/plan with validate.",
            "Development is used for model selection, not an unbiased final estimate.",
            "Synthetic families excluded; vehicle-group validation is pending.",
            "Late probability is calibrated on the development fold, not on an unseen day.",
        ],
    }
    for name in ("main", "fallback"):
        print(f"Selecting and fitting {name} model...", flush=True)
        model, spec, candidates = select_model(
            features, target, fit, development, config, no_hint=name == "fallback"
        )
        path = directory / f"{name}.cbm"
        model.save_model(str(path))
        manifest["models"][name] = {**spec, "file": path.name, "sha256": sha256(path)}
        report["selection"]["candidates"][name] = candidates
        pd.Series(
            model.feature_importances_, index=spec["features"], name="importance"
        ).sort_values(ascending=False).to_csv(
            directory / f"{name}_importance.csv", index_label="feature"
        )
    late = (target > config.late_threshold_s).astype(float)
    manifest["late_rate"] = float(late.mean())
    report["late_probability"] = {
        "threshold_s": config.late_threshold_s,
        "train_positive_rate": float(late.mean()),
        "development_positive_rows": int(late[development].sum()),
        "candidates": {},
        "development": {},
    }
    for name in ("late", "late_fallback"):
        print(f"Selecting and calibrating {name} classifier...", flush=True)
        model, spec, trace, fold_report = select_classifier(
            features, late, fit, development, config, no_hint=name == "late_fallback"
        )
        path = directory / f"{name}.cbm"
        model.save_model(str(path))
        manifest["classifiers"][name] = {**spec, "file": path.name, "sha256": sha256(path)}
        report["late_probability"]["candidates"][name] = trace
        report["late_probability"]["development"][name] = fold_report
    sources = [
        root / split / name
        for split in ("train", "test")
        for name in ("traffic.csv", "schedule.csv")
    ]
    sources += [point_path(root, split) for split in ("train", "test")]
    manifest["data_sha256"] = {str(path): sha256(path) for path in sources}
    source = Path(__file__).parent
    manifest["code_sha256"] = {path.name: sha256(path) for path in sorted(source.glob("*.py"))}
    manifest["environment"] = {name: version(name) for name in ("catboost", "pandas", "numpy")}
    manifest["environment"].update(python=platform.python_version(), platform=platform.platform())
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    manifest["source_note"] = (
        "code_sha256 identifies the actual sources, including uncommitted code."
    )
    write_json(directory / "manifest.json", manifest)
    print("Evaluating frozen models on test...", flush=True)
    report["test"] = evaluate(root, directory)
    report["train_seconds"] = perf_counter() - started
    write_json(directory / "metrics.json", report)
    return report
