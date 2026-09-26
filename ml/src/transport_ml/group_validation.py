"""Train-only vehicle holdout with temporal selection inside the remaining vehicles."""

import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from transport_ml.data import load_inputs, load_labels, load_points, point_path
from transport_ml.features import SCHEMA_VERSION, FeatureBuilder, FeatureConfig
from transport_ml.model import sha256, write_json
from transport_ml.training import TrainConfig, evaluation, select_model, temporal_masks


@dataclass(frozen=True)
class GroupConfig:
    holdout_fraction: float = 0.25
    seed: int = 42


def group_mask(points: pd.DataFrame, config: GroupConfig) -> np.ndarray:
    """Choose whole real vehicles independently of row order, timestamps and labels."""
    if not 0 < config.holdout_fraction < 1:
        raise ValueError("Holdout fraction must be in (0, 1)")
    if points.tr_id.isna().any() or (points.tr_id.astype("int64") >= 9_000_000).any():
        raise ValueError("Group validation requires real vehicles only")
    vehicles = np.array(sorted(points.tr_id.astype(str).unique()))
    count = int(np.ceil(len(vehicles) * config.holdout_fraction))
    if count < 1 or len(vehicles) - count < 2:
        raise ValueError("Need at least two training vehicles and one holdout vehicle")
    selected = np.random.default_rng(config.seed).permutation(vehicles)[:count]
    return points.tr_id.astype(str).isin(selected).to_numpy()


def fit_outer_train(points, features, target, holdout, config):
    """Never pass outer holdout rows to selection, early stopping or the final refit."""
    train_points = points.loc[~holdout].reset_index(drop=True)
    train_features = features.loc[~holdout].reset_index(drop=True)
    train_target = target[~holdout]
    fit, development, boundary = temporal_masks(
        train_points, train_target, config.development_fraction
    )
    models, specs, candidates = {}, {}, {}
    for name in ("main", "fallback"):
        models[name], specs[name], candidates[name] = select_model(
            train_features, train_target, fit, development, config, no_hint=name == "fallback"
        )
    selection = {
        "boundary": str(boundary),
        "fit_rows": int(fit.sum()),
        "development_rows": int(development.sum()),
        "purged_rows": int((~fit & ~development).sum()),
        "candidates": candidates,
    }
    return models, specs, selection


def validate_groups(
    root: Path, directory: Path, config: TrainConfig, group_config: GroupConfig
) -> dict:
    """Write a reproducible experiment without reading test/validate or changing shipped models."""
    if config.iterations < 1 or config.threads < 1 or config.learning_rate <= 0:
        raise ValueError("Iterations, threads and learning rate must be positive")
    all_points = load_points(root, "train")
    real = all_points.tr_id.astype("int64") < 9_000_000
    points = all_points.loc[real].reset_index(drop=True)
    holdout = group_mask(points, group_config)
    target = load_labels(root, "train", points)
    train_points = points.loc[~holdout].reset_index(drop=True)
    fit, development, boundary = temporal_masks(
        train_points, target[~holdout], config.development_fraction
    )
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    # Persist the outer split before training; the inner role is for selection only.
    roles = np.full(len(points), "holdout", dtype=object)
    roles[~holdout] = np.where(development, "development", np.where(fit, "fit", "purged"))
    points.assign(outer_role=np.where(holdout, "holdout", "train"), selection_role=roles).to_csv(
        directory / "split.csv", index=False
    )
    feature_config = FeatureConfig()
    traffic, plan = load_inputs(root, "train")
    features = FeatureBuilder(traffic, plan, feature_config).transform(points)
    models, specs, selection = fit_outer_train(points, features, target, holdout, config)
    train_median = float(np.median(target[~holdout]))
    manifest = {
        "experiment": "real_vehicle_holdout",
        "feature_schema_version": SCHEMA_VERSION,
        "features": list(features),
        "feature_config": feature_config.to_dict(),
        "train_config": asdict(config),
        "group_config": asdict(group_config),
        "time_basis": "dataset_naive_ns",
        "training_group": "tr_id < 9000000 (real-only), excluding outer holdout vehicles",
        "excluded_synthetic_rows": int((~real).sum()),
        "train_median_s": train_median,
        "models": {},
        "groups": {
            name: {
                "rows": int(mask.sum()),
                "vehicles": sorted(points.loc[mask, "tr_id"].unique().tolist()),
                "rows_by_vehicle": points.loc[mask].groupby("tr_id").size().to_dict(),
            }
            for name, mask in (("train", ~holdout), ("holdout", holdout))
        },
        "inner_boundary": str(boundary),
        "split_sha256": sha256(directory / "split.csv"),
    }
    predictions = {}
    for name, model in models.items():
        spec = specs[name]
        path = directory / f"{name}.cbm"
        model.save_model(str(path))
        manifest["models"][name] = {**spec, "file": path.name, "sha256": sha256(path)}
        offset = features.loc[holdout, "cur_dev_s"].to_numpy() if spec["residual"] else 0
        predictions[name] = model.predict(features.loc[holdout, spec["features"]]) + offset
    sources = [root / "train" / name for name in ("traffic.csv", "schedule.csv")]
    sources.append(point_path(root, "train"))
    manifest["data_sha256"] = {str(path): sha256(path) for path in sources}
    manifest["code_sha256"] = {
        path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    manifest["environment"] = {name: version(name) for name in ("catboost", "pandas", "numpy")}
    manifest["environment"].update(python=platform.python_version(), platform=platform.platform())
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    manifest["source_note"] = "code_sha256 includes uncommitted sources used for this experiment."
    write_json(directory / "manifest.json", manifest)
    report = {
        "selection": selection,
        "holdout": evaluation(
            points.loc[holdout],
            target[holdout],
            predictions["main"],
            predictions["fallback"],
            train_median,
        ),
        "limitations": [
            "A single seeded vehicle holdout from one day, not an unseen-day estimate.",
            "Existing feature and candidate choices predate this holdout; not a blind benchmark.",
            "Synthetic vehicles excluded because their families are unknown.",
            "Temporal selection uses only outer-train vehicles; refit includes its purged rows.",
            "No test/validate labels or shipped models used; no probability calibration evaluated.",
        ],
        "elapsed_seconds": perf_counter() - started,
    }
    points.loc[holdout, ["sample_id", "tr_id", "T"]].assign(
        target_delay_s=target[holdout],
        prediction=predictions["main"],
        no_hint_prediction=predictions["fallback"],
    ).to_csv(directory / "holdout_predictions.csv", index=False)
    write_json(directory / "metrics.json", report)
    return report
