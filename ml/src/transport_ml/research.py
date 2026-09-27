"""Reproducible train-only selection and refitting; test/validate are never loaded here."""

import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from transport_ml.data import load_inputs, load_labels, load_points, point_path
from transport_ml.features import FeatureBuilder, FeatureConfig
from transport_ml.model import sha256, write_json
from transport_ml.schedule_context import CONTEXT_COLUMNS
from transport_ml.synthetic import is_synthetic, keep_synthetic, synthetic_families
from transport_ml.trip_context import HINT_COLUMNS


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str
    features: str
    depth: int
    iterations: int
    loss: str = "MAE"
    residual: bool = False
    l2: float = 5.0
    seed: int = 42
    augmentation: bool = False
    learning_rate: float = 0.05

    def __post_init__(self):
        if self.mode not in {"main", "fallback"}:
            raise ValueError("Unknown prediction mode")
        if self.mode == "fallback" and self.residual:
            raise ValueError("An autonomous model cannot depend on supplied hints")
        if self.augmentation and self.mode != "fallback":
            raise ValueError("Augmented points have no supplied hints")
        if not 1 <= self.depth <= 10 or self.iterations < 1 or self.l2 < 0:
            raise ValueError("Invalid candidate parameters")
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("Candidate needs a plain filename as its name")


def feature_columns(features: pd.DataFrame, kind: str, mode: str) -> list[str]:
    base = [name for name in features if name not in CONTEXT_COLUMNS]
    extra = [name for name in features if name in CONTEXT_COLUMNS]
    if kind == "base":
        result = base
    elif kind == "relative":
        absolute = {"target_lon", "target_lat", "time_sin", "time_cos"}
        result = [name for name in base if name not in absolute] + extra
    elif kind == "full":
        result = list(features)
    elif kind == "compact":
        result = [
            "cur_dev_s",
            "horizon_s",
            "gps_age_s",
            "observed_stop_s",
            "target_distance_m",
            "last_speed",
            "speed_mean_600s",
            "stopped_fraction_600s",
        ] + extra
    else:
        raise ValueError(f"Unknown feature set: {kind}")
    return [name for name in result if mode != "fallback" or name not in HINT_COLUMNS]


def inner_folds(points: pd.DataFrame, target: np.ndarray, protocol: dict):
    """Whole vehicles and forward time; labels are used only to purge unavailable outcomes."""
    ids = np.array(sorted(points.tr_id.unique()))
    count = protocol.get("group_folds", 5)
    if len(ids) < count:
        raise ValueError("Too few vehicles for the requested group folds")
    groups = np.array_split(np.random.default_rng(protocol["seed"]).permutation(ids), count)
    result = []
    for index, group in enumerate(groups):
        valid = points.tr_id.isin(group).to_numpy()
        result.append((f"group{index}", ~valid, valid, None))
    boundaries = list(
        map(
            pd.Timestamp,
            protocol.get(
                "time_boundaries",
                [
                    "2026-01-06 12:00:00",
                    "2026-01-06 17:00:00",
                    "2026-01-07 00:00:00",
                ],
            ),
        )
    )
    outcome = points.target_time_begin + pd.to_timedelta(target, unit="s")
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        cutoff = start - pd.Timedelta(minutes=30)
        fit = ((points["T"] < cutoff) & (outcome < start)).to_numpy()
        valid = ((points["T"] >= start) & (points["T"] < end)).to_numpy()
        if not fit.any() or not valid.any():
            raise ValueError("Empty forward validation fold")
        result.append((f"time{start.hour}", fit, valid, cutoff))
    return result


def build_augmented(points: pd.DataFrame, builder: FeatureBuilder):
    """Additional legal cutoffs for labelled TRAIN visits, with all future facts excluded.

    Parent indices keep every copy in the same fold; hints are never backfilled from a
    later point. No new visit labels are extracted from the raw schedule (which overlaps
    other splits). Each original target retains total training weight one.
    """
    rows = []
    for parent, point in enumerate(points.to_dict("records")):
        for horizon in (630, 690, 750, 810, 870):
            at = point["target_time_begin"] - pd.Timedelta(seconds=horizon)
            if at == point["T"]:
                continue
            visit = builder.target(point["tr_id"], at)
            if visit is None or visit.time_begin != point["target_time_begin"]:
                continue
            rows.append(
                {
                    **point,
                    "sample_id": f"aug:{point['sample_id']}:{horizon}",
                    "T": at,
                    "cur_dev_s": np.nan,
                    "parent": parent,
                }
            )
    augmented = pd.DataFrame(rows)
    if augmented.empty:
        raise ValueError("No eligible augmented cutoffs")
    return augmented, builder.transform(augmented)


def fit_candidate(candidate, features, target, fit, *, augmented=None, cutoff=None, threads=4):
    columns = feature_columns(features, candidate.features, candidate.mode)
    offset = features.cur_dev_s.to_numpy() if candidate.residual else np.zeros(len(target))
    x = features.loc[fit, columns]
    y = (target - offset)[fit]
    weights = None
    if candidate.augmentation:
        if augmented is None:
            raise ValueError("Augmented training inputs are required")
        points, extra = augmented
        use = fit[points.parent.to_numpy()].copy()
        if cutoff is not None:
            use &= (points["T"] < cutoff).to_numpy()
        parents = points.loc[use, "parent"].to_numpy()
        counts = np.bincount(parents, minlength=len(target)) + 1
        x = pd.concat([x, extra.loc[use, columns]], ignore_index=True)
        y = np.r_[y, target[parents]]
        weights = np.r_[1 / counts[fit], 1 / counts[parents]]
    model = CatBoostRegressor(
        iterations=candidate.iterations,
        depth=candidate.depth,
        learning_rate=candidate.learning_rate,
        loss_function=candidate.loss,
        random_seed=candidate.seed,
        thread_count=threads,
        l2_leaf_reg=candidate.l2,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(x, y, sample_weight=weights)
    return model, columns


def candidate_prediction(model, columns, candidate, features):
    values = model.predict(features[columns])
    if candidate.residual:
        values += features.cur_dev_s.to_numpy()
    return values


def provenance(root: Path) -> dict:
    source = Path(__file__).parent
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    inputs = [root / "train" / name for name in ("traffic.csv", "schedule.csv")]
    inputs.append(point_path(root, "train"))
    return {
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "code_sha256": {path.name: sha256(path) for path in sorted(source.glob("*.py"))},
        "data_sha256": {str(path): sha256(path) for path in inputs},
        "environment": {
            **{name: version(name) for name in ("catboost", "numpy", "pandas")},
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def load_research_data(root):
    points = load_points(root, "train")
    points = points[points.tr_id.astype("int64") < 9_000_000].reset_index(drop=True)
    traffic, plan = load_inputs(root, "train")
    builder = FeatureBuilder(traffic, plan, FeatureConfig(schedule_context=True))
    return points, builder.transform(points), builder


def selection_metrics(predictions: pd.DataFrame) -> dict:
    metrics = {}
    for kind in ("group", "time"):
        rows = predictions[predictions.fold.str.startswith(kind)]
        metrics[kind] = float(np.abs(rows.target - rows.prediction).mean())
    return {**metrics, "score": (metrics["group"] + metrics["time"]) / 2}


def run_research(root: Path, directory: Path, protocol: dict, configs: list[dict]) -> dict:
    """Save all candidates and out-of-fold predictions, excluding outer rows before labels."""
    candidates = [Candidate(**config) for config in configs]
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "provenance.json", provenance(root))
    write_json(directory / "candidates.json", configs)
    points, features, builder = load_research_data(root)
    inner = ~points.tr_id.isin(protocol["outer_vehicles"])
    points = points[inner].reset_index(drop=True)
    features = features[inner].reset_index(drop=True)
    target = load_labels(root, "train", points)
    folds = inner_folds(points, target, protocol)
    pd.concat(
        [
            points[["sample_id", "tr_id", "T"]].assign(
                fold=name, role=np.where(valid, "validation", np.where(fit, "fit", "unused"))
            )
            for name, fit, valid, _ in folds
        ]
    ).to_csv(directory / "split.csv", index=False)
    augmented = (
        build_augmented(points, builder) if any(c.augmentation for c in candidates) else None
    )
    reports = []
    for candidate in candidates:
        started = perf_counter()
        rows = []
        for name, fit, valid, cutoff in folds:
            model, columns = fit_candidate(
                candidate, features, target, fit, augmented=augmented, cutoff=cutoff
            )
            prediction = candidate_prediction(model, columns, candidate, features.loc[valid])
            rows.append(
                pd.DataFrame(
                    {
                        "sample_id": points.sample_id[valid],
                        "tr_id": points.tr_id[valid],
                        "fold": name,
                        "target": target[valid],
                        "prediction": prediction,
                        "cur_dev_s": points.cur_dev_s[valid],
                    }
                )
            )
        predictions = pd.concat(rows, ignore_index=True)
        report = {
            "config": asdict(candidate),
            **selection_metrics(predictions),
            "seconds": perf_counter() - started,
        }
        stem = candidate.name.replace(":", "-")
        predictions.to_csv(directory / f"{stem}.csv", index=False)
        write_json(directory / f"{stem}.json", report)
        reports.append(report)
        print(f"{candidate.name}: {report['score']:.4f} s", flush=True)
    summary = {"candidates": sorted(reports, key=lambda row: row["score"])}
    write_json(directory / "summary.json", summary)
    return summary


def load_training_data(root: Path, config: FeatureConfig, spec: dict | None):
    """Assemble labelled rows; each split's features come from its own traffic and plan.

    Default (``spec is None``) is the v3 behaviour: real train vehicles only. A spec may add
    labelled test points (same day as validate, disjoint visits) and synthetic train
    vehicles with their copies of protected validate moments removed (see ``synthetic``).
    Validate is read only as ``points.csv`` inputs (tr_id, T) for that purge; never labels.
    """
    spec = spec or {}
    splits = spec.get("splits", ["train"])
    if not set(splits) <= {"train", "test"} or len(set(splits)) != len(splits):
        raise ValueError("training_splits must be distinct labelled splits: train, test")
    if "train" not in splits:
        raise ValueError("training_splits must include train")
    frames, matrices, targets = [], [], []
    report = {"splits": splits, "synthetic": spec.get("synthetic", "exclude")}
    for split in splits:
        points = load_points(root, split)
        traffic, plan = load_inputs(root, split)
        synthetic = is_synthetic(points.tr_id)
        keep = ~synthetic
        if split == "train" and spec.get("synthetic", "exclude") == "purged":
            families = synthetic_families(plan)
            guarded = spec.get("protect", ["validate"])
            if "validate" not in guarded:
                raise ValueError("Synthetic copies of validate moments must always be purged")
            protected = pd.concat(
                [load_points(root, name)[["tr_id", "T"]] for name in guarded], ignore_index=True
            )
            report["protected"] = guarded
            report["protected_sha256"] = {name: sha256(point_path(root, name)) for name in guarded}
            copies = points[synthetic]
            keep[np.flatnonzero(synthetic)] = keep_synthetic(
                copies, families, protected, float(spec["purge_margin_s"])
            )
            report["synthetic_families"] = {key: list(value) for key, value in families.items()}
            report["synthetic_kept"] = int(keep[synthetic].sum())
            report["synthetic_dropped"] = int((~keep[synthetic]).sum())
        elif spec.get("synthetic", "exclude") not in {"exclude", "purged"}:
            raise ValueError("synthetic must be 'exclude' or 'purged'")
        points = points[keep].reset_index(drop=True)
        builder = FeatureBuilder(traffic, plan, config)
        frames.append(points.assign(split=split))
        matrices.append(builder.transform(points))
        targets.append(load_labels(root, split, points))
        report[f"{split}_rows"] = len(points)
        report[f"{split}_sha256"] = {
            str(path): sha256(path)
            for path in (point_path(root, split), root / split / "traffic.csv")
        }
    points = pd.concat(frames, ignore_index=True)
    features = pd.concat(matrices, ignore_index=True)
    return points, features, np.concatenate(targets), report


def train_recipe(root: Path, directory: Path, recipe: dict) -> dict:
    """Refit a frozen recipe on real labelled points; preserve the existing risk classifiers.

    Selection and final evaluation are separate commands. In particular this function
    cannot silently inspect test or change a candidate based on its score. By default only
    train is used; ``training_splits`` may add the labelled test points for a final refit,
    after which test is no longer an independent check of that bundle. Only plan columns
    and label files are read: schedule facts never enter features or targets.
    """
    import shutil

    from transport_ml.model import DelayModel

    if set(recipe["models"]) != {"main", "fallback"}:
        raise ValueError("A recipe must define main and fallback")
    if recipe.get("hint_policy") != "supplied_only":
        raise ValueError("Context recipes require separate supplied and GPS estimates")
    config = FeatureConfig(**recipe["feature_config"])
    if not config.schedule_context:
        raise ValueError("A context recipe requires schedule_context")
    source = DelayModel(Path(recipe["classifier_source"]))
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "recipe.json", recipe)
    started = perf_counter()
    training_data = recipe.get("training_data")
    if training_data is None and "training_splits" in recipe:
        # Recipe format of ML-V4-19: real points of the listed splits, synthetic excluded.
        training_data = {"splits": recipe["training_splits"], "synthetic": "exclude"}
    needs_augmented = any(
        member["candidate"].get("augmentation", False)
        for members in recipe["models"].values()
        for member in members
    )
    if training_data is not None and needs_augmented:
        raise ValueError("Augmentation is defined for the real-train-only recipe")
    points, features, target, data_report = load_training_data(root, config, training_data)
    if points.sample_id.duplicated().any():
        raise ValueError("Training splits share a sample_id")
    splits = data_report["splits"]
    fit = np.ones(len(points), dtype=bool)
    augmented = None
    if needs_augmented:
        traffic, plan = load_inputs(root, "train")
        augmented = build_augmented(points, FeatureBuilder(traffic, plan, config))
    manifest = {
        "feature_schema_version": config.schema_version,
        "features": list(features),
        "feature_config": config.to_dict(),
        "time_basis": "dataset_naive_ns",
        "hint_policy": recipe["hint_policy"],
        "training_group": recipe.get(
            "training_group",
            f"real-only {'+'.join(splits)}; frozen train-only group/temporal selection",
        ),
        "training_splits": splits,
        "training_data": data_report,
        "train_rows": len(points),
        "train_median_s": float(np.median(target)),
        "late_rate": float((target > 120).mean()),
        "models": {},
        "classifiers": {},
        "recipe_sha256": sha256(directory / "recipe.json"),
        **provenance(root),
    }
    for mode, members in recipe["models"].items():
        weights = np.asarray([member["weight"] for member in members], dtype=float)
        if not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid recipe weights")
        if not np.isclose(weights.sum(), 1):
            raise ValueError("Recipe weights must sum to one")
        saved = []
        importance = pd.Series(0.0, index=features.columns)
        for index, member in enumerate(members):
            candidate = Candidate(**member["candidate"])
            if candidate.mode != mode:
                raise ValueError("Candidate mode differs from its recipe branch")
            model, columns = fit_candidate(candidate, features, target, fit, augmented=augmented)
            path = directory / f"{mode}-{index}.cbm"
            model.save_model(str(path))
            saved.append(
                {
                    **asdict(candidate),
                    "features": columns,
                    "file": path.name,
                    "sha256": sha256(path),
                    "weight": float(member["weight"]),
                }
            )
            importance.loc[columns] += member["weight"] * model.feature_importances_
        manifest["models"][mode] = {"members": saved}
        importance.sort_values(ascending=False).to_csv(
            directory / f"{mode}_importance.csv", index_label="feature", header=["importance"]
        )
    for name, spec in source.manifest.get("classifiers", {}).items():
        shutil.copyfile(source.directory / spec["file"], directory / spec["file"])
        manifest["classifiers"][name] = spec
    manifest["classifier_provenance"] = {
        "source_manifest_sha256": source.version,
        "note": "Risk classifiers and their development calibration are unchanged from v2.",
    }
    points.assign(role="fit").to_csv(directory / "split.csv", index=False)
    write_json(directory / "manifest.json", manifest)
    restored = DelayModel(directory)
    prediction = restored.predict(features)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions after saving the bundle")
    report = {
        "train_rows": len(points),
        "train_seconds": perf_counter() - started,
        "recipe_sha256": manifest["recipe_sha256"],
        "model_version": restored.version,
        "probability_note": manifest["classifier_provenance"]["note"],
    }
    write_json(directory / "training.json", report)
    return report


def validate_recipe(root: Path, directory: Path, recipe: dict, protocol: dict) -> dict:
    """One outer evaluation of a previously frozen recipe, against the fixed v2 regressors.

    This is an audit command, not another search loop. Do not use its results to select
    different members or retune the recipe. Outer outcomes are accessed after all fits.
    """
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "frozen.json",
        {
            "recipe": recipe,
            "protocol": protocol,
            "provenance": provenance(root),
        },
    )
    points, features, builder = load_research_data(root)
    held = points.tr_id.isin(protocol["outer_vehicles"]).to_numpy()
    fit = ~held
    if not held.any() or not fit.any():
        raise ValueError("Empty train or outer holdout")
    target = np.zeros(len(points))
    target[fit] = load_labels(root, "train", points[fit])
    needs_augmented = any(
        member["candidate"].get("augmentation", False)
        for members in recipe["models"].values()
        for member in members
    )
    augmented = build_augmented(points, builder) if needs_augmented else None
    predictions = {}
    for mode, members in recipe["models"].items():
        values = np.zeros(int(held.sum()))
        for index, member in enumerate(members):
            candidate = Candidate(**member["candidate"])
            model, columns = fit_candidate(candidate, features, target, fit, augmented=augmented)
            model.save_model(str(directory / f"{mode}-{index}.cbm"))
            values += member["weight"] * candidate_prediction(
                model, columns, candidate, features[held]
            )
        predictions[mode] = values
        baseline = Candidate(
            "v2",
            mode,
            "base",
            4 if mode == "main" else 6,
            368 if mode == "main" else 293,
            residual=mode == "main",
        )
        model, columns = fit_candidate(baseline, features, target, fit)
        model.save_model(str(directory / f"v2-{mode}.cbm"))
        predictions[f"v2_{mode}"] = candidate_prediction(model, columns, baseline, features[held])
    outcome = load_labels(root, "train", points[held])
    predictions["cur_dev"] = points.loc[held, "cur_dev_s"].to_numpy()
    predictions["zero"] = np.zeros(int(held.sum()))
    frame = (
        points.loc[held, ["sample_id", "tr_id", "T"]].copy().assign(target=outcome, **predictions)
    )
    frame.to_csv(directory / "predictions.csv", index=False)
    report = {
        "rows": int(held.sum()),
        "vehicles": protocol["outer_vehicles"],
        "metrics": {
            name: {
                "mae_s": float(np.abs(outcome - values).mean()),
                "p50_s": float(np.median(np.abs(outcome - values))),
                "p90_s": float(np.quantile(np.abs(outcome - values), 0.9)),
            }
            for name, values in predictions.items()
        },
        "by_vehicle": {
            str(tr_id): {
                name: float(np.abs(group.target - group[name]).mean()) for name in predictions
            }
            for tr_id, group in frame.groupby("tr_id")
        },
    }
    write_json(directory / "metrics.json", report)
    return report
