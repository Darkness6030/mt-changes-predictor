"""Build ``ml/pretrained/v6``: equal-weight union of two independently selected ensembles.

* ``v4`` (ML-V4-19): the v3 recipe (schema 2 features) refitted on real train + test;
  platform score 0.86126.
* ``trip`` (ML-IMPROVE-13): schema 3 trip-structure features, real train + test + synthetic
  train with copies of validate moments purged; platform score 0.85649.

Both are causal models on the same shared FeatureBuilder; schema 3 is a superset of
schema 2, so every member reads its own stored column list. Risk classifiers and their
out-of-fold calibration are copied from v5. No weights are tuned on any score: 1/6 each.

    .venv/bin/python ml/experiments/improve13_build_v6.py --trip artifacts/v6-trip \
        --output ml/pretrained/v6
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from transport_ml.features import FeatureConfig
from transport_ml.model import DelayModel, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--v4", type=Path, default=Path("ml/pretrained/v4"))
    parser.add_argument("--v5", type=Path, default=Path("ml/pretrained/v5"))
    parser.add_argument("--trip", type=Path, required=True, help="trained improve13 recipe")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="PREFIX=DIR",
        help="additional trained bundles for a wider equal-weight union (v7: c, d)",
    )
    parser.add_argument("--recipe", type=Path, default=Path("ml/experiments/improve13-recipe.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--zero-gate",
        action="store_true",
        help="v8: add the hurdle gate trained on the trip recipe's training data",
    )
    args = parser.parse_args()
    if not args.trip.exists():
        subprocess.run(
            [
                "python", "-m", "transport_ml", "train-recipe",
                "--data", str(args.data), "--recipe", str(args.recipe), "--model", str(args.trip),
            ],
            check=True,
        )  # fmt: skip
    sources = {"a": DelayModel(args.v4), "b": DelayModel(args.trip)}
    for item in args.extra:
        prefix, path = item.split("=", 1)
        if prefix in sources:
            raise ValueError(f"Duplicate source prefix: {prefix}")
        sources[prefix] = DelayModel(Path(path))
    classifiers = DelayModel(args.v5)
    config = FeatureConfig(**sources["b"].manifest["feature_config"])
    if any(not set(source.features) <= set(sources["b"].features) for source in sources.values()):
        raise ValueError("Schema 3 must contain every member feature")
    args.output.mkdir(parents=True, exist_ok=False)
    models = {}
    for mode in ("main", "fallback"):
        members = []
        for prefix, source in sources.items():
            for spec in source.manifest["models"][mode]["members"]:
                file = f"{prefix}-{spec['file']}"
                shutil.copyfile(source.directory / spec["file"], args.output / file)
                members.append({**spec, "file": file, "sha256": sha256(args.output / file)})
        for member in members:
            member["weight"] = 1 / len(members)
        models[mode] = {"members": members}
    for spec in classifiers.manifest["classifiers"].values():
        shutil.copyfile(classifiers.directory / spec["file"], args.output / spec["file"])
    manifest = {
        "feature_schema_version": config.schema_version,
        "features": sources["b"].features,
        "feature_config": config.to_dict(),
        "time_basis": "dataset_naive_ns",
        "hint_policy": "supplied_only",
        "training_group": "equal-weight union of independently trained ensembles: "
        + ", ".join(sources),
        "train_rows": sources["b"].manifest["train_rows"],
        "models": models,
        "classifiers": classifiers.manifest["classifiers"],
        "sources": {
            prefix: {
                "directory": str(source.directory),
                "manifest_sha256": source.version,
                "training_group": source.manifest.get("training_group"),
            }
            for prefix, source in sources.items()
        },
        "classifier_provenance": {
            "source_manifest_sha256": classifiers.version,
            "note": "Risk classifiers and out-of-fold calibration copied from v5 unchanged.",
        },
    }
    if args.zero_gate:
        from transport_ml.zero_gate import fit_zero_gate, save_zero_gate

        recipe = json.loads(args.recipe.read_text())
        gate = fit_zero_gate(args.data, config, recipe["training_data"])
        manifest["zero_gate"] = save_zero_gate(*gate, args.output)
        manifest["training_group"] += "; hurdle zero gate (P(delay == 0) > 0.5 -> 0)"
    write_json(args.output / "manifest.json", manifest)
    restored = DelayModel(args.output)
    print(
        json.dumps({"model_version": restored.version, "members": len(models["main"]["members"])})
    )


if __name__ == "__main__":
    main()
