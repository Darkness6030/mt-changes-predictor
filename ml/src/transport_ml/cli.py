"""Offline commands; all paths are explicit and relative to the current directory."""

import argparse
import json
from pathlib import Path

from transport_ml.group_validation import GroupConfig, validate_groups
from transport_ml.research import run_research, train_recipe, validate_recipe
from transport_ml.submission import predict
from transport_ml.training import TrainConfig, evaluate, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Select on train development, refit, evaluate on test")
    fit.add_argument("--iterations", type=int, default=600)
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--seed", type=int, default=42)
    check = commands.add_parser("evaluate", help="Evaluate the stored model on labelled test")
    infer = commands.add_parser("predict", help="Generate validate submission without labels")
    infer.add_argument("--output", type=Path, default=Path("artifacts/ml-v1/submission.csv"))
    groups = commands.add_parser("validate-groups", help="Train-only real-vehicle holdout")
    groups.add_argument("--data", type=Path, default=Path("dataset"))
    groups.add_argument("--output", type=Path, required=True)
    groups.add_argument("--iterations", type=int, default=600)
    groups.add_argument("--threads", type=int, default=4)
    groups.add_argument("--seed", type=int, default=42)
    groups.add_argument("--holdout-fraction", type=float, default=0.25)
    research = commands.add_parser("research", help="Train-only group/forward candidate comparison")
    research.add_argument("--data", type=Path, default=Path("dataset"))
    research.add_argument("--output", type=Path, required=True)
    research.add_argument("--protocol", type=Path, required=True)
    research.add_argument("--candidates", type=Path, required=True)
    recipe = commands.add_parser("train-recipe", help="Refit a frozen recipe on real train only")
    recipe.add_argument("--data", type=Path, default=Path("dataset"))
    recipe.add_argument("--model", type=Path, required=True)
    recipe.add_argument("--recipe", type=Path, required=True)
    holdout = commands.add_parser(
        "validate-recipe", help="Audit a frozen recipe on reserved vehicles"
    )
    holdout.add_argument("--data", type=Path, default=Path("dataset"))
    holdout.add_argument("--output", type=Path, required=True)
    holdout.add_argument("--recipe", type=Path, required=True)
    holdout.add_argument("--protocol", type=Path, required=True)
    for command in (fit, check, infer):
        command.add_argument("--data", type=Path, default=Path("dataset"))
        command.add_argument("--model", type=Path, default=Path("artifacts/ml-v1"))
    args = parser.parse_args()
    if args.command == "research":
        result = run_research(
            args.data,
            args.output,
            json.loads(args.protocol.read_text()),
            json.loads(args.candidates.read_text()),
        )
    elif args.command == "train-recipe":
        result = train_recipe(args.data, args.model, json.loads(args.recipe.read_text()))
    elif args.command == "validate-recipe":
        result = validate_recipe(
            args.data,
            args.output,
            json.loads(args.recipe.read_text()),
            json.loads(args.protocol.read_text()),
        )
    elif args.command == "train":
        result = train(
            args.data,
            args.model,
            TrainConfig(iterations=args.iterations, threads=args.threads, seed=args.seed),
        )
    elif args.command == "validate-groups":
        result = validate_groups(
            args.data,
            args.output,
            TrainConfig(iterations=args.iterations, threads=args.threads, seed=args.seed),
            GroupConfig(holdout_fraction=args.holdout_fraction, seed=args.seed),
        )
    elif args.command == "evaluate":
        result = evaluate(args.data, args.model)
    else:
        result = predict(args.data, args.model, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
