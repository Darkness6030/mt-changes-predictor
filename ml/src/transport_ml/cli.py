"""Offline commands; all paths are explicit and relative to the current directory."""

import argparse
import json
from pathlib import Path

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
    for command in (fit, check, infer):
        command.add_argument("--data", type=Path, default=Path("dataset"))
        command.add_argument("--model", type=Path, default=Path("artifacts/ml-v1"))
    args = parser.parse_args()
    if args.command == "train":
        result = train(
            args.data,
            args.model,
            TrainConfig(iterations=args.iterations, threads=args.threads, seed=args.seed),
        )
    elif args.command == "evaluate":
        result = evaluate(args.data, args.model)
    else:
        result = predict(args.data, args.model, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
