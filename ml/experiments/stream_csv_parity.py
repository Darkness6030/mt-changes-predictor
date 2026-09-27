"""Does the streaming Backend reproduce the submitted CSV? (answer to the organisers' check)

Start a Backend that replays the validate day with the supplied forecast points and the
same ML service as the CSV (default model), e.g.::

    BACKEND_MODE=replay BACKEND_SPLIT=validate BACKEND_USE_POINTS=true BACKEND_LABELS= \\
    BACKEND_ML_URL=http://localhost:8011 BACKEND_REPLAY_SPEED=240 \\
    "BACKEND_REPLAY_START=2026-01-06 02:30:00" BACKEND_PREDICT_INTERVAL_S=3600 \\
    .venv/bin/transport-backend serve

When ``/api/v1/status`` shows all 151 points delivered, run::

    .venv/bin/python ml/experiments/stream_csv_parity.py http://localhost:8000 \\
        ml/pretrained/v8/submission.csv docs/stream-csv-parity.json

Only the Backend's published predictions and the CSV are compared; no labels are read.
"""

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd


def main(backend: str, submission: Path, output: Path) -> None:
    with urllib.request.urlopen(f"{backend}/api/v1/predictions?limit=1000", timeout=10) as answer:
        log = json.load(answer)["predictions"]
    with urllib.request.urlopen(f"{backend}/api/v1/status", timeout=10) as answer:
        status = json.load(answer)
    first: dict[str, dict] = {}
    for item in log:
        if item.get("trigger") == "point":
            first.setdefault(item["sample_id"], item)
    csv = pd.read_csv(submission, sep=";").set_index("sample_id").prediction
    streamed = pd.Series({key: item["delay_s"] for key, item in first.items()}, dtype=float)
    difference = (streamed - csv.reindex(streamed.index)).abs()
    missing = sorted(set(csv.index) - set(streamed.index))
    report = {
        "submission": str(submission),
        "model_version": status["ml"].get("model_version"),
        "points": status["points"],
        "streamed_with_number": len(streamed),
        "equal_within_0_01_s": int((difference < 0.01).sum()),
        "max_abs_difference_s": float(difference.max()),
        "median_abs_difference_s": float(difference.median()),
        "largest_differences_s": difference.sort_values().tail(5).round(4).to_dict(),
        "without_number": len(missing),
        "without_number_ids": missing,
        "statuses": dict(Counter(item.get("status") for item in first.values())),
        "note": (
            "Points without a number got an honest status (for example stale: no trusted GPS "
            "fix within 120 s of T) instead of a forecast; the offline CSV still has one."
        ),
    }
    output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "without_number_ids"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), Path(sys.argv[2]), Path(sys.argv[3]))
