import pandas as pd
import pytest
from transport_backend.live_evaluation import score_predictions


def test_periodic_evaluation_weights_visits_separately_and_uses_only_labelled_targets():
    predictions = pd.DataFrame(
        {
            "tr_id": ["bus"] * 4,
            "target_stop_id": ["a", "a", "b", "unlabelled"],
            "candidate": [10.0, 10.0, 100.0, 999.0],
            "baseline_estimated": [0.0] * 4,
            "baseline_no_hint": [0.0] * 4,
            "zero": [0.0] * 4,
            "probability": [0.1] * 4,
            "horizon_s": [660.0] * 4,
        }
    )
    labels = pd.DataFrame(
        {"tr_id": ["bus"] * 3, "target_stop_id": ["a", "b", "c"], "target_delay_s": [0.0, 0.0, 0.0]}
    )
    report = score_predictions(predictions, labels)
    assert report["scored_rows"] == 3
    assert report["scored_visits"] == 2
    assert report["visit_coverage"] == pytest.approx(2 / 3)
    assert report["metrics"]["candidate"]["mae_s"] == 40
    assert report["metrics"]["candidate"]["visit_weighted_mae_s"] == 55
