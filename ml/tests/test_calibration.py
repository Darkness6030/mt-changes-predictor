"""Properties of the probability calibration and its quality report."""

import numpy as np
import pytest
from transport_ml.calibration import fit_platt, reliability, roc_auc


def test_platt_recovers_a_known_sigmoid():
    generator = np.random.default_rng(7)
    scores = generator.normal(size=4000)
    labels = (generator.uniform(size=4000) < 1 / (1 + np.exp(-(1.5 * scores - 0.4)))).astype(float)
    platt = fit_platt(scores, labels)
    assert platt.a == pytest.approx(1.5, abs=0.2)
    assert platt.b == pytest.approx(-0.4, abs=0.2)
    probability = platt.apply(scores)
    assert probability.min() >= 0 and probability.max() <= 1
    assert reliability(labels, probability)["status"] == "validated"


def test_calibration_needs_both_classes_and_finite_input():
    scores = np.linspace(-2, 2, 20)
    with pytest.raises(ValueError, match="both classes"):
        fit_platt(scores, np.zeros(20))
    with pytest.raises(ValueError, match="ten aligned"):
        fit_platt(scores[:5], np.ones(5))
    platt = fit_platt(scores, (scores > 0).astype(float))
    with pytest.raises(ValueError, match="finite"):
        platt.apply(np.array([np.inf]))


def test_reliability_flags_a_useless_probability():
    labels = np.repeat([0.0, 1.0], 50)
    constant = np.full(100, 0.5)
    report = reliability(labels, constant)
    assert report["status"] == "weak"
    assert report["bins"][0]["rows"] == 100
    assert np.isnan(roc_auc(np.zeros(10), np.linspace(0, 1, 10)))
    assert roc_auc(labels, np.concatenate([np.zeros(50), np.ones(50)])) == 1.0


def test_reliability_rejects_impossible_probabilities():
    with pytest.raises(ValueError, match="within"):
        reliability(np.array([0.0, 1.0]), np.array([0.0, 1.5]))
