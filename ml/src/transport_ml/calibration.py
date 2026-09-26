"""Platt scaling and honest probability quality checks for the late classifier.

Two parameters are fitted on a held-out fold, so the reported reliability is not the
output of the same rows the trees were grown on. Nothing here reads the target of the
evaluation fold that is later used for reporting.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Platt:
    """Sigmoid calibration ``p = 1 / (1 + exp(-(a * score + b)))``."""

    a: float
    b: float
    fit_rows: int
    positive_rows: int

    def to_dict(self) -> dict:
        return {"method": "platt", "a": self.a, "b": self.b, "fit_rows": self.fit_rows}

    def apply(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Calibration input must be finite")
        return sigmoid(self.a * scores + self.b)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> Platt:
    """Newton fit on Platt's smoothed targets, which keeps small folds from saturating."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.shape != labels.shape or scores.ndim != 1 or len(scores) < 10:
        raise ValueError("Calibration needs at least ten aligned scores and labels")
    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    if positive < 1 or negative < 1:
        raise ValueError("Calibration needs both classes")
    smoothed = np.where(labels > 0, (positive + 1) / (positive + 2), 1 / (negative + 2))
    slope, intercept = 1.0, 0.0
    # Two-parameter Newton written out: the 2x2 system stays exact and free of matmul noise.
    for _ in range(200):
        probability = sigmoid(slope * scores + intercept)
        residual = probability - smoothed
        curvature = probability * (1 - probability) + 1e-9
        gradient = np.array([float(np.sum(residual * scores)), float(residual.sum())])
        weighted = curvature * scores
        hessian = np.array(
            [
                [float(np.sum(weighted * scores)) + 1e-8, float(weighted.sum())],
                [float(weighted.sum()), float(curvature.sum()) + 1e-8],
            ]
        )
        step = np.linalg.solve(hessian, gradient)
        if not np.isfinite(step).all():
            raise ValueError("Calibration did not converge")
        slope -= float(step[0])
        intercept -= float(step[1])
        if np.abs(step).max() < 1e-10:
            break
    return Platt(slope, intercept, len(scores), int(positive))


def roc_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    """Rank-based AUC with ties averaged; undefined for a single class."""
    labels = np.asarray(labels, dtype=float)
    positive = labels > 0
    if positive.all() or not positive.any():
        return float("nan")
    order = np.argsort(probability, kind="stable")
    ranks = np.empty(len(probability), dtype=float)
    sorted_values = np.asarray(probability, dtype=float)[order]
    start = 0
    for index in range(1, len(sorted_values) + 1):
        if index == len(sorted_values) or sorted_values[index] != sorted_values[start]:
            ranks[order[start:index]] = 0.5 * (start + index - 1) + 1
            start = index
    count_positive = positive.sum()
    count_negative = len(labels) - count_positive
    return float(
        (ranks[positive].sum() - count_positive * (count_positive + 1) / 2)
        / (count_positive * count_negative)
    )


def reliability(labels: np.ndarray, probability: np.ndarray, bins: int = 5) -> dict:
    """Brier/log loss plus equal-width bins with counts, so small bins stay visible."""
    labels = np.asarray(labels, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if labels.shape != probability.shape:
        raise ValueError("Reliability needs aligned inputs")
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probabilities must be finite and within [0, 1]")
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    base_rate = float(labels.mean())
    report = {
        "rows": int(len(labels)),
        "positive_rate": base_rate,
        "brier": float(np.mean((probability - labels) ** 2)),
        "base_rate_brier": float(np.mean((base_rate - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "roc_auc": roc_auc(labels, probability),
        "bins": [],
    }
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= low) & (probability < high if high < 1.0 else probability <= 1.0)
        if not mask.any():
            continue
        report["bins"].append(
            {
                "range": [float(low), float(high)],
                "rows": int(mask.sum()),
                "mean_predicted": float(probability[mask].mean()),
                "observed_rate": float(labels[mask].mean()),
            }
        )
    # A model that cannot beat the constant base rate is reported as weak, never as calibrated.
    report["status"] = "validated" if report["brier"] < report["base_rate_brier"] else "weak"
    return report
