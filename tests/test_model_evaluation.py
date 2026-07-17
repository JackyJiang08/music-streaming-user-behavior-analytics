"""Unit tests for src/model_evaluation.py on tiny synthetic datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.model_evaluation import (
    calibration_table,
    evaluate_models,
    paired_bootstrap_auc_delta,
    precision_at_recall,
    threshold_metrics,
)


@pytest.fixture
def synthetic_problem():
    """Small binary problem where signal strength is controllable."""
    rng = np.random.default_rng(0)
    n = 1_500
    x_strong = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    logit = 1.8 * x_strong
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"x_strong": x_strong, "x_noise": x_noise})
    return X, y


def test_evaluate_models_shape_and_columns(synthetic_problem):
    X, y = synthetic_problem
    good = LogisticRegression().fit(X, y)
    noise_only = LogisticRegression().fit(X[["x_noise"]].to_numpy(), y)

    class _NoiseWrapper:
        def predict_proba(self, X_in):
            return noise_only.predict_proba(X_in[["x_noise"]].to_numpy())

    table = evaluate_models({"good": good, "noise": _NoiseWrapper()}, X, y)
    assert list(table["model"]) == ["good", "noise"]
    assert set(table.columns) == {
        "model", "roc_auc", "pr_auc", "brier", "precision_at_threshold",
        "recall_at_threshold", "f1_at_threshold", "flagged_share",
    }
    assert table.loc[0, "roc_auc"] > table.loc[1, "roc_auc"]


def test_evaluate_models_validates_inputs(synthetic_problem):
    X, y = synthetic_problem
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_models({}, X, y)
    with pytest.raises(TypeError, match="predict_proba"):
        evaluate_models({"bad": object()}, X, y)


def test_threshold_metrics_match_confusion_matrix_arithmetic(synthetic_problem):
    X, y = synthetic_problem
    proba = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    threshold = 0.4
    got = threshold_metrics(y, proba, threshold)

    tn, fp, fn, tp = confusion_matrix(y, (proba >= threshold).astype(int)).ravel()
    assert got["precision"] == pytest.approx(tp / (tp + fp))
    assert got["recall"] == pytest.approx(tp / (tp + fn))
    assert got["flagged_share"] == pytest.approx((tp + fp) / len(y))


def test_paired_bootstrap_detects_a_real_auc_gap(synthetic_problem):
    X, y = synthetic_problem
    strong = LogisticRegression().fit(X[["x_strong"]].to_numpy(), y)
    weak = LogisticRegression().fit(X[["x_noise"]].to_numpy(), y)
    proba_strong = strong.predict_proba(X[["x_strong"]].to_numpy())[:, 1]
    proba_weak = weak.predict_proba(X[["x_noise"]].to_numpy())[:, 1]

    result = paired_bootstrap_auc_delta(y, proba_weak, proba_strong,
                                        n_bootstrap=500, seed=1)
    true_delta = roc_auc_score(y, proba_strong) - roc_auc_score(y, proba_weak)
    assert result.delta == pytest.approx(true_delta)
    assert result.ci_low <= true_delta <= result.ci_high
    assert result.significant and result.ci_low > 0


def test_paired_bootstrap_finds_no_gap_between_identical_models(synthetic_problem):
    X, y = synthetic_problem
    proba = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    result = paired_bootstrap_auc_delta(y, proba, proba, n_bootstrap=200, seed=2)
    assert result.delta == 0
    assert not result.significant


def test_paired_bootstrap_validates_inputs(synthetic_problem):
    X, y = synthetic_problem
    proba = np.full(len(y), 0.5)
    with pytest.raises(ValueError, match="n_bootstrap"):
        paired_bootstrap_auc_delta(y, proba, proba, n_bootstrap=10)
    with pytest.raises(ValueError, match="metric"):
        paired_bootstrap_auc_delta(y, proba, proba, metric="accuracy")


def test_calibration_table_bins_sum_to_population(synthetic_problem):
    X, y = synthetic_problem
    proba = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    table = calibration_table(y, proba, n_bins=10)
    assert table["count"].sum() == len(y)
    assert ((table["mean_predicted"] >= 0) & (table["mean_predicted"] <= 1)).all()
    assert ((table["observed_rate"] >= 0) & (table["observed_rate"] <= 1)).all()


def test_precision_at_recall_meets_target(synthetic_problem):
    X, y = synthetic_problem
    proba = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    op = precision_at_recall(y, proba, target_recall=0.75)
    assert op["recall"] >= 0.75
    check = threshold_metrics(y, proba, op["threshold"])
    assert check["precision"] == pytest.approx(op["precision"])
    assert op["contacts_per_catch"] >= 1.0


def test_precision_at_recall_validates_target(synthetic_problem):
    X, y = synthetic_problem
    proba = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    with pytest.raises(ValueError, match="target_recall"):
        precision_at_recall(y, proba, target_recall=1.5)


def test_lift_table_matches_hand_computation():
    from src.model_evaluation import lift_table

    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    table = lift_table(y, proba, fractions=[0.2, 0.5])
    top2 = table.iloc[0]
    assert top2["n_contacted"] == 2
    assert top2["precision"] == 1.0            # both top-2 are positives
    assert top2["share_of_positives_captured"] == pytest.approx(2 / 3)
    assert top2["lift_vs_random"] == pytest.approx(1.0 / 0.3)
    with pytest.raises(ValueError, match="fractions"):
        lift_table(y, proba, fractions=[0, 0.5])
