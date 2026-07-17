"""Unit tests for src/uplift_modeling.py on synthetic randomized data."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from src.uplift_modeling import (
    SLearner,
    TLearner,
    XLearner,
    cumulative_gain_curve,
    qini_coefficient,
    qini_curve,
    uplift_by_decile,
    validate_against_true_uplift,
)

FAST_CLF = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7, random_state=0)
FAST_REG = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=7, random_state=0)


@pytest.fixture(scope="module")
def randomized_data():
    """Randomized experiment with a known two-segment uplift structure.

    Segment A (x0 < 0): true uplift +0.20; segment B (x0 >= 0): -0.10.
    Base retention 0.40 either side.
    """
    rng = np.random.default_rng(3)
    n = 6_000
    X = rng.normal(size=(n, 3))
    treatment = (rng.random(n) < 0.5).astype(int)
    true_uplift = np.where(X[:, 0] < 0, 0.20, -0.10)
    p = 0.40 + treatment * true_uplift
    y = (rng.random(n) < p).astype(int)
    return X, treatment, y, true_uplift


@pytest.mark.parametrize(
    "learner_factory",
    [
        lambda: SLearner(base_classifier=FAST_CLF),
        lambda: TLearner(base_classifier=FAST_CLF),
        lambda: XLearner(base_classifier=FAST_CLF, base_regressor=FAST_REG),
    ],
    ids=["s_learner", "t_learner", "x_learner"],
)
def test_learners_recover_segment_effect_signs(randomized_data, learner_factory):
    X, treatment, y, true_uplift = randomized_data
    learner = learner_factory().fit(X, treatment, y)
    predicted = learner.predict_uplift(X)

    positive_segment = X[:, 0] < 0
    assert predicted[positive_segment].mean() > 0.08  # true +0.20
    assert predicted[~positive_segment].mean() < 0.02  # true -0.10
    assert predicted[positive_segment].mean() > predicted[~positive_segment].mean()


def test_learner_refuses_non_randomized_assignment(randomized_data):
    X, _, y, _ = randomized_data
    rigged = (np.arange(len(y)) % 10 == 0).astype(int)  # 10/90 split
    with pytest.raises(ValueError, match="sample-ratio-mismatch"):
        TLearner(base_classifier=FAST_CLF).fit(X, rigged, y)


def test_learner_validates_inputs(randomized_data):
    X, treatment, y, _ = randomized_data
    with pytest.raises(ValueError, match="binary"):
        SLearner(base_classifier=FAST_CLF).fit(X, treatment * 2, y)
    with pytest.raises(ValueError, match="length mismatch"):
        SLearner(base_classifier=FAST_CLF).fit(X, treatment[:-1], y)


def test_oracle_qini_beats_random_scores(randomized_data):
    X, treatment, y, true_uplift = randomized_data
    rng = np.random.default_rng(0)
    oracle = qini_coefficient(y, treatment, true_uplift)
    random_scores = qini_coefficient(y, treatment, rng.random(len(y)))
    assert oracle > 0.01
    assert oracle > random_scores + 0.005


def test_qini_curve_shape_and_endpoint(randomized_data):
    X, treatment, y, true_uplift = randomized_data
    curve = qini_curve(y, treatment, true_uplift)
    assert list(curve.columns) == ["fraction_targeted", "incremental_responders"]
    assert curve["fraction_targeted"].iloc[-1] == pytest.approx(1.0)
    # Endpoint equals the overall scaled incremental responders regardless
    # of ordering — the curve only reshapes the path, not the destination.
    shuffled = qini_curve(y, treatment, np.zeros(len(y)))
    assert curve["incremental_responders"].iloc[-1] == pytest.approx(
        shuffled["incremental_responders"].iloc[-1]
    )


def test_uplift_by_decile_is_monotone_for_oracle(randomized_data):
    X, treatment, y, true_uplift = randomized_data
    table = uplift_by_decile(y, treatment, true_uplift, n_bins=5)
    assert len(table) == 5
    assert table["n_users"].sum() == len(y)
    # Oracle ranking: top bins hold the +0.20 segment, bottom the -0.10 one.
    assert table["observed_uplift"].iloc[0] > 0.10
    assert table["observed_uplift"].iloc[-1] < 0.0


def test_cumulative_gain_positive_for_oracle(randomized_data):
    X, treatment, y, true_uplift = randomized_data
    curve = cumulative_gain_curve(y, treatment, true_uplift)
    # Targeting the positive-uplift segment first accumulates real gains.
    mid = curve[curve["fraction_targeted"] <= 0.5]["incremental_responders"]
    assert mid.iloc[-1] > 0


def test_validate_against_true_uplift_scores_perfection_and_noise(randomized_data):
    _, _, _, true_uplift = randomized_data
    perfect = validate_against_true_uplift(true_uplift, true_uplift)
    assert perfect["mse"] == 0
    assert perfect["rank_correlation"] == pytest.approx(1.0)

    rng = np.random.default_rng(1)
    noise = validate_against_true_uplift(rng.random(len(true_uplift)), true_uplift)
    assert abs(noise["rank_correlation"]) < 0.1


def test_validate_against_true_uplift_guards():
    with pytest.raises(ValueError, match="length mismatch"):
        validate_against_true_uplift([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match=">= 3"):
        validate_against_true_uplift([0.1, 0.2], [0.1, 0.2])
