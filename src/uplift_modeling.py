"""Uplift (heterogeneous treatment effect) modeling with plain scikit-learn.

Hand-rolled S-, T-, and X-learner meta-learners plus the evaluation
utilities notebook 09 uses to turn notebook 06's average treatment effect
into a targeting policy: uplift-by-decile tables, Qini and cumulative-gain
curves, and validation against the simulation's ground-truth uplift. No
dedicated causal-inference dependency — the meta-learners are a few lines
of scikit-learn each, and the project keeps its footprint small.

All learners share the interface

    learner.fit(X, treatment, y)          # treatment/y are 0/1 arrays
    learner.predict_uplift(X)             # P(y=1 | treat) - P(y=1 | control)

and refuse to fit when the treatment assignment looks non-randomized
(sample-ratio-mismatch check reused from src.ab_testing): meta-learners
estimate causal effects only under randomization.

Typical usage:

    from src.uplift_modeling import TLearner, qini_curve, qini_coefficient

    learner = TLearner().fit(X_train, treatment_train, y_train)
    uplift = learner.predict_uplift(X_test)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.ab_testing import srm_check
from src.config import RANDOM_SEED

# The winning gradient-boosting configuration from notebook 07's tuned
# search (HistGradientBoosting family), reused as the default base learner.
_DEFAULT_BASE_PARAMS = {
    "learning_rate": 0.0742,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 0.0057,
    "max_iter": 150,
}


def _default_classifier(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=random_state,
                                          **_DEFAULT_BASE_PARAMS)


def _default_regressor(random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=random_state,
                                         **_DEFAULT_BASE_PARAMS)


def _validate_fit_inputs(
    X: np.ndarray, treatment: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    treatment = np.asarray(treatment)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    if not (len(X) == len(treatment) == len(y)):
        raise ValueError(
            f"length mismatch: X={len(X)}, treatment={len(treatment)}, y={len(y)}"
        )
    if not set(np.unique(treatment)) <= {0, 1}:
        raise ValueError("treatment must be binary 0/1 (1 = treated)")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y must be binary 0/1")

    # Meta-learners are causal only under randomization: refuse clearly
    # broken assignments instead of returning confounded estimates.
    n_treated = int(treatment.sum())
    srm = srm_check(n_control=len(treatment) - n_treated, n_treatment=n_treated)
    if not srm.passed:
        raise ValueError(
            "treatment assignment fails the sample-ratio-mismatch check "
            f"(p={srm.p_value:.2e}); the data does not look randomized 50/50, "
            "so meta-learner estimates would be confounded"
        )
    return X, treatment.astype(int), y.astype(int)


class SLearner:
    """Single-model learner: treatment enters as one extra feature.

    Lowest variance (one model sees all data) but biased toward zero
    heterogeneity when the base learner underuses the treatment feature.
    """

    def __init__(self, base_classifier=None, random_state: int = RANDOM_SEED):
        self.base_classifier = base_classifier
        self.random_state = random_state

    def fit(self, X, treatment, y) -> "SLearner":
        X, treatment, y = _validate_fit_inputs(X, treatment, y)
        base = self.base_classifier or _default_classifier(self.random_state)
        self.model_ = clone(base).fit(
            np.column_stack([X, treatment]), y
        )
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        as_treated = np.column_stack([X, np.ones(len(X))])
        as_control = np.column_stack([X, np.zeros(len(X))])
        return (
            self.model_.predict_proba(as_treated)[:, 1]
            - self.model_.predict_proba(as_control)[:, 1]
        )


class TLearner:
    """Two-model learner: separate outcome models per arm.

    Unbiased for arm-specific response surfaces but differences two noisy
    models, so uplift estimates carry both models' variance.
    """

    def __init__(self, base_classifier=None, random_state: int = RANDOM_SEED):
        self.base_classifier = base_classifier
        self.random_state = random_state

    def fit(self, X, treatment, y) -> "TLearner":
        X, treatment, y = _validate_fit_inputs(X, treatment, y)
        base = self.base_classifier or _default_classifier(self.random_state)
        self.model_treated_ = clone(base).fit(X[treatment == 1], y[treatment == 1])
        self.model_control_ = clone(base).fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (
            self.model_treated_.predict_proba(X)[:, 1]
            - self.model_control_.predict_proba(X)[:, 1]
        )


class XLearner:
    """Crossed learner (Künzel et al. 2019): impute each user's missing
    potential outcome with the opposite arm's T-learner model, regress the
    imputed individual effects, and blend the two effect models by the
    treatment propensity. Typically the best of both worlds when arms are
    imbalanced or effects are smooth; costs four model fits.
    """

    def __init__(self, base_classifier=None, base_regressor=None,
                 random_state: int = RANDOM_SEED):
        self.base_classifier = base_classifier
        self.base_regressor = base_regressor
        self.random_state = random_state

    def fit(self, X, treatment, y) -> "XLearner":
        X, treatment, y = _validate_fit_inputs(X, treatment, y)
        clf = self.base_classifier or _default_classifier(self.random_state)
        reg = self.base_regressor or _default_regressor(self.random_state)

        X_treated, y_treated = X[treatment == 1], y[treatment == 1]
        X_control, y_control = X[treatment == 0], y[treatment == 0]

        model_treated = clone(clf).fit(X_treated, y_treated)
        model_control = clone(clf).fit(X_control, y_control)

        # Imputed individual effects using the opposite arm's model.
        d_treated = y_treated - model_control.predict_proba(X_treated)[:, 1]
        d_control = model_treated.predict_proba(X_control)[:, 1] - y_control

        self.effect_model_treated_ = clone(reg).fit(X_treated, d_treated)
        self.effect_model_control_ = clone(reg).fit(X_control, d_control)
        # Randomized assignment: a constant propensity is the correct weight.
        self.propensity_ = float(treatment.mean())
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        tau_treated = self.effect_model_treated_.predict(X)
        tau_control = self.effect_model_control_.predict(X)
        e = self.propensity_
        return e * tau_control + (1 - e) * tau_treated


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _validate_eval_inputs(y, treatment, uplift_scores):
    y = np.asarray(y)
    treatment = np.asarray(treatment)
    scores = np.asarray(uplift_scores, dtype=float)
    if not (len(y) == len(treatment) == len(scores)):
        raise ValueError(
            f"length mismatch: y={len(y)}, treatment={len(treatment)}, "
            f"scores={len(scores)}"
        )
    if len(y) == 0:
        raise ValueError("inputs are empty")
    return y, treatment, scores


def uplift_by_decile(
    y: Sequence[int],
    treatment: Sequence[int],
    uplift_scores: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Observed treatment-vs-control outcome gap per predicted-uplift decile.

    Decile 1 holds the highest predicted uplift. A model that ranks well
    shows observed uplift falling monotonically across deciles.
    """
    y, treatment, scores = _validate_eval_inputs(y, treatment, uplift_scores)
    frame = pd.DataFrame({"y": y, "t": treatment, "score": scores})
    # rank first so duplicate score values cannot collapse bins
    frame["decile"] = pd.qcut(
        frame["score"].rank(method="first", ascending=False),
        q=n_bins, labels=range(1, n_bins + 1),
    )
    rows = []
    for decile, group in frame.groupby("decile", observed=True):
        treated = group[group["t"] == 1]
        control = group[group["t"] == 0]
        rows.append(
            {
                "decile": int(decile),
                "n_users": len(group),
                "mean_predicted_uplift": group["score"].mean(),
                "treated_rate": treated["y"].mean() if len(treated) else np.nan,
                "control_rate": control["y"].mean() if len(control) else np.nan,
                "observed_uplift": (
                    treated["y"].mean() - control["y"].mean()
                    if len(treated) and len(control)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def qini_curve(
    y: Sequence[int],
    treatment: Sequence[int],
    uplift_scores: Sequence[float],
    n_points: int = 100,
) -> pd.DataFrame:
    """Qini curve: incremental responders when targeting top-f by score.

    At each targeted fraction f (users sorted by descending predicted
    uplift), Qini(f) = R_t(f) - R_c(f) * N_t(f) / N_c(f): treated responders
    in the targeted slice minus the control responders scaled to the treated
    exposure. Random targeting traces a straight line to the endpoint.
    """
    y, treatment, scores = _validate_eval_inputs(y, treatment, uplift_scores)
    order = np.argsort(-scores, kind="stable")
    y_sorted, t_sorted = y[order], treatment[order]

    cum_treated = np.cumsum(t_sorted)
    cum_control = np.cumsum(1 - t_sorted)
    cum_resp_treated = np.cumsum(y_sorted * t_sorted)
    cum_resp_control = np.cumsum(y_sorted * (1 - t_sorted))

    n = len(y)
    idx = np.unique(np.linspace(1, n, min(n_points, n)).astype(int)) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = cum_resp_treated[idx] - np.where(
            cum_control[idx] > 0,
            cum_resp_control[idx] * cum_treated[idx] / cum_control[idx],
            0.0,
        )
    return pd.DataFrame(
        {"fraction_targeted": (idx + 1) / n, "incremental_responders": qini}
    )


def qini_coefficient(
    y: Sequence[int],
    treatment: Sequence[int],
    uplift_scores: Sequence[float],
) -> float:
    """Area between the Qini curve and the random-targeting line, per user.

    Positive = the model ranks persuadable users better than chance; zero =
    no better than random targeting. Units: incremental responders per user
    (so values are comparable across sample sizes).
    """
    curve = qini_curve(y, treatment, uplift_scores, n_points=200)
    x = curve["fraction_targeted"].to_numpy()
    q = curve["incremental_responders"].to_numpy()
    random_line = x * q[-1]
    return float(np.trapezoid(q - random_line, x) / len(np.asarray(y)))


def cumulative_gain_curve(
    y: Sequence[int],
    treatment: Sequence[int],
    uplift_scores: Sequence[float],
    n_points: int = 100,
) -> pd.DataFrame:
    """Cumulative gain: estimated incremental responders if the top-f slice
    were fully treated — (rate_t(f) - rate_c(f)) x users targeted."""
    y, treatment, scores = _validate_eval_inputs(y, treatment, uplift_scores)
    order = np.argsort(-scores, kind="stable")
    y_sorted, t_sorted = y[order], treatment[order]

    cum_treated = np.cumsum(t_sorted)
    cum_control = np.cumsum(1 - t_sorted)
    cum_resp_treated = np.cumsum(y_sorted * t_sorted)
    cum_resp_control = np.cumsum(y_sorted * (1 - t_sorted))

    n = len(y)
    idx = np.unique(np.linspace(1, n, min(n_points, n)).astype(int)) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_gap = np.where(cum_treated[idx] > 0, cum_resp_treated[idx] / cum_treated[idx], 0.0) \
            - np.where(cum_control[idx] > 0, cum_resp_control[idx] / cum_control[idx], 0.0)
    return pd.DataFrame(
        {
            "fraction_targeted": (idx + 1) / n,
            "incremental_responders": rate_gap * (idx + 1),
        }
    )


def validate_against_true_uplift(
    predicted_uplift: Sequence[float],
    true_uplift: Sequence[float],
) -> dict[str, float]:
    """Score predictions against simulation ground truth.

    Only possible because the experiment is simulated with a stored per-user
    true uplift; production uplift models never get this luxury.
    """
    predicted = np.asarray(predicted_uplift, dtype=float)
    true = np.asarray(true_uplift, dtype=float)
    if len(predicted) != len(true):
        raise ValueError(
            f"length mismatch: predicted={len(predicted)}, true={len(true)}"
        )
    if len(predicted) < 3:
        raise ValueError("need >= 3 values to correlate")
    spearman = stats.spearmanr(predicted, true)
    return {
        "mse": float(np.mean((predicted - true) ** 2)),
        "rank_correlation": float(spearman.statistic),
        "pearson_correlation": float(np.corrcoef(predicted, true)[0, 1]),
    }
