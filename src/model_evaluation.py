"""Shared model-evaluation harness for churn-model comparisons.

Makes notebook-05-style comparisons reproducible: a tidy metrics table for
any set of fitted models on one fixed test split, calibration tables, an
operating-point (precision at a business recall) helper, and a paired
bootstrap on AUC differences — resampling the same test indices for both
models so the confidence interval is on the *difference*, not two marginal
CIs whose overlap says little.

Typical usage:

    from src.model_evaluation import evaluate_models, paired_bootstrap_auc_delta

    table = evaluate_models({"logistic": log_model, "hist_gb": gb_model},
                            X_test, y_test, threshold=0.5)
    delta = paired_bootstrap_auc_delta(y_test, log_proba, gb_proba)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import RANDOM_SEED


def _validate_binary_arrays(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    if len(y_true) != len(y_proba):
        raise ValueError(
            f"y_true and y_proba lengths differ: {len(y_true)} vs {len(y_proba)}"
        )
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("y_true must be binary 0/1")
    if np.min(y_proba) < 0 or np.max(y_proba) > 1:
        raise ValueError("y_proba must be probabilities in [0, 1]")


def threshold_metrics(
    y_true: Sequence[int], y_proba: Sequence[float], threshold: float
) -> dict[str, float]:
    """Precision/recall/F1 and flagged share at one decision threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    _validate_binary_arrays(y_true, y_proba)
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    y_pred = (y_proba >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flagged_share": float(y_pred.mean()),
    }


def evaluate_models(
    models: Mapping[str, object],
    X_test: pd.DataFrame,
    y_test: Sequence[int],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Tidy metrics table for fitted models on one fixed test split.

    Ranking metrics (ROC-AUC, PR-AUC), calibration (Brier score), and the
    business operating point (precision/recall/F1 at `threshold`). Every
    model must implement predict_proba.
    """
    if not models:
        raise ValueError("models must be non-empty")
    y_true = np.asarray(y_test)

    rows = []
    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"model {name!r} has no predict_proba method")
        y_proba = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
        _validate_binary_arrays(y_true, y_proba)
        at_threshold = threshold_metrics(y_true, y_proba, threshold)
        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_true, y_proba),
                "pr_auc": average_precision_score(y_true, y_proba),
                "brier": brier_score_loss(y_true, y_proba),
                "precision_at_threshold": at_threshold["precision"],
                "recall_at_threshold": at_threshold["recall"],
                "f1_at_threshold": at_threshold["f1"],
                "flagged_share": at_threshold["flagged_share"],
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BootstrapDelta:
    """Paired-bootstrap estimate of metric_b - metric_a on a shared test set."""

    metric: str
    value_a: float
    value_b: float
    delta: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    significant: bool  # CI excludes zero


def paired_bootstrap_auc_delta(
    y_true: Sequence[int],
    proba_a: Sequence[float],
    proba_b: Sequence[float],
    n_bootstrap: int = 2_000,
    alpha: float = 0.05,
    metric: str = "roc_auc",
    seed: int = RANDOM_SEED,
) -> BootstrapDelta:
    """Percentile CI for AUC(model_b) - AUC(model_a) via paired bootstrap.

    Each replicate resamples test indices once and scores both models on the
    same resample, so model-to-model correlation is preserved and the CI is
    on the difference itself. `metric` is "roc_auc" or "pr_auc".
    """
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)
    _validate_binary_arrays(y_true, proba_a)
    _validate_binary_arrays(y_true, proba_b)
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap must be >= 100, got {n_bootstrap}")
    scorers = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}
    if metric not in scorers:
        raise ValueError(f"metric must be one of {list(scorers)}, got {metric!r}")
    score = scorers[metric]

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_bootstrap)
    filled = 0
    while filled < n_bootstrap:
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        if y_boot.min() == y_boot.max():  # single-class resample: AUC undefined
            continue
        deltas[filled] = score(y_boot, proba_b[idx]) - score(y_boot, proba_a[idx])
        filled += 1

    ci_low, ci_high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return BootstrapDelta(
        metric=metric,
        value_a=float(score(y_true, proba_a)),
        value_b=float(score(y_true, proba_b)),
        delta=float(score(y_true, proba_b) - score(y_true, proba_a)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_bootstrap=n_bootstrap,
        significant=bool(ci_low > 0 or ci_high < 0),
    )


def calibration_table(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Quantile-binned calibration: mean predicted vs observed churn per bin.

    Quantile bins keep counts comparable across bins (uniform bins can leave
    sparse tails). A perfectly calibrated model has mean_predicted equal to
    observed_rate in every bin.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    _validate_binary_arrays(y_true, y_proba)
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    frame = pd.DataFrame({"y": y_true, "proba": y_proba})
    frame["bin"] = pd.qcut(frame["proba"], q=n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        mean_predicted=("proba", "mean"),
        observed_rate=("y", "mean"),
        count=("y", "size"),
    )
    return grouped.reset_index(drop=True)


def lift_table(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50),
) -> pd.DataFrame:
    """Campaign view of a ranking model: contact the top-f by score.

    For each fraction f: precision within the targeted slice, share of all
    positives captured, and lift versus contacting users at random (whose
    precision is the base rate).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    _validate_binary_arrays(y_true, y_proba)
    fractions = list(fractions)
    if not fractions or not all(0 < f <= 1 for f in fractions):
        raise ValueError(f"fractions must be in (0, 1], got {fractions}")

    order = np.argsort(-y_proba, kind="stable")
    y_sorted = y_true[order]
    base_rate = y_true.mean()
    total_positives = int(y_true.sum())

    rows = []
    for fraction in fractions:
        k = max(1, int(round(len(y_true) * fraction)))
        captured = int(y_sorted[:k].sum())
        precision = captured / k
        rows.append(
            {
                "fraction_targeted": fraction,
                "n_contacted": k,
                "precision": precision,
                "share_of_positives_captured": captured / total_positives
                if total_positives
                else np.nan,
                "lift_vs_random": precision / base_rate if base_rate else np.nan,
            }
        )
    return pd.DataFrame(rows)


def precision_at_recall(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    target_recall: float,
) -> dict[str, float]:
    """Operating point: the highest threshold whose recall meets the target.

    Returns that threshold with its precision, achieved recall, flagged
    share, and contacts_per_catch (users flagged per churner caught = the
    outreach cost of one save at this operating point).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    _validate_binary_arrays(y_true, y_proba)
    if not 0 < target_recall <= 1:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns recalls descending toward 0 as the
    # threshold rises; pick the highest threshold still meeting the target.
    meets = recalls[:-1] >= target_recall
    if not meets.any():
        raise ValueError(
            f"no threshold reaches recall {target_recall}; max recall is 1.0 "
            "only when flagging everyone — check the inputs"
        )
    best = np.where(meets)[0][-1]
    threshold = float(thresholds[best])
    at = threshold_metrics(y_true, y_proba, threshold)
    caught = at["recall"] * y_true.mean()
    return {
        "threshold": threshold,
        "precision": at["precision"],
        "recall": at["recall"],
        "flagged_share": at["flagged_share"],
        "contacts_per_catch": at["flagged_share"] / caught if caught else float("inf"),
    }
