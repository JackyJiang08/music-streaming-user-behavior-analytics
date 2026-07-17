"""Thin, tested wrappers around lifelines for time-to-churn analysis.

Used by notebook 08: Kaplan-Meier fits by segment, a tidy log-rank test,
Cox proportional-hazards with standardized inputs, the Schoenfeld-residual
proportional-hazards check, and held-out concordance evaluation. The
survival table itself is built by sql/build_survival_table.sql (see
scripts/build_survival_table.py); `qa_check_survival_table` holds its
QA guards so they are unit-testable.

Typical usage:

    from src.survival_analysis import km_by_segment, logrank_by_segment

    curves = km_by_segment(df, "acquisition_channel")
    logrank = logrank_by_segment(df, "acquisition_channel")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index, median_survival_times

DURATION_COL = "duration_days"
EVENT_COL = "churn_event"


def qa_check_survival_table(df: pd.DataFrame) -> None:
    """Raise with an actionable message if the survival table is malformed."""
    for col in ("user_id", DURATION_COL, EVENT_COL):
        if col not in df.columns:
            raise ValueError(f"survival table is missing required column {col!r}")
    duplicated = len(df) - df["user_id"].nunique()
    if duplicated:
        raise ValueError(
            f"QA failed: {duplicated} duplicated user rows — check JOIN keys"
        )
    negative = int((df[DURATION_COL] <= 0).sum())
    if negative:
        raise ValueError(
            f"QA failed: {negative} non-positive durations — check the horizon "
            "and signup dates"
        )
    if not set(df[EVENT_COL].unique()) <= {0, 1}:
        raise ValueError("QA failed: churn_event must be 0/1")


def fit_km(
    df: pd.DataFrame,
    label: str = "all users",
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
) -> KaplanMeierFitter:
    """Fit one Kaplan-Meier curve."""
    if df.empty:
        raise ValueError("cannot fit a Kaplan-Meier curve on an empty frame")
    kmf = KaplanMeierFitter(label=label)
    kmf.fit(df[duration_col], event_observed=df[event_col])
    return kmf


def km_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
    min_group_size: int = 50,
) -> dict[str, KaplanMeierFitter]:
    """Fit one Kaplan-Meier curve per segment value.

    Groups below `min_group_size` are dropped (their curves are noise);
    raises if that leaves fewer than two groups to compare.
    """
    if segment_col not in df.columns:
        raise ValueError(f"segment column {segment_col!r} not found")
    curves: dict[str, KaplanMeierFitter] = {}
    for value, group in df.groupby(segment_col, observed=True):
        if len(group) >= min_group_size:
            curves[str(value)] = fit_km(group, str(value), duration_col, event_col)
    if len(curves) < 2:
        raise ValueError(
            f"fewer than two segments of {segment_col!r} have >= "
            f"{min_group_size} users; nothing to compare"
        )
    return curves


def median_survival_summary(kmf: KaplanMeierFitter) -> dict[str, float]:
    """Median survival time with its 95% CI (NaN when the curve never
    crosses 50%)."""
    ci = median_survival_times(kmf.confidence_interval_)
    return {
        "median": float(kmf.median_survival_time_),
        "ci_low": float(ci.iloc[0, 0]),
        "ci_high": float(ci.iloc[0, 1]),
    }


@dataclass(frozen=True)
class LogRankResult:
    """Tidy multivariate log-rank test outcome for one segmentation."""

    segment_col: str
    n_groups: int
    test_statistic: float
    degrees_of_freedom: int
    p_value: float
    significant: bool


def logrank_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
    alpha: float = 0.05,
    min_group_size: int = 50,
) -> LogRankResult:
    """Multivariate log-rank test: do the segment survival curves differ?

    Applies the same `min_group_size` filter as `km_by_segment` so the test
    covers exactly the curves being plotted.
    """
    if segment_col not in df.columns:
        raise ValueError(f"segment column {segment_col!r} not found")
    sizes = df[segment_col].value_counts()
    kept = sizes[sizes >= min_group_size].index
    subset = df[df[segment_col].isin(kept)]
    n_groups = subset[segment_col].nunique()
    if n_groups < 2:
        raise ValueError(
            f"need >= 2 segments with >= {min_group_size} users, got {n_groups}"
        )
    result = multivariate_logrank_test(
        subset[duration_col], subset[segment_col], subset[event_col]
    )
    return LogRankResult(
        segment_col=segment_col,
        n_groups=n_groups,
        test_statistic=float(result.test_statistic),
        degrees_of_freedom=n_groups - 1,
        p_value=float(result.p_value),
        significant=bool(result.p_value < alpha),
    )


def build_cox_design(
    df: pd.DataFrame,
    numeric_covariates: list[str],
    categorical_covariates: list[str],
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
    numeric_stats: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Design matrix for Cox: standardized numerics + drop-first dummies.

    Numerics are z-scored so hazard ratios read "per one standard
    deviation". Pass the returned `numeric_stats` back in when transforming
    a held-out split, so test data is standardized with training statistics.
    """
    missing = [
        c for c in numeric_covariates + categorical_covariates + [duration_col, event_col]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"dataframe is missing columns: {missing}")

    if numeric_stats is None:
        numeric_stats = pd.DataFrame(
            {"mean": df[numeric_covariates].mean(), "std": df[numeric_covariates].std()}
        )
        if (numeric_stats["std"] == 0).any():
            constant = numeric_stats.index[numeric_stats["std"] == 0].tolist()
            raise ValueError(f"constant numeric covariates cannot be scaled: {constant}")

    design = pd.DataFrame(index=df.index)
    for col in numeric_covariates:
        design[col] = (df[col] - numeric_stats.loc[col, "mean"]) / numeric_stats.loc[
            col, "std"
        ]
    if categorical_covariates:
        dummies = pd.get_dummies(
            df[categorical_covariates], drop_first=True, dtype=float
        )
        design = pd.concat([design, dummies], axis=1)
    design[duration_col] = df[duration_col].to_numpy()
    design[event_col] = df[event_col].to_numpy()
    return design, numeric_stats


def fit_cox(
    design: pd.DataFrame,
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
    penalizer: float = 0.0,
    strata: list[str] | None = None,
) -> CoxPHFitter:
    """Fit a Cox proportional-hazards model on a prepared design matrix.

    `strata` moves covariates out of the linear predictor into stratified
    baseline hazards — the standard remedy when a covariate violates the
    proportional-hazards assumption.
    """
    fitter = CoxPHFitter(penalizer=penalizer)
    fitter.fit(design, duration_col=duration_col, event_col=event_col, strata=strata)
    return fitter


def hazard_ratio_table(fitter: CoxPHFitter) -> pd.DataFrame:
    """Tidy hazard ratios: HR with 95% CI and p-value per covariate."""
    summary = fitter.summary
    return pd.DataFrame(
        {
            "covariate": summary.index.get_level_values(-1),
            "hazard_ratio": summary["exp(coef)"].to_numpy(),
            "ci_low": summary["exp(coef) lower 95%"].to_numpy(),
            "ci_high": summary["exp(coef) upper 95%"].to_numpy(),
            "p_value": summary["p"].to_numpy(),
        }
    ).reset_index(drop=True)


def check_proportional_hazards(
    fitter: CoxPHFitter,
    design: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Schoenfeld-residual test of the PH assumption, one row per covariate.

    `violated=True` means the covariate's effect drifts over time; handle it
    (stratify, or interpret as a time-averaged effect) rather than ignoring it.
    """
    result = proportional_hazard_test(fitter, design, time_transform="rank")
    summary = result.summary.reset_index()
    covariate_col = summary.columns[0]
    return pd.DataFrame(
        {
            "covariate": summary[covariate_col].astype(str),
            "test_statistic": summary["test_statistic"].to_numpy(),
            "p_value": summary["p"].to_numpy(),
            "violated": (summary["p"] < alpha).to_numpy(),
        }
    )


def concordance_on_holdout(
    fitter: CoxPHFitter,
    design_test: pd.DataFrame,
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
) -> float:
    """C-index on held-out users: higher predicted hazard should mean
    shorter observed survival."""
    if design_test.empty:
        raise ValueError("design_test is empty")
    partial_hazard = fitter.predict_partial_hazard(design_test)
    return float(
        concordance_index(
            design_test[duration_col], -partial_hazard, design_test[event_col]
        )
    )
