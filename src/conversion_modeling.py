"""Single source of the paid-conversion modeling protocol (notebook 10).

Freezes the landmark design built by sql/build_conversion_table.sql:
population = users who had never paid as of the 2026-03-02 landmark;
label = paid start within the following 30 days (base rate 21.1%);
features = behavior strictly before the landmark. The snapshot framing
("free/trial at 2026-04-01") is untrainable in this data — converters are
already premium at the snapshot, leaving 30 positives — so the landmark
shift is what makes the same business question well-posed; notebook 10's
audit section shows the reconciliation.

Leakage policy: every feature is aggregated from events dated before the
landmark (enforced in the SQL), and payment/plan/revenue columns are banned
outright — `assert_leakage_safe` refuses any feature list that touches
them, because they encode the label or post-landmark information.

Baseline headline metrics (fitted by notebook 10 on the frozen split; a
future notebook can gate against these the way notebook 07 gated on 05):

    logistic regression    PR-AUC 0.7274, ROC-AUC 0.8939, Brier 0.0994
    hist gradient boosting PR-AUC 0.7255, ROC-AUC 0.8945, Brier 0.0994

Typical usage:

    from src.conversion_modeling import load_conversion_split

    split = load_conversion_split()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.churn_modeling import ModelingSplit
from src.config import RANDOM_SEED
from src.data_loader import connect

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET_COL = "paid_conversion_next_30d"
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "tenure_days_at_landmark",
    "active_days_w",
    "listen_events_w",
    "listen_minutes_w",
    "sessions_w",
    "skip_rate_w",
    "completion_rate_w",
    "liked_songs_w",
    "playlist_adds_w",
    "search_events_w",
    "ad_impressions_w",
    "ad_clicks_w",
    "ad_click_rate_w",
    "ad_revenue_w",
    "trial_exposed_pre",
    "trial_started_pre",
    "trial_expired_pre",
    "student_eligible",
    "marketing_opt_in",
]

CATEGORICAL_FEATURES = [
    "device",
    "acquisition_channel",
    "country",
    "age_group",
    "music_persona",
]

# Columns that must never be features: the label itself, anything measured
# after the landmark, and every payment/plan/revenue field from the other
# project tables (they encode the conversion event or post-label state).
FORBIDDEN_FEATURES = frozenset(
    {
        TARGET_COL,
        "paid_conversion_30d",
        "paid_started_30d",
        "paid_started_in_observation_30d",
        "renewal_success_30d",
        "renewal_success_14d",
        "renewal_success_count_30d",
        "payment_failed_30d",
        "payment_failed_count_30d",
        "cancel_30d",
        "cancel_count_30d",
        "subscription_revenue_30d",
        "subscription_revenue_observation_30d",
        "total_revenue_30d",
        "subscription_type",
        "current_subscription_type",
        "lifecycle_stage",
        "churn_label_14d",
        "listen_events_14d",
    }
)


def assert_leakage_safe(feature_columns: list[str]) -> None:
    """Raise if a proposed feature list contains a label-encoding column."""
    leaky = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if leaky:
        raise ValueError(
            f"leakage guard: {leaky} encode the conversion label or "
            "post-landmark state and must not be used as features"
        )


def qa_check_conversion_table(df: pd.DataFrame) -> None:
    """Raise with an actionable message if the conversion table is malformed."""
    for col in ("user_id", TARGET_COL):
        if col not in df.columns:
            raise ValueError(f"conversion table is missing required column {col!r}")
    duplicated = len(df) - df["user_id"].nunique()
    if duplicated:
        raise ValueError(
            f"QA failed: {duplicated} duplicated user rows — check JOIN keys"
        )
    if not set(df[TARGET_COL].unique()) <= {0, 1}:
        raise ValueError(f"QA failed: {TARGET_COL} must be 0/1")
    rate = df[TARGET_COL].mean()
    if not 0.02 <= rate <= 0.60:
        raise ValueError(
            f"QA failed: label rate {rate:.3f} is outside the plausible range — "
            "check the landmark windows in sql/build_conversion_table.sql"
        )


def split_conversion_table(df: pd.DataFrame) -> ModelingSplit:
    """Stratified 80/20 split with the project seed on a prepared table."""
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    assert_leakage_safe(numeric + categorical)

    X = df[numeric + categorical].copy()
    y = df[TARGET_COL].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    return ModelingSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_features=numeric,
        categorical_features=categorical,
    )


def load_conversion_split(data_dir: Path | str | None = None) -> ModelingSplit:
    """Build the conversion table from SQL and return the frozen split."""
    _, sql, run_script = connect(data_dir)
    run_script((REPO_ROOT / "sql" / "build_conversion_table.sql").read_text())
    table = sql("SELECT * FROM conversion_table")
    qa_check_conversion_table(table)
    return split_conversion_table(table)
