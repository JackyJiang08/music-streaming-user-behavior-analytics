"""Unit tests for the conversion protocol (SQL + src/conversion_modeling.py)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.conversion_modeling import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    assert_leakage_safe,
    qa_check_conversion_table,
    split_conversion_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERSION_SQL = (REPO_ROOT / "sql" / "build_conversion_table.sql").read_text()


# ---------------------------------------------------------------------------
# The SQL itself on a synthetic event log (landmark = 2026-03-02)
# ---------------------------------------------------------------------------

@pytest.fixture
def conversion_from_synthetic_log() -> pd.DataFrame:
    users = pd.DataFrame(
        [
            # A: never paid, converts in label window -> label 1
            ("A", "2026-01-05", "mobile_ios", "referral", "US", "18-24", "explorer", 1, 1),
            # B: paid BEFORE the landmark -> excluded from population
            ("B", "2026-01-05", "web", "paid_social", "UK", "25-34", "commuter", 0, 0),
            # C: never paid, no conversion -> label 0
            ("C", "2026-02-10", "desktop", "app_store", "US", "35-44", "casual", 0, 1),
            # D: signed up after the landmark -> excluded
            ("D", "2026-03-10", "mobile_ios", "referral", "US", "18-24", "explorer", 0, 0),
        ],
        columns=["user_id", "signup_date", "primary_device", "acquisition_channel",
                 "country", "age_group", "music_persona", "student_eligible",
                 "marketing_opt_in"],
    )
    subscription_events = pd.DataFrame(
        [
            ("B", "2026-02-01 10:00:00", "paid_started"),
            ("A", "2026-02-15 10:00:00", "trial_started"),   # pre-landmark funnel
            ("A", "2026-03-10 10:00:00", "paid_started"),    # label window
            ("C", "2026-04-20 10:00:00", "paid_started"),    # after label window
        ],
        columns=["user_id", "event_timestamp", "event_type"],
    )
    listening_events = pd.DataFrame(
        [
            # A: one event inside the feature window, one AFTER the landmark
            # (the second must not count toward features).
            ("A", "2026-02-20 09:00:00", "A_s1", 200, 180, 0, 0, 1, 1, 0),
            ("A", "2026-03-05 09:00:00", "A_s2", 200, 100, 0, 1, 0, 0, 0),
            # C: event before the feature window opens (must not count).
            ("C", "2026-01-15 09:00:00", "C_s1", 200, 50, 0, 1, 0, 0, 0),
        ],
        columns=["user_id", "event_timestamp", "session_id", "track_duration_sec",
                 "play_duration_sec", "completed_flag", "skipped_flag",
                 "liked_flag", "playlist_add_flag", "search_used_flag"],
    )
    ad_events = pd.DataFrame(
        [("A", "2026-02-21 09:00:00", 1, 0.01), ("A", "2026-03-06 09:00:00", 1, 0.01)],
        columns=["user_id", "event_timestamp", "clicked_flag", "revenue_usd"],
    )
    conn = sqlite3.connect(":memory:")
    users.to_sql("users", conn, index=False)
    subscription_events.to_sql("subscription_events", conn, index=False)
    listening_events.to_sql("listening_events", conn, index=False)
    ad_events.to_sql("ad_events", conn, index=False)
    conn.executescript(CONVERSION_SQL)
    return pd.read_sql_query("SELECT * FROM conversion_table", conn).set_index("user_id")


def test_sql_population_excludes_already_paid_and_post_landmark_signups(
    conversion_from_synthetic_log,
):
    assert set(conversion_from_synthetic_log.index) == {"A", "C"}  # B, D excluded


def test_sql_label_windows(conversion_from_synthetic_log):
    assert conversion_from_synthetic_log.loc["A", TARGET_COL] == 1
    # C's paid start falls after the 30-day label window -> label 0.
    assert conversion_from_synthetic_log.loc["C", TARGET_COL] == 0


def test_sql_features_stop_at_the_landmark(conversion_from_synthetic_log):
    row = conversion_from_synthetic_log.loc["A"]
    assert row["listen_events_w"] == 1      # post-landmark event excluded
    assert row["ad_impressions_w"] == 1     # post-landmark impression excluded
    assert row["trial_started_pre"] == 1
    # C's only listening event predates the feature window.
    assert conversion_from_synthetic_log.loc["C", "listen_events_w"] == 0


def test_feature_lists_resolve_on_the_real_sql_schema(conversion_from_synthetic_log):
    # Schema-only check: every declared feature must exist in the view the
    # real SQL produces (no full-dataset training involved).
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
               if c not in conversion_from_synthetic_log.reset_index().columns]
    assert missing == []


# ---------------------------------------------------------------------------
# Leakage guard and QA
# ---------------------------------------------------------------------------

def test_leakage_guard_raises_on_label_encoding_column():
    with pytest.raises(ValueError, match="leakage guard"):
        assert_leakage_safe(["active_days_w", "paid_started_30d"])
    with pytest.raises(ValueError, match="leakage guard"):
        assert_leakage_safe(["subscription_revenue_30d"])
    assert_leakage_safe(["active_days_w", "device"])  # clean list passes


def test_qa_check_rejects_bad_tables():
    with pytest.raises(ValueError, match="duplicated"):
        qa_check_conversion_table(
            pd.DataFrame({"user_id": ["U1", "U1"], TARGET_COL: [0, 1]})
        )
    with pytest.raises(ValueError, match="label rate"):
        qa_check_conversion_table(
            pd.DataFrame({"user_id": ["U1", "U2"], TARGET_COL: [1, 1]})
        )


# ---------------------------------------------------------------------------
# Split determinism
# ---------------------------------------------------------------------------

@pytest.fixture
def prepared_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(n)],
        TARGET_COL: (rng.random(n) < 0.2).astype(int),
        "active_days_w": rng.integers(0, 28, n),
        "listen_minutes_w": rng.gamma(2, 20, n),
        "device": rng.choice(["mobile_ios", "web"], n),
        "acquisition_channel": rng.choice(["referral", "paid_social"], n),
    })
    return frame


def test_split_is_deterministic_and_stratified(prepared_table):
    a = split_conversion_table(prepared_table)
    b = split_conversion_table(prepared_table)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_series_equal(a.y_test, b.y_test)
    assert a.y_train.mean() == pytest.approx(a.y_test.mean(), abs=0.02)
    # Only features present in the frame are kept.
    assert a.numeric_features == ["active_days_w", "listen_minutes_w"]
    assert a.categorical_features == ["device", "acquisition_channel"]
