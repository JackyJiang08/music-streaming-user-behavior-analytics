"""Unit tests for the survival table SQL and src/survival_analysis.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.survival_analysis import (
    build_cox_design,
    check_proportional_hazards,
    concordance_on_holdout,
    fit_cox,
    fit_km,
    hazard_ratio_table,
    km_by_segment,
    logrank_by_segment,
    median_survival_summary,
    qa_check_survival_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SURVIVAL_SQL = (REPO_ROOT / "sql" / "build_survival_table.sql").read_text()


# ---------------------------------------------------------------------------
# Duration/event construction (the SQL itself) on a synthetic event log
# ---------------------------------------------------------------------------

@pytest.fixture
def survival_from_synthetic_log() -> pd.DataFrame:
    """Run the real SQL on a tiny event log with hand-computed answers.

    Anchor user Z (pre-window signup) pins the observation window to
    2026-01-01..2026-03-01 and must be excluded from the cohort itself.
    """
    users = pd.DataFrame(
        [
            # user, signup, channel, device, age, persona
            ("Z", "2025-12-01", "organic_search", "web", "25-34", "background"),
            ("A", "2026-01-01", "referral", "mobile_ios", "18-24", "explorer"),
            ("B", "2026-01-10", "paid_social", "desktop", "25-34", "commuter"),
            ("C", "2026-01-15", "app_store", "mobile_android", "35-44", "casual"),
            ("D", "2026-01-05", "referral", "mobile_ios", "18-24", "explorer"),
        ],
        columns=["user_id", "signup_date", "acquisition_channel",
                 "primary_device", "age_group", "music_persona"],
    )
    events = []

    def add(user, day, playlist=0, liked=0, skipped=0):
        events.append((user, f"{day} 12:00:00", skipped, liked, playlist))

    add("Z", "2026-01-01")           # anchors obs_start
    add("Z", "2026-03-01")           # anchors horizon
    # A: gap Jan 5 -> Feb 1 (27d >= 14): event at Jan 5 + 14 = day 18.
    add("A", "2026-01-02"); add("A", "2026-01-05"); add("A", "2026-02-01")
    # B: no gap >= 14 and last event 9d before horizon: censored at day 50.
    add("B", "2026-01-12", playlist=1); add("B", "2026-01-20")
    add("B", "2026-02-01"); add("B", "2026-02-10"); add("B", "2026-02-20")
    # C: never listened: event at signup + 14 = day 14.
    # D: tail gap Jan 10 -> horizon (50d >= 14): event at Jan 10 + 14 = day 19.
    add("D", "2026-01-06", liked=1); add("D", "2026-01-10")

    events_df = pd.DataFrame(
        events,
        columns=["user_id", "event_timestamp", "skipped_flag", "liked_flag",
                 "playlist_add_flag"],
    )
    conn = sqlite3.connect(":memory:")
    users.to_sql("users", conn, index=False)
    events_df.to_sql("listening_events", conn, index=False)
    conn.executescript(SURVIVAL_SQL)
    return pd.read_sql_query("SELECT * FROM survival_table", conn).set_index("user_id")


def test_sql_excludes_pre_window_signups(survival_from_synthetic_log):
    assert "Z" not in survival_from_synthetic_log.index
    assert set(survival_from_synthetic_log.index) == {"A", "B", "C", "D"}


def test_sql_mid_history_gap_is_an_event(survival_from_synthetic_log):
    row = survival_from_synthetic_log.loc["A"]
    assert row["churn_event"] == 1
    assert row["duration_days"] == 18  # Jan 5 + 14 days, despite Feb 1 return


def test_sql_active_user_is_censored_at_horizon(survival_from_synthetic_log):
    row = survival_from_synthetic_log.loc["B"]
    assert row["churn_event"] == 0
    assert row["duration_days"] == 50  # Jan 10 signup -> Mar 1 horizon


def test_sql_never_listened_user_churns_at_day_14(survival_from_synthetic_log):
    row = survival_from_synthetic_log.loc["C"]
    assert row["churn_event"] == 1
    assert row["duration_days"] == 14


def test_sql_tail_gap_is_an_event(survival_from_synthetic_log):
    row = survival_from_synthetic_log.loc["D"]
    assert row["churn_event"] == 1
    assert row["duration_days"] == 19  # Jan 10 + 14 days from Jan 5 signup


def test_sql_week1_landmark_covariates(survival_from_synthetic_log):
    assert survival_from_synthetic_log.loc["A", "active_days_w1"] == 2
    assert survival_from_synthetic_log.loc["B", "playlist_or_like_w1"] == 1
    assert survival_from_synthetic_log.loc["C", "active_days_w1"] == 0
    assert survival_from_synthetic_log.loc["D", "playlist_or_like_w1"] == 1


# ---------------------------------------------------------------------------
# QA guards
# ---------------------------------------------------------------------------

def test_qa_guard_fires_on_negative_duration():
    bad = pd.DataFrame(
        {"user_id": ["U1", "U2"], "duration_days": [10, -3], "churn_event": [1, 0]}
    )
    with pytest.raises(ValueError, match="non-positive durations"):
        qa_check_survival_table(bad)


def test_qa_guard_fires_on_duplicate_users():
    bad = pd.DataFrame(
        {"user_id": ["U1", "U1"], "duration_days": [10, 12], "churn_event": [1, 0]}
    )
    with pytest.raises(ValueError, match="duplicated"):
        qa_check_survival_table(bad)


# ---------------------------------------------------------------------------
# lifelines wrappers on synthetic survival data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_survival() -> pd.DataFrame:
    """Two groups with a known hazard gap plus censoring at t=60."""
    rng = np.random.default_rng(0)
    n = 800
    fast = rng.random(n) < 0.5  # fast-churn group: half the users
    scale = np.where(fast, 10.0, 60.0)
    t_true = rng.exponential(scale)
    duration = np.minimum(t_true, 60.0)
    return pd.DataFrame(
        {
            "user_id": [f"U{i}" for i in range(n)],
            "duration_days": np.maximum(duration, 0.5),
            "churn_event": (t_true <= 60).astype(int),
            "group": np.where(fast, "fast", "slow"),
            "x_risk": fast.astype(float) + rng.normal(0, 0.1, n),
        }
    )


def test_km_by_segment_and_median(synthetic_survival):
    curves = km_by_segment(synthetic_survival, "group")
    assert set(curves) == {"fast", "slow"}
    fast = median_survival_summary(curves["fast"])
    slow = median_survival_summary(curves["slow"])
    assert fast["median"] < slow["median"]
    assert fast["ci_low"] <= fast["median"] <= fast["ci_high"]


def test_km_by_segment_needs_two_groups(synthetic_survival):
    lone = synthetic_survival[synthetic_survival["group"] == "fast"]
    with pytest.raises(ValueError, match="fewer than two"):
        km_by_segment(lone, "group")


def test_logrank_detects_the_known_gap(synthetic_survival):
    result = logrank_by_segment(synthetic_survival, "group")
    assert result.n_groups == 2
    assert result.degrees_of_freedom == 1
    assert result.p_value < 1e-6
    assert result.significant


def test_cox_recovers_risk_direction_and_generalizes(synthetic_survival):
    train = synthetic_survival.iloc[:600]
    test = synthetic_survival.iloc[600:]
    design_train, stats = build_cox_design(train, ["x_risk"], [])
    design_test, _ = build_cox_design(test, ["x_risk"], [], numeric_stats=stats)

    fitter = fit_cox(design_train)
    table = hazard_ratio_table(fitter)
    assert list(table.columns) == ["covariate", "hazard_ratio", "ci_low",
                                   "ci_high", "p_value"]
    risk_row = table.set_index("covariate").loc["x_risk"]
    assert risk_row["hazard_ratio"] > 1.5  # higher risk score -> faster churn
    assert risk_row["ci_low"] > 1

    c_index = concordance_on_holdout(fitter, design_test)
    assert c_index > 0.65

    ph = check_proportional_hazards(fitter, design_train)
    assert list(ph.columns) == ["covariate", "test_statistic", "p_value", "violated"]
    assert len(ph) == 1


def test_build_cox_design_rejects_constant_numeric(synthetic_survival):
    frame = synthetic_survival.assign(constant=1.0)
    with pytest.raises(ValueError, match="constant"):
        build_cox_design(frame, ["constant"], [])


def test_fit_km_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        fit_km(pd.DataFrame(columns=["duration_days", "churn_event"]))
