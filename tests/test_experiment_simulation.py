"""Unit tests for src/experiment_simulation.py on a small synthetic population."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiment_simulation import (
    SEGMENT_UPLIFT_PP,
    label_effect_segment,
    simulate_experiment,
    simulate_novelty_trajectory,
)


@pytest.fixture
def population() -> pd.DataFrame:
    """Synthetic population shaped like ab_test_population, ~8K users.

    Activity/skip profiles are drawn so all four effect segments are
    populated and the weighted uplift stays near the 2.2pp design target.
    """
    rng = np.random.default_rng(1)
    n = 8_000
    active_days = rng.choice(
        [0, 1, 2, 4, 6, 8, 12, 20],
        size=n,
        p=[0.10, 0.12, 0.10, 0.14, 0.14, 0.12, 0.14, 0.14],
    )
    skip = np.clip(rng.normal(0.16, 0.10, n), 0, 0.9)
    skip[rng.random(n) < 0.13] = 0.5  # carve out a high-skipper segment
    churn_prob = np.clip(0.8 - 0.04 * active_days, 0.1, 0.9)
    return pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(n)],
            "current_subscription_type": rng.choice(["free", "trial"], size=n),
            "tenure_days": rng.integers(18, 400, size=n),
            "active_days_30d": active_days,
            "listen_minutes_30d": rng.gamma(2, 30, size=n) * (active_days > 0),
            "skip_rate_30d": skip,
            "playlist_adds_30d": rng.poisson(0.4, size=n),
            "ad_revenue_30d": rng.gamma(1, 0.06, size=n),
            "cancel_count_30d": (rng.random(n) < 0.03).astype(int),
            "churn_label_14d": (rng.random(n) < churn_prob).astype(int),
        }
    )


def test_segments_cover_everyone_and_respect_definitions(population):
    seg = label_effect_segment(population)
    assert set(seg.unique()) == set(SEGMENT_UPLIFT_PP)
    high_skippers = population["skip_rate_30d"] > 0.35
    assert (seg[high_skippers] == "high_skipper").all()
    low = ~high_skippers & (population["active_days_30d"] <= 2)
    assert (seg[low] == "low_activity").all()


def test_control_arm_matches_observed_labels(population):
    out = simulate_experiment(population)
    control = out[out["group"] == "control"]
    assert (control["retained_14d_post"] == 1 - control["churn_label_14d"]).all()


def test_heterogeneous_uplift_is_recovered_per_segment(population):
    out = simulate_experiment(population, seed=11)
    treated = out[out["group"] == "treatment"]
    # Compare each treated segment against its own counterfactual baseline so
    # the check isolates the injected effect from between-arm sampling noise.
    for segment, uplift_pp in SEGMENT_UPLIFT_PP.items():
        seg = treated[treated["effect_segment"] == segment]
        realized = (
            seg["retained_14d_post"].mean() - (1 - seg["churn_label_14d"]).mean()
        ) * 100
        assert realized == pytest.approx(uplift_pp, abs=1.6), segment
    assert (out["true_uplift_pp"] == out["effect_segment"].map(SEGMENT_UPLIFT_PP)).all()


def test_non_retained_users_have_no_post_activity(population):
    out = simulate_experiment(population)
    gone = out[out["retained_14d_post"] == 0]
    assert (gone["listen_minutes_14d_post"] == 0).all()
    assert (gone["active_days_14d_post"] == 0).all()
    assert (gone["playlist_added_14d_post"] == 0).all()
    assert gone["skip_rate_14d_post"].isna().all()
    assert (gone["ad_revenue_14d_post"] == 0).all()


def test_cancel_only_defined_for_trial_users(population):
    out = simulate_experiment(population)
    assert (
        out.loc[out["current_subscription_type"] == "free", "cancel_14d_post"]
        .isna()
        .all()
    )
    trial = out.loc[out["current_subscription_type"] == "trial", "cancel_14d_post"]
    assert trial.notna().all()
    assert trial.isin([0.0, 1.0]).all()


def test_simulation_is_deterministic(population):
    a = simulate_experiment(population, seed=5)
    b = simulate_experiment(population, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_simulation_rejects_missing_columns(population):
    with pytest.raises(ValueError, match="missing required columns"):
        simulate_experiment(population.drop(columns=["skip_rate_30d"]))


def test_novelty_trajectory_decays_toward_long_run():
    df = simulate_novelty_trajectory(
        n_per_arm=20_000, mean_weekly_minutes=25, n_weeks=6, seed=9
    )
    assert len(df) == 6
    assert df["true_rel_lift"].is_monotonic_decreasing
    assert df["rel_lift"].iloc[0] > df["rel_lift"].iloc[-1]
    # week 1 overstates the long-run truth by well over half
    assert df["rel_lift"].iloc[0] > 1.5 * df["true_rel_lift"].iloc[-1]


def test_novelty_trajectory_validates_inputs():
    with pytest.raises(ValueError):
        simulate_novelty_trajectory(n_per_arm=1, mean_weekly_minutes=25)
    with pytest.raises(ValueError):
        simulate_novelty_trajectory(n_per_arm=100, mean_weekly_minutes=0)
