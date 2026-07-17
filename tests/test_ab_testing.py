"""Unit tests for src/ab_testing.py on small synthetic fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ab_testing import (
    MetricSpec,
    assign_groups,
    mean_diff_test,
    minimum_detectable_effect,
    peeking_false_positive_rate,
    proportion_ztest,
    required_sample_size,
    sequential_pvalues,
    srm_check,
    summarize_experiment,
)


# ---------------------------------------------------------------------------
# Sample size and MDE
# ---------------------------------------------------------------------------

def test_sample_size_matches_textbook_value():
    # Fleiss et al.: p=0.50 vs 0.52, alpha=0.05, power=0.80 -> ~9,807 per arm.
    n = required_sample_size(baseline_rate=0.50, mde_pp=2.0)
    assert 9_800 <= n <= 9_815


def test_sample_size_grows_for_smaller_effects():
    assert required_sample_size(0.45, 1.0) > required_sample_size(0.45, 5.0)


def test_sample_size_rejects_impossible_rates():
    with pytest.raises(ValueError):
        required_sample_size(0.99, 5.0)
    with pytest.raises(ValueError):
        required_sample_size(0.5, 0.0)


def test_mde_roundtrips_with_sample_size():
    baseline, mde_pp = 0.45, 2.0
    n = required_sample_size(baseline, mde_pp)
    assert minimum_detectable_effect(n, baseline) == pytest.approx(mde_pp, rel=0.05)


def test_mde_shrinks_with_more_users():
    assert minimum_detectable_effect(20_000, 0.45) < minimum_detectable_effect(
        5_000, 0.45
    )


# ---------------------------------------------------------------------------
# Assignment and SRM
# ---------------------------------------------------------------------------

def test_assignment_is_deterministic_and_balanced():
    ids = [f"U{i:05d}" for i in range(10_000)]
    a = assign_groups(ids, seed=7)
    b = assign_groups(ids, seed=7)
    pd.testing.assert_frame_equal(a, b)
    share = (a["group"] == "treatment").mean()
    assert 0.48 <= share <= 0.52
    assert set(a["group"]) == {"control", "treatment"}


def test_assignment_rejects_duplicate_users():
    with pytest.raises(ValueError, match="duplicates"):
        assign_groups(["U1", "U2", "U1"])


def test_srm_passes_a_fair_split_and_flags_a_rigged_one():
    fair = srm_check(n_control=5_000, n_treatment=5_050)
    assert fair.passed
    rigged = srm_check(n_control=6_000, n_treatment=4_000)  # 60/40
    assert not rigged.passed
    assert rigged.p_value < 1e-6


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def test_proportion_ztest_matches_hand_computed_case():
    # 100/1000 vs 150/1000: pooled p=0.125, se=0.014790 -> z=3.381, p=0.00072.
    result = proportion_ztest(100, 1_000, 150, 1_000)
    assert result.lift_abs == pytest.approx(0.05)
    assert result.lift_rel == pytest.approx(0.50)
    assert result.statistic == pytest.approx(3.381, abs=0.005)
    assert result.p_value == pytest.approx(0.00072, abs=0.0001)
    assert result.ci_low < 0.05 < result.ci_high
    assert result.ci_low > 0  # significant difference excludes zero


def test_proportion_ztest_validates_inputs():
    with pytest.raises(ValueError):
        proportion_ztest(11, 10, 5, 10)
    with pytest.raises(ValueError):
        proportion_ztest(5, 10, 5, 0)


def test_mean_diff_test_detects_a_known_shift():
    rng = np.random.default_rng(0)
    control = rng.normal(100, 20, size=500)
    treatment = rng.normal(110, 20, size=500)
    result = mean_diff_test(control, treatment)
    assert result.lift_abs == pytest.approx(10, abs=3)
    assert result.p_value < 1e-6
    assert result.ci_low < result.lift_abs < result.ci_high


def test_mean_diff_test_requires_enough_values():
    with pytest.raises(ValueError):
        mean_diff_test([1.0], [2.0, 3.0])


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

@pytest.fixture
def experiment_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 2_000
    group = np.where(rng.random(n) < 0.5, "treatment", "control")
    treated = group == "treatment"
    retained = rng.random(n) < np.where(treated, 0.55, 0.45)
    skip = rng.normal(0.18, 0.05, n) + np.where(treated, 0.03, 0.0)
    return pd.DataFrame(
        {"group": group, "retained": retained.astype(int), "skip_rate": skip}
    )


def test_summarize_experiment_scorecard(experiment_frame):
    scorecard = summarize_experiment(
        experiment_frame,
        [
            MetricSpec("retention_14d", "retained", "proportion", "primary", "increase"),
            MetricSpec("skip_rate", "skip_rate", "mean", "guardrail", "decrease"),
        ],
    )
    assert list(scorecard["metric"]) == ["retention_14d", "skip_rate"]
    retention = scorecard.iloc[0]
    assert retention["significant"] and retention["healthy"]
    skip = scorecard.iloc[1]
    assert skip["significant"] and not skip["healthy"]  # guardrail breach


def test_summarize_experiment_validates_inputs(experiment_frame):
    with pytest.raises(ValueError, match="non-empty"):
        summarize_experiment(experiment_frame, [])
    with pytest.raises(ValueError, match="not found"):
        summarize_experiment(
            experiment_frame,
            [MetricSpec("x", "no_such_column", "mean", "primary", "increase")],
        )


# ---------------------------------------------------------------------------
# Peeking
# ---------------------------------------------------------------------------

def test_sequential_pvalues_trajectory_shape_and_final_look():
    rng = np.random.default_rng(4)
    control = (rng.random(2_000) < 0.40).astype(int)
    treatment = (rng.random(2_000) < 0.50).astype(int)  # large true effect
    trajectory = sequential_pvalues(control, treatment, n_looks=8)
    assert list(trajectory["look"]) == list(range(1, 9))
    assert trajectory["n_control"].iloc[-1] == 2_000
    assert trajectory["p_value"].iloc[-1] < 0.001  # full sample detects it


def test_sequential_pvalues_validates_inputs():
    with pytest.raises(ValueError, match="n_looks"):
        sequential_pvalues([0, 1] * 10, [0, 1] * 10, n_looks=0)
    with pytest.raises(ValueError):
        sequential_pvalues([0, 1], [0, 1], n_looks=5)


def test_peeking_inflates_false_positives():
    fpr = peeking_false_positive_rate(
        n_per_arm=5_000, n_looks=10, n_simulations=1_000, seed=2
    )
    assert fpr["false_positive_rate"].iloc[0] == pytest.approx(0.05, abs=0.02)
    assert fpr["false_positive_rate"].is_monotonic_increasing
    assert fpr["false_positive_rate"].iloc[-1] > 0.10
