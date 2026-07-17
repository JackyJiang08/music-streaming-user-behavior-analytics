"""A/B experiment design and analysis toolkit.

Statistical machinery for the experiment lifecycle used in notebook 06:
sample-size and minimum-detectable-effect planning, deterministic user-level
randomization, the sample-ratio-mismatch (SRM) check, two-proportion and
Welch tests with confidence intervals, a metric scorecard, and sequential
p-value helpers for the peeking demonstration. Inference is delegated to
statsmodels/scipy; formulas cite their standard references.

Typical usage:

    from src.ab_testing import required_sample_size, srm_check

    n = required_sample_size(baseline_rate=0.455, mde_pp=2.0)
    srm = srm_check(n_control=16_700, n_treatment=16_750)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from src.config import RANDOM_SEED

# ---------------------------------------------------------------------------
# Design: sample size and minimum detectable effect
# ---------------------------------------------------------------------------


def required_sample_size(
    baseline_rate: float,
    mde_pp: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Users needed per arm to detect an absolute lift on a proportion metric.

    `mde_pp` is the minimum detectable effect in percentage points (2.0 means
    a 2pp absolute lift). Two-sided two-proportion formula with pooled
    variance under H0 and unpooled variance under H1 (Fleiss, Levin & Paik,
    *Statistical Methods for Rates and Proportions*, eq. 4.14 without
    continuity correction).
    """
    if not 0 < baseline_rate < 1:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if mde_pp == 0:
        raise ValueError("mde_pp must be non-zero")
    treated_rate = baseline_rate + mde_pp / 100
    if not 0 < treated_rate < 1:
        raise ValueError(
            f"baseline_rate + mde_pp/100 must be in (0, 1), got {treated_rate}"
        )
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError(f"alpha and power must be in (0, 1), got {alpha}, {power}")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    pooled = (baseline_rate + treated_rate) / 2
    se0 = math.sqrt(2 * pooled * (1 - pooled))
    se1 = math.sqrt(
        baseline_rate * (1 - baseline_rate) + treated_rate * (1 - treated_rate)
    )
    n = ((z_alpha * se0 + z_power * se1) / abs(mde_pp / 100)) ** 2
    return math.ceil(n)


def minimum_detectable_effect(
    n_per_arm: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest absolute lift (in percentage points) detectable at the given
    per-arm sample size.

    Inverse of `required_sample_size` under the planning approximation that
    both arms share the baseline variance (accurate for small effects).
    """
    if n_per_arm <= 0:
        raise ValueError(f"n_per_arm must be positive, got {n_per_arm}")
    if not 0 < baseline_rate < 1:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    mde = (z_alpha + z_power) * math.sqrt(
        2 * baseline_rate * (1 - baseline_rate) / n_per_arm
    )
    return 100 * mde


# ---------------------------------------------------------------------------
# Assignment and sample-ratio-mismatch check
# ---------------------------------------------------------------------------


def assign_groups(
    user_ids: Sequence[str],
    treatment_share: float = 0.5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Randomize users into control/treatment at the user level.

    Mirrors a production hash-based splitter: each user draws independently,
    so arm sizes are binomial rather than exactly proportional and the SRM
    check below stays meaningful. Deterministic for a given seed and input
    order. Returns a DataFrame with columns `user_id` and `group`.
    """
    ids = pd.Series(user_ids)
    if ids.duplicated().any():
        dupes = ids[ids.duplicated()].head(3).tolist()
        raise ValueError(
            f"user_ids contains duplicates (e.g. {dupes}); the randomization "
            "unit must be unique users"
        )
    if not 0 < treatment_share < 1:
        raise ValueError(f"treatment_share must be in (0, 1), got {treatment_share}")

    rng = np.random.default_rng(seed)
    is_treatment = rng.random(len(ids)) < treatment_share
    return pd.DataFrame(
        {
            "user_id": ids.to_numpy(),
            "group": np.where(is_treatment, "treatment", "control"),
        }
    )


@dataclass(frozen=True)
class SRMResult:
    """Sample-ratio-mismatch verdict for observed arm sizes."""

    n_control: int
    n_treatment: int
    expected_ratio: float
    chi2: float
    p_value: float
    passed: bool
    p_threshold: float


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_ratio: float = 0.5,
    p_threshold: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit test that arm sizes match the design.

    `expected_ratio` is the designed treatment share. Uses the conventional
    strict alarm threshold (p < 0.001, Fabijan et al. 2019): an SRM means
    assignment, logging, or eligibility is broken and no downstream result
    should be trusted.
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError(
            f"both arms need users, got control={n_control}, treatment={n_treatment}"
        )
    if not 0 < expected_ratio < 1:
        raise ValueError(f"expected_ratio must be in (0, 1), got {expected_ratio}")
    total = n_control + n_treatment
    observed = np.array([n_control, n_treatment])
    expected = np.array([total * (1 - expected_ratio), total * expected_ratio])
    chi2, p_value = stats.chisquare(observed, expected)
    return SRMResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_ratio=expected_ratio,
        chi2=float(chi2),
        p_value=float(p_value),
        passed=bool(p_value >= p_threshold),
        p_threshold=p_threshold,
    )


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """Two-sample comparison on one metric (treatment minus control)."""

    control_value: float
    treatment_value: float
    lift_abs: float
    lift_rel: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float


def proportion_ztest(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided z-test for a difference in proportions.

    Test statistic from `statsmodels.stats.proportion.proportions_ztest`
    (pooled standard error under H0); Wald confidence interval on the
    absolute lift from `confint_proportions_2indep`.
    """
    for label, successes, n in (
        ("control", control_successes, control_n),
        ("treatment", treatment_successes, treatment_n),
    ):
        if n <= 0:
            raise ValueError(f"{label}_n must be positive, got {n}")
        if not 0 <= successes <= n:
            raise ValueError(f"{label}_successes must be in [0, {n}], got {successes}")

    p_c = control_successes / control_n
    p_t = treatment_successes / treatment_n
    lift = p_t - p_c

    z, p_value = proportions_ztest(
        count=[treatment_successes, control_successes],
        nobs=[treatment_n, control_n],
    )
    ci_low, ci_high = confint_proportions_2indep(
        treatment_successes,
        treatment_n,
        control_successes,
        control_n,
        method="wald",
        compare="diff",
        alpha=alpha,
    )
    return TestResult(
        control_value=p_c,
        treatment_value=p_t,
        lift_abs=lift,
        lift_rel=lift / p_c if p_c else float("nan"),
        statistic=float(z),
        p_value=float(p_value),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
    )


def mean_diff_test(
    control_values: Sequence[float],
    treatment_values: Sequence[float],
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided Welch t-test for a difference in means (unequal variances).

    Statistic and p-value from `scipy.stats.ttest_ind(equal_var=False)`;
    the confidence interval uses the Welch–Satterthwaite degrees of freedom.
    """
    x_c = pd.Series(control_values, dtype=float).dropna().to_numpy()
    x_t = pd.Series(treatment_values, dtype=float).dropna().to_numpy()
    if len(x_c) < 2 or len(x_t) < 2:
        raise ValueError(
            f"need >= 2 non-null values per arm, got {len(x_c)} and {len(x_t)}"
        )

    mean_c, mean_t = x_c.mean(), x_t.mean()
    diff = mean_t - mean_c
    t_stat, p_value = stats.ttest_ind(x_t, x_c, equal_var=False)

    var_c, var_t = x_c.var(ddof=1), x_t.var(ddof=1)
    se = math.sqrt(var_c / len(x_c) + var_t / len(x_t))
    df = (var_c / len(x_c) + var_t / len(x_t)) ** 2 / (
        (var_c / len(x_c)) ** 2 / (len(x_c) - 1)
        + (var_t / len(x_t)) ** 2 / (len(x_t) - 1)
    )
    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    return TestResult(
        control_value=float(mean_c),
        treatment_value=float(mean_t),
        lift_abs=float(diff),
        lift_rel=float(diff / mean_c) if mean_c else float("nan"),
        statistic=float(t_stat),
        p_value=float(p_value),
        ci_low=float(diff - t_alpha * se),
        ci_high=float(diff + t_alpha * se),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """One experiment metric: where it lives and how to read it."""

    name: str
    column: str
    kind: Literal["proportion", "mean"]
    role: Literal["primary", "secondary", "guardrail"]
    desired_direction: Literal["increase", "decrease"]


def summarize_experiment(
    df: pd.DataFrame,
    metrics: Sequence[MetricSpec],
    group_col: str = "group",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build the experiment scorecard: one tested row per metric.

    Proportion metrics use `proportion_ztest`; continuous metrics use
    `mean_diff_test`. Users with a null metric value (e.g. skip rate for
    users who never returned) are excluded from that metric only. `healthy`
    is False only when a metric moved significantly against its desired
    direction — a guardrail breach when the role is guardrail.
    """
    if not metrics:
        raise ValueError("metrics must be non-empty")
    control = df[df[group_col] == "control"]
    treatment = df[df[group_col] == "treatment"]
    if control.empty or treatment.empty:
        raise ValueError("both control and treatment rows are required")

    rows = []
    for spec in metrics:
        if spec.column not in df.columns:
            raise ValueError(f"metric column {spec.column!r} not found in dataframe")
        x_c = control[spec.column].dropna()
        x_t = treatment[spec.column].dropna()
        if spec.kind == "proportion":
            result = proportion_ztest(
                int(x_c.sum()), len(x_c), int(x_t.sum()), len(x_t), alpha=alpha
            )
        else:
            result = mean_diff_test(x_c, x_t, alpha=alpha)
        significant = result.p_value < alpha
        moved_desired = (
            result.lift_abs > 0
            if spec.desired_direction == "increase"
            else result.lift_abs < 0
        )
        rows.append(
            {
                "metric": spec.name,
                "role": spec.role,
                "control": result.control_value,
                "treatment": result.treatment_value,
                "lift_abs": result.lift_abs,
                "lift_rel": result.lift_rel,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "p_value": result.p_value,
                "significant": significant,
                "healthy": moved_desired or not significant,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Peeking (sequential looks)
# ---------------------------------------------------------------------------


def sequential_pvalues(
    control_outcomes: Sequence[int],
    treatment_outcomes: Sequence[int],
    n_looks: int = 10,
) -> pd.DataFrame:
    """P-value trajectory of one experiment read at equally spaced looks.

    Outcomes are binary arrays in enrollment order; look k tests the first
    k/n_looks share of each arm with the two-proportion z-test. Used to show
    how a single experiment's p-value wanders across interim analyses.
    """
    x_c = np.asarray(control_outcomes)
    x_t = np.asarray(treatment_outcomes)
    if n_looks < 1:
        raise ValueError(f"n_looks must be >= 1, got {n_looks}")
    if len(x_c) < n_looks or len(x_t) < n_looks:
        raise ValueError(
            f"each arm needs >= n_looks outcomes, got {len(x_c)}, {len(x_t)}"
        )

    rows = []
    for k in range(1, n_looks + 1):
        n_c = int(round(len(x_c) * k / n_looks))
        n_t = int(round(len(x_t) * k / n_looks))
        result = proportion_ztest(int(x_c[:n_c].sum()), n_c, int(x_t[:n_t].sum()), n_t)
        rows.append(
            {"look": k, "n_control": n_c, "n_treatment": n_t, "p_value": result.p_value}
        )
    return pd.DataFrame(rows)


def peeking_false_positive_rate(
    n_per_arm: int = 10_000,
    true_rate: float = 0.5,
    n_looks: int = 10,
    n_simulations: int = 2_000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Replicated A/A experiments read at interim looks ("peeking").

    Both arms share the same true rate, so every significant result is a
    false positive. Returns, for each number of looks k, the share of
    simulations declared significant at any of the first k equally spaced
    looks — the Type I error inflation from stopping at the first p < alpha.
    Vectorized equivalent of running `sequential_pvalues` per replication.
    """
    if n_looks < 1 or n_simulations < 1 or n_per_arm < n_looks:
        raise ValueError(
            "need n_looks >= 1, n_simulations >= 1 and n_per_arm >= n_looks, got "
            f"{n_looks}, {n_simulations}, {n_per_arm}"
        )
    if not 0 < true_rate < 1:
        raise ValueError(f"true_rate must be in (0, 1), got {true_rate}")

    rng = np.random.default_rng(seed)
    look_sizes = (
        np.linspace(n_per_arm / n_looks, n_per_arm, n_looks).round().astype(int)
    )
    increments = np.diff(look_sizes, prepend=0)

    # successes[sim, look] = cumulative conversions per arm at each look
    succ_c = rng.binomial(increments, true_rate, size=(n_simulations, n_looks)).cumsum(
        axis=1
    )
    succ_t = rng.binomial(increments, true_rate, size=(n_simulations, n_looks)).cumsum(
        axis=1
    )
    n_cum = look_sizes[np.newaxis, :]

    p_c = succ_c / n_cum
    p_t = succ_t / n_cum
    pooled = (succ_c + succ_t) / (2 * n_cum)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n_cum))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (p_t - p_c) / se, 0.0)
    significant = 2 * stats.norm.sf(np.abs(z)) < alpha

    stopped_by_look = np.maximum.accumulate(significant, axis=1)
    return pd.DataFrame(
        {
            "looks": np.arange(1, n_looks + 1),
            "false_positive_rate": stopped_by_look.mean(axis=0),
        }
    )
