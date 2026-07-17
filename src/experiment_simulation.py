"""Simulated A/B experiment: personalized-playlist module vs. current home.

The dataset is a static snapshot, so the experiment is simulated on top of
the real eligible population with *known, injected* treatment effects — the
analysis in notebook 06 is then validated by whether it recovers them.

Scenario
--------
Treatment shows a personalized-playlist module on the home screen; control
keeps the existing recommendation page. Primary metric: 14-day retention
(listened at least once in the 14 days after exposure), matching the
project's churn-label definition.

Eligible population (see sql/ab_test_population.sql)
----------------------------------------------------
Non-paying listeners (free tier and in-trial) at the snapshot. The ideal
target — users who signed up within days of the experiment start — is empty
in this snapshot (minimum observed tenure is 18 days), so tenure is kept as
a covariate/segment rather than an eligibility filter.

Effect-generation model (ground truth)
--------------------------------------
Each user's observed 14-day retention label is their control potential
outcome. Treated users receive a segment-dependent additive retention effect
(`true_uplift_pp`, stored per user for later uplift-model validation):

    high_skipper   (skip_rate_30d > 0.35)        -1.5pp  module crowds out
                                                         their own navigation
    low_activity   (active_days_30d <= 2)        +5.0pp  discovery helps most
    mid_activity   (active_days_30d in 3..9)     +2.2pp  moderate help
    high_activity  (active_days_30d >= 10)       +0.5pp  habits already formed

Population-weighted average: ~+2.2pp. Segments are assigned in the order
listed (a high skipper stays high_skipper regardless of activity). Positive
effects rescue would-be churners with probability uplift/segment churn rate;
negative effects drop would-be retainers with probability |uplift|/segment
retention rate, so the expected within-segment lift equals the target.

Other injected effects (uniform, not heterogeneous):
    listen minutes   +6.5% for treated retained users
    active days      +4.0% for treated retained users
    playlist adds    Poisson-thinned from observed 30d adds; treatment
                     multiplies the rate by 1.15
    skip rate        +3.0pp for treated retained users (the guardrail story:
                     the module boosts usage but recommends imperfectly)
    trial cancels    flat (base rate both arms) — no injected effect
    ad revenue       flat per retained user — no injected effect

Typical usage:

    from src.experiment_simulation import load_eligible_population, simulate_experiment

    population = load_eligible_population()
    results = simulate_experiment(population)

Run as a script to export data/experiment_results.csv.gz:

    python -m src.experiment_simulation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ab_testing import assign_groups
from src.config import RANDOM_SEED
from src.data_loader import connect

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "experiment_results.csv.gz"

# Post-exposure measurement window (days), matching the retention label.
POST_PERIOD_DAYS: int = 14
_OBSERVATION_DAYS: int = 30  # length of the pre-snapshot behavioral window
_POST_SCALE = POST_PERIOD_DAYS / _OBSERVATION_DAYS

# Ground-truth retention uplift per segment, in percentage points.
SEGMENT_UPLIFT_PP: dict[str, float] = {
    "high_skipper": -1.5,
    "low_activity": +5.0,
    "mid_activity": +2.2,
    "high_activity": +0.5,
}
TARGET_OVERALL_UPLIFT_PP: float = 2.2  # population-weighted design target

# Uniform injected effects.
LISTEN_MINUTES_REL_LIFT: float = 0.065
ACTIVE_DAYS_REL_LIFT: float = 0.040
PLAYLIST_RATE_MULTIPLIER: float = 1.15
SKIP_RATE_LIFT_PP: float = 3.0

# Novelty-effect model for the weekly trajectory demonstration: the relative
# engagement lift starts high and decays toward its long-run level.
NOVELTY_INITIAL_REL_LIFT: float = 0.12
NOVELTY_LONG_RUN_REL_LIFT: float = 0.05
NOVELTY_DECAY_WEEKS: float = 1.5

_REQUIRED_COLUMNS = (
    "user_id",
    "current_subscription_type",
    "active_days_30d",
    "listen_minutes_30d",
    "skip_rate_30d",
    "playlist_adds_30d",
    "ad_revenue_30d",
    "cancel_count_30d",
    "churn_label_14d",
)


def load_eligible_population(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Build the eligible population from the versioned SQL definitions."""
    _, sql, run_script = connect(data_dir)
    run_script((REPO_ROOT / "sql" / "build_user_feature_table.sql").read_text())
    run_script((REPO_ROOT / "sql" / "ab_test_population.sql").read_text())
    population = sql("SELECT * FROM ab_test_population")

    duplicated = len(population) - population["user_id"].nunique()
    if duplicated:
        raise ValueError(
            f"QA failed: {duplicated} duplicated users in ab_test_population — "
            "check the view definition"
        )
    return population


def label_effect_segment(population: pd.DataFrame) -> pd.Series:
    """Observable segment driving the heterogeneous retention effect."""
    missing = [c for c in ("skip_rate_30d", "active_days_30d")
               if c not in population.columns]
    if missing:
        raise ValueError(f"population is missing required columns: {missing}")
    return pd.Series(
        np.select(
            [
                population["skip_rate_30d"] > 0.35,
                population["active_days_30d"] <= 2,
                population["active_days_30d"] <= 9,
            ],
            ["high_skipper", "low_activity", "mid_activity"],
            default="high_activity",
        ),
        index=population.index,
        name="effect_segment",
    )


def simulate_experiment(
    population: pd.DataFrame,
    treatment_share: float = 0.5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Randomize the population and generate post-period outcomes.

    Returns one row per user: pre-experiment covariates, `group`,
    `effect_segment`, ground-truth `true_uplift_pp`, and simulated
    post-period outcome columns:

      retained_14d_post          1 if the user listened in the post-period
      listen_minutes_14d_post    minutes listened (0 if not retained)
      active_days_14d_post       days with listening activity (0 if not retained)
      playlist_added_14d_post    1 if the user added to a playlist
      skip_rate_14d_post         share of plays skipped (NaN if not retained)
      cancel_14d_post            1 if an in-trial user canceled (NaN for free)
      ad_revenue_14d_post        ad revenue in USD (0 if not retained)
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in population.columns]
    if missing:
        raise ValueError(f"population is missing required columns: {missing}")

    df = population.copy()
    df = df.merge(assign_groups(df["user_id"], treatment_share, seed), on="user_id")
    df["effect_segment"] = label_effect_segment(df)
    df["true_uplift_pp"] = df["effect_segment"].map(SEGMENT_UPLIFT_PP)

    implied = df["true_uplift_pp"].mean()
    if abs(implied - TARGET_OVERALL_UPLIFT_PP) > 0.5:
        raise ValueError(
            f"QA failed: population-weighted uplift {implied:.2f}pp is far from "
            f"the {TARGET_OVERALL_UPLIFT_PP}pp design target — recalibrate "
            "SEGMENT_UPLIFT_PP for this population"
        )

    # Outcome noise must come from a stream independent of the assignment
    # draws in assign_groups (which consume default_rng(seed) directly);
    # reusing that stream would correlate treatment status with the rescue
    # draws and double the realized effect.
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    treated = (df["group"] == "treatment").to_numpy()
    retained = (1 - df["churn_label_14d"]).to_numpy().astype(int)

    # Segment-level base rates used to convert the additive uplift into
    # per-user flip probabilities.
    seg_churn = df.groupby("effect_segment")["churn_label_14d"].transform("mean")
    uplift = df["true_uplift_pp"].to_numpy() / 100
    churn_rate = seg_churn.to_numpy()
    retention_rate = 1 - churn_rate

    rescue_prob = np.where(uplift > 0, uplift / churn_rate, 0.0)
    drop_prob = np.where(uplift < 0, -uplift / retention_rate, 0.0)
    if (rescue_prob > 1).any() or (drop_prob > 1).any():
        raise ValueError(
            "QA failed: an injected uplift implies a flip probability above 1 "
            "for some segment — lower SEGMENT_UPLIFT_PP"
        )
    draws = rng.random(len(df))
    rescued = treated & (retained == 0) & (draws < rescue_prob)
    dropped = treated & (retained == 1) & (draws < drop_prob)
    retained = np.where(rescued, 1, np.where(dropped, 0, retained))

    # Post-period engagement: observed 30-day behavior scaled to the 14-day
    # window with multiplicative noise; zero for users who did not return.
    noise = rng.lognormal(mean=-(0.25**2) / 2, sigma=0.25, size=len(df))
    minutes = df["listen_minutes_30d"].to_numpy() * _POST_SCALE * noise
    minutes = np.where(treated, minutes * (1 + LISTEN_MINUTES_REL_LIFT), minutes)
    minutes = np.where(retained == 1, minutes, 0.0)

    day_noise = rng.lognormal(mean=-(0.2**2) / 2, sigma=0.2, size=len(df))
    days = df["active_days_30d"].to_numpy() * _POST_SCALE * day_noise
    days = np.where(treated, days * (1 + ACTIVE_DAYS_REL_LIFT), days)
    days = np.clip(np.round(days), 1, POST_PERIOD_DAYS)  # retained => >= 1 day
    days = np.where(retained == 1, days, 0).astype(int)

    # Playlist adds: thin the observed 30-day add count to the post window as
    # a Poisson rate (plus a small discovery floor); treatment scales the rate.
    add_rate = df["playlist_adds_30d"].to_numpy() * _POST_SCALE + 0.02
    add_rate = np.where(treated, add_rate * PLAYLIST_RATE_MULTIPLIER, add_rate)
    added = (rng.random(len(df)) < 1 - np.exp(-add_rate)) & (retained == 1)

    skip = df["skip_rate_30d"].to_numpy() + rng.normal(0, 0.05, size=len(df))
    skip = np.where(treated, skip + SKIP_RATE_LIFT_PP / 100, skip)
    skip = np.clip(skip, 0, 1)
    skip = np.where(retained == 1, skip, np.nan)

    # Trial cancellation: flat base rate in both arms (no injected effect);
    # not applicable to free-tier users. Independent of listening retention —
    # a trial can be canceled without listening at all.
    is_trial = (df["current_subscription_type"] == "trial").to_numpy()
    base_cancel = df.loc[is_trial, "cancel_count_30d"].gt(0).mean() * _POST_SCALE
    cancel = np.where(
        is_trial, (rng.random(len(df)) < base_cancel).astype(float), np.nan
    )

    ad_noise = rng.lognormal(mean=-(0.25**2) / 2, sigma=0.25, size=len(df))
    ad_revenue = df["ad_revenue_30d"].to_numpy() * _POST_SCALE * ad_noise
    ad_revenue = np.where(retained == 1, ad_revenue, 0.0)

    df["retained_14d_post"] = retained
    df["listen_minutes_14d_post"] = minutes
    df["active_days_14d_post"] = days
    df["playlist_added_14d_post"] = added.astype(int)
    df["skip_rate_14d_post"] = skip
    df["cancel_14d_post"] = cancel
    df["ad_revenue_14d_post"] = ad_revenue
    return df


def simulate_novelty_trajectory(
    n_per_arm: int,
    mean_weekly_minutes: float,
    n_weeks: int = 6,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Weekly engagement lift with a decaying novelty component.

    The true relative lift in week w is
    long_run + (initial - long_run) * exp(-(w - 1) / decay), so a week-1-only
    readout overstates the long-run effect. Returns per-week arm means and
    the measured relative lift with a 95% CI (Welch approximation).
    """
    if n_per_arm < 2 or n_weeks < 1:
        raise ValueError(
            f"need n_per_arm >= 2 and n_weeks >= 1, got {n_per_arm}, {n_weeks}"
        )
    if mean_weekly_minutes <= 0:
        raise ValueError(
            f"mean_weekly_minutes must be positive, got {mean_weekly_minutes}"
        )

    rng = np.random.default_rng(seed)
    sigma = 0.8  # user-level heterogeneity in weekly minutes
    rows = []
    for week in range(1, n_weeks + 1):
        true_lift = NOVELTY_LONG_RUN_REL_LIFT + (
            NOVELTY_INITIAL_REL_LIFT - NOVELTY_LONG_RUN_REL_LIFT
        ) * np.exp(-(week - 1) / NOVELTY_DECAY_WEEKS)
        base = mean_weekly_minutes * rng.lognormal(
            -(sigma**2) / 2, sigma, size=(2, n_per_arm)
        )
        control = base[0]
        treatment = base[1] * (1 + true_lift)

        diff = treatment.mean() - control.mean()
        se = np.sqrt(
            control.var(ddof=1) / n_per_arm + treatment.var(ddof=1) / n_per_arm
        )
        rows.append(
            {
                "week": week,
                "true_rel_lift": true_lift,
                "control_mean": control.mean(),
                "treatment_mean": treatment.mean(),
                "rel_lift": diff / control.mean(),
                "rel_lift_ci_low": (diff - 1.96 * se) / control.mean(),
                "rel_lift_ci_high": (diff + 1.96 * se) / control.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Build the population, simulate the experiment, and export the results."""
    population = load_eligible_population()
    results = simulate_experiment(population)

    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, compression="gzip")

    lift = (
        results.groupby("group")["retained_14d_post"].mean().pipe(
            lambda s: s["treatment"] - s["control"]
        )
    )
    print(
        f"Wrote {len(results):,} users x {results.shape[1]} columns to {output}\n"
        f"Measured retention lift: {lift * 100:+.2f}pp "
        f"(designed {TARGET_OVERALL_UPLIFT_PP:+.1f}pp population-weighted)"
    )


if __name__ == "__main__":
    main()
