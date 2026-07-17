# Music-Streaming User Behavior Analytics

**End-to-end analytics on music-streaming user behavior: retention diagnostics, free-to-paid conversion, and a reproducible user-level feature/label pipeline.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Stack](https://img.shields.io/badge/stack-SQL%20%7C%20pandas%20%7C%20scikit--learn-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-complete-blue)

> Independent project built on simulated data. Not affiliated with, endorsed by, or based on internal data from any music-streaming service.

## Overview

This project analyzes the two core commercial questions for a subscription streaming service:

- **Retention** — which behavioral signals (activity level, ad pressure, content interaction) predict short-term churn, and which segments are at risk.
- **Monetization** — where the free-to-paid subscription funnel breaks, and which levers (trial exposure, device experience, engagement depth) move conversion.

Delivery follows the standard analytics workflow: SQL diagnostics → leakage-safe user-level feature/label data asset → exploratory analysis and business visualization → churn prediction modeling and a gradient-boosting bake-off → A/B test design and analysis → survival analysis of time-to-churn → uplift modeling and targeting → a paid-conversion ranking model for upsell campaigns.

**Causal analysis storyline (notebooks 06 → 09):** a simulated randomized experiment with known injected effects first answers *whether* to launch (notebook 06: +2.6pp retention on average, but a skip-rate guardrail breach — don't ship to everyone), then *to whom* (notebook 09: hand-rolled S/T/X meta-learners estimate each user's individual treatment effect, validated against the simulation's stored ground truth, and yield a targeting policy that captures 70% of the launch value with 38% of the exposure and none of the known-harmed users).

## Key Findings

Against a baseline of **47.6%** 14-day churn and **17.4%** 30-day paid conversion:

- **Usage habit dominates both problems.** Churn runs 69.6% for low-activity users vs. 11.6% for high-activity ones — and the low-activity segment also contributes **61%** of all churned users, so risk rate and volume point at the same intervention target.
- **Content engagement halves churn.** Users with any playlist or liked-song activity churn at 29.4% vs. 62.8% without.
- **The subscription funnel breaks at the last step:** 31,255 free/trial users narrow to 2,681 trial starts but only 30 in-window paid conversions.
- **Referral is the quality acquisition channel** (22.8% conversion, 36.1% churn); paid social underperforms on every metric (12.6% conversion, 60.2% churn).
- **Recent conversion gains are performance-driven, not mix-driven:** +3.5pp overall decomposes into +3.7pp within-group improvement and −0.2pp user-structure change, with no cohort-quality drift.
- **Churn is predictable with an interpretable model:** a class-balanced logistic regression reaches ROC-AUC 0.788, catching 78% of churners at 66% precision — and beats a tuned random forest, so explainability costs nothing. Model-based risk tiers are monotonically calibrated (17% / 55% / 79% actual churn).
- **Model complexity bought nothing:** three tuned gradient-boosting models (XGBoost, LightGBM, HistGB) land within ±0.07pp ROC-AUC of the logistic baseline, with paired-bootstrap CIs straddling zero and identical top-4 churn drivers — the interpretable model keeps the job.
- **Statistical significance is not a launch decision:** a simulated home-screen experiment lifts 14-day retention by +2.6pp (p < 0.001) yet breaches its skip-rate guardrail (+13% relative), so the verdict is iterate-and-retest, not ship. The win concentrates in low-activity listeners (+4.4pp); daily peeking would have inflated the false-positive rate from 5% to ~20%.
- **Upsell outreach can cost 3x less:** a calibrated conversion ranking (PR-AUC 0.727 vs a 21.1% base rate, Brier 0.099) makes a top-20% campaign convert at 66% — 1.52 contacts per conversion vs 4.74 at random — while covering 62% of all converters. Conversion is driven by listening intensity and ad pressure: the mirror image of churn's too-little-usage story.
- **Targeting beats blanket rollout:** an X-learner uplift model (rank correlation 0.45 with the simulation's true effects) plus the experiment-proven skip-rate exclusion targets 38% of users for ~70% of the full-launch retention value — and shows why model scores never replace guardrails: a −1.5pp harmed minority stays invisible to the model at this sample size.
- **Churn has a clock:** half of a signup cohort disengages within 35 days, and 45% of churn-defining silences begin in week 1 — so onboarding nudges belong in days 0–3, not at day 14. The referral vs paid-social quality gap *widens* over time (Cox HR 0.77 vs 1.23 after day 30), while device makes no difference to churn timing (log-rank p = 0.37).

Full evidence, charts, and caveats in the Key Findings sections of [notebooks 03–06](notebooks/).

## Quickstart

```bash
git clone https://github.com/JackyJiang08/music-streaming-user-behavior-analytics.git
cd music-streaming-user-behavior-analytics
pip install -r requirements.txt

# Rebuild the user-level wide table from the raw event tables
python scripts/build_user_feature_table.py

# Regenerate the simulated A/B experiment dataset
python -m src.experiment_simulation

# Run the unit tests
pytest

# Or explore interactively
jupyter lab notebooks/
```

Runs on `pandas + sqlite3`; no database setup required. All notebooks and scripts load data through `src/data_loader.py`, which auto-discovers `./data` locally or `/content/data` on Colab (override with `STREAMING_DATA_DIR`) and reads the gzipped events file directly — nothing needs to be decompressed.

## Dataset

Five simulated tables covering **50,000 users** and **~1.5M events**:

| Table | Grain | Contents |
|---|---|---|
| `users.csv` | user | Demographics, device, acquisition channel, signup date |
| `listening_events.csv.gz` | event | 1.07M listening logs: sessions, duration, skips, likes, playlist adds |
| `subscription_events.csv` | event | Trial exposure/start, payments, renewals, cancellations |
| `ad_events.csv` | event | Ad impressions, clicks, completions, revenue |
| `feature_table.csv` | user | Prebuilt user-level feature snapshot |

`listening_events` is stored gzipped to stay under GitHub's 100 MB file limit. `data/experiment_results.csv.gz` is a generated artifact — the simulated A/B experiment outcomes (with per-user ground-truth uplift) produced by `python -m src.experiment_simulation` and committed for downstream analyses.

## Analysis

| Notebook | Purpose |
|---|---|
| [`01_retention_and_conversion_analysis`](notebooks/01_retention_and_conversion_analysis.ipynb) | SQL diagnostics: baseline metrics, five behavioral hypotheses, funnel break points |
| [`02_user_feature_table_and_labels`](notebooks/02_user_feature_table_and_labels.ipynb) | Builds the leakage-safe, one-row-per-user feature table with churn and conversion labels |
| [`03_eda_and_visualization`](notebooks/03_eda_and_visualization.ipynb) | Visual diagnostics of churn and conversion drivers across segments, devices, channels, and cohorts |
| [`04_advanced_eda_contribution_and_mix_shift`](notebooks/04_advanced_eda_contribution_and_mix_shift.ipynb) | Segment prioritization (rate vs. contribution), cohort quality drift, mix-shift decomposition |
| [`05_churn_model_training_and_evaluation`](notebooks/05_churn_model_training_and_evaluation.ipynb) | Churn classifiers (logistic regression vs. random forest): evaluation, threshold tuning, drivers, risk tiers |
| [`06_ab_test_design_and_analysis`](notebooks/06_ab_test_design_and_analysis.ipynb) | A/B test with simulated treatment effects: power analysis, SRM gate, metric scorecard, pitfalls (peeking, multiple testing, novelty), segment drill-down, launch decision |
| [`07_gradient_boosting_churn_comparison`](notebooks/07_gradient_boosting_churn_comparison.ipynb) | GBM bake-off (HistGB, XGBoost, LightGBM) vs the logistic baseline on the frozen notebook-05 protocol: paired-bootstrap AUC deltas, calibration, driver agreement, operating-point economics |
| [`08_survival_analysis_time_to_churn`](notebooks/08_survival_analysis_time_to_churn.ipynb) | Time-to-churn: Kaplan-Meier by segment, log-rank tests, Cox hazards with PH diagnostics, held-out C-index, and survival-derived intervention windows |
| [`09_uplift_modeling_targeting`](notebooks/09_uplift_modeling_targeting.ipynb) | Individual treatment effects with hand-rolled S/T/X meta-learners: Qini evaluation, ground-truth validation, and a guardrail-aware targeting policy |
| [`10_paid_conversion_model`](notebooks/10_paid_conversion_model.ipynb) | Paid-conversion ranking on a leakage-safe landmark design: PR-AUC-first evaluation, calibration, lift/capture campaign economics, drivers vs churn |

Each notebook opens with its scope; notebooks 03–10 close with data-grounded Key Findings. The wide-table SQL is versioned once in `sql/build_user_feature_table.sql` and runs end to end via `scripts/build_user_feature_table.py`.

## Repository Structure

```
├── data/                          # Source tables (see Dataset)
├── notebooks/                     # Analysis notebooks (01-06)
├── src/
│   ├── config.py                  # Project constants: random seed, snapshot dates
│   ├── data_loader.py             # Shared loader: CSVs -> pandas -> in-memory SQLite
│   ├── ab_testing.py              # Experiment stats: power, SRM, tests, scorecard, peeking
│   ├── experiment_simulation.py   # Simulated experiment with injected ground-truth effects
│   ├── churn_modeling.py          # Frozen notebook-05 modeling protocol (split, features, preprocessing)
│   ├── model_evaluation.py        # Evaluation harness: metrics table, calibration, paired bootstrap
│   ├── survival_analysis.py       # lifelines wrappers: KM, log-rank, Cox, PH check, C-index
│   ├── uplift_modeling.py         # S/T/X meta-learners + Qini/decile/gain evaluation (pure sklearn)
│   └── conversion_modeling.py     # Frozen conversion protocol: landmark design + leakage guard
├── sql/
│   ├── build_user_feature_table.sql   # Wide-table definition (single source of truth)
│   ├── ab_test_population.sql         # Experiment eligibility cohort
│   ├── build_survival_table.sql       # Per-user time-to-churn durations and events
│   └── build_conversion_table.sql     # Landmark conversion population, features, and label
├── scripts/
│   ├── build_user_feature_table.py    # CLI pipeline: build + QA + export the wide table
│   └── build_survival_table.py        # CLI pipeline: build + QA + export the survival table
├── tests/                         # Unit tests for src/ (pytest)
└── requirements.txt
```

## Project Status

- [x] Retention and conversion diagnostics (SQL)
- [x] User-level feature table and label engineering
- [x] Exploratory data analysis and visualization
- [x] Advanced EDA: contribution analysis, cohort drift, mix-shift decomposition
- [x] Churn prediction model: training, evaluation, threshold tuning, risk segmentation
- [x] A/B test design and analysis for retention/conversion levers (simulated treatment effects)
- [x] Gradient-boosting comparison (HistGB, XGBoost, LightGBM) with paired-bootstrap evaluation
- [x] Survival analysis: time-to-churn, Cox hazards, intervention windows
- [x] Uplift modeling: individual treatment effects and guardrail-aware targeting policy
- [x] Paid-conversion prediction model: leakage-safe landmark design, calibrated ranking, campaign economics

## License

[MIT](LICENSE)
