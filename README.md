# Spotify User Behavior Analytics

**End-to-end analytics on music-streaming user behavior: retention diagnostics, free-to-paid conversion, and a reproducible user-level feature/label pipeline.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Stack](https://img.shields.io/badge/stack-SQL%20%7C%20pandas%20%7C%20matplotlib-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-brightgreen)

> **Disclaimer:** This is an independent analytics project built on **simulated data** modeled after a music-streaming service. It is not affiliated with, endorsed by, or based on internal data from Spotify.

## Overview

A streaming service lives or dies on two numbers: how many users keep listening, and how many free listeners become paying subscribers. This project treats both as measurable, diagnosable problems:

1. **Retention** — which behavioral signals (activity level, ad pressure, content interaction) predict short-term churn, and which user segments are at risk?
2. **Monetization** — where does the free-to-paid subscription funnel break, and which levers (trial exposure, device experience, engagement depth) move conversion?

The work follows the standard industry arc: SQL-based diagnostics → a leakage-safe user-level feature and label data asset → exploratory analysis and business visualization → (upcoming) churn/conversion models and experiment design.

## Dataset

Five simulated tables covering **50,000 users** and **~1.5M events**:

| Table | Grain | Contents |
|---|---|---|
| `users.csv` | user | Demographics, device, acquisition channel, signup date |
| `listening_events.csv.gz` | event | 1.07M listening logs: sessions, duration, skips, likes, playlist adds |
| `subscription_events.csv` | event | Trial exposure/start, payments, renewals, cancellations |
| `ad_events.csv` | event | Ad impressions, clicks, completions, revenue |
| `feature_table.csv` | user | Prebuilt user-level feature snapshot |

## Repository Structure

```
├── data/                          # Source tables (see Dataset above)
├── notebooks/
│   ├── 01_retention_and_conversion_analysis.ipynb
│   ├── 02_user_feature_table_and_labels.ipynb
│   └── 03_eda_and_visualization.ipynb
├── src/
│   └── data_loader.py             # Shared loader: CSVs -> pandas -> in-memory SQLite
├── sql/
│   └── build_user_feature_table.sql   # Wide-table definition (single source of truth)
├── scripts/
│   └── build_user_feature_table.py    # CLI pipeline: build + QA + export the wide table
└── requirements.txt
```

## Analysis

**`01_retention_and_conversion_analysis.ipynb` — Retention Overview and Conversion Hypothesis Analysis.**
Establishes the business baseline (activity, churn, conversion, cancellation rates), segments users by subscription status and lifecycle stage, and tests five behavioral hypotheses: low activity as a churn signal, ad overload vs. retention (with activity-level controls), trial exposure vs. paid conversion, device differences, and content interaction vs. stickiness. Maps the subscription conversion funnel to locate drop-off points.

**`02_user_feature_table_and_labels.ipynb` — User-Level Feature Table and Label Engineering.**
Designs the snapshot / observation / prediction time-window framework, aggregates listening, ad, and subscription events into user-level features, defines leakage-safe churn (14-day) and paid-conversion (30-day) labels from the prediction window, and QA-checks the assembled wide table (JOIN inflation, label and feature distributions) before exporting it for downstream EDA and modeling. The wide-table SQL is versioned in `sql/build_user_feature_table.sql` and can be run end to end with `python scripts/build_user_feature_table.py`.

**`03_eda_and_visualization.ipynb` — Exploratory Analysis and Business Visualization.**
Turns the feature snapshot into business-facing charts: data quality gates (granularity, missing rates, label levels), derived audience segments (activity level, content engagement, ad-load buckets), and visual diagnostics — churn/conversion baseline, activity vs. churn, the conversion funnel, ad pressure, content engagement, device and acquisition-channel quality, listening-time distributions, engagement scatter, and signup-cohort trends. Each chart is framed by the business question, reading, candidate action, and limitations.

## Roadmap

- [x] Retention and conversion diagnostics (SQL)
- [x] User-level feature table and label engineering
- [x] Exploratory data analysis and visualization
- [ ] Churn and paid-conversion prediction models
- [ ] Experiment design for conversion levers (trial exposure, ad load)

## Getting Started

Everything runs on `pandas + sqlite3` — no database setup required.

```bash
git clone https://github.com/JackyJiang08/spotify-user-behavior-analytics.git
cd spotify-user-behavior-analytics
pip install -r requirements.txt

# Rebuild the user-level wide table from the raw event tables:
python scripts/build_user_feature_table.py

# Or explore interactively:
jupyter lab notebooks/
```

All notebooks and scripts share one data path (`src/data_loader.py`), which auto-discovers the data in `./data` (local) or `/content/data` (Colab) and reads the gzipped `listening_events.csv.gz` directly. Set `SPOTIFY_DATA_DIR` to point somewhere else.

> `listening_events.csv.gz` is stored gzipped to stay under GitHub's 100 MB file limit; nothing needs to be decompressed to run the project.

## License

[MIT](LICENSE)
