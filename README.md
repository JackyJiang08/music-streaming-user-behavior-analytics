# Spotify User Behavior Analytics

**End-to-end analytics on music-streaming user behavior: retention diagnostics, free-to-paid conversion, and a reproducible user-level feature/label pipeline.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Stack](https://img.shields.io/badge/stack-SQL%20%7C%20pandas%20%7C%20matplotlib-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-brightgreen)

> Independent project built on simulated data. Not affiliated with, endorsed by, or based on internal data from Spotify.

## Overview

This project analyzes the two core commercial questions for a subscription streaming service:

- **Retention** — which behavioral signals (activity level, ad pressure, content interaction) predict short-term churn, and which segments are at risk.
- **Monetization** — where the free-to-paid subscription funnel breaks, and which levers (trial exposure, device experience, engagement depth) move conversion.

Delivery follows the standard analytics workflow: SQL diagnostics → leakage-safe user-level feature/label data asset → exploratory analysis and business visualization → predictive modeling and experiment design (planned).

## Quickstart

```bash
git clone https://github.com/JackyJiang08/spotify-user-behavior-analytics.git
cd spotify-user-behavior-analytics
pip install -r requirements.txt

# Rebuild the user-level wide table from the raw event tables
python scripts/build_user_feature_table.py

# Or explore interactively
jupyter lab notebooks/
```

Runs on `pandas + sqlite3`; no database setup required. All notebooks and scripts load data through `src/data_loader.py`, which auto-discovers `./data` locally or `/content/data` on Colab (override with `SPOTIFY_DATA_DIR`) and reads the gzipped events file directly — nothing needs to be decompressed.

## Dataset

Five simulated tables covering **50,000 users** and **~1.5M events**:

| Table | Grain | Contents |
|---|---|---|
| `users.csv` | user | Demographics, device, acquisition channel, signup date |
| `listening_events.csv.gz` | event | 1.07M listening logs: sessions, duration, skips, likes, playlist adds |
| `subscription_events.csv` | event | Trial exposure/start, payments, renewals, cancellations |
| `ad_events.csv` | event | Ad impressions, clicks, completions, revenue |
| `feature_table.csv` | user | Prebuilt user-level feature snapshot |

`listening_events` is stored gzipped to stay under GitHub's 100 MB file limit.

## Analysis

| Notebook | Scope | Deliverables |
|---|---|---|
| [`01_retention_and_conversion_analysis`](notebooks/01_retention_and_conversion_analysis.ipynb) | Business baseline; lifecycle segmentation; five behavioral hypotheses — activity vs. churn, ad overload, trial exposure, device, content interaction | Retention/conversion baselines, hypothesis readouts, funnel break-point map |
| [`02_user_feature_table_and_labels`](notebooks/02_user_feature_table_and_labels.ipynb) | Snapshot/observation/prediction window design; event-to-user feature aggregation; leakage-safe 14-day churn and 30-day conversion labels | QA-verified one-row-per-user wide table |
| [`03_eda_and_visualization`](notebooks/03_eda_and_visualization.ipynb) | Data-quality gates; derived segments (activity, content engagement, ad load); visual diagnostics of churn and conversion drivers | Business charts, each framed as question → reading → action → limitation |

The wide-table definition is versioned once in `sql/build_user_feature_table.sql` and consumed by both notebook 02 and the CLI pipeline `scripts/build_user_feature_table.py` (build → QA → export).

## Repository Structure

```
├── data/                          # Source tables (see Dataset)
├── notebooks/                     # Analysis notebooks (01-03)
├── src/
│   └── data_loader.py             # Shared loader: CSVs -> pandas -> in-memory SQLite
├── sql/
│   └── build_user_feature_table.sql   # Wide-table definition (single source of truth)
├── scripts/
│   └── build_user_feature_table.py    # CLI pipeline: build + QA + export the wide table
└── requirements.txt
```

## Project Status

- [x] Retention and conversion diagnostics (SQL)
- [x] User-level feature table and label engineering
- [x] Exploratory data analysis and visualization
- [ ] Churn and paid-conversion prediction models
- [ ] Experiment design for conversion levers (trial exposure, ad load)

## License

[MIT](LICENSE)
