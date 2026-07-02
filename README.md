# Spotify User Behavior & Conversion Optimization

An end-to-end data analytics project on Spotify-style user behavior, focused on two business problems:

1. **Retention** — which behavioral signals (activity level, ad load, content interaction) predict short-term churn?
2. **Monetization** — where does the free-to-paid subscription funnel break, and which levers (trial exposure, device, engagement) move conversion?

The project builds from SQL-based diagnostics toward a full modeling pipeline: hypothesis-driven analysis, a reproducible user-level feature/label data asset, and (upcoming) EDA, churn/conversion models, and experiment design.

> **Status: early stage.** The diagnostic analysis and feature-table construction are in place; EDA, modeling, and experimentation phases are planned next.

## Repository Structure

```
├── data/
│   ├── users.csv                  # One row per user: demographics, device, acquisition channel
│   ├── listening_events.csv.gz    # Event-level listening logs (gzipped; ~118 MB uncompressed)
│   ├── subscription_events.csv    # Trial, payment, renewal, and cancellation events
│   ├── ad_events.csv              # Ad impressions, clicks, completions, revenue
│   └── feature_table.csv          # Prebuilt user-level feature snapshot (one row per user)
├── notebooks/
│   ├── 01_retention_and_conversion_analysis.ipynb
│   └── 02_user_feature_table_and_labels.ipynb
├── src/
│   └── data_loader.py             # Shared loader: CSVs -> pandas -> in-memory SQLite
├── sql/
│   └── build_user_feature_table.sql   # Wide-table definition (single source of truth)
├── scripts/
│   └── build_feature_table.py     # CLI pipeline: build + QA + export the wide table
└── requirements.txt
```

## Analysis

**`01_retention_and_conversion_analysis.ipynb` — Retention Overview and Conversion Hypothesis Analysis.**
Establishes the business baseline (activity, churn, conversion, cancellation rates), segments users by subscription status and lifecycle stage, and tests five behavioral hypotheses: low activity as a churn signal, ad overload vs. retention (with activity-level controls), trial exposure vs. paid conversion, device differences, and content interaction vs. stickiness. Maps the subscription conversion funnel to locate drop-off points.

**`02_user_feature_table_and_labels.ipynb` — User-Level Feature Table and Label Engineering.**
Designs the snapshot / observation / prediction time-window framework, aggregates listening, ad, and subscription events into user-level features, defines leakage-safe churn (14-day) and paid-conversion (30-day) labels from the prediction window, and QA-checks the assembled wide table (JOIN inflation, label and feature distributions) before exporting it for downstream EDA and modeling. The wide-table SQL is versioned in `sql/build_user_feature_table.sql` and can be run end to end with `python scripts/build_feature_table.py`.

## Roadmap

- [x] Retention and conversion diagnostics (SQL)
- [x] User-level feature table and label engineering
- [ ] Exploratory data analysis and visualization
- [ ] Churn and paid-conversion prediction models
- [ ] Experiment design for conversion levers (trial exposure, ad load)

## Getting Started

Everything runs on `pandas + sqlite3` — no database setup required.

```bash
git clone https://github.com/JackyJiang08/spotify-user-conversion-optimization.git
cd spotify-user-conversion-optimization
pip install -r requirements.txt

# Rebuild the user-level wide table from the raw event tables:
python scripts/build_feature_table.py

# Or explore interactively:
jupyter lab notebooks/
```

All notebooks and scripts share one data path (`src/data_loader.py`), which auto-discovers the data in `./data` (local) or `/content/data` (Colab) and reads the gzipped `listening_events.csv.gz` directly. Set `SPOTIFY_DATA_DIR` to point somewhere else.

> `listening_events.csv.gz` is stored gzipped to stay under GitHub's 100 MB file limit; nothing needs to be decompressed to run the project.

## License

[MIT](LICENSE)
