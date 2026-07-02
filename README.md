# Spotify User Behavior & Conversion Optimization

> **Work in progress** — this project is at an early stage; more analysis, EDA, and modeling notebooks will be added.

A data analysis project on Spotify-style user behavior: retention diagnostics, subscription conversion funnels, and building a user-level feature/label wide table for downstream churn and conversion modeling.

## Repository Structure

```
├── data/
│   ├── users.csv                  # One row per user: demographics, device, acquisition channel
│   ├── listening_events.csv.gz    # Event-level listening logs (gzipped, ~118 MB uncompressed)
│   ├── subscription_events.csv    # Trial / payment / renewal / cancellation events
│   ├── ad_events.csv              # Ad impressions, clicks, completions, revenue
│   └── feature_table.csv          # Prebuilt user-level feature table (one row per user)
└── notebooks/
    ├── Lesson2_SQL_I.ipynb        # Retention overview & hypothesis testing with SQL
    └── Lesson3_SQL_II_User_Level_Wide_Table_and_Label_Design.ipynb
                                   # Building the user-level wide table and labels
```

## Notebooks

Both notebooks are designed for Google Colab and use `pandas + sqlite3` (no DuckDB required):

- **Lesson 2 — SQL I: Retention Overview and Hypothesis Testing.** Table granularity checks, business baseline metrics, and five hypotheses: low activity as a churn signal, ad overload vs. retention, trial exposure vs. paid conversion, device differences, and content interaction vs. stickiness. Includes a subscription conversion funnel.
- **Lesson 3 — SQL II: User-Level Wide Table and Label Design.** Time window design (snapshot / observation / prediction), feature aggregation from listening, ad, and subscription events, churn/conversion label definitions, JOIN-inflation QA, and export of the final wide table.

## Usage (Colab)

1. Create a `/content/data` folder in the Colab file sidebar.
2. Upload the five data files. Note: `listening_events.csv.gz` is gzipped to stay under GitHub's 100 MB file limit — decompress it first (`gunzip listening_events.csv.gz`), or load it directly with `pd.read_csv('listening_events.csv.gz')` (pandas handles gzip transparently).
3. Run the data-loading cell, then work through the sections.

## License

[MIT](LICENSE)
