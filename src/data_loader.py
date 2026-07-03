"""Data loading utilities for the Spotify user conversion project.

Loads the five project tables into pandas and registers them in an in-memory
SQLite database, so every notebook and script queries the same schema through
one code path.

Typical usage (notebook or script):

    from src.data_loader import connect

    conn, sql, run_script = connect()
    sql("SELECT COUNT(*) FROM users")
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

TABLES = {
    "users": "users.csv",
    "listening_events": "listening_events.csv",
    "subscription_events": "subscription_events.csv",
    "ad_events": "ad_events.csv",
    "feature_table": "feature_table.csv",
}

REPO_ROOT = Path(__file__).resolve().parents[1]

# Searched in order; the SPOTIFY_DATA_DIR environment variable wins if set.
_CANDIDATE_DIRS = [
    REPO_ROOT / "data",
    Path("/content/data"),  # Colab upload target
]


def find_data_dir() -> Path:
    """Locate the directory holding the project CSVs."""
    env_dir = os.environ.get("SPOTIFY_DATA_DIR")
    candidates = [Path(env_dir)] if env_dir else _CANDIDATE_DIRS
    for d in candidates:
        if d.is_dir():
            return d
    raise FileNotFoundError(
        "Could not find the data directory. Expected one of: "
        + ", ".join(str(d) for d in candidates)
        + ". Set the SPOTIFY_DATA_DIR environment variable to override."
    )


def _resolve_file(data_dir: Path, filename: str) -> Path:
    """Return the CSV path, accepting a gzipped variant (e.g. .csv.gz)."""
    for candidate in (data_dir / filename, data_dir / f"{filename}.gz"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing data file: {filename} (or {filename}.gz) in {data_dir}"
    )


def load_dataframes(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load all project tables as pandas DataFrames, keyed by table name."""
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    return {
        table: pd.read_csv(_resolve_file(data_dir, filename))
        for table, filename in TABLES.items()
    }


def load_table(table: str, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load a single project table (cheaper than load_dataframes when the
    analysis only needs one, e.g. feature_table)."""
    if table not in TABLES:
        raise KeyError(f"Unknown table {table!r}; expected one of {list(TABLES)}")
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    return pd.read_csv(_resolve_file(data_dir, TABLES[table]))


def connect(data_dir: Path | str | None = None):
    """Load the tables into in-memory SQLite and return query helpers.

    Returns:
        conn: the sqlite3 connection with all tables registered.
        sql: function running a SELECT and returning a DataFrame.
        run_script: function executing a multi-statement SQL script
            (e.g. CREATE VIEW definitions).
    """
    dataframes = load_dataframes(data_dir)
    conn = sqlite3.connect(":memory:")
    for table, df in dataframes.items():
        df.to_sql(table, conn, if_exists="replace", index=False)

    def sql(query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, conn)

    def run_script(script: str) -> None:
        conn.executescript(script)

    print(f"Loaded tables: {', '.join(TABLES)}")
    return conn, sql, run_script
