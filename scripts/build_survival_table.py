#!/usr/bin/env python3
"""Build the per-user time-to-churn survival table and export it to CSV.

Runs the SQL used by notebooks/08_survival_analysis_time_to_churn.ipynb
(sql/build_survival_table.sql) end to end from the command line:

    python scripts/build_survival_table.py
    python scripts/build_survival_table.py --output /path/to/output.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import connect  # noqa: E402

DDL_PATH = REPO_ROOT / "sql" / "build_survival_table.sql"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "survival_table.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    conn, sql, run_script = connect()
    run_script(DDL_PATH.read_text())
    survival_df = sql("SELECT * FROM survival_table")

    # QA guards: one row per user, no negative durations, sane event flags.
    duplicated = len(survival_df) - survival_df["user_id"].nunique()
    if duplicated:
        raise SystemExit(
            f"QA failed: {duplicated} duplicated user rows — check JOIN keys."
        )
    negative = int((survival_df["duration_days"] <= 0).sum())
    if negative:
        raise SystemExit(
            f"QA failed: {negative} non-positive durations — check the horizon "
            "and signup dates."
        )
    if not set(survival_df["churn_event"].unique()) <= {0, 1}:
        raise SystemExit("QA failed: churn_event must be 0/1.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    survival_df.to_csv(args.output, index=False)
    censored_share = 1 - survival_df["churn_event"].mean()
    print(
        f"Wrote {len(survival_df):,} users x {survival_df.shape[1]} columns "
        f"to {args.output}\n"
        f"Events: {int(survival_df['churn_event'].sum()):,} "
        f"({survival_df['churn_event'].mean():.1%}) | "
        f"censored: {censored_share:.1%} | "
        f"duration range: {survival_df['duration_days'].min():.0f}-"
        f"{survival_df['duration_days'].max():.0f} days"
    )


if __name__ == "__main__":
    main()
