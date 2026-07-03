#!/usr/bin/env python3
"""Build the user-level feature/label wide table and export it to CSV.

Runs the same SQL as notebooks/02_user_feature_table_and_labels.ipynb
(sql/build_user_feature_table.sql) end to end from the command line:

    python scripts/build_user_feature_table.py
    python scripts/build_user_feature_table.py --output /path/to/output.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import connect  # noqa: E402

DDL_PATH = REPO_ROOT / "sql" / "build_user_feature_table.sql"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "user_feature_table.csv"


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
    wide_df = sql("SELECT * FROM user_level_feature_table")

    # QA guard: the wide table must be one row per user.
    duplicated = len(wide_df) - wide_df["user_id"].nunique()
    if duplicated:
        raise SystemExit(
            f"QA failed: {duplicated} duplicated user rows — check JOIN keys."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(args.output, index=False)
    print(
        f"Wrote {len(wide_df):,} users x {wide_df.shape[1]} columns "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
