"""Locks the project's headline baselines so CI catches silent drift.

Every number here is quoted in the README or gated on by a notebook. If one
of these tests fails, the committed data or a protocol module changed in a
way that invalidates the published narrative — fix the regression, do not
update the expected value.

Only cheap invariants are recomputed (flat reads of committed data). The
expensive ones are locked elsewhere: notebook 07 hard-fails unless the
logistic churn baseline reproduces notebook 05 exactly, and the SQL table
constructions are verified on synthetic event logs in the other test files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def test_feature_table_base_rates():
    """The README's headline baselines: 47.6% churn, 17.4% conversion."""
    table = pd.read_csv(
        DATA_DIR / "feature_table.csv",
        usecols=["user_id", "churn_label_14d", "paid_conversion_30d"],
    )
    assert len(table) == 50_000
    assert table["user_id"].nunique() == 50_000
    assert table["churn_label_14d"].mean() == pytest.approx(0.47558, abs=1e-5)
    assert table["paid_conversion_30d"].mean() == pytest.approx(0.17392, abs=1e-5)


def test_experiment_dataset_headlines():
    """Notebook 06's measured ATE and the injected ground-truth average."""
    experiment = pd.read_csv(
        DATA_DIR / "experiment_results.csv.gz",
        usecols=["group", "retained_14d_post", "true_uplift_pp"],
    )
    assert len(experiment) == 33_473
    by_arm = experiment.groupby("group")["retained_14d_post"].mean()
    ate = by_arm["treatment"] - by_arm["control"]
    assert ate == pytest.approx(0.02576, abs=1e-4)  # the +2.58pp headline
    assert experiment["true_uplift_pp"].mean() == pytest.approx(2.2252, abs=1e-3)


def test_churn_baseline_gate_is_documented():
    """The notebook-07 parity-gate numbers must stay in the protocol module.

    The gate itself runs when notebook 07 executes (it refits the model);
    this test only pins the documented values against accidental edits.
    """
    docstring = (REPO_ROOT / "src" / "churn_modeling.py").read_text()
    for value in ("0.7876", "0.7817", "0.6704"):
        assert value in docstring, f"churn baseline {value} missing from docstring"


def test_conversion_baseline_gate_is_documented():
    """Same pinning for the notebook-10 conversion baselines."""
    docstring = (REPO_ROOT / "src" / "conversion_modeling.py").read_text()
    for value in ("0.7274", "0.8939", "0.0994"):
        assert value in docstring, f"conversion baseline {value} missing from docstring"
