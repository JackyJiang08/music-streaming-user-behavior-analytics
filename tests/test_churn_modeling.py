"""Unit tests for src/churn_modeling.py on a tiny synthetic frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.churn_modeling import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    prepare_modeling_table,
)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "primary_device": rng.choice(["mobile_ios", "web"], size=n),
            "playlist_adds_30d": rng.poisson(1, size=n),
            "liked_songs_30d": rng.poisson(2, size=n),
            "active_days_30d": rng.integers(0, 30, size=n),
            "skip_rate_30d": rng.uniform(0, 1, size=n),
            "acquisition_channel": rng.choice(["organic", "paid"], size=n),
            "country": rng.choice(["US", "UK"], size=n),
            "churn_label_14d": rng.integers(0, 2, size=n).astype(float),
        }
    )


def test_prepare_aliases_defaults_and_target(raw_frame):
    raw_frame.loc[0, "churn_label_14d"] = np.nan
    df = prepare_modeling_table(raw_frame)
    assert len(df) == len(raw_frame) - 1  # missing-target row dropped
    assert df["churn_label_14d"].dtype.kind == "i"
    assert (df["device"] == raw_frame["primary_device"].iloc[1:]).all()  # alias
    assert (df["tenure_days"] == 0).all()  # missing column -> constant default
    assert (df["subscription_type"] == "unknown").all()


def test_prepare_requires_target(raw_frame):
    with pytest.raises(KeyError, match="churn_label_14d"):
        prepare_modeling_table(raw_frame.drop(columns=["churn_label_14d"]))


def test_preprocessor_transforms_available_features(raw_frame):
    df = prepare_modeling_table(raw_frame)
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    preprocess = build_preprocessor(numeric, categorical)
    transformed = preprocess.fit_transform(df[numeric + categorical])
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= len(numeric)  # one-hot expands categoricals
