"""Single source of the notebook-05 churn-modeling protocol.

Freezes the exact data preparation, feature lists, train/test split, and
preprocessing used by notebooks/05_churn_model_training_and_evaluation.ipynb
so later model comparisons (notebook 07+) run on the identical protocol —
a comparison on a different split or feature set would be invalid.

On this repo's feature_table.csv every listed feature resolves to a real
column (some via the alias map, e.g. top_genre_30d -> top_genre); the
constant defaults below only kick in on reduced data versions, keeping the
protocol runnable there. Reproduction check: the logistic baseline must
match notebook 05 exactly (ROC-AUC 0.7876, recall 0.7817, precision 0.6704
at the 0.5 threshold).

Typical usage:

    from src.churn_modeling import build_preprocessor, load_modeling_split

    split = load_modeling_split()
    preprocess = build_preprocessor()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED
from src.data_loader import load_table

TARGET_COL = "churn_label_14d"
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "active_days_30d",
    "listen_minutes_30d",
    "sessions_30d",
    "playlist_count_30d",
    "liked_song_count_30d",
    "search_events_30d",
    "skip_rate_30d",
    "ad_load_per_active_day",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "subscription_type",
    "device",
    "acquisition_channel",
    "country",
    "top_genre",
]

_RENAME_MAP = {
    "primary_device": "device",
    "playlist_adds_30d": "playlist_count_30d",
    "liked_songs_30d": "liked_song_count_30d",
    "ad_impressions_per_active_day": "ad_load_per_active_day",
    "top_genre_30d": "top_genre",
}

_DEFAULTS = {
    "sessions_30d": 0,
    "search_events_30d": 0,
    "skip_rate_30d": 0,
    "tenure_days": 0,
    "ad_load_per_active_day": 0,
    "playlist_count_30d": 0,
    "liked_song_count_30d": 0,
    "device": "unknown",
    "acquisition_channel": "unknown",
    "subscription_type": "unknown",
    "country": "unknown",
    "top_genre": "unknown",
}


def prepare_modeling_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare the user-level modeling table with stable column aliases.

    Verbatim logic from notebook 05: alias data-version column names, fill
    missing optional columns with conservative defaults, and drop rows with
    a missing target.
    """
    df = df_raw.copy()
    for old_col, new_col in _RENAME_MAP.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    for col, default_value in _DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_value
    if TARGET_COL not in df.columns:
        raise KeyError(f"Missing required target column: {TARGET_COL}")
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


@dataclass(frozen=True)
class ModelingSplit:
    """The frozen notebook-05 train/test split plus its feature lists."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


def load_modeling_split(data_dir: Path | str | None = None) -> ModelingSplit:
    """Load feature_table and reproduce the exact notebook-05 split.

    Identical protocol: same source table, same feature selection (columns
    that actually exist after aliasing/defaults), same stratified 80/20
    split with the project seed.
    """
    df = prepare_modeling_table(load_table("feature_table", data_dir))

    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[numeric + categorical].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    return ModelingSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_features=numeric,
        categorical_features=categorical,
    )


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    dense: bool = False,
) -> ColumnTransformer:
    """The notebook-05 preprocessing: scaled numerics + one-hot categoricals.

    Scaling is a no-op for tree models but is kept for every model so all
    candidates see byte-identical inputs. `dense=True` makes the one-hot
    output a dense array — same values, different storage — for estimators
    that reject sparse input (HistGradientBoostingClassifier).
    """
    numeric = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical = (
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline(steps=[("onehot", onehot)]), categorical),
        ]
    )
