"""Project-wide constants shared by notebooks, scripts, and src modules.

Centralizes the analysis snapshot dates and the default random seed so every
stochastic step in the project is reproducible from one place.

Typical usage:

    from src.config import RANDOM_SEED, SNAPSHOT_DATE
"""

from __future__ import annotations

# Default seed for every stochastic step (train/test splits, simulations).
RANDOM_SEED: int = 42

# Analysis snapshot: features are computed on data up to this date.
SNAPSHOT_DATE: str = "2026-04-01"

# Behavioral observation window feeding the user-level feature table.
OBSERVATION_START: str = "2026-03-02"
OBSERVATION_END: str = SNAPSHOT_DATE

# Label windows measured after the snapshot.
CHURN_WINDOW_DAYS: int = 14  # churn = no listening activity in this window
CONVERSION_WINDOW_DAYS: int = 30  # conversion = paid start within this window
