"""Schema validation for Ocean Temperature Reconstruction (Problem 2)."""

from __future__ import annotations

import logging
import pandas as pd

LOGGER = logging.getLogger("cosmos_os.reconstruction.schema")


def validate_reconstruction_schema(df: pd.DataFrame, is_test: bool = False) -> bool:
    """Validate dataframe structure for ocean temperature reconstruction.
    
    Required columns:
    - time (or date)
    - station
    - layer
    - temp (required for train, optional/NaN for test missing layers)
    """
    required_cols = ["station", "layer"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Reconstruction dataset missing required columns: {missing}")

    if not is_test and "temp" not in df.columns:
        raise ValueError("Reconstruction train dataset must contain 'temp' column.")

    if len(df) == 0:
        raise ValueError("Reconstruction dataset is empty.")

    LOGGER.info("Reconstruction schema validated successfully (%d rows, is_test=%s).", len(df), is_test)
    return True
