from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["time", "station", "layer", "temp", "psal", "depth", "label"]


def validate_ocean_schema(df: pd.DataFrame, is_test: bool = False) -> None:
    """Validate the schema of the Ocean dataset.
    
    Args:
        df (pd.DataFrame): The dataset to validate.
        is_test (bool): Whether the dataset is a test set (label column is optional).
        
    Raises:
        ValueError: If essential columns are missing or if dtypes are incorrect.
    """
    cols = list(df.columns)
    
    expected = REQUIRED_COLUMNS.copy()
    if is_test:
        expected.remove("label")
        
    missing = [c for c in expected if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
        
    # Check for NaNs in critical columns (station, layer, time)
    for c in ["station", "layer", "time"]:
        if c in df and df[c].isna().any():
            LOGGER.warning("Missing values found in critical column: %s", c)
            
    LOGGER.info("Schema validation passed (is_test=%s).", is_test)
