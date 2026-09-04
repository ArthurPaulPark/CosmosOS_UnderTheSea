"""Submission generator for Ocean Temperature Reconstruction (Problem 2)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("cosmos_os.reconstruction.submission")


def generate_reconstruction_submission(
    sample_sub: pd.DataFrame,
    predictions: Sequence[float] | pd.DataFrame,
    output_path: str | Path = "submission_reconstruction.csv",
    target_col: str = "temp",
) -> pd.DataFrame:
    """Generate final submission file matching sample_submission.csv format."""
    sub = sample_sub.copy()
    keys = ["station", "layer", "time"]

    if isinstance(predictions, pd.DataFrame) and all(k in predictions.columns for k in keys):
        # Join on the key. The feature transform reorders rows into timestamp-major
        # order, so lining predictions up by position puts every value on the wrong row.
        if all(k in sub.columns for k in keys):
            merged = sub[keys].merge(
                predictions[keys + [target_col]], on=keys, how="left", validate="one_to_one"
            )
            if len(merged) != len(sub):
                raise ValueError(f"Key join changed row count: {len(sub)} -> {len(merged)}")
            missing = int(merged[target_col].isna().sum())
            if missing:
                raise ValueError(f"{missing} submission rows had no matching prediction key")
            preds_arr = merged[target_col].to_numpy()
        else:
            preds_arr = predictions[target_col].to_numpy()
    elif isinstance(predictions, pd.DataFrame):
        if target_col in predictions.columns:
            preds_arr = predictions[target_col].values
        else:
            preds_arr = predictions.iloc[:, -1].values
    else:
        preds_arr = np.asarray(predictions, dtype=float)

    if len(preds_arr) != len(sub):
        raise ValueError(
            f"Prediction length mismatch: sample_submission has {len(sub)} rows, "
            f"but predictions has {len(preds_arr)} rows."
        )

    # Check for NaNs and handle safely
    if np.isnan(preds_arr).any():
        nan_count = int(np.isnan(preds_arr).sum())
        LOGGER.warning("Found %d NaN values in predictions. Filling with mean.", nan_count)
        valid_mean = float(np.nanmean(preds_arr)) if not np.isnan(preds_arr).all() else 15.0
        preds_arr = np.nan_to_num(preds_arr, nan=valid_mean)

    sub[target_col] = preds_arr

    # Ensure required columns if present
    required_keys = ["station", "layer", "time", target_col]
    if all(k in sub.columns for k in required_keys):
        sub = sub[required_keys]

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_p, index=False)
    LOGGER.info("Saved reconstruction submission to %s (%d rows).", out_p, len(sub))
    return sub
