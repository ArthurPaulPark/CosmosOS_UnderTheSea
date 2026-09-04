from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def generate_ocean_submission(
    sample_sub: pd.DataFrame, 
    predictions: pd.DataFrame, 
    output_path: str | Path = "submission.csv",
    fill_anomaly_type: bool = False
) -> Path:
    """Generate the submission file for the Ocean competition.
    
    Args:
        sample_sub (pd.DataFrame): The sample submission template.
        predictions (pd.DataFrame): DataFrame containing at least 'id' and 'label' (or array of predictions).
                                    If array, it will be mapped sequentially.
        output_path (str | Path): Path to save the final submission.csv.
        fill_anomaly_type (bool): Whether to optionally write anomaly types.
        
    Returns:
        Path: Path to the generated submission file.
    """
    out_path = Path(output_path)
    sub = sample_sub.copy()
    
    # If predictions is a dataframe with id and label, merge it.
    # Otherwise, assume it's a 1D array of labels in the same order as test data.
    if isinstance(predictions, pd.DataFrame) and "label" in predictions:
        if "id" in predictions and "id" in sub:
            # Map by id if available
            mapping = dict(zip(predictions["id"], predictions["label"]))
            sub["label"] = sub["id"].map(mapping).fillna(0).astype(int)
        else:
            sub["label"] = predictions["label"].values
    else:
        # Array-like
        sub["label"] = predictions

    if fill_anomaly_type and "anomaly_type" in sub.columns:
        # Just a placeholder for optional logic
        LOGGER.info("Filling anomaly_type column (optional logic).")
        # In reality, this would map anomaly types based on feature contributions
        # Currently, just set to empty or default
        pass

    sub.to_csv(out_path, index=False)
    LOGGER.info("Submission saved to %s", out_path)
    return out_path
