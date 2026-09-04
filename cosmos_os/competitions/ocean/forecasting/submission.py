"""Submission generation module for Problem 3 Wave Forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


SUBMISSION_COLUMNS = ["case_id", "station", "lead_h", "hs_pred"]


def generate_forecast_submission(
    df_test_index: pd.DataFrame,
    predictions_map: Dict[Tuple[str, int], float],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generates a submission DataFrame adhering strictly to Problem 3 specifications."""
    df_sub = df_test_index[["case_id", "station", "lead_h"]].copy()

    preds = []
    for _, row in df_sub.iterrows():
        cid = str(row["case_id"])
        lead = int(row["lead_h"])
        val = predictions_map.get((cid, lead), np.nan)
        preds.append(val)

    # Sanity clip
    preds_arr = np.array(preds, dtype=float)
    preds_clipped = np.clip(preds_arr, 0.0, 30.0)

    df_sub["hs_pred"] = preds_clipped

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_sub[SUBMISSION_COLUMNS].to_csv(p, index=False)

    return df_sub[SUBMISSION_COLUMNS]
