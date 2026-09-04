"""Schema validation module for Ocean Significant Wave Height Forecasting (Problem 3)."""

from __future__ import annotations

from typing import List, Tuple
import pandas as pd


WAVE_REQUIRED_COLUMNS = ["station", "time", "hs", "tp", "hmax", "wvdir"]
ATMOS_REQUIRED_COLUMNS = ["station", "time", "wspd", "gust", "wdir", "airt", "relh", "caph"]
TEST_CONTEXT_REQUIRED_COLUMNS = [
    "case_id",
    "station",
    "step_minute",
    "hs",
    "tp",
    "hmax",
    "wvdir",
    "wspd",
    "gust",
    "wdir",
    "airt",
    "relh",
    "caph",
]
TEST_INDEX_REQUIRED_COLUMNS = ["case_id", "station", "lead_h"]
SUBMISSION_REQUIRED_COLUMNS = ["case_id", "station", "lead_h", "hs_pred"]
VALID_LEAD_HOURS = [3, 6, 9, 12, 18, 24]
VALID_STATIONS = ["G-ORS", "I-ORS", "S-ORS"]


def validate_forecasting_schema(
    df_wave: pd.DataFrame,
    df_atmos: pd.DataFrame,
    df_test_context: pd.DataFrame | None = None,
    df_test_index: pd.DataFrame | None = None,
    df_sample_sub: pd.DataFrame | None = None,
) -> Tuple[bool, List[str]]:
    """Validate all Problem 3 dataset schemas."""
    errors: List[str] = []

    # 1. Validate train_wave
    missing_wave = [c for c in WAVE_REQUIRED_COLUMNS if c not in df_wave.columns]
    if missing_wave:
        errors.append(f"train_wave missing required columns: {missing_wave}")
    if not df_wave["station"].isin(VALID_STATIONS).all():
        errors.append(f"train_wave contains invalid station IDs: {set(df_wave['station']) - set(VALID_STATIONS)}")

    # 2. Validate train_atmos
    missing_atmos = [c for c in ATMOS_REQUIRED_COLUMNS if c not in df_atmos.columns]
    if missing_atmos:
        errors.append(f"train_atmos missing required columns: {missing_atmos}")
    if not df_atmos["station"].isin(VALID_STATIONS).all():
        errors.append(f"train_atmos contains invalid station IDs: {set(df_atmos['station']) - set(VALID_STATIONS)}")

    # 3. Validate test_context
    if df_test_context is not None:
        missing_ctx = [c for c in TEST_CONTEXT_REQUIRED_COLUMNS if c not in df_test_context.columns]
        if missing_ctx:
            errors.append(f"test_context missing required columns: {missing_ctx}")
        if df_test_context["step_minute"].min() != -2880 or df_test_context["step_minute"].max() != 0:
            errors.append(
                f"test_context step_minute range invalid: [{df_test_context['step_minute'].min()}, {df_test_context['step_minute'].max()}]"
            )
        # Check rows per case
        counts = df_test_context.groupby("case_id").size()
        if not (counts == 289).all():
            errors.append("test_context cases do not all have exactly 289 rows")

    # 4. Validate test_index
    if df_test_index is not None:
        missing_idx = [c for c in TEST_INDEX_REQUIRED_COLUMNS if c not in df_test_index.columns]
        if missing_idx:
            errors.append(f"test_index missing required columns: {missing_idx}")
        if not df_test_index["lead_h"].isin(VALID_LEAD_HOURS).all():
            errors.append(f"test_index contains invalid lead_h: {set(df_test_index['lead_h']) - set(VALID_LEAD_HOURS)}")
        if len(df_test_index) != 1200:
            errors.append(f"test_index expected 1,200 rows, got {len(df_test_index)}")

    # 5. Validate sample_submission
    if df_sample_sub is not None:
        if list(df_sample_sub.columns) != SUBMISSION_REQUIRED_COLUMNS:
            errors.append(f"sample_submission column mismatch: {list(df_sample_sub.columns)} vs {SUBMISSION_REQUIRED_COLUMNS}")
        if len(df_sample_sub) != 1200:
            errors.append(f"sample_submission expected 1,200 rows, got {len(df_sample_sub)}")

    return len(errors) == 0, errors
