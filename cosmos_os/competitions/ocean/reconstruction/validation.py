"""Validation strategies and evaluation metrics for Ocean Temperature Reconstruction (Problem 2).

Provides:
1. Random Mask Validation (general random layer masking)
2. Primary Competition Simulation Validation (S-ORS station, 2024/2025 Sep~Oct, layers 2/3/4 temp+psal masked)
3. Secondary Blocked Validation (multiple contiguous windows)
4. Linear baseline & reconstruction metrics calculation
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER = logging.getLogger("cosmos_os.reconstruction.validation")


def _to_naive_datetime(s: pd.Series | str) -> pd.Series | pd.Timestamp:
    """Safely convert datetime series or string to timezone-naive Timestamp/Series."""
    if isinstance(s, str):
        ts = pd.to_datetime(s)
        if hasattr(ts, "tz") and ts.tz is not None:
            ts = ts.tz_localize(None)
        return ts
    dt = pd.to_datetime(s)
    if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
        return dt.dt.tz_localize(None)
    return dt


def calculate_reconstruction_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    layers: Sequence[int] | None = None,
) -> Dict[str, float]:
    """Calculate flat scalar metrics: RMSE, MAE, R2, count, and layer-specific RMSE metrics."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if mask.sum() == 0:
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "count": 0.0}

    y_t = y_t[mask]
    y_p = y_p[mask]

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.var(y_t) > 1e-9 else 1.0

    metrics: Dict[str, float] = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "count": float(len(y_t)),
    }

    if layers is not None:
        layers_arr = np.asarray(layers)[mask]
        for lyr in sorted(np.unique(layers_arr)):
            lyr_mask = layers_arr == lyr
            if lyr_mask.sum() > 0:
                lyr_rmse = float(np.sqrt(mean_squared_error(y_t[lyr_mask], y_p[lyr_mask])))
                metrics[f"layer_{int(lyr)}_rmse"] = lyr_rmse

    return metrics


def build_random_mask_validation(
    df: pd.DataFrame,
    test_size: float = 0.2,
    mask_ratio: float = 0.3,
    time_col: str = "time",
    layer_col: str = "layer",
    temp_col: str = "temp",
    psal_col: str = "psal",
    target_layers: Sequence[int] = (2, 3, 4),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split dataset chronologically and apply random masking on intermediate layers."""
    df = df.copy()
    if time_col in df.columns:
        df["_dt_parsed"] = _to_naive_datetime(df[time_col])
        df = df.sort_values("_dt_parsed").reset_index(drop=True)

    unique_times = df["_dt_parsed"].drop_duplicates().values if "_dt_parsed" in df.columns else df.index.values
    n_times = len(unique_times)
    split_idx = int(n_times * (1.0 - test_size))
    split_time = unique_times[split_idx]

    train_mask = df["_dt_parsed"] < split_time if "_dt_parsed" in df.columns else df.index < split_idx
    val_mask = ~train_mask

    if "_dt_parsed" in df.columns:
        df = df.drop(columns=["_dt_parsed"])

    raw_train = df[train_mask].copy().reset_index(drop=True)
    raw_val = df[val_mask].copy().reset_index(drop=True)

    # In validation, select rows with target_layers to mask
    rng = np.random.default_rng(random_state)
    is_target_layer = raw_val[layer_col].isin(target_layers)
    target_indices = raw_val[is_target_layer].index

    n_mask = int(len(target_indices) * mask_ratio)
    masked_indices = rng.choice(target_indices, size=n_mask, replace=False) if n_mask > 0 else []

    val_masked = raw_val.copy()
    ground_truth_series = raw_val.loc[masked_indices, temp_col].copy()
    ground_truth_df = raw_val.loc[masked_indices].copy()

    val_masked.loc[masked_indices, temp_col] = np.nan
    if psal_col in val_masked.columns:
        val_masked.loc[masked_indices, psal_col] = np.nan

    return raw_train, val_masked, ground_truth_df, ground_truth_series


def build_competition_simulation_validation(
    df: pd.DataFrame,
    target_station: str = "S-ORS",
    sim_start_date: str = "2024-09-01",
    sim_end_date: str = "2024-10-31",
    missing_layers: Sequence[int] = (2, 3, 4),
    time_col: str = "time",
    station_col: str = "station",
    layer_col: str = "layer",
    temp_col: str = "temp",
    psal_col: str = "psal",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Reproduce official competition condition on complete historical/validation data with simultaneous temp+psal masking."""
    df = df.copy()
    if time_col in df.columns:
        df["_dt_parsed"] = _to_naive_datetime(df[time_col])
        df = df.sort_values("_dt_parsed").reset_index(drop=True)

    sim_mask = pd.Series(False, index=df.index)
    if "_dt_parsed" in df.columns:
        t_start = _to_naive_datetime(sim_start_date)
        t_end = _to_naive_datetime(sim_end_date)
        time_in_range = (df["_dt_parsed"] >= t_start) & (df["_dt_parsed"] <= t_end)
    else:
        time_in_range = pd.Series(True, index=df.index)

    if station_col in df.columns and target_station in df[station_col].values:
        station_match = df[station_col] == target_station
    else:
        station_match = pd.Series(True, index=df.index)

    sim_period_mask = time_in_range & station_match

    # If the exact simulation slice is empty, fall back to latest 20% slice
    if sim_period_mask.sum() == 0:
        LOGGER.warning("Exact simulation date range not found. Falling back to latest 20% slice.")
        unique_times = df["_dt_parsed"].drop_duplicates().values if "_dt_parsed" in df.columns else df.index.values
        split_idx = int(len(unique_times) * 0.8)
        split_time = unique_times[split_idx]
        sim_period_mask = df["_dt_parsed"] >= split_time if "_dt_parsed" in df.columns else df.index >= split_idx

    if "_dt_parsed" in df.columns:
        df = df.drop(columns=["_dt_parsed"])

    raw_train = df[~sim_period_mask].copy().reset_index(drop=True)
    val_complete = df[sim_period_mask].copy().reset_index(drop=True)

    is_missing_layer = val_complete[layer_col].isin(missing_layers)
    val_target_df = val_complete[is_missing_layer].copy()
    val_target_series = val_target_df[temp_col].copy()

    val_masked_df = val_complete.copy()
    val_masked_df.loc[is_missing_layer, temp_col] = np.nan
    if psal_col in val_masked_df.columns:
        val_masked_df.loc[is_missing_layer, psal_col] = np.nan

    return raw_train, val_masked_df, val_target_df, val_target_series


def build_secondary_blocked_validation(
    df: pd.DataFrame,
    block_windows: Sequence[Tuple[str, str]] | None = None,
    target_station: str = "S-ORS",
    missing_layers: Sequence[int] = (2, 3, 4),
) -> List[Dict[str, Any]]:
    """Generate multiple contiguous blocked validation slices."""
    if block_windows is None:
        block_windows = [
            ("2024-05-01", "2024-06-30"),  # Spring early stratification
            ("2024-07-01", "2024-08-31"),  # Mid-summer peak stratification
            ("2024-09-01", "2024-10-31"),  # Autumn primary transition
            ("2025-05-01", "2025-06-30"),  # 2025 Spring
            ("2025-07-01", "2025-08-31"),  # 2025 Summer
        ]

    slices = []
    for start_d, end_d in block_windows:
        raw_tr, val_m, val_tgt, val_s = build_competition_simulation_validation(
            df,
            target_station=target_station,
            sim_start_date=start_d,
            sim_end_date=end_d,
            missing_layers=missing_layers,
        )
        valid_targets = val_s.dropna()
        if len(valid_targets) > 100:
            slices.append({
                "window": f"{start_d}~{end_d}",
                "start_date": start_d,
                "end_date": end_d,
                "raw_train": raw_tr,
                "val_masked": val_m,
                "val_target_df": val_tgt,
                "val_target_series": val_s,
                "evaluable_count": len(valid_targets),
            })
    return slices


def calculate_baseline_linear_interpolation_rmse(
    val_df_with_features: pd.DataFrame,
    target_indices: Sequence[Any],
    true_target: Sequence[float],
) -> float:
    """Calculate RMSE of linear vertical interpolation baseline on the target evaluation slice."""
    if "vertical_linear_interp" not in val_df_with_features.columns:
        return 999.0
    preds = val_df_with_features.loc[target_indices, "vertical_linear_interp"].values
    truth = np.asarray(true_target, dtype=float)
    mask = np.isfinite(preds) & np.isfinite(truth)
    if mask.sum() == 0:
        return 999.0
    return float(np.sqrt(mean_squared_error(truth[mask], preds[mask])))
