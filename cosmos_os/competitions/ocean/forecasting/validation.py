"""Validation logic and metrics computation for Problem 3 Wave Forecasting."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

from .cases import ForecastCase, HistoricalForecastCaseBuilder, VALID_LEAD_HOURS


def build_primary_forecast_validation(
    grid_by_station: Dict[str, pd.DataFrame],
    train_start: str = "2024-01-01 00:00:00+09:00",
    train_end: str = "2025-02-28 23:50:00+09:00",
    val_start: str = "2025-03-04 00:00:00+09:00",
    val_end: str = "2025-06-30 23:50:00+09:00",
    dense_stride_steps: int = 2,  # every 20 minutes for train
) -> Tuple[List[ForecastCase], List[ForecastCase]]:
    """Builds primary temporal forward train/validation split strictly within historical train data."""
    builder = HistoricalForecastCaseBuilder()

    t_train_start = pd.to_datetime(train_start)
    t_train_end = pd.to_datetime(train_end)
    t_val_start = pd.to_datetime(val_start)
    t_val_end = pd.to_datetime(val_end)

    # 1. Validation cases: strict >=78h separation per station
    val_cases = builder.build_cases(
        grid_by_station,
        start_time=t_val_start,
        end_time=t_val_end,
        min_separation_hours=78.0,
    )

    # 2. Training cases: dense anchors
    raw_train_cases = builder.build_cases(
        grid_by_station,
        start_time=t_train_start,
        end_time=t_train_end,
        min_separation_hours=None,
        dense_anchor_stride_steps=dense_stride_steps,
    )

    # 3. Purge training cases overlapping with validation
    train_cases = builder.purge_overlapping_train_cases(raw_train_cases, val_cases, safety_buffer_hours=6.0)

    return train_cases, val_cases


def build_rolling_forecast_backtest(
    grid_by_station: Dict[str, pd.DataFrame],
    dense_stride_steps: int = 3,
) -> List[Dict[str, Any]]:
    """Builds 3 temporal forward backtest folds strictly within 2024-01 to 2025-06-30."""
    builder = HistoricalForecastCaseBuilder()

    folds_def = [
        {
            "fold": 1,
            "name": "2024_H2_Holdout",
            "train_start": "2024-01-01 00:00:00+09:00",
            "train_end": "2024-07-31 23:50:00+09:00",
            "val_start": "2024-08-04 00:00:00+09:00",
            "val_end": "2024-11-30 23:50:00+09:00",
        },
        {
            "fold": 2,
            "name": "2024_Winter_Holdout",
            "train_start": "2024-01-01 00:00:00+09:00",
            "train_end": "2024-11-30 23:50:00+09:00",
            "val_start": "2024-12-04 00:00:00+09:00",
            "val_end": "2025-03-31 23:50:00+09:00",
        },
        {
            "fold": 3,
            "name": "2025_Spring_Holdout",
            "train_start": "2024-01-01 00:00:00+09:00",
            "train_end": "2025-02-28 23:50:00+09:00",
            "val_start": "2025-03-04 00:00:00+09:00",
            "val_end": "2025-06-30 23:50:00+09:00",
        },
    ]

    folds: List[Dict[str, Any]] = []
    for f in folds_def:
        t_tr_start = pd.to_datetime(f["train_start"])
        t_tr_end = pd.to_datetime(f["train_end"])
        t_v_start = pd.to_datetime(f["val_start"])
        t_v_end = pd.to_datetime(f["val_end"])

        val_cases = builder.build_cases(
            grid_by_station,
            start_time=t_v_start,
            end_time=t_v_end,
            min_separation_hours=78.0,
        )

        raw_tr = builder.build_cases(
            grid_by_station,
            start_time=t_tr_start,
            end_time=t_tr_end,
            min_separation_hours=None,
            dense_anchor_stride_steps=dense_stride_steps,
        )

        train_cases = builder.purge_overlapping_train_cases(raw_tr, val_cases)

        folds.append({
            "fold": f["fold"],
            "name": f["name"],
            "train_period": f"{f['train_start'][:10]} ~ {f['train_end'][:10]}",
            "val_period": f"{f['val_start'][:10]} ~ {f['val_end'][:10]}",
            "train_cases": train_cases,
            "val_cases": val_cases,
        })

    return folds


def calculate_forecast_metrics(
    val_cases: List[ForecastCase],
    preds_dict: Dict[Tuple[str, int], float],  # (case_id, lead_h) -> pred_hs
) -> Dict[str, float]:
    """Calculates pooled RMSE, lead-wise RMSE, station-wise RMSE, and persistence baseline comparison."""
    y_true_list = []
    y_pred_list = []
    y_pers_list = []
    leads_list = []
    stations_list = []

    for c in val_cases:
        for lead in VALID_LEAD_HOURS:
            y_t = c.targets[lead]
            y_p = preds_dict.get((c.case_id, lead), c.latest_hs)
            y_base = c.latest_hs

            y_true_list.append(y_t)
            y_pred_list.append(y_p)
            y_pers_list.append(y_base)
            leads_list.append(lead)
            stations_list.append(c.station)

    df_eval = pd.DataFrame({
        "true": y_true_list,
        "pred": y_pred_list,
        "pers": y_pers_list,
        "lead": leads_list,
        "station": stations_list,
    })

    err_model = df_eval["pred"] - df_eval["true"]
    err_pers = df_eval["pers"] - df_eval["true"]

    overall_rmse = float(np.sqrt(np.mean(err_model**2)))
    pers_rmse = float(np.sqrt(np.mean(err_pers**2)))

    metrics: Dict[str, float] = {
        "rmse": overall_rmse,
        "baseline_persistence_rmse": pers_rmse,
        "rmse_improvement_over_baseline": float(pers_rmse - overall_rmse),
        "rmse_improvement_ratio_pct": float((pers_rmse - overall_rmse) / pers_rmse * 100.0) if pers_rmse > 0 else 0.0,
        "case_count": float(len(val_cases)),
        "row_count": float(len(df_eval)),
    }

    # Per-lead RMSE
    for lead in VALID_LEAD_HOURS:
        sub_df = df_eval[df_eval["lead"] == lead]
        sub_err = sub_df["pred"] - sub_df["true"]
        sub_pers_err = sub_df["pers"] - sub_df["true"]
        metrics[f"lead_{lead}_rmse"] = float(np.sqrt(np.mean(sub_err**2)))
        metrics[f"lead_{lead}_persistence_rmse"] = float(np.sqrt(np.mean(sub_pers_err**2)))

    # Per-station RMSE
    for st in sorted(df_eval["station"].unique()):
        sub_df = df_eval[df_eval["station"] == st]
        sub_err = sub_df["pred"] - sub_df["true"]
        sub_pers_err = sub_df["pers"] - sub_df["true"]
        st_clean = st.replace("-", "_")
        metrics[f"rmse_{st_clean}"] = float(np.sqrt(np.mean(sub_err**2)))
        metrics[f"persistence_rmse_{st_clean}"] = float(np.sqrt(np.mean(sub_pers_err**2)))

    return metrics
