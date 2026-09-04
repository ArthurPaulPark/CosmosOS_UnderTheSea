"""Historical Pseudo-Case Construction and Purging for Wave Forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


VALID_LEAD_HOURS = (3, 6, 9, 12, 18, 24)
CONTEXT_STEPS = 288  # 48 hours in 10-minute intervals
CONTEXT_ROWS = 289  # -2880 to 0 inclusive


@dataclass
class ForecastCase:
    """Represents a single 48h context forecasting case."""
    case_id: str
    station: str
    reference_time: pd.Timestamp
    context_df: pd.DataFrame  # 289 rows, step_minute from -2880 to 0
    targets: Dict[int, float]  # {3: hs_3h, ..., 24: hs_24h}
    latest_hs: float
    time_since_last_wave_obs: int  # in minutes


def align_historical_grid(
    df_wave: pd.DataFrame,
    df_atmos: pd.DataFrame,
    start_time: str = "2024-01-01 00:00:00+09:00",
    end_time: str = "2025-06-30 23:50:00+09:00",
) -> Dict[str, pd.DataFrame]:
    """Aligns wave and atmospheric observations onto a continuous 10-minute grid per station."""
    wave_df = df_wave.copy()
    atmos_df = df_atmos.copy()

    wave_df["time_dt"] = pd.to_datetime(wave_df["time"])
    atmos_df["time_dt"] = pd.to_datetime(atmos_df["time"])

    # Ensure timezone consistency
    if hasattr(wave_df["time_dt"].dt, "tz") and wave_df["time_dt"].dt.tz is not None:
        idx = pd.date_range(start_time, end_time, freq="10min")
    else:
        idx = pd.date_range(start_time, end_time, freq="10min").tz_localize(None)

    grids: Dict[str, pd.DataFrame] = {}
    stations = sorted(wave_df["station"].unique())

    for st in stations:
        w_st = wave_df[wave_df["station"] == st].set_index("time_dt")
        a_st = atmos_df[atmos_df["station"] == st].set_index("time_dt")

        df_grid = pd.DataFrame(index=idx)
        df_grid.index.name = "time_dt"

        # Join wave
        wave_cols = [c for c in ["hs", "tp", "hmax", "wvdir"] if c in w_st.columns]
        df_grid = df_grid.join(w_st[wave_cols], how="left")

        # Join atmos
        atmos_cols = [c for c in ["wspd", "gust", "wdir", "airt", "relh", "caph"] if c in a_st.columns]
        df_grid = df_grid.join(a_st[atmos_cols], how="left")

        df_grid["station"] = st
        grids[st] = df_grid

    return grids


class HistoricalForecastCaseBuilder:
    """Builds evaluation-like pseudo-cases from continuous historical observations."""

    def __init__(
        self,
        hs_threshold: float = 1.5,
        lead_hours: Sequence[int] = VALID_LEAD_HOURS,
        context_steps: int = CONTEXT_STEPS,
    ):
        self.hs_threshold = hs_threshold
        self.lead_hours = tuple(lead_hours)
        self.lead_steps = tuple(h * 6 for h in self.lead_hours)
        self.context_steps = context_steps

    def build_cases(
        self,
        grid_by_station: Dict[str, pd.DataFrame],
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        min_separation_hours: Optional[float] = 78.0,
        dense_anchor_stride_steps: int = 1,
    ) -> List[ForecastCase]:
        """Extracts valid cases from station grids.
        
        If min_separation_hours is provided (e.g. 78h), enforces strict non-overlapping separation.
        If min_separation_hours is None, uses dense_anchor_stride_steps.
        """
        all_cases: List[ForecastCase] = []

        for st, df_grid in grid_by_station.items():
            n_rows = len(df_grid)
            hs_arr = df_grid["hs"].to_numpy()
            times = df_grid.index

            eligible_indices: List[int] = []
            max_lead_step = max(self.lead_steps)

            for i in range(self.context_steps, n_rows - max_lead_step):
                t_ref = times[i]
                if start_time is not None and t_ref < start_time:
                    continue
                if end_time is not None and t_ref > end_time:
                    continue

                # Find latest finite hs in context (step_minute <= 0)
                ctx_hs = hs_arr[i - self.context_steps : i + 1]
                valid_ctx_hs = ctx_hs[~np.isnan(ctx_hs)]
                if len(valid_ctx_hs) == 0:
                    continue
                latest_hs = valid_ctx_hs[-1]

                # Check eligibility condition: latest_hs >= threshold
                if latest_hs < self.hs_threshold:
                    continue

                # Check all 6 target leads exist and are finite
                targets = [hs_arr[i + s] for s in self.lead_steps]
                if any(np.isnan(t) for t in targets):
                    continue

                eligible_indices.append(i)

            # Apply separation if requested
            selected_indices: List[int] = []
            if min_separation_hours is not None and min_separation_hours > 0:
                sep_steps = int(min_separation_hours * 6)
                last_i = -sep_steps - 1
                for idx in eligible_indices:
                    if idx - last_i >= sep_steps:
                        selected_indices.append(idx)
                        last_i = idx
            else:
                selected_indices = eligible_indices[::dense_anchor_stride_steps]

            # Construct ForecastCase objects
            for idx in selected_indices:
                t_ref = times[idx]
                slice_df = df_grid.iloc[idx - self.context_steps : idx + 1].copy()
                slice_df["step_minute"] = np.arange(-self.context_steps * 10, 10, 10)

                # Latest finite wave observation
                ctx_hs = slice_df["hs"].to_numpy()
                finite_idx = np.where(~np.isnan(ctx_hs))[0]
                latest_hs_val = float(ctx_hs[finite_idx[-1]])
                last_wave_step = int(slice_df["step_minute"].iloc[finite_idx[-1]])
                time_since_obs = -last_wave_step

                targets_dict = {
                    h: float(hs_arr[idx + s]) for h, s in zip(self.lead_hours, self.lead_steps)
                }

                case_id = f"H_{st.replace('-', '_')}_{t_ref.strftime('%Y%m%d%H%M')}"
                case = ForecastCase(
                    case_id=case_id,
                    station=st,
                    reference_time=t_ref,
                    context_df=slice_df,
                    targets=targets_dict,
                    latest_hs=latest_hs_val,
                    time_since_last_wave_obs=time_since_obs,
                )
                all_cases.append(case)

        return all_cases

    @staticmethod
    def purge_overlapping_train_cases(
        train_cases: List[ForecastCase],
        val_cases: List[ForecastCase],
        safety_buffer_hours: float = 6.0,
    ) -> List[ForecastCase]:
        """Purges any training case whose context or target window overlaps with a validation case."""
        purged_train: List[ForecastCase] = []

        # Build validation exclusion intervals per station
        val_intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for vc in val_cases:
            st = vc.station
            t_ref = vc.reference_time
            # Validation context start (-48h) to target end (+24h + buffer)
            start_excl = t_ref - pd.Timedelta(hours=48)
            end_excl = t_ref + pd.Timedelta(hours=24 + safety_buffer_hours)
            val_intervals.setdefault(st, []).append((start_excl, end_excl))

        for tc in train_cases:
            st = tc.station
            intervals = val_intervals.get(st, [])
            tc_start = tc.reference_time - pd.Timedelta(hours=48)
            tc_end = tc.reference_time + pd.Timedelta(hours=24)

            # Check overlap with any validation interval
            overlaps = False
            for v_start, v_end in intervals:
                if not (tc_end < v_start or tc_start > v_end):
                    overlaps = True
                    break

            if not overlaps:
                purged_train.append(tc)

        return purged_train
