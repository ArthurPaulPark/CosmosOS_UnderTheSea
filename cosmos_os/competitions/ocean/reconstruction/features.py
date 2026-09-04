"""Vertical Profile Feature Engineering for Ocean Temperature Reconstruction (Problem 2).

This module provides leakage-free physical oceanographic feature engineering
for reconstructing missing intermediate depth temperatures (e.g., layers 2, 3, 4)
given available surface (layer 1) and subsurface/deep (layers 5, 6, 7, 8) observations.

Supports:
- Legacy Presets: light, standard, full (100% backward compatible)
- Controlled Residual Presets: r1 (basic), r2 (vertical structure), r3 (temporal context), h1 (historical)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("cosmos_os.reconstruction.features")


def compute_residual_target(
    df: pd.DataFrame,
    temp_col: str = "temp",
    interp_col: str = "vertical_linear_interp",
) -> pd.Series:
    """Compute residual target: y = T_true - T_linear."""
    if temp_col not in df.columns or interp_col not in df.columns:
        raise KeyError(f"Columns {temp_col} and {interp_col} must be present in DataFrame.")
    return df[temp_col] - df[interp_col]


@dataclass
class FeatureSpec:
    """Feature specification container."""
    frame: pd.DataFrame
    categorical_cols: List[str] = field(default_factory=list)
    numerical_cols: List[str] = field(default_factory=list)
    target_col: str | None = None
    notes: List[str] = field(default_factory=list)


class VerticalProfileFeaturePlugin:
    """Feature engineering plugin for ocean temperature vertical profile reconstruction.
    
    Strict leakage-free architecture:
    - fit(train_df): Computes climatology lookup tables and boundary observation caches.
    - transform(df, boundary_context=None): Generates vertical gradient, layer boundary interpolation,
      temporal boundary lags/rolling stats, and merges train climatology.
    """

    leakage_unit: str = "group_time"

    VALID_PRESETS = ("light", "standard", "full", "r1", "r2", "r3", "h1")

    def __init__(
        self,
        time_col: str = "time",
        station_col: str = "station",
        layer_col: str = "layer",
        depth_col: str = "depth",
        temp_col: str = "temp",
        psal_col: str = "psal",
        preset: str = "standard",  # "light", "standard", "full", "r1", "r2", "r3", "h1"
        surface_layer: int = 1,
        bottom_layer: int = 5,
        boundary_gap_feature: bool = False,
    ) -> None:
        self.time_col = time_col
        self.station_col = station_col
        self.layer_col = layer_col
        self.depth_col = depth_col
        self.temp_col = temp_col
        self.psal_col = psal_col
        self.preset = preset.lower()
        self.surface_layer = surface_layer
        self.boundary_gap_feature = boundary_gap_feature
        self.bottom_layer = bottom_layer

        if self.preset not in self.VALID_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from {self.VALID_PRESETS}.")

        self._is_fitted = False
        self._climatology_station_layer_month: Dict[tuple, float] = {}
        self._climatology_layer_month: Dict[tuple, float] = {}
        self._climatology_layer: Dict[int, float] = {}
        self._global_mean_temp: float = 15.0
        self._nominal_depth_map: Dict[int, float] = {}
        self._boundary_cache: pd.DataFrame | None = None
        # Thermocline shape ratio per deep layer, learned in fit(). Empty means no
        # correction, i.e. a plain straight line.
        self._bottom_shape_ratio: Dict[int, float] = {}

    @property
    def name(self) -> str:
        return f"VerticalProfileFeaturePlugin(preset={self.preset})"

    def fit(self, train_df: pd.DataFrame) -> "VerticalProfileFeaturePlugin":
        """Compute climatology statistics strictly on train data."""
        df = train_df.copy()
        if self.time_col in df.columns:
            ts = pd.to_datetime(df[self.time_col])
            df["_month"] = ts.dt.month
        else:
            df["_month"] = 1

        valid_temps = df[self.temp_col].dropna() if self.temp_col in df.columns else pd.Series([], dtype=float)
        self._global_mean_temp = float(valid_temps.mean()) if len(valid_temps) > 0 else 15.0

        if self.depth_col in df.columns and self.layer_col in df.columns:
            for l, grp in df.groupby(self.layer_col):
                d_vals = grp[self.depth_col].dropna()
                if len(d_vals) > 0:
                    self._nominal_depth_map[int(l)] = float(d_vals.median())
        elif "nominal_depth" in df.columns and self.layer_col in df.columns:
            for l, grp in df.groupby(self.layer_col):
                d_vals = grp["nominal_depth"].dropna()
                if len(d_vals) > 0:
                    self._nominal_depth_map[int(l)] = float(d_vals.median())

        default_depths = {1: 4.18, 2: 6.87, 3: 9.83, 4: 14.87, 5: 19.15, 6: 30.21, 7: 49.05, 8: 49.35}
        for l, d in default_depths.items():
            if l not in self._nominal_depth_map:
                self._nominal_depth_map[l] = d

        for l in range(1, 15):
            if l not in self._nominal_depth_map:
                self._nominal_depth_map[l] = float(l * 10.0)

        # Climatology: (station, layer, month)
        if self.station_col in df.columns and self.layer_col in df.columns and self.temp_col in df.columns:
            group_slm = df.groupby([self.station_col, self.layer_col, "_month"])[self.temp_col].mean()
            self._climatology_station_layer_month = {k: float(v) for k, v in group_slm.items() if np.isfinite(v)}

            group_lm = df.groupby([self.layer_col, "_month"])[self.temp_col].mean()
            self._climatology_layer_month = {k: float(v) for k, v in group_lm.items() if np.isfinite(v)}

            group_l = df.groupby(self.layer_col)[self.temp_col].mean()
            self._climatology_layer = {int(k): float(v) for k, v in group_l.items() if np.isfinite(v)}

            self._boundary_cache = self._extract_boundary_observations(df)
            self._fit_bottom_shape_ratio(df)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, boundary_context: pd.DataFrame | None = None) -> pd.DataFrame:
        """Transform dataframe and attach vertical reconstruction features."""
        if not self._is_fitted:
            raise RuntimeError("VerticalProfileFeaturePlugin must be fit() on train data before transform().")

        out = df.copy()

        if self.time_col in out.columns:
            ts = pd.to_datetime(out[self.time_col])
            out["_month"] = ts.dt.month
            out["_day"] = ts.dt.day
            out["_hour"] = ts.dt.hour
            out["_doy"] = ts.dt.dayofyear
        else:
            out["_month"] = 1
            out["_day"] = 1
            out["_hour"] = 0
            out["_doy"] = 1

        if self.layer_col in out.columns:
            out["layer_num"] = pd.to_numeric(out[self.layer_col], errors="coerce").fillna(1).astype(int)
        else:
            out["layer_num"] = 1

        if self.depth_col in out.columns and not out[self.depth_col].isna().all():
            out["effective_depth"] = pd.to_numeric(out[self.depth_col], errors="coerce")
            out["effective_depth"] = out["effective_depth"].fillna(out["layer_num"].map(self._nominal_depth_map))
        elif "nominal_depth" in out.columns and not out["nominal_depth"].isna().all():
            out["effective_depth"] = pd.to_numeric(out["nominal_depth"], errors="coerce")
            out["effective_depth"] = out["effective_depth"].fillna(out["layer_num"].map(self._nominal_depth_map))
        else:
            out["effective_depth"] = out["layer_num"].map(self._nominal_depth_map).fillna(10.0)

        out["nominal_depth"] = out["layer_num"].map(self._nominal_depth_map).fillna(10.0)

        if boundary_context is not None:
            boundary_df = self._extract_boundary_observations(boundary_context)
        elif self.temp_col in out.columns and out[self.temp_col].notna().any():
            boundary_df = self._extract_boundary_observations(out)
        elif self._boundary_cache is not None:
            boundary_df = self._boundary_cache
        else:
            boundary_df = pd.DataFrame(columns=[self.time_col, self.station_col, "surf_temp", "bot_temp"])

        if not boundary_df.empty and self.time_col in out.columns and self.station_col in out.columns:
            out = out.merge(boundary_df, on=[self.time_col, self.station_col], how="left")

        surf_d = self._nominal_depth_map.get(self.surface_layer, 4.18)
        bot_d = self._nominal_depth_map.get(self.bottom_layer, 19.15)
        depth_span = max(bot_d - surf_d, 1.0)

        out["rel_depth_pos"] = (out["effective_depth"] - surf_d) / depth_span
        out["rel_depth_pos"] = out["rel_depth_pos"].clip(0.0, 1.0)

        t_surf = out.get("surf_temp", pd.Series(self._global_mean_temp, index=out.index)).fillna(self._global_mean_temp)
        t_bot_raw = out.get("bot_temp", pd.Series(np.nan, index=out.index))

        # Layer 5 is absent for 29.5% of the scoring window. A global mean there is not
        # a temperature at 19 m on that day, so rebuild the boundary from the next
        # observed layer down, correcting for thermocline curvature.
        t_bot, bot_source = self._reconstruct_bottom_boundary(out, t_surf, t_bot_raw, surf_d)

        out["surf_temp_val"] = t_surf
        out["bot_temp_val"] = t_bot
        out["bot_boundary_is_observed"] = bot_source
        out["vertical_temp_diff_span"] = t_surf - t_bot
        out["vertical_linear_interp"] = t_surf + out["rel_depth_pos"] * (t_bot - t_surf)
        out["vertical_mean_boundary"] = (t_surf + t_bot) / 2.0
        out["vertical_gradient"] = (t_surf - t_bot) / depth_span

        out["diff_to_surf"] = out["vertical_linear_interp"] - t_surf
        out["diff_to_bot"] = out["vertical_linear_interp"] - t_bot

        # 4. Climatology Anomaly (Train-fitted)
        clim_vals = []
        for idx, row in out.iterrows():
            st = row.get(self.station_col)
            ly = int(row.get("layer_num", 1))
            mo = int(row.get("_month", 1))
            c_val = self._climatology_station_layer_month.get(
                (st, ly, mo),
                self._climatology_layer_month.get((ly, mo), self._climatology_layer.get(ly, self._global_mean_temp))
            )
            clim_vals.append(c_val)
        out["climatology_temp"] = clim_vals
        out["climatology_diff_linear"] = out["vertical_linear_interp"] - out["climatology_temp"]

        # Temporal Harmonics (shared across standard, full, r1, r2, r3, h1)
        if self.preset in ("standard", "full", "r1", "r2", "r3", "h1"):
            out["sin_hour"] = np.sin(2 * np.pi * out["_hour"] / 24.0)
            out["cos_hour"] = np.cos(2 * np.pi * out["_hour"] / 24.0)
            out["sin_doy"] = np.sin(2 * np.pi * out["_doy"] / 365.25)
            out["cos_doy"] = np.cos(2 * np.pi * out["_doy"] / 365.25)

        # 5. Temporal Lags and Rolling on Boundary Observations
        if self.preset in ("standard", "full", "r3", "h1"):
            out = self._add_temporal_boundary_features(out)

        # 6. Physical proxies, Harmonics, Non-linear profile curvature
        if self.preset in ("standard", "full", "r2", "r3", "h1"):
            out["stratification_index"] = np.maximum(out["surf_temp_val"] - out["bot_temp_val"], 0.0)
            out["thermocline_intensity_proxy"] = np.abs(out["surf_temp_val"] - out["bot_temp_val"]) / depth_span
            out["is_thermocline_season"] = out["_month"].isin([6, 7, 8, 9, 10]).astype(float)

            out["mid_layer_curvature_weight"] = 4.0 * out["rel_depth_pos"] * (1.0 - out["rel_depth_pos"])
            out["curved_profile_estimate"] = out["vertical_linear_interp"] - (
                out["stratification_index"] * 0.3 * out["mid_layer_curvature_weight"]
            )

            if "deep_temp_6" in out.columns:
                out["deep_temp_6_val"] = out["deep_temp_6"].fillna(out["bot_temp_val"])
                out["gradient_surf_to_deep6"] = (out["surf_temp_val"] - out["deep_temp_6_val"]) / max(30.21 - surf_d, 1.0)

                # Layer 4 sits on the sharp gradient (4->5 runs five times 3->4) and
                # dominates the pooled error. Layers 5 and 6 measure that gradient
                # directly; its ratio against the shallow gradient gives the thermocline
                # position. A tree cannot form either quotient from the raw columns.
                eps = 1e-6
                shallow_grad = (out["surf_temp_val"] - out["bot_temp_val"]) / max(bot_d - surf_d, 1.0)
                deep_grad = (out["bot_temp_val"] - out["deep_temp_6_val"]) / max(30.21 - bot_d, 1.0)
                out["grad_deep_5_6"] = deep_grad
                out["grad_ratio_deep_shallow"] = deep_grad / (shallow_grad.abs() + eps)
                out["thermocline_sharpness"] = deep_grad - shallow_grad
                out["depth_rel_thermocline"] = (
                    out["effective_depth"] / max(bot_d, 1.0)
                    * deep_grad / (deep_grad.abs() + shallow_grad.abs() + eps)
                )
            if "deep_temp_8" in out.columns:
                out["deep_temp_8_val"] = out["deep_temp_8"].fillna(out["bot_temp_val"])
                out["gradient_surf_to_deep8"] = (out["surf_temp_val"] - out["deep_temp_8_val"]) / max(49.35 - surf_d, 1.0)

            if "surf_psal" in out.columns:
                out["surf_psal_val"] = out["surf_psal"].fillna(32.0)
            if "bot_psal" in out.columns:
                out["bot_psal_val"] = out["bot_psal"].fillna(32.2)

        if self.preset == "full":
            out["depth_temp_interaction"] = out["effective_depth"] * out["surf_temp_val"]
            out["climatology_ratio"] = (out["vertical_linear_interp"] + 1e-5) / (out["climatology_temp"] + 1e-5)

        drop_internal = ["_month", "_day", "_hour", "_doy"]
        out = out.drop(columns=[c for c in drop_internal if c in out.columns])
        return out



    def _fit_bottom_shape_ratio(self, df: pd.DataFrame) -> None:
        """Learn how far the lower boundary sits off a straight surface-to-deep line.

        Measured only on training rows where both the boundary and the deeper layer were
        observed, then reused to rebuild the boundary where it is missing. Storing a ratio
        rather than a temperature keeps it valid across seasons.
        """
        self._bottom_shape_ratio = {}
        need = {self.time_col, self.layer_col, self.temp_col}
        if not need.issubset(df.columns):
            return

        wide = df.pivot_table(index=self.time_col, columns=self.layer_col,
                              values=self.temp_col, aggfunc="first")
        surf_d = self._nominal_depth_map.get(self.surface_layer, 4.18)
        bot_d = self._nominal_depth_map.get(self.bottom_layer, 19.15)
        if self.surface_layer not in wide.columns or self.bottom_layer not in wide.columns:
            return

        t_surf, t_bot = wide[self.surface_layer], wide[self.bottom_layer]
        for deep_layer, deep_depth in ((6, 30.21), (7, 49.05), (8, 49.35)):
            if deep_layer not in wide.columns:
                continue
            t_deep = wide[deep_layer]
            span = t_surf - t_deep
            frac = (bot_d - surf_d) / max(deep_depth - surf_d, 1.0)
            straight = t_surf + frac * (t_deep - t_surf)
            ratio = (t_bot - straight) / span.where(span.abs() > 1e-6)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) >= 100:
                self._bottom_shape_ratio[deep_layer] = float(ratio.median())

        LOGGER.info("Bottom-boundary shape ratios (train-fitted): %s", self._bottom_shape_ratio)

    def _reconstruct_bottom_boundary(self, out, t_surf, t_bot_raw, surf_d):
        """Fill a missing lower boundary from the next observed layer down.

        Returns (temperature, is_observed_flag).

        Layer 5 (19.15 m) brackets the masked layers from below. When it is absent the
        deepest usable substitute is layer 6 (30.21 m), but that sits below the
        thermocline, so a straight surface-to-layer-6 line runs cold at 19 m. The gap
        between the true layer-5 value and that straight line, expressed as a fraction of
        the surface-to-layer-6 span, is stable enough to be learned on training rows where
        layer 5 was reporting and reapplied where it was not.
        """
        observed = t_bot_raw.notna()
        if observed.all():
            return t_bot_raw, observed.astype(np.int8)

        bot_d = self._nominal_depth_map.get(self.bottom_layer, 19.15)
        filled = t_bot_raw.copy()

        for deep_layer, deep_depth in ((6, 30.21), (7, 49.05), (8, 49.35)):
            col = f"deep_temp_{deep_layer}"
            if col not in out.columns or filled.notna().all():
                continue
            t_deep = out[col]
            usable = filled.isna() & t_deep.notna() & t_surf.notna()
            if not usable.any():
                continue

            frac = (bot_d - surf_d) / max(deep_depth - surf_d, 1.0)
            straight = t_surf + frac * (t_deep - t_surf)
            shape = self._bottom_shape_ratio.get(deep_layer, 0.0)
            filled.loc[usable] = (straight + shape * (t_surf - t_deep))[usable]

        # Nothing observed below the surface: fall back to the global mean, which now
        # covers a handful of rows rather than 29%.
        filled = filled.fillna(self._global_mean_temp)
        return filled, observed.astype(np.int8)

    def _extract_boundary_observations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract observed surface and bottom layers per (time, station)."""
        time_st_cols = [self.time_col, self.station_col]
        if not all(c in df.columns for c in time_st_cols) or self.temp_col not in df.columns:
            return pd.DataFrame(columns=time_st_cols + ["surf_temp", "bot_temp"])
        
        # Layer 1 surface
        l1 = df[df[self.layer_col] == self.surface_layer].dropna(subset=[self.temp_col])
        l1_cols = [self.time_col, self.station_col, self.temp_col]
        if self.psal_col in df.columns:
            l1_cols.append(self.psal_col)
        l1_agg = l1.groupby(time_st_cols)[[c for c in l1_cols if c in l1.columns and c not in time_st_cols]].first().reset_index()
        l1_agg.rename(columns={self.temp_col: "surf_temp", self.psal_col: "surf_psal"}, inplace=True)

        # Layer 5 bottom
        l5 = df[df[self.layer_col] == self.bottom_layer].dropna(subset=[self.temp_col])
        l5_cols = [self.time_col, self.station_col, self.temp_col]
        if self.psal_col in df.columns:
            l5_cols.append(self.psal_col)
        l5_agg = l5.groupby(time_st_cols)[[c for c in l5_cols if c in l5.columns and c not in time_st_cols]].first().reset_index()
        l5_agg.rename(columns={self.temp_col: "bot_temp", self.psal_col: "bot_psal"}, inplace=True)

        merged = pd.merge(l1_agg, l5_agg, on=time_st_cols, how="outer")

        # Layer 6 deep if available
        l6 = df[df[self.layer_col] == 6].dropna(subset=[self.temp_col])
        if len(l6) > 0:
            l6_agg = l6.groupby(time_st_cols)[self.temp_col].first().reset_index()
            l6_agg.rename(columns={self.temp_col: "deep_temp_6"}, inplace=True)
            merged = pd.merge(merged, l6_agg, on=time_st_cols, how="left")

        # Layer 8 deep if available
        l8 = df[df[self.layer_col] == 8].dropna(subset=[self.temp_col])
        if len(l8) > 0:
            l8_agg = l8.groupby(time_st_cols)[self.temp_col].first().reset_index()
            l8_agg.rename(columns={self.temp_col: "deep_temp_8"}, inplace=True)
            merged = pd.merge(merged, l8_agg, on=time_st_cols, how="left")

        return merged

    def _add_temporal_boundary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute lag and rolling stats strictly on observed boundary temperatures."""
        out = df.copy()
        if self.station_col not in out.columns or self.time_col not in out.columns:
            return out

        out = out.sort_values(by=[self.station_col, self.time_col]).reset_index(drop=True)

        out["surf_temp_lag1"] = out.groupby(self.station_col)["surf_temp_val"].shift(1).fillna(out["surf_temp_val"])
        out["surf_temp_lag2"] = out.groupby(self.station_col)["surf_temp_val"].shift(2).fillna(out["surf_temp_val"])
        out["surf_temp_lag6"] = out.groupby(self.station_col)["surf_temp_val"].shift(6).fillna(out["surf_temp_val"])

        out["bot_temp_lag1"] = out.groupby(self.station_col)["bot_temp_val"].shift(1).fillna(out["bot_temp_val"])
        out["bot_temp_lag2"] = out.groupby(self.station_col)["bot_temp_val"].shift(2).fillna(out["bot_temp_val"])
        out["bot_temp_lag6"] = out.groupby(self.station_col)["bot_temp_val"].shift(6).fillna(out["bot_temp_val"])

        out["surf_temp_roll_mean_6"] = (
            out.groupby(self.station_col)["surf_temp_val"]
            .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
            .fillna(out["surf_temp_val"])
        )
        out["surf_temp_roll_std_6"] = (
            out.groupby(self.station_col)["surf_temp_val"]
            .transform(lambda s: s.rolling(window=6, min_periods=1).std())
            .fillna(0.0)
        )
        out["bot_temp_roll_mean_6"] = (
            out.groupby(self.station_col)["bot_temp_val"]
            .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
            .fillna(out["bot_temp_val"])
        )
        out["bot_temp_roll_std_6"] = (
            out.groupby(self.station_col)["bot_temp_val"]
            .transform(lambda s: s.rolling(window=6, min_periods=1).std())
            .fillna(0.0)
        )

        if self.boundary_gap_feature and "bot_boundary_is_observed" in out.columns:
            out["bot_missing_run_length"] = self._boundary_missing_run_length(out)

        return out

    def _boundary_missing_run_length(self, out: pd.DataFrame) -> np.ndarray:
        """How many consecutive prior steps the lower boundary (layer 5) has been missing.

        The outage is not scattered rows - its longest run is 402 hours - so without
        this a fresh gap and a step deep inside a long outage look identical: both get
        the same reconstructed boundary from _reconstruct_bottom_boundary, whose shape
        ratio was fit on rows where the boundary was typically present. `out` carries one row per
        (station, layer, time), but boundary presence is a property of (station, time) alone
        - counting on the raw rows would multiply each step by however many layers were
        reported that timestamp, so the run is computed per unique (station, time) first and
        then broadcast back. `out` must already be sorted by (station, time).
        """
        key_cols = [self.station_col, self.time_col]
        keyed = (
            out[key_cols + ["bot_boundary_is_observed"]]
            .drop_duplicates(subset=key_cols)
            .sort_values(key_cols)
        )
        observed = keyed["bot_boundary_is_observed"] == 1
        new_station = keyed[self.station_col] != keyed[self.station_col].shift(1)
        reset = observed | new_station
        group_id = reset.cumsum()
        missing = ~observed
        keyed = keyed.assign(_run=missing.groupby(group_id).cumsum().astype(float))
        merged = out[key_cols].merge(keyed[key_cols + ["_run"]], on=key_cols, how="left")
        return merged["_run"].to_numpy()

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on train and transform."""
        return self.fit(train_df).transform(train_df)
