"""Wave Forecast Feature Engineering Plugin (Strictly SAFE_AT_TEST)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


SAFE_WINDOWS_MINUTES = [60, 180, 360, 720, 1440, 2880]  # 1h, 3h, 6h, 12h, 24h, 48h


def extract_latest_finite(series: pd.Series) -> Tuple[float, int]:
    """Returns the latest finite value and the step_minute at which it occurred."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan, -9999
    val = float(valid.iloc[-1])
    step = int(valid.index[-1]) if hasattr(valid.index, "name") and valid.index.name == "step_minute" else 0
    return val, step


class WaveForecastFeaturePlugin:
    """Extracts strictly test-safe features from 48-hour context frames."""

    def __init__(self, windows_minutes: Sequence[int] = SAFE_WINDOWS_MINUTES):
        self.windows_minutes = list(windows_minutes)
        self.feature_names_: List[str] = []
        self._fitted = False

    def extract_case_features(self, context_df: pd.DataFrame, station: str) -> Dict[str, float]:
        """Extracts a flat feature dictionary from a single 289-row context frame."""
        df = context_df.sort_values("step_minute").set_index("step_minute")
        feats: Dict[str, float] = {}

        feats["is_G_ORS"] = 1.0 if station == "G-ORS" else 0.0
        feats["is_I_ORS"] = 1.0 if station == "I-ORS" else 0.0
        feats["is_S_ORS"] = 1.0 if station == "S-ORS" else 0.0

        hs_valid = df["hs"].dropna()
        if len(hs_valid) > 0:
            feats["latest_hs"] = float(hs_valid.iloc[-1])
            feats["last_wave_step"] = float(hs_valid.index[-1])
            feats["time_since_last_wave_obs"] = float(-hs_valid.index[-1])
            feats["has_wave_obs_at_step0"] = 1.0 if 0 in hs_valid.index else 0.0
        else:
            feats["latest_hs"] = 1.5  # safe fallback
            feats["last_wave_step"] = -9999.0
            feats["time_since_last_wave_obs"] = 9999.0
            feats["has_wave_obs_at_step0"] = 0.0

        tp_valid = df["tp"].dropna() if "tp" in df else pd.Series(dtype=float)
        feats["latest_tp"] = float(tp_valid.iloc[-1]) if len(tp_valid) > 0 else np.nan

        hmax_valid = df["hmax"].dropna() if "hmax" in df else pd.Series(dtype=float)
        feats["latest_hmax"] = float(hmax_valid.iloc[-1]) if len(hmax_valid) > 0 else np.nan

        wvdir_valid = df["wvdir"].dropna() if "wvdir" in df else pd.Series(dtype=float)
        latest_wvdir = float(wvdir_valid.iloc[-1]) if len(wvdir_valid) > 0 else np.nan
        feats["latest_wvdir"] = latest_wvdir

        for col in ["wspd", "gust", "wdir", "airt", "relh", "caph"]:
            if col in df.columns:
                c_valid = df[col].dropna()
                feats[f"latest_{col}"] = float(c_valid.iloc[-1]) if len(c_valid) > 0 else np.nan
            else:
                feats[f"latest_{col}"] = np.nan

        if not np.isnan(latest_wvdir):
            rad_wv = np.radians(latest_wvdir)
            feats["sin_wvdir"] = float(np.sin(rad_wv))
            feats["cos_wvdir"] = float(np.cos(rad_wv))
        else:
            feats["sin_wvdir"] = np.nan
            feats["cos_wvdir"] = np.nan

        latest_wdir = feats.get("latest_wdir", np.nan)
        if not np.isnan(latest_wdir):
            rad_w = np.radians(latest_wdir)
            feats["sin_wdir"] = float(np.sin(rad_w))
            feats["cos_wdir"] = float(np.cos(rad_w))
        else:
            feats["sin_wdir"] = np.nan
            feats["cos_wdir"] = np.nan

        if not np.isnan(latest_wvdir) and not np.isnan(latest_wdir):
            diff_ang = np.radians(latest_wdir - latest_wvdir)
            feats["sin_wind_wave_diff"] = float(np.sin(diff_ang))
            feats["cos_wind_wave_diff"] = float(np.cos(diff_ang))
        else:
            feats["sin_wind_wave_diff"] = np.nan
            feats["cos_wind_wave_diff"] = np.nan

        latest_hs = feats["latest_hs"]
        feats["hs_energy_proxy"] = float(latest_hs**2)
        feats["hs_steepness_proxy"] = (
            float(feats["latest_hmax"] / (latest_hs + 1e-4)) if not np.isnan(feats.get("latest_hmax", np.nan)) else np.nan
        )
        feats["hs_power_proxy"] = (
            float(latest_hs**2 * feats["latest_tp"]) if not np.isnan(feats.get("latest_tp", np.nan)) else np.nan
        )

        for win_min in self.windows_minutes:
            win_df = df.loc[-win_min:0]
            win_hs = win_df["hs"].dropna()
            label_h = win_min // 60

            if len(win_hs) > 0:
                feats[f"hs_mean_{label_h}h"] = float(win_hs.mean())
                feats[f"hs_std_{label_h}h"] = float(win_hs.std()) if len(win_hs) > 1 else 0.0
                feats[f"hs_min_{label_h}h"] = float(win_hs.min())
                feats[f"hs_max_{label_h}h"] = float(win_hs.max())
                feats[f"hs_range_{label_h}h"] = float(win_hs.max() - win_hs.min())
                feats[f"hs_diff_{label_h}h"] = float(latest_hs - win_hs.iloc[0])
                if len(win_hs) > 1:
                    x = win_hs.index.to_numpy() / 60.0  # in hours
                    y = win_hs.to_numpy()
                    slope = np.polyfit(x, y, 1)[0]
                    feats[f"hs_slope_{label_h}h"] = float(slope)
                else:
                    feats[f"hs_slope_{label_h}h"] = 0.0
            else:
                feats[f"hs_mean_{label_h}h"] = latest_hs
                feats[f"hs_std_{label_h}h"] = 0.0
                feats[f"hs_min_{label_h}h"] = latest_hs
                feats[f"hs_max_{label_h}h"] = latest_hs
                feats[f"hs_range_{label_h}h"] = 0.0
                feats[f"hs_diff_{label_h}h"] = 0.0
                feats[f"hs_slope_{label_h}h"] = 0.0

            # Atmos Window stats
            if "wspd" in win_df:
                win_wspd = win_df["wspd"].dropna()
                if len(win_wspd) > 0:
                    feats[f"wspd_mean_{label_h}h"] = float(win_wspd.mean())
                    feats[f"wspd_max_{label_h}h"] = float(win_wspd.max())
                    feats[f"wspd_std_{label_h}h"] = float(win_wspd.std()) if len(win_wspd) > 1 else 0.0
                else:
                    feats[f"wspd_mean_{label_h}h"] = np.nan
                    feats[f"wspd_max_{label_h}h"] = np.nan
                    feats[f"wspd_std_{label_h}h"] = np.nan

            if "gust" in win_df:
                win_gust = win_df["gust"].dropna()
                feats[f"gust_max_{label_h}h"] = float(win_gust.max()) if len(win_gust) > 0 else np.nan

            if "caph" in win_df:
                win_caph = win_df["caph"].dropna()
                if len(win_caph) > 1:
                    feats[f"caph_diff_{label_h}h"] = float(win_caph.iloc[-1] - win_caph.iloc[0])
                else:
                    feats[f"caph_diff_{label_h}h"] = 0.0

        # 6. Wind-forcing ratios
        # A fully developed sea sits at roughly hs ~ wspd^2, so whether the current sea is
        # below or above that equilibrium tells the model which way hs is about to move.
        # The trees have wspd and hs already but cannot form the quotient themselves, and
        # the same applies to "how big is now against the last day / two days".
        eps = 1e-6
        latest_wspd = feats.get("latest_wspd", np.nan)
        feats["wind_forcing_ratio"] = (
            float(latest_wspd ** 2 / (latest_hs + eps)) if not np.isnan(latest_wspd) else np.nan
        )
        gust_3h = feats.get("gust_max_3h", np.nan)
        feats["gust_forcing_ratio"] = (
            float(gust_3h ** 2 / (latest_hs + eps)) if not np.isnan(gust_3h) else np.nan
        )
        mean_24h = feats.get("hs_mean_24h", np.nan)
        feats["hs_over_mean_24h"] = (
            float(latest_hs / (mean_24h + eps)) if not np.isnan(mean_24h) else np.nan
        )
        max_48h = feats.get("hs_max_48h", np.nan)
        feats["hs_over_max_48h"] = (
            float(latest_hs / (max_48h + eps)) if not np.isnan(max_48h) else np.nan
        )

        # 7. Missingness & Observation Counts
        feats["wave_obs_count_6h"] = float(df.loc[-360:0, "hs"].count())
        feats["wave_obs_count_24h"] = float(df.loc[-1440:0, "hs"].count())
        feats["wave_obs_count_48h"] = float(df["hs"].count())
        feats["wave_missing_ratio_48h"] = float(1.0 - (df["hs"].count() / len(df)))

        # Duration above high-wave threshold
        feats["high_wave_count_1_5m"] = float((df["hs"].dropna() >= 1.5).sum())
        feats["high_wave_count_2_0m"] = float((df["hs"].dropna() >= 2.0).sum())

        return feats

    def fit(self, X: Any = None, y: Any = None) -> WaveForecastFeaturePlugin:
        self._fitted = True
        return self

    def transform_cases(self, cases: Sequence[Any]) -> pd.DataFrame:
        """Transforms a sequence of ForecastCase or test context cases into a feature DataFrame."""
        rows = []
        for c in cases:
            if hasattr(c, "context_df"):
                f = self.extract_case_features(c.context_df, c.station)
                f["case_id"] = c.case_id
                f["station"] = c.station
            elif isinstance(c, tuple):  # (case_id, station, context_df)
                cid, st, cdf = c
                f = self.extract_case_features(cdf, st)
                f["case_id"] = cid
                f["station"] = st
            rows.append(f)

        df_feats = pd.DataFrame(rows)
        if not self.feature_names_:
            exclude = ["case_id", "station"]
            self.feature_names_ = [col for col in df_feats.columns if col not in exclude]
        return df_feats
