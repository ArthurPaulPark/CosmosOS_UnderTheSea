"""시계열 이상 탐지용 피처 자동 생성 플러그인.

범용 시계열 이상 탐지에 필요한 8가지 피처 카테고리를 Generator 아키텍처로 생성한다.
특정 대회 컬럼명을 하드코딩하지 않으며, ``value_cols`` / ``group_col`` 기반으로
IoT 센서·제조·금융·서버 로그·의료 센서 등 모든 시계열에 적용 가능하다.

설계 원칙:
    - **Preset** 시스템으로 Feature Explosion 방지 (light / standard / full)
    - **Generator 모듈화**: Plugin 은 Orchestrator 역할만 수행
    - **Leakage Prevention**: shift(1) + groupby + fit/transform 분리
    - **Slope 최적화**: numpy 벡터화 (LinearRegression 미사용)
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .base import FeatureSpec, ModalityPlugin

logger = logging.getLogger("cosmos_os.anomaly")


# ---------------------------------------------------------------------------
# Preset 정의
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PresetConfig:
    """Preset 이 활성화하는 Generator 와 세부 옵션."""

    enable_difference: bool
    enable_rolling: bool
    rolling_stats: tuple[str, ...]
    enable_trend: bool
    enable_robust: bool
    enable_seasonality: bool
    enable_runlength: bool
    enable_offset: bool
    enable_noise: bool


PRESETS: Dict[str, _PresetConfig] = {
    "light": _PresetConfig(
        enable_difference=True,
        enable_rolling=True,
        rolling_stats=("mean", "std", "min", "max"),
        enable_trend=False,
        enable_robust=False,
        enable_seasonality=True,
        enable_runlength=True,
        enable_offset=False,
        enable_noise=False,
    ),
    "standard": _PresetConfig(
        enable_difference=True,
        enable_rolling=True,
        rolling_stats=("mean", "std", "min", "max", "median", "mad"),
        enable_trend=True,
        enable_robust=True,
        enable_seasonality=True,
        enable_runlength=True,
        enable_offset=True,
        enable_noise=False,
    ),
    "full": _PresetConfig(
        enable_difference=True,
        enable_rolling=True,
        rolling_stats=(
            "mean", "median", "std", "var", "min", "max",
            "range", "q25", "q75", "mad",
        ),
        enable_trend=True,
        enable_robust=True,
        enable_seasonality=True,
        enable_runlength=True,
        enable_offset=True,
        enable_noise=True,
    ),
}


# ---------------------------------------------------------------------------
# Generator 기본 클래스
# ---------------------------------------------------------------------------

class _BaseGenerator(ABC):
    """피처 생성기 기본 클래스."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        """df 에 피처를 추가하고, 추가된 컬럼명 목록을 반환한다."""


class _FittableGenerator(_BaseGenerator, ABC):
    """fit/transform 분리가 필요한 Generator 기본 클래스."""

    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> None:
        """train 에서 통계를 계산·저장한다."""

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        """저장된 통계를 사용해 피처를 생성한다."""

    def generate(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        return self.transform(df, value_cols, group_col, **kwargs)


# ---------------------------------------------------------------------------
# 개별 Generator 구현
# ---------------------------------------------------------------------------

class _DifferenceGenerator(_BaseGenerator):
    """Lag, diff, gradient, 2차 미분 피처."""

    @property
    def name(self) -> str:
        return "difference"

    def __init__(self, lags: Sequence[int] = (1, 2, 3, 6, 12)) -> None:
        self.lags = list(lags)

    def generate(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        out = df
        new_cols: List[str] = []

        for col in value_cols:
            base = (
                out.groupby(group_col, observed=True)[col]
                if group_col
                else out[col]
            )

            # Lag 피처
            for lag in self.lags:
                cname = f"{col}_lag{lag}"
                out[cname] = base.shift(lag)
                new_cols.append(cname)

            # Diff 피처 (shift(1) 기반 — 현재 값 차단)
            shifted = base.shift(1)
            shifted_2 = base.shift(2)

            cname = f"{col}_diff1"
            out[cname] = shifted - shifted_2
            new_cols.append(cname)

            cname = f"{col}_diff2"
            out[cname] = shifted - base.shift(3)
            new_cols.append(cname)

            cname = f"{col}_diff6"
            out[cname] = shifted - base.shift(7)
            new_cols.append(cname)

            # Gradient (1차 기울기 — diff1 그 자체)
            cname = f"{col}_gradient"
            out[cname] = out[f"{col}_diff1"]
            new_cols.append(cname)

            cname = f"{col}_abs_gradient"
            out[cname] = out[f"{col}_gradient"].abs()
            new_cols.append(cname)

            # 2차 미분
            cname = f"{col}_second_deriv"
            out[cname] = out[f"{col}_diff1"] - (shifted_2 - base.shift(3))
            new_cols.append(cname)

        return out, new_cols


class _RollingGenerator(_BaseGenerator):
    """Rolling 통계 피처 (mean, median, std, var, min, max, range, q25, q75, mad)."""

    @property
    def name(self) -> str:
        return "rolling"

    def __init__(
        self,
        windows: Sequence[int] = (12, 48, 288),
        stats: Sequence[str] = ("mean", "std", "min", "max"),
    ) -> None:
        self.windows = list(windows)
        self.stats = list(stats)

    def generate(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        out = df
        new_cols: List[str] = []

        for col in value_cols:
            shifted = (
                out.groupby(group_col, observed=True)[col].shift(1)
                if group_col
                else out[col].shift(1)
            )
            for w in self.windows:
                roll = shifted.rolling(w, min_periods=1)
                stat_map: Dict[str, pd.Series] = {}

                if "mean" in self.stats:
                    stat_map[f"{col}_roll_mean_{w}"] = roll.mean()
                if "median" in self.stats:
                    stat_map[f"{col}_roll_median_{w}"] = roll.median()
                if "std" in self.stats:
                    stat_map[f"{col}_roll_std_{w}"] = roll.std()
                if "var" in self.stats:
                    stat_map[f"{col}_roll_var_{w}"] = roll.var()
                if "min" in self.stats:
                    stat_map[f"{col}_roll_min_{w}"] = roll.min()
                if "max" in self.stats:
                    stat_map[f"{col}_roll_max_{w}"] = roll.max()
                if "range" in self.stats:
                    r_min = roll.min()
                    r_max = roll.max()
                    stat_map[f"{col}_roll_range_{w}"] = r_max - r_min
                if "q25" in self.stats:
                    stat_map[f"{col}_roll_q25_{w}"] = roll.quantile(0.25)
                if "q75" in self.stats:
                    stat_map[f"{col}_roll_q75_{w}"] = roll.quantile(0.75)
                if "mad" in self.stats:
                    med = roll.median()
                    # MAD = median(|x - median(x)|)
                    # 근사: rolling std * 0.6745 대신 정확한 계산
                    stat_map[f"{col}_roll_mad_{w}"] = (
                        (shifted - med).abs().rolling(w, min_periods=1).median()
                    )

                for cname, series in stat_map.items():
                    out[cname] = series
                    new_cols.append(cname)

        return out, new_cols


class _TrendGenerator(_BaseGenerator):
    """EMA, slope, rolling mean 잔차 피처.

    Slope 계산은 numpy 벡터화 방식을 사용한다 (LinearRegression 미사용).
    """

    @property
    def name(self) -> str:
        return "trend"

    def __init__(self, windows: Sequence[int] = (12, 48, 288)) -> None:
        self.windows = list(windows)

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """numpy 벡터화 rolling linear slope.

        slope = (n·Σ(x·y) - Σx·Σy) / (n·Σ(x²) - (Σx)²)
        x = 0, 1, ..., n-1 이므로 x 관련 항은 상수로 사전 계산한다.
        """
        n = window
        x = np.arange(n, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        denom = n * sum_x2 - sum_x ** 2

        def _slope(vals: np.ndarray) -> float:
            if len(vals) < n:
                return np.nan
            sum_y = vals.sum()
            sum_xy = (x[: len(vals)] * vals).sum()
            return (n * sum_xy - sum_x * sum_y) / denom

        return series.rolling(window, min_periods=window).apply(
            _slope, raw=True
        )

    def generate(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        out = df
        new_cols: List[str] = []

        for col in value_cols:
            shifted = (
                out.groupby(group_col, observed=True)[col].shift(1)
                if group_col
                else out[col].shift(1)
            )
            for w in self.windows:
                # EMA
                cname = f"{col}_ema_{w}"
                out[cname] = shifted.ewm(span=w, min_periods=1).mean()
                new_cols.append(cname)

                # EMA diff (현재 shifted 값 - EMA)
                cname_diff = f"{col}_ema_diff_{w}"
                out[cname_diff] = shifted - out[cname]
                new_cols.append(cname_diff)

                # Slope (벡터화)
                cname_slope = f"{col}_slope_{w}"
                if group_col:
                    out[cname_slope] = (
                        out.groupby(group_col, observed=True)[col]
                        .shift(1)
                        .groupby(out[group_col], observed=True)
                        .transform(lambda s: self._rolling_slope(s, w))
                    )
                else:
                    out[cname_slope] = self._rolling_slope(shifted, w)
                new_cols.append(cname_slope)

                # Rolling mean 잔차
                roll_mean = shifted.rolling(w, min_periods=1).mean()
                cname_resid = f"{col}_resid_rollmean_{w}"
                out[cname_resid] = shifted - roll_mean
                new_cols.append(cname_resid)

        return out, new_cols


class _RobustStatisticsGenerator(_FittableGenerator):
    """Z-score, robust z-score, IQR flag. train 통계를 fit 에서 저장한다."""

    @property
    def name(self) -> str:
        return "robust_statistics"

    def __init__(self) -> None:
        self._stats: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> None:
        self._stats.clear()
        for col in value_cols:
            s = train[col].dropna()
            median = float(s.median())
            mad = float((s - median).abs().median())
            self._stats[col] = {
                "mean": float(s.mean()),
                "std": max(float(s.std()), 1e-10),
                "median": median,
                "mad": max(mad, 1e-10),
                "q1": float(s.quantile(0.25)),
                "q3": float(s.quantile(0.75)),
            }

    def transform(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        if not self._stats:
            raise RuntimeError(
                "RobustStatisticsGenerator.fit(train)을 먼저 호출하라"
            )
        out = df
        new_cols: List[str] = []

        for col in value_cols:
            st = self._stats[col]

            cname = f"{col}_zscore"
            out[cname] = (out[col] - st["mean"]) / st["std"]
            new_cols.append(cname)

            cname = f"{col}_robust_zscore"
            out[cname] = (out[col] - st["median"]) / st["mad"]
            new_cols.append(cname)

            cname = f"{col}_iqr_flag"
            iqr = st["q3"] - st["q1"]
            out[cname] = (
                (out[col] < st["q1"] - 1.5 * iqr)
                | (out[col] > st["q3"] + 1.5 * iqr)
            ).astype(np.int8)
            new_cols.append(cname)

        return out, new_cols


class _SeasonalityGenerator(_BaseGenerator):
    """시간 요소 + sin/cos 순환 인코딩 피처."""

    @property
    def name(self) -> str:
        return "seasonality"

    def generate(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        *,
        time_col: str = "",
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        out = df
        new_cols: List[str] = []
        ts = out[time_col]

        parts: Dict[str, tuple[pd.Series, float]] = {
            "tp_month": (ts.dt.month.astype(float), 12.0),
            "tp_dayofyear": (ts.dt.dayofyear.astype(float), 365.0),
            "tp_hour": (ts.dt.hour.astype(float), 24.0),
            "tp_weekday": (ts.dt.dayofweek.astype(float), 7.0),
        }

        for cname, (series, period) in parts.items():
            if series.nunique() > 1:
                out[cname] = series
                new_cols.append(cname)

                sin_name = f"{cname}_sin"
                cos_name = f"{cname}_cos"
                out[sin_name] = np.sin(2 * np.pi * series / period)
                out[cos_name] = np.cos(2 * np.pi * series / period)
                new_cols.extend([sin_name, cos_name])

        return out, new_cols


class _RunLengthGenerator(_BaseGenerator):
    """동일 값 연속 길이(run-length) + window 내 고유값 수."""

    @property
    def name(self) -> str:
        return "runlength"

    def __init__(self, windows: Sequence[int] = (12, 48, 288)) -> None:
        self.windows = list(windows)

    @staticmethod
    def _compute_runlength(series: pd.Series) -> pd.Series:
        """동일 값이 연속되는 길이를 계산한다."""
        diff_mask = series.ne(series.shift(1))
        groups = diff_mask.cumsum()
        return groups.groupby(groups).cumcount() + 1

    def generate(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        out = df
        new_cols: List[str] = []

        for col in value_cols:
            # Run-length
            cname = f"{col}_runlength"
            if group_col:
                out[cname] = out.groupby(group_col, observed=True)[col].transform(
                    self._compute_runlength
                )
            else:
                out[cname] = self._compute_runlength(out[col])
            new_cols.append(cname)

            # Unique count in window
            for w in self.windows:
                cname_uniq = f"{col}_uniq_count_{w}"
                shifted = (
                    out.groupby(group_col, observed=True)[col].shift(1)
                    if group_col
                    else out[col].shift(1)
                )
                out[cname_uniq] = (
                    shifted.rolling(w, min_periods=1)
                    .apply(lambda x: len(np.unique(x[~np.isnan(x)])), raw=True)
                )
                new_cols.append(cname_uniq)

        return out, new_cols


class _OffsetGenerator(_FittableGenerator):
    """그룹 평균 대비 편차. fit 에서 train 그룹 통계를 저장한다.

    ``group_cols`` 는 외부에서 주입받는다 — station/layer 같은 대회 전용 이름을
    하드코딩하지 않는다.
    """

    @property
    def name(self) -> str:
        return "offset"

    def __init__(self, windows: Sequence[int] = (12, 48, 288)) -> None:
        self.windows = list(windows)
        self._group_means: Dict[str, Dict[str, float | Dict]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        *,
        offset_group_cols: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._group_means.clear()
        cols = list(offset_group_cols or [])
        if group_col and group_col not in cols:
            cols.append(group_col)

        for col in value_cols:
            self._group_means[col] = {
                "global_mean": float(train[col].mean()),
            }
            for gcol in cols:
                if gcol in train.columns:
                    gm = train.groupby(gcol, observed=True)[col].mean().to_dict()
                    self._group_means[col][f"group_{gcol}"] = gm

    def transform(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        *,
        offset_group_cols: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        if not self._group_means:
            raise RuntimeError("OffsetGenerator.fit(train)을 먼저 호출하라")
        out = df
        new_cols: List[str] = []

        cols = list(offset_group_cols or [])
        if group_col and group_col not in cols:
            cols.append(group_col)

        for col in value_cols:
            stats = self._group_means[col]

            for gcol in cols:
                key = f"group_{gcol}"
                if key in stats and gcol in out.columns:
                    cname = f"{col}_dev_{gcol}_mean"
                    mapped = out[gcol].map(stats[key])
                    out[cname] = out[col] - mapped.fillna(stats["global_mean"])
                    new_cols.append(cname)

            # Rolling mean 대비 편차
            shifted = (
                out.groupby(group_col, observed=True)[col].shift(1)
                if group_col
                else out[col].shift(1)
            )
            for w in self.windows:
                roll_mean = shifted.rolling(w, min_periods=1).mean()
                cname = f"{col}_dev_rollmean_{w}"
                out[cname] = out[col] - roll_mean
                new_cols.append(cname)

        return out, new_cols


class _NoiseGenerator(_BaseGenerator):
    """Coefficient of Variation 피처."""

    @property
    def name(self) -> str:
        return "noise"

    def __init__(self, windows: Sequence[int] = (12, 48, 288)) -> None:
        self.windows = list(windows)

    def generate(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_col: str | None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, List[str]]:
        out = df
        new_cols: List[str] = []

        for col in value_cols:
            shifted = (
                out.groupby(group_col, observed=True)[col].shift(1)
                if group_col
                else out[col].shift(1)
            )
            for w in self.windows:
                roll = shifted.rolling(w, min_periods=1)
                mean = roll.mean()
                std = roll.std()
                cname = f"{col}_cv_{w}"
                out[cname] = std / mean.replace(0, np.nan).abs()
                new_cols.append(cname)

        return out, new_cols


# ---------------------------------------------------------------------------
# AnomalyFeaturePlugin — Orchestrator
# ---------------------------------------------------------------------------

class AnomalyFeaturePlugin(ModalityPlugin):
    """시계열 이상 탐지용 피처 자동 생성 플러그인.

    Generator 아키텍처로 8가지 피처 카테고리를 조합한다.
    Preset 시스템(light / standard / full)으로 Feature Explosion 을 제어한다.
    특정 대회 컬럼명을 하드코딩하지 않으며, ``value_cols`` / ``group_col`` 기반으로
    IoT·제조·금융·서버·의료 등 모든 시계열에 적용 가능하다.

    **누수 방지**:
        - 모든 lag/rolling 은 shift(1) 이후 계산 — 현재 시점 값 차단
        - group_col 지정 시 그룹별 독립 계산
        - Robust Statistics / Offset 의 기준값은 fit() 에서만 계산

    Args:
        time_col: 타임스탬프 컬럼명.
        value_cols: 피처를 생성할 수치 컬럼명 목록.
        group_col: 패널 데이터의 그룹 컬럼 (station·기기·환자 등).
        preset: ``"light"`` / ``"standard"`` (기본) / ``"full"``.
        windows: Rolling window 크기 목록. 기본 ``(12, 48, 288)``.
        lags: Lag 시점 목록. 기본 ``(1, 2, 3, 6, 12)``.
        offset_group_cols: Offset 편차를 계산할 추가 그룹 컬럼 목록.
            예) ``["station", "layer"]``. 지정하지 않으면 group_col 만 사용.

    Example::

        plugin = AnomalyFeaturePlugin(
            time_col="time",
            value_cols=["temp"],
            group_col="sensor_id",
            preset="standard",
        )
        train_df = plugin.fit_transform(train)
        test_df = plugin.transform(test)
    """

    leakage_unit = "time"

    def __init__(
        self,
        time_col: str,
        value_cols: Sequence[str],
        group_col: str | None = None,
        preset: str = "standard",
        windows: Sequence[int] = (12, 48, 288),
        lags: Sequence[int] = (1, 2, 3, 6, 12),
        offset_group_cols: Sequence[str] | None = None,
    ) -> None:
        if preset not in PRESETS:
            raise ValueError(
                f"preset은 {list(PRESETS)} 중 하나여야 한다: {preset!r}"
            )
        self.time_col = time_col
        self.value_cols = list(value_cols)
        self.group_col = group_col
        self.preset = preset
        self.windows = list(windows)
        self.lags = list(lags)
        self.offset_group_cols = list(offset_group_cols or [])
        self._fitted = False
        self._feature_names: List[str] = []

        if group_col:
            self.leakage_unit = "group_time"

        # Preset 에 따라 Generator 조합
        cfg = PRESETS[preset]
        self._generators: List[_BaseGenerator] = []
        self._fittable_generators: List[_FittableGenerator] = []

        if cfg.enable_difference:
            self._generators.append(_DifferenceGenerator(lags=self.lags))
        if cfg.enable_rolling:
            self._generators.append(
                _RollingGenerator(windows=self.windows, stats=list(cfg.rolling_stats))
            )
        if cfg.enable_trend:
            self._generators.append(_TrendGenerator(windows=self.windows))
        if cfg.enable_robust:
            gen = _RobustStatisticsGenerator()
            self._generators.append(gen)
            self._fittable_generators.append(gen)
        if cfg.enable_seasonality:
            self._generators.append(_SeasonalityGenerator())
        if cfg.enable_runlength:
            self._generators.append(_RunLengthGenerator(windows=self.windows))
        if cfg.enable_offset:
            gen_o = _OffsetGenerator(windows=self.windows)
            self._generators.append(gen_o)
            self._fittable_generators.append(gen_o)
        if cfg.enable_noise:
            self._generators.append(_NoiseGenerator(windows=self.windows))

    @property
    def name(self) -> str:
        return "anomaly"

    def fit(self, train: pd.DataFrame) -> "AnomalyFeaturePlugin":
        """train 에서 Robust Statistics / Offset 기준값을 계산·저장한다."""
        started = time.perf_counter()
        df = self._prepare(train)

        for gen in self._fittable_generators:
            gen.fit(
                df,
                self.value_cols,
                self.group_col,
                offset_group_cols=self.offset_group_cols,
            )

        self._fitted = True
        elapsed = time.perf_counter() - started
        logger.info(
            "AnomalyFeaturePlugin.fit: %d컬럼, %.2f초",
            len(self.value_cols),
            elapsed,
        )
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """저장된 통계를 사용해 피처를 생성한다.

        fit() 없이 호출하면 RuntimeError.
        """
        if not self._fitted:
            raise RuntimeError(
                "AnomalyFeaturePlugin.fit(train)을 먼저 호출하라"
            )
        return self._build(data)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """fit + transform 을 한 번에 수행한다."""
        return self.fit(train).transform(train)

    def build_features(
        self, data: pd.DataFrame, fit: bool = False, **kwargs: Any
    ) -> FeatureSpec:
        """ModalityPlugin 인터페이스 구현."""
        frame = self.fit_transform(data) if fit else self.transform(data)
        existing_num = [
            c
            for c in data.columns
            if c not in {self.time_col, self.group_col}
            and pd.api.types.is_numeric_dtype(data[c])
            and not pd.api.types.is_bool_dtype(data[c])
        ]
        existing_cat = [
            c
            for c in data.columns
            if c not in {self.time_col, self.group_col}
            and (
                pd.api.types.is_bool_dtype(data[c])
                or not pd.api.types.is_numeric_dtype(data[c])
            )
        ]
        return FeatureSpec(
            frame=frame,
            categorical_cols=[c for c in existing_cat if not c.startswith("_")],
            numerical_cols=sorted(set(existing_num + self._feature_names)),
            notes=[
                f"preset={self.preset!r}, windows={self.windows}",
                f"{len(self._feature_names)}개 이상 탐지 피처 생성",
                "모든 lag/rolling은 shift(1) 이후 계산 — 자기 누수 차단",
            ],
        )

    def candidate_models(self) -> List[str]:
        return ["lgbm"]

    def caveats(self) -> List[str]:
        return [
            "무작위/그룹 분할 금지 — 반드시 시간 절단(temporal_split)을 쓸 것.",
            "lag/rolling 피처는 shift(1) 이후 계산해야 현재 값이 새지 않는다.",
            "패널 데이터면 group_col을 반드시 지정 — 다른 개체의 과거가 섞이면 안 된다.",
            "Robust Statistics와 Offset의 기준값은 fit(train) 통계만 사용한다.",
        ]

    @property
    def feature_names(self) -> List[str]:
        """마지막 transform 에서 생성된 피처 이름 목록."""
        return list(self._feature_names)

    # -- internal --------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """타임스탬프 파싱 + 정렬."""
        out = df.copy()
        out[self.time_col] = pd.to_datetime(
            out[self.time_col], errors="coerce", format="mixed"
        )
        sort_keys = (
            [self.group_col, self.time_col] if self.group_col else [self.time_col]
        )
        return out.sort_values(sort_keys).reset_index(drop=True)

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generator 들을 순차 실행해 피처를 생성한다."""
        started = time.perf_counter()
        out = self._prepare(data)
        all_new: List[str] = []

        for gen in self._generators:
            kwargs: Dict[str, Any] = {}
            if isinstance(gen, _SeasonalityGenerator):
                kwargs["time_col"] = self.time_col
            if isinstance(gen, _OffsetGenerator):
                kwargs["offset_group_cols"] = self.offset_group_cols
            out, new_cols = gen.generate(
                out, self.value_cols, self.group_col, **kwargs
            )
            all_new.extend(new_cols)

        self._feature_names = all_new
        elapsed = time.perf_counter() - started
        logger.info(
            "AnomalyFeaturePlugin.transform: %d행 × %d피처, %.2f초",
            len(out),
            len(all_new),
            elapsed,
        )
        logger.debug("생성된 피처: %s", all_new)
        return out


__all__ = [
    "AnomalyFeaturePlugin",
    "PRESETS",
]
