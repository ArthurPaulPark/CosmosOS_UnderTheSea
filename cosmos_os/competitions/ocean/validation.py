from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


def add_station_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the physical station-layer series identifier."""
    out = df.copy()
    out["station_layer"] = out["station"].astype(str) + "_" + out["layer"].astype(str)
    return out


def add_profile_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add features relating a layer to the rest of its profile at the same timestamp.

    Offset and drift move a whole window together, so statistics computed within one
    series cannot see them - the rolling baseline drifts along with the anomaly. The
    sibling layers do not move, so the gap to the column exposes both types.

    Stateless and same-timestamp only, so it is safe to apply before the temporal split.
    """
    out = df.copy()
    profile = out.groupby(["station", "time"])["temp"]
    # Median over the column including self: with 5+ layers one bad reading barely
    # moves it, and it stays defined when only a single layer reported.
    out["temp_layer_median"] = profile.transform("median")
    out["temp_layer_spread"] = profile.transform(lambda s: s.max() - s.min())
    out["temp_dev_layers"] = out["temp"] - out["temp_layer_median"]
    out["temp_layer_count"] = profile.transform("count")
    return out


add_cross_layer_deviation = add_profile_context


def add_injection_shape_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (144, 288),
    value_cols: Sequence[str] = ("temp", "temp_dev_layers"),
    group_col: str = "station_layer",
) -> pd.DataFrame:
    """Describe how well a straight line explains a centred window, not just its slope.

    The plugin supplies `slope_12/48/288` - how fast a window moves - but nothing for
    how well a line fits it, which is what separates an injected ramp from drift-like
    weather. Windows are centred rather than trailing so they see both sides of a
    segment; the scored file is delivered complete, so this reads only its own values.

    Attaches `<col>_ctr_span_<w>`, `_ctr_r2_<w>` and `_ctr_flat_<w>` in input row order.
    Rejected on the leaderboard (Phase 37); kept off by default.
    """
    if missing := {"time", group_col}.difference(df.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if any(window < 2 for window in windows):
        raise ValueError("windows must be at least 2 samples wide")

    out = df.sort_values([group_col, "time"])
    for column in value_cols:
        if column not in out.columns:
            continue
        for window in windows:
            spans, fits, flats = [], [], []
            for _, series in out.groupby(group_col, observed=True)[column]:
                span, fit, flat = _centred_line_fit(series, window)
                spans.append(span)
                fits.append(fit)
                flats.append(flat)
            out[f"{column}_ctr_span_{window}"] = pd.concat(spans)
            out[f"{column}_ctr_r2_{window}"] = pd.concat(fits)
            out[f"{column}_ctr_flat_{window}"] = pd.concat(flats)
    # reindex, not sort_index: sorting by index value restores the caller's order only
    # when the index happens to be an ordered range.
    return out.reindex(df.index)


def _centred_line_fit(values: pd.Series, window: int):
    """Least squares over a centred window, as rolling moments rather than a loop.

    Returns how far the fitted line travels across the window (a ramp's amplitude),
    how much of the window it explains, and the window's own spread.
    """
    index = pd.Series(np.arange(len(values)), index=values.index, dtype=float)
    roll = {"window": window, "center": True, "min_periods": max(8, window // 3)}
    mean_x = index.rolling(**roll).mean()
    mean_y = values.rolling(**roll).mean()
    covariance = (index * values).rolling(**roll).mean() - mean_x * mean_y
    variance_x = (index * index).rolling(**roll).mean() - mean_x * mean_x
    variance_y = values.rolling(**roll).var()
    slope = covariance / variance_x.replace(0, np.nan)
    fit = ((covariance ** 2) / (variance_x * variance_y).replace(0, np.nan)).clip(0, 1)
    return (slope * window).abs(), fit, variance_y.pow(0.5)


def add_targeted_anomaly_features(
    df: pd.DataFrame, short_window: int = 18, long_window: int = 1008
) -> pd.DataFrame:
    """Two features for the types the generic statistics miss.

    `vol_ratio` divides recent step size by the series' own week-long typical step. A
    tree cannot divide two rolling standard deviations, so "unusually volatile *for this
    series*" has to be supplied as one number. `temp_resid_long` is the raw-series
    counterpart of `dev_resid_long`. Both are causal and stateless.
    """
    out = df.sort_values(["station_layer", "time"])
    groups = out["station_layer"]

    step = out.groupby(groups, observed=True)["temp"].diff().abs()
    short = step.groupby(groups).transform(
        lambda s: s.rolling(short_window, min_periods=6).median()
    )
    long = step.groupby(groups).transform(
        lambda s: s.rolling(long_window, min_periods=48).median()
    )
    out["vol_ratio"] = short / (long + 1e-6)

    baseline = out.groupby(groups, observed=True)["temp"].transform(
        lambda s: s.rolling(long_window, min_periods=48).median()
    )
    out["temp_resid_long"] = out["temp"] - baseline
    return out.sort_index()


def add_long_baseline_residual(
    df: pd.DataFrame, window: int = 1008, min_periods: int = 48
) -> pd.DataFrame:
    """Cross-layer deviation against its own week-long causal baseline.

    The plugin's longest window is 288 steps (48 h) but an offset runs up to 86.5 h, so
    it sits inside that window and drags the baseline with it. A 7-day trailing median
    leaves even the longest offset a minority of the window, so a residual survives.

    Widening the plugin's own `windows` instead was worse: it adds ~40 features at once
    and the dilution cost more than the signal.
    """
    out = df.sort_values(["station_layer", "time"])
    baseline = out.groupby("station_layer", observed=True)["temp_dev_layers"].transform(
        lambda s: s.rolling(window, min_periods=min_periods).median()
    )
    out["dev_resid_long"] = out["temp_dev_layers"] - baseline
    return out.sort_index()


def fit_group_scale(
    df: pd.DataFrame, cols: Sequence[str], group_col: str = "station_layer"
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Learn a robust centre and spread per series, from training rows only.

    The plugin's own z-scores pool the whole frame, so one global spread has to serve a
    warm surface layer and a cold 49 m layer alike. A deviation that is routine for one
    series then looks identical to one that is genuinely anomalous for another. Median
    and MAD per station_layer give each series its own yardstick, which is what the
    competition notes recommend over a single shared threshold.
    """
    stats: dict[str, tuple[pd.Series, pd.Series]] = {}
    for col in cols:
        median = df.groupby(group_col)[col].median()
        deviation = (df[col] - df[group_col].map(median)).abs()
        mad = deviation.groupby(df[group_col]).median().clip(lower=1e-6)
        stats[col] = (median, mad)
    return stats


def apply_group_scale(
    df: pd.DataFrame,
    stats: dict[str, tuple[pd.Series, pd.Series]],
    group_col: str = "station_layer",
) -> pd.DataFrame:
    """Attach `<col>_gz`, each column expressed in its own series' robust units."""
    out = df.copy()
    for col, (median, mad) in stats.items():
        # Series unseen in train map to NaN, which LightGBM handles natively.
        out[f"{col}_gz"] = (out[col] - out[group_col].map(median)) / out[group_col].map(mad)
    return out


def calculate_competition_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
) -> dict[str, float]:
    """Calculate competition metrics with label 1 as the anomaly class."""
    metrics = {
        "anomaly_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "anomaly_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "anomaly_recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_prob is not None and np.unique(np.asarray(y_true)).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics


def build_time_based_validation(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset temporally; a random split is not allowed for this competition."""
    if "time" not in df.columns:
        raise ValueError("Cannot perform time-based split without 'time' column.")
        
    df_sorted = df.sort_values("time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train = df_sorted.iloc[:split_idx].copy()
    val = df_sorted.iloc[split_idx:].copy()
    
    LOGGER.info("Time-based validation split: Train=%d, Val=%d", len(train), len(val))
    return train, val


def add_neighbour_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Temperature gap to the layer directly above and directly below.

    `temp_dev_layers` measures a layer against its whole column's median; the gap to the
    immediate neighbour is a local quantity between a fixed pair of sensors, so a warmer
    year moves both ends together. NaN where there is no neighbour (layer 1, the deepest
    layer, and single-layer G-ORS), which LightGBM splits on natively.

    Rejected on the leaderboard (Phase 43) - it is missing from 63% of test rows against
    39% of train. Kept off by default. Stateless and same-timestamp only.
    """
    if missing := {"station", "time", "layer", "temp"}.difference(df.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    column = df.pivot_table(index=["station", "time"], columns="layer", values="temp")
    out = df.copy()
    # shift along the layer axis: column order is sorted, so shift(1) is the layer above.
    for name, step in (("temp_grad_upper", 1), ("temp_grad_lower", -1)):
        gap = (column - column.shift(step, axis=1)).stack().rename(name).reset_index()
        out = out.merge(gap, on=["station", "time", "layer"], how="left")
    return out.set_index(df.index)


def pseudo_label_mask(
    probabilities: np.ndarray,
    frame: pd.DataFrame,
    positive_floor: float,
    negative_ceiling: float,
    jump: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a scored frame into confidently-positive and confidently-negative rows.

    A positive must be both improbable and visibly discontinuous.
    """
    if not 0 <= negative_ceiling < positive_floor <= 1:
        raise ValueError("need 0 <= negative_ceiling < positive_floor <= 1")
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(frame):
        raise ValueError("probabilities and frame must align")
    step = (
        frame.sort_values(["station_layer", "time"])
        .groupby("station_layer", observed=True)["temp"]
        .diff()
        .abs()
        .reindex(frame.index)
        .to_numpy()
    )
    positive = (probabilities > positive_floor) & (step >= jump)
    negative = probabilities < negative_ceiling
    return positive, negative


ANOMALY_CLASSES = ("normal", "flatline", "noise", "drift", "offset", "spike")


def multiclass_target(frame: pd.DataFrame) -> np.ndarray:
    """Map each row to its anomaly type as a class index, normal being 0.

    Compound labels ("noise+drift") take their first component; unrecognised strings
    fall to normal rather than silently opening a class. The anomaly score is then
    1 - P(normal).

    Rejected: six classes leave four to ten events per class in a window, and the gate
    fails worst on the smallest windows (Phase 45). Kept off by default.
    """
    if not {"label", "anomaly_type"}.issubset(frame.columns):
        raise ValueError("multiclass target needs label and anomaly_type")
    index = {name: position for position, name in enumerate(ANOMALY_CLASSES)}
    primary = frame["anomaly_type"].fillna("").astype(str).str.split("+").str[0].str.strip()
    target = primary.map(index).fillna(0).to_numpy(dtype=int)
    # A row labelled normal stays normal even if it carries a stray type string.
    return np.where(frame["label"].to_numpy() == 1, target, 0)
