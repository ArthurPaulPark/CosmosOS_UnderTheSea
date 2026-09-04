"""Ocean Significant Wave Height Multi-Horizon Forecasting Adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from ...base import BaseCompetitionAdapter
from ...registry import CompetitionRegistry
from .cases import (
    ForecastCase,
    HistoricalForecastCaseBuilder,
    VALID_LEAD_HOURS,
    align_historical_grid,
)
from .features import WaveForecastFeaturePlugin
from .schema import validate_forecasting_schema
from .submission import generate_forecast_submission
from .validation import (
    build_primary_forecast_validation,
    build_rolling_forecast_backtest,
    calculate_forecast_metrics,
)

LOGGER = logging.getLogger("cosmos_os.forecasting.adapter")


@CompetitionRegistry.register("ocean_forecast")
@CompetitionRegistry.register("ocean_p3")
class OceanForecastAdapter(BaseCompetitionAdapter):
    """Adapter for Problem 3 Significant Wave Height Multi-Horizon Forecasting."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_mode = config.get("model_mode", "per_horizon")  # unified, per_horizon, persistence_residual
        self.strategy = config.get("strategy", "direct")  # direct vs residual
        self.feature_plugin = WaveForecastFeaturePlugin()

        self.df_wave: Optional[pd.DataFrame] = None
        self.df_atmos: Optional[pd.DataFrame] = None
        self.df_test_context: Optional[pd.DataFrame] = None
        self.df_test_index: Optional[pd.DataFrame] = None
        self.df_sample_sub: Optional[pd.DataFrame] = None
        self.df_baseline_persistence: Optional[pd.DataFrame] = None

        self.grid_by_station: Dict[str, pd.DataFrame] = {}
        self.train_cases: List[ForecastCase] = []
        self.val_cases: List[ForecastCase] = []

        self.models_: Dict[Any, lgb.LGBMRegressor] = {}
        self.val_metrics: Dict[str, float] = {}
        self.predictions_map: Dict[Tuple[str, int], float] = {}
        self.submission_df: Optional[pd.DataFrame] = None

    def load_dataset(self) -> None:
        """Loads wave, atmos, test context, test index, sample sub, and baseline persistence."""
        ds_cfg = self.config.get("dataset", {})
        wave_p = ds_cfg.get("train_wave")
        atmos_p = ds_cfg.get("train_atmos")
        test_ctx_p = ds_cfg.get("test_context")
        test_idx_p = ds_cfg.get("test_index")
        sample_p = ds_cfg.get("sample_submission")
        base_pers_p = ds_cfg.get("baseline_persistence")

        if wave_p and Path(wave_p).exists():
            self.df_wave = pd.read_csv(wave_p)
        if atmos_p and Path(atmos_p).exists():
            self.df_atmos = pd.read_csv(atmos_p)
        if test_ctx_p and Path(test_ctx_p).exists():
            self.df_test_context = pd.read_parquet(test_ctx_p) if str(test_ctx_p).endswith(".parquet") else pd.read_csv(test_ctx_p)
        if test_idx_p and Path(test_idx_p).exists():
            self.df_test_index = pd.read_csv(test_idx_p)
        if sample_p and Path(sample_p).exists():
            self.df_sample_sub = pd.read_csv(sample_p)
        if base_pers_p and Path(base_pers_p).exists():
            self.df_baseline_persistence = pd.read_csv(base_pers_p)

    def validate_schema(self) -> None:
        """Validates all loaded datasets against Problem 3 specifications."""
        if self.df_wave is not None and self.df_atmos is not None:
            is_valid, errors = validate_forecasting_schema(
                self.df_wave,
                self.df_atmos,
                self.df_test_context,
                self.df_test_index,
                self.df_sample_sub,
            )
            if not is_valid:
                raise ValueError(f"Schema validation failed: {errors}")

    def run_eda(self) -> None:
        """Run exploratory data analysis for wave and atmospheric observations."""
        pass

    def prepare_features(self) -> None:
        """Fit feature plugin on training observations."""
        self.feature_plugin.fit()

    def verify_official_persistence_baseline(self) -> Dict[str, Any]:
        """Verifies that our local persistence logic reproduces the official baseline_persistence.csv 100%."""
        if self.df_test_context is None or self.df_baseline_persistence is None:
            return {"verified": True, "note": "test_context or baseline_persistence not provided"}

        test_cases_dict = {}
        for cid, grp in self.df_test_context.groupby("case_id"):
            hs_valid = grp["hs"].dropna()
            if len(hs_valid) > 0:
                latest_hs = float(hs_valid.iloc[-1])
            else:
                latest_hs = 1.5
            test_cases_dict[cid] = latest_hs

        df_base = self.df_baseline_persistence.copy()
        df_base["our_pers"] = df_base["case_id"].map(test_cases_dict)
        diff = (df_base["hs_pred"] - df_base["our_pers"]).abs()

        exact_match = int((diff == 0.0).sum())
        mismatches = int((diff > 0.0).sum())
        max_diff = float(diff.max())

        res = {
            "verified": mismatches == 0,
            "total_rows": len(df_base),
            "exact_match_rows": exact_match,
            "mismatch_rows": mismatches,
            "max_absolute_difference": max_diff,
        }

        if mismatches > 0:
            raise RuntimeError(f"Official persistence reproduction check failed! {mismatches} mismatches found (max diff: {max_diff}m)")

        return res

    def build_validation(self) -> None:
        """Constructs station grids and primary evaluation-like historical pseudo-cases."""
        if self.df_wave is None or self.df_atmos is None:
            return

        self.grid_by_station = align_historical_grid(self.df_wave, self.df_atmos)

        val_cfg = self.config.get("validation", {})
        v_type = val_cfg.get("type", "primary_holdout")

        if v_type == "primary_holdout":
            tr_start = val_cfg.get("train_start", "2024-01-01 00:00:00+09:00")
            tr_end = val_cfg.get("train_end", "2025-02-28 23:50:00+09:00")
            v_start = val_cfg.get("val_start", "2025-03-04 00:00:00+09:00")
            v_end = val_cfg.get("val_end", "2025-06-30 23:50:00+09:00")
            dense_stride = val_cfg.get("dense_stride_steps", 2)

            self.train_cases, self.val_cases = build_primary_forecast_validation(
                self.grid_by_station,
                train_start=tr_start,
                train_end=tr_end,
                val_start=v_start,
                val_end=v_end,
                dense_stride_steps=dense_stride,
            )

    def _wind_regime_weights(self, cases) -> np.ndarray | None:
        """Weight down training cases that carry no wind record.

        I-ORS and S-ORS only have atmospheric records from 2025-01 while their wave records
        start 2024-01, so 45% of training cases are built with no wind at all - yet 195 of
        the 200 scored cases have it at 97% coverage. Nearly half of training therefore
        sits in a regime that is never graded, and the wind-forcing features go blank
        there.

        Dropping those cases was measured first and lost (rolling 0.7558 -> 0.7685): they
        still carry wave dynamics, and removing them halves the training set. Weighting
        them down keeps the rows while shifting the effective distribution toward the
        graded regime, and wins all four seeds on both validations (primary 0.7359 ->
        0.7261, rolling scored over wind-bearing cases 0.7574 -> 0.7494).

        The 0.25 is empirical, not a matched rate: reproducing the scored set's 97% share
        exactly would need about 0.04, which is close to the dropping that already failed.
        """
        weight = float(self.config.get("model", {}).get("wind_blind_weight", 0.25))
        if not cases or not 0 < weight < 1:
            return None
        has_wind = np.array([
            "wspd" in c.context_df.columns and c.context_df["wspd"].notna().mean() > 0.5
            for c in cases
        ])
        if has_wind.all() or not has_wind.any():
            return None
        weights = np.where(has_wind, 1.0, weight)
        LOGGER.info("Wind-blind weight %.2f applied to %d of %d training cases",
                    weight, int((~has_wind).sum()), len(cases))
        return weights / weights.mean()

    def train(self) -> None:
        """Trains multi-horizon forecasting models according to selected model_mode."""
        if not self.train_cases:
            return

        m_cfg = self.config.get("model", {})
        # Neighbouring anchors overlap heavily across only ~10k cases, so an
        # unsubsampled fit memorises the training window. Subsampling costs a little on
        # the 54-case primary holdout and improves every rolling fold (0.8345 -> 0.8204).
        params = dict(m_cfg.get("params", {
            "learning_rate": 0.03, "num_leaves": 31, "random_state": 42,
            "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
        }))
        # n_estimators is passed separately as n_rounds; keeping it here would duplicate
        # the keyword when constructing LGBMRegressor.
        n_rounds = m_cfg.get("n_rounds", params.pop("n_estimators", 200))
        params.pop("n_estimators", None)

        df_tr_feats = self.feature_plugin.transform_cases(self.train_cases)
        feat_cols = self.feature_plugin.feature_names_
        case_weight = self._wind_regime_weights(self.train_cases)

        if self.model_mode == "unified":
            rows_x = []
            rows_y = []
            for idx, c in enumerate(self.train_cases):
                base_row = df_tr_feats.iloc[idx][feat_cols].to_dict()
                for lead in VALID_LEAD_HOURS:
                    r = dict(base_row)
                    r["lead_h"] = lead
                    rows_x.append(r)
                    y_val = c.targets[lead]
                    if self.strategy == "residual":
                        y_val -= c.latest_hs
                    rows_y.append(y_val)

            X_tr = pd.DataFrame(rows_x)
            y_tr = np.array(rows_y)
            clf = lgb.LGBMRegressor(**params, n_estimators=n_rounds, verbose=-1)
            # Each case became one row per lead, so its weight repeats the same way.
            clf.fit(X_tr, y_tr,
                    sample_weight=None if case_weight is None
                    else np.repeat(case_weight, len(VALID_LEAD_HOURS)))
            self.models_["unified"] = clf

        elif self.model_mode in ("per_horizon", "persistence_residual"):
            is_residual = (self.model_mode == "persistence_residual" or self.strategy == "residual")
            X_tr = df_tr_feats[feat_cols]

            for lead in VALID_LEAD_HOURS:
                y_tr_list = []
                for c in self.train_cases:
                    target_val = c.targets[lead]
                    if is_residual:
                        target_val -= c.latest_hs
                    y_tr_list.append(target_val)

                y_tr = np.array(y_tr_list)
                clf = lgb.LGBMRegressor(**params, n_estimators=n_rounds, verbose=-1)
                clf.fit(X_tr, y_tr, sample_weight=case_weight)
                self.models_[lead] = clf

        if self.val_cases:
            df_val_feats = self.feature_plugin.transform_cases(self.val_cases)
            preds_dict = {}

            if self.model_mode == "unified":
                clf = self.models_["unified"]
                for idx, c in enumerate(self.val_cases):
                    base_row = df_val_feats.iloc[idx][feat_cols].to_dict()
                    for lead in VALID_LEAD_HOURS:
                        r = dict(base_row)
                        r["lead_h"] = lead
                        pred_val = float(clf.predict(pd.DataFrame([r]))[0])
                        if self.strategy == "residual":
                            pred_val += c.latest_hs
                        preds_dict[(c.case_id, lead)] = max(0.0, pred_val)

            elif self.model_mode in ("per_horizon", "persistence_residual"):
                is_residual = (self.model_mode == "persistence_residual" or self.strategy == "residual")
                X_val = df_val_feats[feat_cols]
                for lead in VALID_LEAD_HOURS:
                    clf = self.models_[lead]
                    preds_arr = clf.predict(X_val)
                    for idx, c in enumerate(self.val_cases):
                        pred_val = float(preds_arr[idx])
                        if is_residual:
                            pred_val += c.latest_hs
                        preds_dict[(c.case_id, lead)] = max(0.0, pred_val)

            self.val_metrics = calculate_forecast_metrics(self.val_cases, preds_dict)

    def predict(self) -> None:
        """Infers multi-horizon predictions on official test_context.parquet."""
        if self.df_test_context is None or self.df_test_index is None:
            return

        test_cases_list = []
        for cid, grp in self.df_test_context.groupby("case_id", sort=False):
            st = str(grp["station"].iloc[0])
            test_cases_list.append((cid, st, grp))

        df_test_feats = self.feature_plugin.transform_cases(test_cases_list)
        feat_cols = self.feature_plugin.feature_names_

        preds_map = {}
        if self.model_mode == "unified":
            clf = self.models_["unified"]
            for idx, (cid, st, grp) in enumerate(test_cases_list):
                base_row = df_test_feats.iloc[idx][feat_cols].to_dict()
                hs_valid = grp["hs"].dropna()
                latest_hs = float(hs_valid.iloc[-1]) if len(hs_valid) > 0 else 1.5
                for lead in VALID_LEAD_HOURS:
                    r = dict(base_row)
                    r["lead_h"] = lead
                    pred_val = float(clf.predict(pd.DataFrame([r]))[0])
                    if self.strategy == "residual":
                        pred_val += latest_hs
                    preds_map[(cid, lead)] = max(0.0, pred_val)

        elif self.model_mode in ("per_horizon", "persistence_residual"):
            is_residual = (self.model_mode == "persistence_residual" or self.strategy == "residual")
            X_test = df_test_feats[feat_cols]

            latest_hs_map = {}
            for cid, grp in self.df_test_context.groupby("case_id", sort=False):
                hs_valid = grp["hs"].dropna()
                latest_hs_map[cid] = float(hs_valid.iloc[-1]) if len(hs_valid) > 0 else 1.5

            for lead in VALID_LEAD_HOURS:
                clf = self.models_[lead]
                preds_arr = clf.predict(X_test)
                for idx, (cid, st, grp) in enumerate(test_cases_list):
                    pred_val = float(preds_arr[idx])
                    if is_residual:
                        pred_val += latest_hs_map[cid]
                    preds_map[(cid, lead)] = max(0.0, pred_val)

        self.predictions_map = preds_map

    def create_submission(self) -> None:
        """Creates and validates the submission file."""
        if not self.predictions_map or self.df_test_index is None:
            return

        out_path = self.config.get("submission", {}).get("output")
        self.submission_df = generate_forecast_submission(
            self.df_test_index,
            self.predictions_map,
            output_path=out_path,
        )

    def run(self) -> None:
        """Executes full end-to-end forecasting pipeline."""
        self.load_dataset()
        self.validate_schema()
        self.verify_official_persistence_baseline()
        self.build_validation()
        self.train()
        self.predict()
        self.create_submission()
