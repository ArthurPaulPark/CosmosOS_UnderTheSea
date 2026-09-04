from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from cosmos_os import prepare_features, train_lgbm
from cosmos_os.modalities.anomaly import AnomalyFeaturePlugin

from ..base import BaseCompetitionAdapter
from ..registry import CompetitionRegistry
from .eda import generate_ocean_eda
from .gru_auxiliary import blend_one_directional
from .schema import validate_ocean_schema
from .submission import generate_ocean_submission
from .validation import (
    add_injection_shape_features,
    add_long_baseline_residual,
    add_neighbour_gradient,
    add_targeted_anomaly_features,
    add_profile_context,
    pseudo_label_mask,
    add_station_layer,
    build_time_based_validation,
    calculate_competition_metrics,
)

LOGGER = logging.getLogger(__name__)

# Identifier and label columns are never model inputs.
DROP_COLS = ["id", "time", "station", "layer", "station_layer", "anomaly_type"]


@CompetitionRegistry.register("ocean")
class OceanCompetitionAdapter(BaseCompetitionAdapter):
    """Adapter for the Ocean AI Competition."""
    
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.feature_plugin: AnomalyFeaturePlugin | None = None
        self.validation_feature_plugin: AnomalyFeaturePlugin | None = None
        self.model_booster = None
        self.submission_model_booster = None
        # Retained so every preset's booster can be exported, not just the first.
        self.ensemble_models: dict[str, Any] = {}

    def load_dataset(self) -> None:
        """Load the required CSV files."""
        dataset_cfg = self.config.get("dataset", {})
        train_path = Path(dataset_cfg.get("train", "train.csv"))
        test_path = Path(dataset_cfg.get("test", "test.csv"))
        sub_path = Path(dataset_cfg.get("submission", "sample_submission.csv"))
        
        if not train_path.exists():
            raise FileNotFoundError(f"Train dataset not found at {train_path}")
            
        self.train_data = pd.read_csv(train_path)
        LOGGER.info("Train dataset loaded: %d rows", len(self.train_data))
        
        if test_path.exists():
            self.test_data = pd.read_csv(test_path)
            LOGGER.info("Test dataset loaded: %d rows", len(self.test_data))
            
        if sub_path.exists():
            self.sample_submission = pd.read_csv(sub_path)
            LOGGER.info("Sample submission loaded: %d rows", len(self.sample_submission))

    def validate_schema(self) -> None:
        """Validate the loaded datasets."""
        if self.train_data is not None:
            validate_ocean_schema(self.train_data, is_test=False)
        if self.test_data is not None:
            validate_ocean_schema(self.test_data, is_test=True)

    def run_eda(self) -> None:
        """Run EDA on the train dataset."""
        if self.train_data is not None:
            generate_ocean_eda(self.train_data, output_dir=self.config.get("experiment", {}).get("reports_dir", "reports"))

    def prepare_features(self) -> None:
        """Build validation features without fitting on the validation period."""
        feat_cfg = self.config.get("feature", {})
        # `full` is stronger on offset, `standard` on drift; averaging beats either alone.
        self.presets = list(dict.fromkeys(
            feat_cfg.get("presets") or [feat_cfg.get("preset", "standard"), "full"]
        ))
        preset = self.presets[0]
        if self.train_data is None:
            return

        train_data = self._context(self.train_data)
        test_size = self.config.get("validation", {}).get("test_size", 0.2)
        raw_train, raw_val = build_time_based_validation(train_data, test_size=test_size)

        # Built per frame after the split so each baseline uses only its own history.
        raw_train = self._shape(add_targeted_anomaly_features(add_long_baseline_residual(raw_train)))
        raw_val = self._shape(add_targeted_anomaly_features(add_long_baseline_residual(raw_val)))

        # temp_dev_layers carries the same lag/rolling/trend treatment as temp; a drift
        # appears as that inter-layer gap growing steadily.
        value_cols = ["temp", "temp_dev_layers"]
        self.value_cols = value_cols
        self.raw_train, self.raw_val = raw_train, raw_val

        self.validation_feature_plugin = AnomalyFeaturePlugin(
            time_col="time",
            value_cols=value_cols,
            group_col="station_layer",
            preset=preset,
        )
        LOGGER.info("Fitting validation features on raw train split using preset '%s'", preset)
        self.validation_feature_plugin.fit(raw_train)
        self.train_split = self.validation_feature_plugin.transform(raw_train)
        self.val_split = self.validation_feature_plugin.transform(raw_val)
        self.features = pd.concat([self.train_split, self.val_split], ignore_index=True)

        self.feature_plugin = AnomalyFeaturePlugin(
            time_col="time",
            value_cols=value_cols,
            group_col="station_layer",
            preset=preset,
        )
        self.raw_full_train = self._shape(
            add_targeted_anomaly_features(add_long_baseline_residual(train_data))
        )
        self.full_train_features = self.feature_plugin.fit_transform(self.raw_full_train)

        if self.test_data is not None:
            LOGGER.info("Generating test features with full-train statistics")
            self.raw_test = self._shape(
                add_targeted_anomaly_features(
                    add_long_baseline_residual(self._context(self.test_data))
                )
            )
            self.test_features = self.feature_plugin.transform(self.raw_test)

    def _context(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the same-timestamp profile context, and the neighbour gap if asked.

        Off unless configured, so a plain run still rebuilds the registered candidate
        byte-for-byte - run_all.sh checks that on every rebuild.
        """
        out = add_profile_context(add_station_layer(frame))
        if self.config.get("feature", {}).get("neighbours"):
            out = add_neighbour_gradient(out)
        return out

    def _shape(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the injection-shape features, if the config asks for them.

        Off unless configured, so a plain run still rebuilds the registered
        candidate byte-for-byte - run_all.sh checks that on every rebuild.
        """
        settings = self.config.get("feature", {}).get("injection_shape")
        if not settings:
            return frame
        windows = tuple(settings.get("windows", (144, 288))) if isinstance(settings, dict) else (144, 288)
        return add_injection_shape_features(frame, windows=windows)

    def _preset_probabilities(self, preset, fit_frame, predict_frame, store_key=None):
        """Fit one preset's plugin and model on fit_frame, return P(anomaly) for predict_frame."""
        plugin = AnomalyFeaturePlugin(
            time_col="time",
            value_cols=self.value_cols,
            group_col="station_layer",
            preset=preset,
        )
        plugin.fit(fit_frame)
        fit_feats = plugin.transform(fit_frame)
        pred_feats = plugin.transform(predict_frame)

        num_cols = [c for c in fit_feats.columns if c not in DROP_COLS and c != "label"]
        bundle = prepare_features(
            train_df=fit_feats,
            test_df=pred_feats,
            categorical_cols=[],
            numerical_cols=num_cols,
            target_col="label",
        )
        model = train_lgbm(
            bundle.x_train,
            bundle.y_train,
            labels=["0", "1"],
            params=self.config.get("model", {}).get("params", {}),
        )
        if store_key is not None:
            self.ensemble_models[store_key] = model
        probs = model.predict_proba(bundle.x_test)
        probabilities = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        return self._pseudo_round(bundle, probabilities, pred_feats)

    def _pseudo_round(self, bundle, probabilities, predict_frame):
        """Refit on the scored frame's own confident rows, then score it again.

        The scored file is delivered complete and unlabelled, so the first pass can
        label it and the model can be refit on distributed data alone. Only rows the
        first pass is already sure about are used; the contested band is left out.

        Rejected on the leaderboard (Phase 44); off unless configured.
        """
        settings = self.config.get("feature", {}).get("pseudo_labels")
        if not settings:
            return probabilities
        options = settings if isinstance(settings, dict) else {}
        positive, negative = pseudo_label_mask(
            probabilities,
            predict_frame,
            options.get("positive_floor", 0.98),
            options.get("negative_ceiling", 0.005),
            options.get("jump", 1.2),
        )
        keep = positive | negative
        LOGGER.info(
            "Pseudo-labels: %d positive, %d negative, %d rows left uncertain",
            int(positive.sum()),
            int(negative.sum()),
            int((~keep).sum()),
        )
        if not keep.any():
            LOGGER.warning("No rows passed the confidence filters; skipping the refit")
            return probabilities
        model = train_lgbm(
            pd.concat([bundle.x_train, bundle.x_test[keep]], ignore_index=True),
            pd.concat(
                [pd.Series(bundle.y_train), pd.Series(positive[keep].astype(int))],
                ignore_index=True,
            ),
            labels=["0", "1"],
            params=self.config.get("model", {}).get("params", {}),
        )
        probs = model.predict_proba(bundle.x_test)
        return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    def _blend_preset_probabilities(
        self, base: np.ndarray, extras: list[np.ndarray]
    ) -> np.ndarray:
        """Blend preset probabilities, defaulting to the existing equal weights."""
        if len(extras) + 1 != len(self.presets):
            raise ValueError("preset probability count does not match configured presets")
        configured = self.config.get("feature", {}).get("ensemble_weights")
        weights = np.ones(len(self.presets), dtype=float) if configured is None else np.asarray(
            configured, dtype=float
        )
        if (
            len(weights) != len(self.presets)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("feature.ensemble_weights must align with presets and be non-negative")
        weights /= weights.sum()
        return sum(weight * probabilities for weight, probabilities in zip(weights, [base, *extras]))

    def _gru_auxiliary_settings(self) -> tuple[float, int, str]:
        settings = self.config.get("submission", {}).get("gru_auxiliary") or {}
        return (
            float(settings.get("weight", 0.0)),
            int(settings.get("seed", 0)),
            str(settings.get("device", "cpu")),
        )

    def _apply_gru_auxiliary(
        self, y_prob: np.ndarray, raw_train: pd.DataFrame, raw_target: pd.DataFrame
    ) -> np.ndarray:
        """Blend in a GRU trained on drift and offset, if the config asks for it.

        Off unless configured, so the registered candidate stays reproducible.

        The GRU runs in a subprocess: importing torch into this process, which has
        already loaded LightGBM, deadlocks on their OpenMP barrier here.
        """
        weight, seed, device = self._gru_auxiliary_settings()
        if weight <= 0:
            return y_prob

        columns = ["station_layer", "time", "temp", "temp_dev_layers"]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            raw_train[[*columns, "label", "anomaly_type"]].reset_index(drop=True).to_parquet(
                work / "train.parquet"
            )
            raw_target[columns].reset_index(drop=True).to_parquet(work / "target.parquet")
            LOGGER.info("Training the GRU auxiliary in a subprocess (weight %.2f)", weight)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cosmos_os.competitions.ocean.gru_auxiliary",
                    "--train", str(work / "train.parquet"),
                    "--target", str(work / "target.parquet"),
                    "--output", str(work / "scores.npy"),
                    "--seed", str(seed),
                    "--device", device,
                ],
                cwd=Path(__file__).resolve().parents[3],
                check=True,
            )
            scores = np.load(work / "scores.npy")
        return blend_one_directional(y_prob, scores, weight)

    def build_validation(self) -> None:
        """Validate that the raw temporal split was prepared before training."""
        if getattr(self, "train_split", None) is None or getattr(self, "val_split", None) is None:
            raise ValueError("Features must be prepared before validation split.")

    def train(self) -> None:
        """Train LightGBM model and record the experiment."""
        if getattr(self, "train_split", None) is None or getattr(self, "val_split", None) is None:
            raise ValueError("Validation split must be built before training.")
            
        target_col = "label"
        drop_cols = ["id", "time", "station", "layer", "station_layer", "anomaly_type"]
        
        num_cols = [c for c in self.train_split.columns if c not in drop_cols and c != target_col]
        bundle = prepare_features(
            train_df=self.train_split,
            test_df=self.val_split,
            categorical_cols=[],
            numerical_cols=num_cols,
            target_col=target_col
        )
        
        X_train, y_train = bundle.x_train, bundle.y_train
        X_val = bundle.x_test
        y_val = self.val_split[target_col]
        
        preset = self.config.get("feature", {}).get("preset", "standard")
        model_name = self.config.get("model", {}).get("name", "LightGBM")
        
        exp_record = self.experiment_manager.start(
            task_type="classification",
            dataset_name="ocean_anomaly",
            model=model_name,
            feature_preset=preset,
            feature_names=list(X_train.columns),
            train_rows=len(X_train),
            validation_rows=len(X_val),
            config=self.config.get("model", {}),
            tags=["ocean_competition"]
        )
        
        try:
            self.model_booster = train_lgbm(
                X_train, 
                y_train,
                labels=["0", "1"],
                params=self.config.get("model", {}).get("params", {})
            )
            
            probs = self.model_booster.predict_proba(X_val)
            y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

            # Equal weights: every ratio tried beat both single models, and the
            # differences between ratios were within noise.
            extra_probabilities = []
            for extra in self.presets[1:]:
                LOGGER.info("Adding ensemble preset '%s' to validation probabilities", extra)
                extra_probabilities.append(
                    self._preset_probabilities(extra, self.raw_train, self.raw_val)
                )
            y_prob = self._blend_preset_probabilities(y_prob, extra_probabilities)
            y_prob = self._apply_gru_auxiliary(y_prob, self.raw_train, self.raw_val)

            # The anomaly class is ~4% of rows, so a 0.5 cut is far too conservative.
            # The threshold is set to reproduce the observed anomaly rate rather than to
            # maximise F1 on this split: the argmax cut is fitted to one split's noise,
            # while the rate is a quantity the problem brief states.
            anomaly_rate = float(np.mean(np.asarray(y_val) == 1))
            self.best_threshold = float(np.quantile(y_prob, 1.0 - anomaly_rate))
            self.anomaly_rate = anomaly_rate
            preds = (y_prob >= self.best_threshold).astype(int)

            competition_metrics = calculate_competition_metrics(y_val, preds, y_prob)
            competition_metrics["decision_threshold"] = self.best_threshold
            score = competition_metrics["anomaly_f1"]
            self.metrics = {**competition_metrics, "score": score}
            self.validation_probabilities = y_prob
            self.validation_predictions = preds

            self.experiment_manager.finish(
                metric_name="anomaly_f1",
                score=score,
                metrics=competition_metrics,
            )
            LOGGER.info("Training completed. Validation anomaly F1: %.4f", score)
        except Exception as e:
            LOGGER.error("Training failed: %s", e)
            raise

    def predict(self) -> None:
        """Predict on the test set."""
        if (
            getattr(self, "full_train_features", None) is None
            or getattr(self, "test_features", None) is None
            or self.model_booster is None
        ):
            LOGGER.warning("Skipping prediction: test features or model not available.")
            return

        drop_cols = ["id", "time", "station", "layer", "station_layer", "anomaly_type"]
        num_cols = [
            c for c in self.full_train_features.columns
            if c not in drop_cols and c != "label"
        ]
        bundle = prepare_features(
            train_df=self.full_train_features,
            test_df=self.test_features,
            categorical_cols=[],
            numerical_cols=num_cols,
            target_col="label",
        )
        if bundle.y_train is None or bundle.x_test is None:
            raise ValueError("Full-train labels and test features are required for prediction.")

        self.submission_model_booster = train_lgbm(
            bundle.x_train,
            bundle.y_train,
            labels=["0", "1"],
            params=self.config.get("model", {}).get("params", {}),
        )

        probs = self.submission_model_booster.predict_proba(bundle.x_test)
        y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        y_prob = self._pseudo_round(bundle, y_prob, self.test_features)

        extra_probabilities = []
        for extra in self.presets[1:]:
            LOGGER.info("Adding ensemble preset '%s' to test probabilities", extra)
            extra_probabilities.append(
                self._preset_probabilities(
                    extra,
                    self.raw_full_train,
                    self.raw_test,
                    store_key=f"submission_{extra}",
                )
            )
        y_prob = self._blend_preset_probabilities(y_prob, extra_probabilities)
        y_prob = self._apply_gru_auxiliary(y_prob, self.raw_full_train, self.raw_test)

        # The submission writer aligns predictions with sample_submission by position,
        # which holds only while the feature transform preserves row order. Assert it
        # rather than let a reordering pass silently.
        self._assert_test_row_order_preserved()

        # Saved so a different cut can be inspected without retraining. Write-only.
        settings = self.config.get("submission", {})
        output_path = Path(settings.get("output", "submission.csv"))
        scores_path = output_path.with_suffix(".probabilities.npy")
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(scores_path, y_prob)
        LOGGER.info("Saved test probabilities to %s", scores_path)

        # An explicit rate wins; otherwise reuse the anomaly share measured on the
        # validation split. The absolute probability cut does not carry across periods,
        # the rate does.
        rate = settings.get("rate") or getattr(self, "anomaly_rate", None)
        if rate:
            threshold = float(np.quantile(y_prob, 1.0 - rate))
            LOGGER.info("Test threshold %.6f matches the %.3f%% anomaly rate", threshold, rate * 100)
        else:
            threshold = getattr(self, "best_threshold", 0.5)
        self.predictions = (y_prob >= threshold).astype(int)
        LOGGER.info(
            "Predictions generated for test data: %d positives", int(self.predictions.sum())
        )


    def _assert_test_row_order_preserved(self) -> None:
        """Verify test_features still lines up row-for-row with the raw test data.

        Compared on the identifying keys with timestamps normalised, because the
        transform rewrites the time column's string format without moving any row.
        """
        if self.test_data is None or getattr(self, "test_features", None) is None:
            return
        if len(self.test_data) != len(self.test_features):
            raise ValueError(
                f"Test feature row count changed: {len(self.test_data)} -> {len(self.test_features)}. "
                "Predictions are written positionally, so this would misalign the submission."
            )

        keys = [c for c in ("station", "year", "layer") if c in self.test_data.columns
                and c in self.test_features.columns]
        left = self.test_data[keys].reset_index(drop=True)
        right = self.test_features[keys].reset_index(drop=True)
        if "time" in self.test_data.columns and "time" in self.test_features.columns:
            left = left.assign(_t=pd.to_datetime(self.test_data["time"]).to_numpy())
            right = right.assign(_t=pd.to_datetime(self.test_features["time"]).to_numpy())
        if not left.equals(right):
            raise ValueError(
                "Feature transform reordered the test rows. Predictions are assigned to "
                "sample_submission by position, so the submission would be scrambled. "
                "Join on the key instead of relying on row order."
            )

    def create_submission(self) -> None:
        """Generate the final submission.csv."""
        if self.sample_submission is None or getattr(self, "predictions", None) is None:
            LOGGER.warning("Skipping submission: sample_submission or predictions not available.")
            return
            
        out_path = self.config.get("submission", {}).get("output", "submission.csv")
        generate_ocean_submission(self.sample_submission, self.predictions, output_path=out_path)
