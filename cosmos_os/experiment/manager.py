from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .compare import compare_experiments
from .models import ExperimentMetric, ExperimentRecord, ExperimentSearchQuery, ExperimentSummary, utc_now_iso
from .storage import ExperimentStorage
from .summary import write_experiment_report

LOGGER = logging.getLogger(__name__)

# Metric names containing any of these substrings are error metrics where a
# lower score is better (e.g. rmse, mae, logloss). Everything else is treated
# as a higher-is-better score (e.g. F1, accuracy, auc).
LOWER_IS_BETTER_KEYWORDS = ("rmse", "mse", "mae", "mape", "error", "loss")


def _is_lower_better(metric_name: str | None) -> bool:
    """Return True when a smaller value of this metric means a better model."""
    name = (metric_name or "").lower()
    return any(keyword in name for keyword in LOWER_IS_BETTER_KEYWORDS)


class ExperimentManager:
    """High-level manager for recording and tracking ML experiments.
    
    This class handles the lifecycle of experiments, including tracking the currently 
    active experiment, saving metrics upon completion, and providing search/comparison APIs.
    """
    
    def __init__(self, root_dir: str | Path = "experiments", reports_dir: str | Path = "reports") -> None:
        """Initialize the ExperimentManager.
        
        Args:
            root_dir (str | Path): Root directory for saving experiments. Defaults to "experiments".
            reports_dir (str | Path): Directory for generating markdown reports. Defaults to "reports".
        """
        self.storage = ExperimentStorage(root=root_dir)
        self.reports_dir = Path(reports_dir)
        self._active: ExperimentRecord | None = None

    def start(
        self,
        *,
        task_type: str,
        dataset_name: str,
        model: str,
        random_seed: int | None = None,
        feature_preset: str | None = None,
        feature_names: list[str] | None = None,
        train_rows: int = 0,
        validation_rows: int = 0,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        user_note: str | None = None,
        git_commit: str | None = None,
    ) -> ExperimentRecord:
        """Start a new experiment and track its metadata.
        
        Args:
            task_type (str): The type of machine learning task (e.g., 'classification', 'regression').
            dataset_name (str): Name of the dataset being used.
            model (str): Name of the model being trained (e.g., 'LightGBM').
            random_seed (int | None): Random seed for reproducibility.
            feature_preset (str | None): Preset name used for feature engineering.
            feature_names (list[str] | None): List of generated feature names.
            train_rows (int): Number of rows in the training set.
            validation_rows (int): Number of rows in the validation set.
            config (dict[str, Any] | None): Dictionary of hyperparameters or configurations.
            tags (list[str] | None): List of tags for categorizing the experiment.
            user_note (str | None): Optional note or description.
            git_commit (str | None): Git commit hash if tracking codebase version.
            
        Returns:
            ExperimentRecord: The created experiment record.
            
        Raises:
            RuntimeError: If there is already an active experiment running.
        """
        if self._active is not None:
            raise RuntimeError("이미 활성화된 실험이 있습니다. 먼저 finish()를 호출하세요.")
            
        experiment_id = self._next_experiment_id()
        if self.storage.metadata_path(experiment_id).exists():
            LOGGER.warning("중복 ID 감지: %s", experiment_id)

        record = ExperimentRecord(
            experiment_id=experiment_id,
            created_at=utc_now_iso(),
            task_type=task_type,
            dataset_name=dataset_name,
            model=model,
            random_seed=random_seed,
            feature_preset=feature_preset,
            feature_count=len(feature_names or []),
            feature_names=list(feature_names or []),
            train_rows=train_rows,
            validation_rows=validation_rows,
            config=dict(config or {}),
            tags=list(tags or []),
            user_note=user_note,
            git_commit=git_commit,
        )
        self._active = record
        self.storage.save(record)
        LOGGER.info("Experiment 시작: %s", record.experiment_id)
        self._refresh_leaderboard()
        return record

    def finish(
        self,
        *,
        metric_name: str,
        score: float,
        metrics: dict[str, float] | None = None,
        training_time: float | None = None,
        inference_time: float | None = None,
        feature_names: list[str] | None = None,
        feature_preset: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        user_note: str | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        """Finish an experiment and save its evaluation metrics.
        
        Args:
            metric_name (str): The name of the evaluation metric (e.g., 'F1', 'RMSE').
            score (float): The evaluation score.
            metrics (dict[str, float] | None): Optional detailed evaluation metrics.
            training_time (float | None): Total training time in seconds.
            inference_time (float | None): Total inference time in seconds.
            feature_names (list[str] | None): List of generated feature names (can override start).
            feature_preset (str | None): Feature preset used (can override start).
            config (dict[str, Any] | None): Hyperparameters (can override start).
            tags (list[str] | None): Tags (can override start).
            user_note (str | None): User note (can override start).
            experiment_id (str | None): Optional specific ID to finish. If None, finishes the active experiment.
            
        Returns:
            ExperimentRecord: The updated and saved experiment record.
            
        Raises:
            RuntimeError: If there is no active experiment and no experiment_id is provided.
        """
        if self._active is None and experiment_id is None:
            raise RuntimeError("start()가 호출되지 않은 상태에서 finish()를 호출할 수 없습니다.")
            
        target_id = experiment_id or (self._active.experiment_id if self._active else None)
        if target_id is None:
            raise RuntimeError("finish 대상 experiment_id가 없다")

        record = self.storage.load(target_id)
        updated = replace(
            record,
            metric=ExperimentMetric(metric_name=metric_name, score=float(score)),
            metrics=(
                {str(k): float(v) for k, v in metrics.items()}
                if metrics is not None
                else record.metrics
            ),
            training_time=training_time if training_time is not None else record.training_time,
            inference_time=inference_time if inference_time is not None else record.inference_time,
            feature_names=list(feature_names) if feature_names is not None else record.feature_names,
            feature_count=len(feature_names) if feature_names is not None else record.feature_count,
            feature_preset=feature_preset if feature_preset is not None else record.feature_preset,
            config=dict(config) if config is not None else record.config,
            tags=list(tags) if tags is not None else record.tags,
            user_note=user_note if user_note is not None else record.user_note,
        )
        self.storage.save(updated)
        write_experiment_report(updated, self.reports_dir)
        self._refresh_leaderboard()
        LOGGER.info("Experiment 종료: %s", updated.experiment_id)
        LOGGER.info("Metric 저장: %s=%.6f", metric_name, score)
        self._active = None
        return updated

    def compare(self, left_id: str, right_id: str) -> str:
        """Compare two experiments and return a Markdown formatted table.
        
        Args:
            left_id (str): Experiment ID for the left column.
            right_id (str): Experiment ID for the right column.
            
        Returns:
            str: A Markdown table comparing metrics, times, and parameters.
        """
        left = self.storage.load(left_id)
        right = self.storage.load(right_id)
        return compare_experiments(left, right)

    def best(self, metric_name: str | None = None) -> ExperimentRecord | None:
        """Find the experiment with the highest metric score.

        Args:
            metric_name: Optional detailed or primary metric name to rank by.
        
        Returns:
            ExperimentRecord | None: The best experiment record, or None if no records exist.
        """
        records = self.storage.all_records()
        if metric_name is None:
            scored = [(r, r.metric.score) for r in records if r.metric is not None]
        else:
            scored = [
                (
                    r,
                    r.metrics.get(
                        metric_name,
                        r.metric.score
                        if r.metric is not None and r.metric.metric_name == metric_name
                        else None,
                    ),
                )
                for r in records
            ]
            scored = [(r, score) for r, score in scored if score is not None]
        return max(scored, key=lambda item: item[1])[0] if scored else None

    def latest(self) -> ExperimentRecord | None:
        """Find the most recently created experiment.
        
        Returns:
            ExperimentRecord | None: The most recent experiment record, or None if no records exist.
        """
        records = self.storage.all_records()
        if not records:
            return None
        return sorted(records, key=lambda r: r.created_at, reverse=True)[0]

    def find(
        self,
        *,
        metric_gt: float | None = None,
        metric_name: str | None = None,
        tag: str | None = None,
        model: str | None = None,
    ) -> list[ExperimentSummary]:
        """Search for experiments matching specific criteria.
        
        Args:
            metric_gt (float | None): Find experiments with a score strictly greater than this value.
            metric_name (str | None): Filter by the name of the evaluation metric.
            tag (str | None): Filter by a specific tag.
            model (str | None): Filter by the name of the model.
            
        Returns:
            list[ExperimentSummary]: A list of summarized experiment data, sorted by creation date (newest first).
        """
        query = ExperimentSearchQuery(metric_gt=metric_gt, metric_name=metric_name, tag=tag, model=model)
        out: list[ExperimentSummary] = []
        for record in self.storage.all_records():
            if query.model and record.model != query.model:
                continue
            if query.tag and query.tag not in record.tags:
                continue
            if query.metric_name and (record.metric is None or record.metric.metric_name != query.metric_name):
                continue
            if query.metric_gt is not None:
                if record.metric is None or record.metric.score <= query.metric_gt:
                    continue
            out.append(
                ExperimentSummary(
                    experiment_id=record.experiment_id,
                    metric_name=record.metric.metric_name if record.metric else None,
                    score=record.metric.score if record.metric else None,
                    created_at=record.created_at,
                    model=record.model,
                    feature_preset=record.feature_preset,
                )
            )
        return sorted(out, key=lambda r: r.created_at, reverse=True)

    def _next_experiment_id(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        prefix = f"EXP_{date_str}_"
        used = []
        for record in self.storage.all_records():
            if record.experiment_id.startswith(prefix):
                suffix = record.experiment_id.replace(prefix, "", 1)
                if suffix.isdigit():
                    used.append(int(suffix))
        next_index = (max(used) + 1) if used else 1
        return f"{prefix}{next_index:03d}"

    def _refresh_leaderboard(self) -> None:
        records = [r for r in self.storage.all_records() if r.metric is not None]
        # Records are grouped by metric name, and within each group the best model comes
        # first. Direction matters: ranking rmse descending would put the worst model on
        # top. Rank numbering stays global because a single ordering across different
        # metrics has no meaning on its own.
        ranked = sorted(
            records,
            key=lambda r: (
                r.metric.metric_name or "",
                r.metric.score if _is_lower_better(r.metric.metric_name) else -r.metric.score,
            ),
        )
        rows = []
        for rank, r in enumerate(ranked, start=1):
            rows.append({
                "Experiment": r.experiment_id,
                "Metric": f"{r.metric.metric_name}:{r.metric.score:.6f}",
                "Date": r.created_at,
                "Model": r.model,
                "Preset": r.feature_preset or "",
                "Rank": rank,
            })
        self.storage.save_leaderboard(rows)
