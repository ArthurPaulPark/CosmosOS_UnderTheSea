from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExperimentMetric:
    metric_name: str
    score: float


@dataclass
class ExperimentRecord:
    experiment_id: str
    created_at: str
    task_type: str
    dataset_name: str
    model: str
    random_seed: int | None
    feature_preset: str | None
    feature_count: int
    feature_names: list[str] = field(default_factory=list)
    train_rows: int = 0
    validation_rows: int = 0
    training_time: float | None = None
    inference_time: float | None = None
    metric: ExperimentMetric | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    git_commit: str | None = None
    tags: list[str] = field(default_factory=list)
    user_note: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "model": self.model,
            "random_seed": self.random_seed,
            "feature_preset": self.feature_preset,
            "feature_count": int(self.feature_count),
            "feature_names": list(self.feature_names),
            "train_rows": int(self.train_rows),
            "validation_rows": int(self.validation_rows),
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "metric": None if self.metric is None else {
                "metric_name": self.metric.metric_name,
                "score": float(self.metric.score),
            },
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
            "git_commit": self.git_commit,
            "tags": list(self.tags),
            "user_note": self.user_note,
            "config": self.config,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        metric_raw = data.get("metric")
        metric = None
        if metric_raw is not None:
            metric = ExperimentMetric(
                metric_name=str(metric_raw["metric_name"]),
                score=float(metric_raw["score"]),
            )
        return cls(
            experiment_id=str(data["experiment_id"]),
            created_at=str(data["created_at"]),
            task_type=str(data["task_type"]),
            dataset_name=str(data["dataset_name"]),
            model=str(data["model"]),
            random_seed=data.get("random_seed"),
            feature_preset=data.get("feature_preset"),
            feature_count=int(data.get("feature_count", 0)),
            feature_names=[str(v) for v in data.get("feature_names", [])],
            train_rows=int(data.get("train_rows", 0)),
            validation_rows=int(data.get("validation_rows", 0)),
            training_time=data.get("training_time"),
            inference_time=data.get("inference_time"),
            metric=metric,
            metrics={str(k): float(v) for k, v in data.get("metrics", {}).items()},
            git_commit=data.get("git_commit"),
            tags=[str(v) for v in data.get("tags", [])],
            user_note=data.get("user_note"),
            config=dict(data.get("config", {})),
        )


@dataclass
class ExperimentSummary:
    experiment_id: str
    metric_name: str | None
    score: float | None
    created_at: str
    model: str
    feature_preset: str | None


@dataclass
class ExperimentSearchQuery:
    metric_gt: float | None = None
    metric_name: str | None = None
    tag: str | None = None
    model: str | None = None


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
