from __future__ import annotations

from pathlib import Path

from .models import ExperimentRecord


def build_experiment_report(record: ExperimentRecord) -> str:
    """Build a detailed Markdown report for a single experiment.
    
    Args:
        record (ExperimentRecord): The experiment record to document.
        
    Returns:
        str: A comprehensive Markdown string detailing the experiment.
    """
    metric_name = record.metric.metric_name if record.metric else "-"
    score = f"{record.metric.score:.6f}" if record.metric else "-"
    tags = ", ".join(record.tags) if record.tags else "-"
    note = record.user_note or "-"

    lines = [
        f"# Experiment {record.experiment_id}",
        "",
        "## Summary",
        f"- DateTime: {record.created_at}",
        f"- TaskType: {record.task_type}",
        f"- Dataset: {record.dataset_name}",
        f"- Model: {record.model}",
        f"- Metric: {metric_name} = {score}",
        f"- Train Rows: {record.train_rows}",
        f"- Validation Rows: {record.validation_rows}",
        f"- Training Time(s): {_fmt(record.training_time)}",
        f"- Inference Time(s): {_fmt(record.inference_time)}",
        f"- Feature Preset: {record.feature_preset or '-'}",
        f"- Feature Count: {record.feature_count}",
        f"- Tags: {tags}",
        "",
        "## Feature Names",
    ]
    if record.feature_names:
        lines.extend([f"- {name}" for name in record.feature_names])
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Detailed Metrics")
    if record.metrics:
        for key in sorted(record.metrics):
            lines.append(f"- {key}: {record.metrics[key]:.6f}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Parameters")
    if record.config:
        for key in sorted(record.config):
            lines.append(f"- {key}: {record.config[key]}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Note")
    lines.append(note)
    return "\n".join(lines) + "\n"


def write_experiment_report(record: ExperimentRecord, reports_dir: str | Path = "reports") -> Path:
    """Write the experiment report to a markdown file.
    
    Args:
        record (ExperimentRecord): The experiment record to save.
        reports_dir (str | Path): Directory to save the markdown report.
        
    Returns:
        Path: The file path of the saved markdown report.
    """
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"Experiment_{record.experiment_id}.md"
    path.write_text(build_experiment_report(record), encoding="utf-8")
    return path


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"
