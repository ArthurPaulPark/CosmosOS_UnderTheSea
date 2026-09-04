from __future__ import annotations

from .models import ExperimentRecord


def compare_experiments(left: ExperimentRecord, right: ExperimentRecord) -> str:
    """Generate a Markdown table comparing two experiments.
    
    Args:
        left (ExperimentRecord): The first experiment record.
        right (ExperimentRecord): The second experiment record to compare against.
        
    Returns:
        str: A Markdown formatted string containing the comparison table.
    """
    left_metric_name = left.metric.metric_name if left.metric else "-"
    left_score = f"{left.metric.score:.6f}" if left.metric else "-"
    right_metric_name = right.metric.metric_name if right.metric else "-"
    right_score = f"{right.metric.score:.6f}" if right.metric else "-"

    rows = [
        ("Metric", f"{left_metric_name}={left_score}", f"{right_metric_name}={right_score}"),
        ("Feature Count", str(left.feature_count), str(right.feature_count)),
        ("Training Time (s)", _fmt_optional(left.training_time), _fmt_optional(right.training_time)),
        ("Inference Time (s)", _fmt_optional(left.inference_time), _fmt_optional(right.inference_time)),
    ]
    param_diff = _diff_config(left.config, right.config)
    rows.append(("Parameter Diff", param_diff, param_diff))

    table = [
        "| Field | " + left.experiment_id + " | " + right.experiment_id + " |",
        "|---|---:|---:|",
    ]
    for field, left_v, right_v in rows:
        table.append(f"| {field} | {left_v} | {right_v} |")
    return "\n".join(table)


def _fmt_optional(value: float | None) -> str:
    """Format an optional float value to 4 decimal places.
    
    Args:
        value (float | None): The value to format.
        
    Returns:
        str: Formatted string or '-' if None.
    """
    if value is None:
        return "-"
    return f"{value:.4f}"


def _diff_config(left: dict, right: dict) -> str:
    """Compare two configuration dictionaries and return differences.
    
    Args:
        left (dict): First configuration dictionary.
        right (dict): Second configuration dictionary.
        
    Returns:
        str: A semicolon-separated string of differences, or 'No difference'.
    """
    keys = sorted(set(left) | set(right))
    diffs = []
    for key in keys:
        if left.get(key) != right.get(key):
            diffs.append(f"{key}: {left.get(key)} -> {right.get(key)}")
    return "; ".join(diffs) if diffs else "No difference"
