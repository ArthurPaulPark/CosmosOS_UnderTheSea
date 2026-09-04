from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def generate_ocean_eda(df: pd.DataFrame, output_dir: str | Path = "reports") -> Path:
    """Run Exploratory Data Analysis and generate a Markdown report.
    
    Args:
        df (pd.DataFrame): The dataset to analyze.
        output_dir (str | Path): Directory to save the EDA report.
        
    Returns:
        Path: Path to the generated Markdown report.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "Ocean_EDA.md"
    
    lines = [
        "# Ocean Competition EDA Report",
        "",
        "## Dataset Summary",
        f"- **Total Rows**: {len(df)}",
        f"- **Total Columns**: {len(df.columns)}",
        "",
        "## Missing Values",
        df.isna().sum().to_frame("Missing Count").to_markdown(),
        "",
        "## Station Distribution",
    ]
    
    if "station" in df:
        lines.append(df["station"].value_counts().to_frame("Count").to_markdown())
    else:
        lines.append("`station` column not found.")
        
    lines.append("")
    lines.append("## Layer Distribution")
    if "layer" in df:
        lines.append(df["layer"].value_counts().to_frame("Count").to_markdown())
    else:
        lines.append("`layer` column not found.")
        
    lines.append("")
    lines.append("## Temperature Distribution (Summary)")
    if "temp" in df:
        lines.append(df["temp"].describe().to_frame("temp").to_markdown())
    else:
        lines.append("`temp` column not found.")
        
    lines.append("")
    lines.append("## Label Distribution")
    if "label" in df:
        lines.append(df["label"].value_counts().to_frame("Count").to_markdown())
    else:
        lines.append("`label` column not found (likely test set).")
        
    lines.append("")
    lines.append("## Time Range")
    if "time" in df:
        lines.append(f"- **Start**: {df['time'].min()}")
        lines.append(f"- **End**: {df['time'].max()}")
    else:
        lines.append("`time` column not found.")
        
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("EDA report generated at %s", report_path)
    return report_path
