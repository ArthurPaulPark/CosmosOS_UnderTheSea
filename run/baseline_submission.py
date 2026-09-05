"""Baseline Submission Pipeline for Ocean AI Competition."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.metrics import confusion_matrix

# Resolve the checkout from this file so the script runs from any working
# directory and on any machine, not just the drive it was developed on.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cosmos_os.competitions import CompetitionRegistry

# We must import the ocean adapter to register it
import cosmos_os.competitions.ocean

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def build_config(
    data_dir: Path,
    submission_path: Path,
    experiment_root: Path,
    full_weight: float = 0.5,
    gru_weight: float = 0.0,
    rate: float | None = None,
    injection_shape: bool = False,
    neighbours: bool = False,
    pseudo_labels: bool = False,
) -> dict:
    """Build the reproducible P1 config without writing any artifact.

    `gru_weight` defaults to 0 so that a plain run rebuilds the registered
    candidate byte-for-byte, which run_all.sh checks. Pass a weight to produce
    the GRU-auxiliary variant instead.
    """
    if not 0 <= full_weight <= 1:
        raise ValueError("full_weight must be between 0 and 1")
    if not 0 <= gru_weight <= 1:
        raise ValueError("gru_weight must be between 0 and 1")
    return {
        "dataset": {
            "train": str(data_dir / "train.csv"),
            "test": str(data_dir / "test.csv"),
            "submission": str(data_dir / "sample_submission.csv"),
        },
        "feature": {
            "presets": ["standard", "full"],
            "ensemble_weights": [1 - full_weight, full_weight],
            **({"injection_shape": {"windows": [144, 288]}} if injection_shape else {}),
            **({"neighbours": True} if neighbours else {}),
            **({"pseudo_labels": True} if pseudo_labels else {}),
        },
        "model": {
            "name": "LightGBM_Baseline",
            "params": {
                "learning_rate": 0.05,
                "n_estimators": 600,
                "verbosity": -1,
            },
        },
        "validation": {"test_size": 0.2},
        "submission": {
            "output": str(submission_path),
            **({"rate": rate} if rate else {}),
            **(
                {"gru_auxiliary": {"weight": gru_weight, "seed": 0, "device": "cpu"}}
                if gru_weight > 0
                else {}
            ),
        },
        "experiment": {
            "root_dir": str(experiment_root),
            "reports_dir": str(experiment_root.parent / "reports"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "P1_qc_anomaly"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/submission_phase10_5_safe.csv")
    )
    parser.add_argument("--experiment-root", type=Path, default=Path("output/experiments"))
    parser.add_argument("--full-weight", type=float, default=0.5)
    parser.add_argument(
        "--pseudo-labels",
        action="store_true",
        help="label test.csv's confident rows with the first pass and refit on them",
    )
    parser.add_argument(
        "--neighbours",
        action="store_true",
        help="add the temperature gap to the layer directly above and below",
    )
    parser.add_argument(
        "--injection-shape",
        action="store_true",
        help=(
            "Add centred linear-fit shape features. The anomalies were injected into "
            "clean segments, so a drift's residual is almost exactly a line and noise "
            "is exactly what a line cannot explain; the plugin measures how fast a "
            "window moves but not how well a line explains it."
        ),
    )
    parser.add_argument(
        "--positive-count",
        type=int,
        default=0,
        help=(
            "Diagnostic override: emit exactly this many positives instead of the "
            "validation split's anomaly rate. The default of 0 keeps the rate fitted "
            "from the distributed data, which is what the candidate uses."
        ),
    )
    parser.add_argument(
        "--gru-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the GRU drift/offset auxiliary, blended one-directionally. "
            "0 (the default) reproduces the registered candidate; 0.6 is the value "
            "the six-window backtest was run at."
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    train_path = data_dir / "train.csv"
    sample_sub_path = data_dir / "sample_submission.csv"

    # If the real datasets don't exist yet (e.g. CI/CD or before mounting),
    # gracefully fallback or exit (for the sake of the environment)
    if not train_path.exists():
        LOGGER.error(f"Dataset not found at {data_dir}. Please ensure the drive is mounted.")
        sys.exit(1)

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output
    config = build_config(
        data_dir,
        submission_path,
        args.experiment_root,
        args.full_weight,
        args.gru_weight,
        (args.positive_count / len(pd.read_csv(sample_sub_path))) if args.positive_count else None,
        args.injection_shape,
        args.neighbours,
        args.pseudo_labels,
    )
    
    start_time = time.time()
    
    # Run Adapter
    adapter = CompetitionRegistry.create("ocean", config=config)
    adapter.run()
    
    elapsed_time = time.time() - start_time
    
    metrics = adapter.metrics
    cm = confusion_matrix(
        adapter.val_split["label"], adapter.validation_predictions, labels=[0, 1]
    )
    
    # Verify Submission
    sub_df = pd.read_csv(submission_path)
    sample_df = pd.read_csv(sample_sub_path)
    
    # Validate rows, columns, nulls
    assert len(sub_df) == len(sample_df), "Row count mismatch"
    assert list(sub_df.columns) == list(sample_df.columns), "Column mismatch"
    assert sub_df["label"].isnull().sum() == 0, "Null values found in label column"
    label_1_count = int((sub_df["label"] == 1).sum())
    label_1_rate = label_1_count / len(sub_df)
    
    # Get Experiment ID from the manager (the latest one)
    # The adapter uses self.experiment_manager which has a _active flag, but after finish it resets.
    # Let's find the most recent directory in experiments.
    exp_dir = Path(config["experiment"]["root_dir"])
    if exp_dir.exists():
        exp_folders = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        exp_id = exp_folders[-1].name if exp_folders else "UNKNOWN"
    else:
        exp_id = "UNKNOWN"
    
    # Print exactly as requested
    print("\n=======================")
    print("Validation")
    print("=======================")
    print(f"Competition Anomaly F1: {metrics['anomaly_f1']:.4f}")
    print(f"Anomaly Precision:      {metrics['anomaly_precision']:.4f}")
    print(f"Anomaly Recall:         {metrics['anomaly_recall']:.4f}")
    print(f"Macro F1 (reference):   {metrics['macro_f1']:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    print("\n=======================")
    print("Submission")
    print("=======================")
    print(str(submission_path.resolve()))
    print(f"Label 1 Count: {label_1_count}")
    print(f"Label 1 Rate:  {label_1_rate:.6f}")
    print(f"Elapsed Time:  {elapsed_time:.2f}s")
    print("Saved")
    
    print("\n=======================")
    print("Experiment")
    print("=======================")
    print(f"{exp_id}")
    print("Saved")
    print("=======================")

if __name__ == "__main__":
    main()
