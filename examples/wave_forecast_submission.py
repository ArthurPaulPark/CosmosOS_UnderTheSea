"""Problem 3 submission pipeline for the Ocean AI Competition.

Running it with no arguments rebuilds the P3 candidate end to end.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import sys
import time

# Resolve the checkout from this file so the script runs from any working
# directory and on any machine, not just the drive it was developed on.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cosmos_os.competitions import CompetitionRegistry
import cosmos_os.competitions.ocean.forecasting  # registers the adapter

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    data_dir = PROJECT_ROOT / "P3_wave_forecast"
    if not (data_dir / "train_wave.csv").exists():
        LOGGER.error("Dataset not found at %s.", data_dir)
        sys.exit(1)

    out_path = PROJECT_ROOT / "output/forecasting/submission_phase13.csv"

    config = {
        "dataset": {
            "train_wave": str(data_dir / "train_wave.csv"),
            "train_atmos": str(data_dir / "train_atmos.csv"),
            "test_context": str(data_dir / "test_context.parquet"),
            "test_index": str(data_dir / "test_index.csv"),
            "sample_submission": str(data_dir / "sample_submission.csv"),
            "baseline_persistence": str(data_dir / "baseline_persistence.csv"),
        },
        # M3: one persistence-residual model per lead time.
        "model_mode": "persistence_residual",
        "strategy": "residual",
        "validation": {"type": "primary_holdout"},
        "submission": {"output": str(out_path)},
    }

    start = time.time()
    adapter = CompetitionRegistry.create("ocean_forecast", config=config)
    adapter.run()
    elapsed = time.time() - start

    m = adapter.val_metrics
    print("\n=======================")
    print("Validation (primary holdout)")
    print("=======================")
    print(f"Pooled RMSE:      {m['rmse']:.6f} m")
    print(f"Persistence RMSE: {m['baseline_persistence_rmse']:.6f} m")
    print(f"Improvement:      {m['rmse_improvement_ratio_pct']:.2f}%")
    print(f"Cases / points:   {int(m['case_count'])} / {int(m['row_count'])}")
    for lead in (3, 6, 9, 12, 18, 24):
        print(f"  +{lead:2d}h  {m[f'lead_{lead}_rmse']:.4f}  (persistence {m[f'lead_{lead}_persistence_rmse']:.4f})")

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print("\n=======================")
    print("Submission")
    print("=======================")
    print(out_path.resolve())
    print(f"Rows:         {len(adapter.submission_df)}")
    print(f"SHA256:       {digest}")
    print(f"Elapsed Time: {elapsed:.2f}s")
    print("=======================")


if __name__ == "__main__":
    main()
