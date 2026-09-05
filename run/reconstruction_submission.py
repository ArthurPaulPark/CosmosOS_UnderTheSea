"""Problem 2 submission pipeline for the Ocean AI Competition.

Running it with no arguments rebuilds the P2 candidate end to end, which the
competition's model-validation step requires.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import sys
import time

import pandas as pd

# Resolve the checkout from this file so the script runs from any working
# directory and on any machine, not just the drive it was developed on.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cosmos_os.competitions import CompetitionRegistry
import cosmos_os.competitions.ocean.reconstruction  # registers the adapter

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    data_dir = PROJECT_ROOT / "P2_profile_restore"
    if not (data_dir / "observations.csv").exists():
        LOGGER.error("Dataset not found at %s.", data_dir)
        sys.exit(1)

    out_path = PROJECT_ROOT / "output/reconstruction/submission_phase12_6.csv"

    config = {
        "dataset": {
            "train": str(data_dir / "observations.csv"),
            "test": str(data_dir / "test_index.csv"),
            "submission": str(data_dir / "sample_submission.csv"),
        },
        "feature": {"preset": "r3"},
        # Per-layer residual: linear-interpolation baseline + LightGBM residual for
        # each missing layer (2, 3, 4).
        "strategy": "residual",
        "model_mode": "per_layer",
        "model": {
            "name": "lightgbm_regressor",
            "params": {"learning_rate": 0.03, "num_leaves": 31},
            "n_rounds": 200,
        },
        "validation": {
            "type": "competition_simulation",
            "station": "S-ORS",
            "sim_start_date": "2024-09-01",
            "sim_end_date": "2024-10-31",
            "missing_layers": [2, 3, 4],
        },
        "submission": {"output": str(out_path)},
    }

    start = time.time()
    adapter = CompetitionRegistry.create("ocean_reconstruction", config=config)
    adapter.run()
    elapsed = time.time() - start

    m = adapter.val_metrics
    print("\n=======================")
    print("Validation (2024-09-01 ~ 2024-10-31 S-ORS simulation)")
    print("=======================")
    print(f"RMSE:              {m['rmse']:.6f} deg C")
    print(f"Baseline (linear):  {m['baseline_linear_interp_rmse']:.6f} deg C")
    print(f"Layer 2 / 3 / 4:    {m.get('layer_2_rmse', float('nan')):.4f} / "
          f"{m.get('layer_3_rmse', float('nan')):.4f} / {m.get('layer_4_rmse', float('nan')):.4f}")

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print("\n=======================")
    print("Submission")
    print("=======================")
    print(out_path.resolve())
    print(f"Rows:         {len(pd.read_csv(out_path))}")
    print(f"SHA256:       {digest}")
    print(f"Elapsed Time: {elapsed:.2f}s")
    print("=======================")


if __name__ == "__main__":
    main()
