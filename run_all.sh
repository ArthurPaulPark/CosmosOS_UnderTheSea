#!/usr/bin/env bash
# Rebuild all three competition submissions from source and verify their hashes.
#
# Everything is trained from the distributed data in this checkout - no external
# observations and no pretrained weights. Runtime is about seven minutes total,
# well inside the competition's six-hour limit.
set -euo pipefail
cd "$(dirname "$0")"

echo "### P1: Anomaly QC (~6 min)"
python3 run/baseline_submission.py

echo
echo "### P2: Vertical Reconstruction (~30 s)"
python3 run/reconstruction_submission.py

echo
echo "### P3: Wave Forecast (~30 s)"
python3 run/wave_forecast_submission.py

echo
echo "### Verifying against the registered candidates"
# Fail loudly rather than leaving a silently different file behind: the point of
# this script is to prove the submissions come from this code. The same check runs
# on Windows as `python verify_hashes.py`.
python3 verify_hashes.py
