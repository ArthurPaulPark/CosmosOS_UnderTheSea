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
# this script is to prove the submissions come from this code.
python3 - <<'PY'
import hashlib
import json
import pathlib
import sys

registry = json.loads(pathlib.Path("artifacts/candidate_registry.json").read_text())
failures = []
for key in ("problem_1", "problem_2", "problem_3"):
    entry = registry[key]
    path = pathlib.Path(entry["submission_path"])
    if not path.exists():
        failures.append(f"{key}: missing {path}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    status = "OK" if digest == entry["sha256"] else "MISMATCH"
    print(f"{key}: {digest}  {path}  [{status}]")
    if status == "MISMATCH":
        failures.append(f"{key}: expected {entry['sha256']}, rebuilt {digest}")

if failures:
    print("\nREPRODUCTION FAILED:", *failures, sep="\n  ")
    sys.exit(1)
print("\nAll three submissions reproduced byte-for-byte.")
PY
