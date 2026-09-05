"""Compare the rebuilt submissions with the hashes registered in the candidate registry.

`run_all.sh` calls this at the end. On Windows, where the shell script does not run,
call it directly after the three entry points:

    python verify_hashes.py

Exits 1 if any file is missing or its hash differs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBLEMS = ("problem_1", "problem_2", "problem_3")


def main() -> int:
    registry = json.loads(
        (ROOT / "artifacts" / "candidate_registry.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []

    for key in PROBLEMS:
        entry = registry[key]
        relative = entry["submission_path"]
        path = ROOT / relative
        if not path.exists():
            failures.append(f"{key}: missing {relative}")
            print(f"{key}: {'-' * 64}  {relative}  [MISSING]")
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matched = digest == entry["sha256"]
        print(f"{key}: {digest}  {relative}  [{'OK' if matched else 'MISMATCH'}]")
        if not matched:
            failures.append(f"{key}: expected {entry['sha256']}, rebuilt {digest}")

    if failures:
        print("\nREPRODUCTION FAILED:", *failures, sep="\n  ")
        return 1

    print("\nAll three submissions reproduced byte-for-byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
