"""Check the rebuilt submissions against the registered answers.

`run_all.sh` calls this at the end. On Windows, where the shell script does not run,
call it directly after the three entry points:

    python verify_hashes.py

The answers shipped here were built on arm64 macOS. Rebuilding on a different CPU
architecture does not give byte-identical files: LightGBM's floating-point arithmetic
differs slightly between architectures, one split threshold lands differently, and the
divergence compounds over hundreds of boosting rounds. The model is the same; the
numbers move in the last digits.

So a hash mismatch alone cannot tell a broken rebuild from a different CPU. This script
therefore compares the files row by row as well, against the tolerances measured on
x86_64 Linux (see README 7장):

    P1  label      disagreement <= 1% of rows
    P2  temp       RMSE <= 0.05 ℃   (measured 0.0118, model's own error 0.6079)
    P3  hs_pred    RMSE <= 0.15 m   (measured 0.0629, model's own error 0.7271)

Exit codes
    0   byte-identical, or different within the tolerance above
    1   a file is missing, or the difference exceeds the tolerance
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBLEMS = ("problem_1", "problem_2", "problem_3")
RELEASED = {
    "problem_1": "submissions/submission_p1.csv",
    "problem_2": "submissions/submission_p2.csv",
    "problem_3": "submissions/submission_p3.csv",
}
TOLERANCE = {
    "problem_1": {"column": "label", "kind": "labels", "limit": 0.01, "unit": "of rows"},
    "problem_2": {"column": "temp", "kind": "rmse", "limit": 0.05, "unit": "℃"},
    "problem_3": {"column": "hs_pred", "kind": "rmse", "limit": 0.15, "unit": "m"},
}


def compare(rebuilt: Path, released: Path, rule: dict) -> tuple[bool, str]:
    """Return (within_tolerance, human readable summary)."""
    try:
        import numpy as np
        import pandas as pd

        left = pd.read_csv(rebuilt)
        right = pd.read_csv(released)
    except Exception as exc:  # diagnostics must never mask the failure
        return False, f"could not compare: {exc}"

    if list(left.columns) != list(right.columns):
        return False, f"columns differ: {list(left.columns)} vs {list(right.columns)}"
    if len(left) != len(right):
        return False, f"row count differs: {len(left)} vs {len(right)}"

    column = rule["column"]
    a, b = left[column], right[column]
    differing = int((a != b).sum())

    if rule["kind"] == "labels":
        share = differing / len(a)
        summary = (
            f"{column}: {differing:,}/{len(a):,} rows differ ({share:.3%}), "
            f"positives {int(a.sum()):,} vs {int(b.sum()):,}"
        )
        return share <= rule["limit"], summary

    delta = (a - b).astype(float)
    rmse = float(np.sqrt((delta**2).mean()))
    summary = (
        f"{column}: {differing:,}/{len(a):,} rows differ, "
        f"RMSE {rmse:.6f} {rule['unit']}, max |Δ| {delta.abs().max():.4f}"
    )
    return rmse <= rule["limit"], summary


def main() -> int:
    registry = json.loads(
        (ROOT / "artifacts" / "candidate_registry.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []
    tolerated: list[str] = []

    for key in PROBLEMS:
        entry = registry[key]
        relative = entry["submission_path"]
        path = ROOT / relative
        if not path.exists():
            print(f"{key}: {'-' * 64}  {relative}  [MISSING]")
            failures.append(f"{key}: missing {relative}")
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == entry["sha256"]:
            print(f"{key}: {digest}  {relative}  [OK]")
            continue

        rule = TOLERANCE[key]
        within, summary = compare(path, ROOT / RELEASED[key], rule)
        label = "WITHIN TOLERANCE" if within else "OUT OF TOLERANCE"
        print(f"{key}: {digest}  {relative}  [{label}]")
        print(f"    vs {RELEASED[key]} — {summary}")
        print(f"    tolerance: {rule['kind']} <= {rule['limit']} {rule['unit']}")
        (tolerated if within else failures).append(f"{key}: {summary}")

    if failures:
        print("\nREPRODUCTION FAILED:", *failures, sep="\n  ")
        return 1

    if tolerated:
        print(
            "\nNot byte-identical, but every difference is within the tolerance measured"
            "\nacross CPU architectures. The registered answers were built on arm64 macOS;"
            "\nrebuilding on x86_64 moves the last digits without changing the model."
        )
        for line in tolerated:
            print(f"  {line}")
        return 0

    print("\nAll three submissions reproduced byte-for-byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
