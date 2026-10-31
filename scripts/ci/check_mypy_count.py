#!/usr/bin/env python3
"""Run the non-required mypy error-count ratchet with the repository config."""

from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
BASELINE = "scripts/ci/mypy_baseline.txt"
ERROR_SUMMARY = re.compile(r"Found (\d+) errors? in \d+ files?(?: \(checked \d+ source files?\))?")
SUCCESS_SUMMARY = re.compile(r"Success: no issues found in \d+ source files?")


def main() -> int:
    try:
        baseline = int((REPO / BASELINE).read_text().strip())
        if baseline < 0:
            raise ValueError("baseline must be nonnegative")
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "rfx"], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except (OSError, ValueError) as error:
        print(f"mypy check failed: {error}", file=sys.stderr)
        return 2

    lines = result.stdout.strip().splitlines()
    summary = lines[-1] if lines else ""
    errors = ERROR_SUMMARY.fullmatch(summary)
    success = SUCCESS_SUMMARY.fullmatch(summary)
    if result.returncode not in (0, 1) or not (errors or success):
        print(result.stdout, end="", file=sys.stderr)
        print("mypy check failed: no valid summary or unexpected exit code", file=sys.stderr)
        return 2

    count = int(errors[1]) if errors else 0
    print(f"mypy errors: {count} (baseline {baseline})")
    if count < baseline:
        print(f"Lower the baseline: write {count} to {BASELINE}")
    return int(count > baseline)


if __name__ == "__main__":
    sys.exit(main())
