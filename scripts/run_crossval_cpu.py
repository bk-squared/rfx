#!/usr/bin/env python3
"""Run the CPU-feasible subset of the crossval suite and print a status table.

This is the single runner referenced by ``docs/public/guide/benchmarks.mdx``
(roadmap task W4.8). The canonical case inventory and CPU disposition live in
``examples/crossval/manifest.json``. This runner executes each CPU-feasible
case in a fresh subprocess with a per-script timeout, classifies the outcome,
and exits 0 iff no script failed for a non-environment reason.

Outcome classification
-----------------------
Each crossval script follows the rfx exit-code convention:

  0 -> all checks passed, including every reference required by the manifest
  1 -> a self-check, numeric accept gate, or required execution failed
  2 -> self-check OK but an external reference or optional dependency is
       missing (inconclusive crossval, NOT a pass)

The manifest may name a failure sentinel for defense in depth. The runner scans
stdout for that sentinel even when a script returns 0, so an inconsistent
headline and process exit cannot silently become PASS.

A separate ENV-SKIP class catches the case where an *optional reference solver*
(currently Meep) is installed but unimportable -- e.g. a Meep built against
NumPy 1.x running under NumPy 2.x. That is an environment/packaging problem,
not an rfx residual, so it must not count as a FAIL.

Exit status of this runner
--------------------------
0  iff no script ended in FAIL or TIMEOUT. PASS / SELF-CHECK-ONLY / ENV-SKIP /
   EXCLUDED do not fail the gate, but remain visible in the summary.
1  if any script failed or timed out.

Usage:
    PYTHONPATH=. python scripts/run_crossval_cpu.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
CROSSVAL_DIR = REPO_ROOT / "examples" / "crossval"
MANIFEST_PATH = CROSSVAL_DIR / "manifest.json"

# Per-script timeout (seconds). The empirically-slowest member of the CPU
# subset is 05 (~6 min when it short-circuits on missing OpenEMS); 900 s gives
# headroom without letting a wedged run hang the suite.
PER_SCRIPT_TIMEOUT_S: Final = 900


@dataclass(frozen=True, slots=True)
class RunnerCase:
    """CPU-runner fields parsed from one manifest entry."""

    filename: str
    cpu_order: int | None
    excluded_reason: str | None
    failure_sentinel: str | None
    expected_exit_codes: frozenset[int]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Classified outcome from one cross-validation subprocess."""

    script: str
    returncode: int
    elapsed: float
    status: str
    note: str


def _load_runner_cases() -> tuple[RunnerCase, ...]:
    """Load the runner projection of the canonical crossval manifest."""
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases: list[RunnerCase] = []
    for raw_case in payload["cases"]:
        cpu_entry = raw_case["cpu_runner"]
        cases.append(
            RunnerCase(
                filename=Path(raw_case["script"]).name,
                cpu_order=cpu_entry.get("order"),
                excluded_reason=cpu_entry.get("excluded_reason"),
                failure_sentinel=raw_case.get("failure_sentinel"),
                expected_exit_codes=frozenset(raw_case["expected_exit_codes"]),
            )
        )
    return tuple(cases)


_RUNNER_CASES: Final = _load_runner_cases()

# CPU-feasible subset, in manifest order. The empirical classification dates to
# the W4.8 survey; the manifest is now the single owner of that policy.
CPU_SUBSET: Final = tuple(
    case.filename
    for case in sorted(
        (case for case in _RUNNER_CASES if case.cpu_order is not None),
        key=lambda case: case.cpu_order or 0,
    )
)

# Cases intentionally not attempted by the CPU runner, with their visible reason.
EXCLUDED: Final = {
    case.filename: case.excluded_reason
    for case in _RUNNER_CASES
    if case.excluded_reason is not None
}

# Defense-in-depth stdout sentinels owned by the manifest.
EXIT0_FAIL_SENTINELS: Final = {
    case.filename: case.failure_sentinel
    for case in _RUNNER_CASES
    if case.failure_sentinel is not None
}

# Per-case exit contract. In particular, exit 2 is valid only for scripts whose
# required external reference or solver dependency can be unavailable.
EXPECTED_EXIT_CODES: Final = {
    case.filename: case.expected_exit_codes for case in _RUNNER_CASES
}

# Substrings that identify an unimportable optional reference solver
# (environment/packaging problem, not an rfx residual).
ENV_BROKEN_REF_MARKERS = (
    "numpy.core.multiarray failed to import",
    "_ARRAY_API not found",
    "A module that was compiled using NumPy 1.x cannot be run",
    "ModuleNotFoundError: No module named 'meep'",
)

# Status labels.
PASS = "PASS"
FAIL = "FAIL"
SELF_ONLY = "SELF-CHECK-ONLY"
ENV_SKIP = "ENV-SKIP"
TIMEOUT = "TIMEOUT"


def classify(
    script: str, returncode: int, output: str, timed_out: bool
) -> tuple[str, str]:
    """Return (status_label, note) for one finished run."""
    if timed_out:
        return TIMEOUT, f"exceeded {PER_SCRIPT_TIMEOUT_S}s"

    expected_exit_codes = EXPECTED_EXIT_CODES.get(script)
    if expected_exit_codes is None:
        return FAIL, "script is not registered in the crossval manifest"
    if returncode not in expected_exit_codes:
        return (
            FAIL,
            f"undeclared exit {returncode}; expected {sorted(expected_exit_codes)}",
        )

    env_broken = any(marker in output for marker in ENV_BROKEN_REF_MARKERS)

    if returncode == 0:
        sentinel = EXIT0_FAIL_SENTINELS.get(script)
        if sentinel and sentinel in output:
            return FAIL, "failure sentinel found after exit 0"
        return PASS, "all gates passed"

    if returncode == 2:
        return SELF_ONLY, "self-check OK; external reference/dependency missing"

    if returncode == 1:
        if env_broken:
            return ENV_SKIP, "optional reference solver unimportable (env/packaging)"
        return FAIL, "self-check / numeric accept gate failed"

    # The registry contract reserves 2 for an inconclusive reference/dependency
    # result. Every other unexpected process exit is a real runner failure.
    return FAIL, f"unexpected script exit {returncode}"


def run_one(script: str) -> RunResult:
    path = CROSSVAL_DIR / script
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    env.setdefault("MPLBACKEND", "Agg")
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=PER_SCRIPT_TIMEOUT_S,
        )
        returncode = proc.returncode
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
    elapsed = time.monotonic() - start
    status, note = classify(script, returncode, output, timed_out)
    return RunResult(
        script=script,
        returncode=returncode,
        elapsed=elapsed,
        status=status,
        note=note,
    )


def main() -> int:
    print("=" * 78)
    print("rfx crossval — CPU-feasible subset runner (W4.8)")
    print(
        f"timeout per script: {PER_SCRIPT_TIMEOUT_S}s   |   subset: {len(CPU_SUBSET)} scripts"
    )
    print("=" * 78)

    results = []
    for script in CPU_SUBSET:
        print(f"\n>>> running {script} ...", flush=True)
        res = run_one(script)
        results.append(res)
        print(
            f"    -> {res.status:<16} exit={res.returncode:<3} "
            f"{res.elapsed:6.1f}s  ({res.note})",
            flush=True,
        )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'case':<32} {'status':<16} {'exit':>4} {'time(s)':>9}")
    print("-" * 78)
    for res in results:
        print(
            f"{res.script:<32} {res.status:<16} {res.returncode:>4} {res.elapsed:>9.1f}"
        )

    if EXCLUDED:
        print("\nexcluded (not attempted):")
        for script, reason in EXCLUDED.items():
            print(f"  - {script}: {reason}")

    n_fail = sum(1 for result in results if result.status == FAIL)
    n_pass = sum(1 for result in results if result.status == PASS)
    n_self = sum(1 for result in results if result.status == SELF_ONLY)
    n_env = sum(1 for result in results if result.status == ENV_SKIP)
    n_timeout = sum(1 for result in results if result.status == TIMEOUT)

    print(
        f"\ntotals: {n_pass} PASS, {n_self} SELF-CHECK-ONLY, "
        f"{n_env} ENV-SKIP, {n_timeout} TIMEOUT, {n_fail} FAIL"
    )

    if n_fail or n_timeout:
        print("\nGATE: FAIL — at least one script failed or timed out.")
        return 1
    print("\nGATE: PASS — no script failed or timed out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
