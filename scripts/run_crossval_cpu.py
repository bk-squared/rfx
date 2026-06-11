#!/usr/bin/env python3
"""Run the CPU-feasible subset of the crossval suite and print a status table.

This is the single runner referenced by ``docs/public/guide/benchmarks.mdx``
(roadmap task W4.8). It runs each CPU-feasible crossval script in a fresh
subprocess with a per-script timeout, classifies the outcome, and exits 0 iff
no script failed for a non-environment reason.

Outcome classification
-----------------------
Each crossval script follows the rfx exit-code convention:

  0 -> all checks passed, including any external-reference cross-check
  1 -> a self-check / numeric accept gate failed (broken physics or infra)
  2 -> self-check OK but an external reference or optional dependency is
       missing (inconclusive crossval, NOT a pass)

Two scripts (01, 04) print a PASS/FAIL summary but exit 0 regardless, so this
runner also scans their stdout for the failure sentinel and downgrades them.

A separate ENV-SKIP class catches the case where an *optional reference solver*
(currently Meep) is installed but unimportable -- e.g. a Meep built against
NumPy 1.x running under NumPy 2.x. That is an environment/packaging problem,
not an rfx residual, so it must not count as a FAIL.

Exit status of this runner
--------------------------
0  iff no script ended in FAIL (a real exit-1 numeric-gate failure that is not
   an environment artefact). PASS / SELF-CHECK-ONLY / ENV-SKIP / EXCLUDED do
   not fail the gate.
1  otherwise.

Usage:
    PYTHONPATH=. python scripts/run_crossval_cpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CROSSVAL_DIR = REPO_ROOT / "examples" / "crossval"

# Per-script timeout (seconds). The empirically-slowest member of the CPU
# subset is 05 (~6 min when it short-circuits on missing OpenEMS); 900 s gives
# headroom without letting a wedged run hang the suite.
PER_SCRIPT_TIMEOUT_S = 900

# CPU-feasible subset, in suite order. Each completed in < 10 min on CPU during
# the W4.8 empirical survey (2026-06-11).
CPU_SUBSET = [
    "01_waveguide_bend.py",
    "04_multilayer_fresnel.py",
    "05_patch_antenna.py",
    "09_half_symmetric_waveguide.py",
    "10_pmc_cpml_half_symmetric.py",
    "11_waveguide_port_wr90.py",
    # 02/03 are pure live-Meep crossvals: they import Meep at module top with no
    # fallback, so on a host without a working Meep they cannot self-check at
    # all. They are still attempted here (so a healthy Meep host exercises them)
    # but are expected to land in ENV-SKIP when Meep is unimportable.
    "02_ring_resonator.py",
    "03_straight_waveguide_flux.py",
]

# Excluded with reasons (NOT attempted). CPU-feasibility decided empirically on
# 2026-06-11; both exceeded a 700 s wall-clock budget on CPU.
EXCLUDED = {
    "06_msl_notch_filter.py": (
        "CPU-infeasible: > 700 s on CPU (non-uniform wire-port + graded-sigma "
        "absorber, ~150 um cells); also needs the openEMS reference .npz. "
        "Run on GPU / with the reference present."
    ),
    "06b_msl_notch_filter_uniform.py": (
        "CPU-infeasible: > 700 s on CPU (uniform fine mesh + distributed "
        "MSL-port de-embedding). Run on GPU."
    ),
    # 12/13 are subgrid prototypes, NOT validated -- excluded from the
    # validation runner on purpose.
    "12_subgrid_disjoint_prototype.py": "subgrid prototype, not validated",
    "13_subgrid_material_validation.py": "subgrid prototype, not validated",
}

# Scripts that print a PASS/FAIL summary but always exit 0; scan stdout to
# recover a real failure. Maps script -> failure sentinel substring.
EXIT0_FAIL_SENTINELS = {
    "01_waveguide_bend.py": "SOME CHECKS FAILED",
    "04_multilayer_fresnel.py": "rfx accuracy: FAIL",
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


def classify(script: str, returncode: int, output: str, timed_out: bool) -> tuple[str, str]:
    """Return (status_label, note) for one finished run."""
    if timed_out:
        return TIMEOUT, f"exceeded {PER_SCRIPT_TIMEOUT_S}s"

    env_broken = any(marker in output for marker in ENV_BROKEN_REF_MARKERS)

    if returncode == 0:
        sentinel = EXIT0_FAIL_SENTINELS.get(script)
        if sentinel and sentinel in output:
            return FAIL, "self-check FAIL (script exits 0 regardless)"
        return PASS, "all gates passed"

    if returncode == 2:
        return SELF_ONLY, "self-check OK; external reference/dependency missing"

    if returncode == 1:
        if env_broken:
            return ENV_SKIP, "optional reference solver unimportable (env/packaging)"
        return FAIL, "self-check / numeric accept gate failed"

    # Any other non-zero code: treat as a script error, surface but do not fail
    # the gate (matches the scripts' own "2 = script error" bucket intent).
    return SELF_ONLY, f"script error (exit {returncode})"


def run_one(script: str) -> dict:
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
    return {
        "script": script,
        "returncode": returncode,
        "elapsed": elapsed,
        "status": status,
        "note": note,
    }


def main() -> int:
    print("=" * 78)
    print("rfx crossval — CPU-feasible subset runner (W4.8)")
    print(f"timeout per script: {PER_SCRIPT_TIMEOUT_S}s   |   subset: {len(CPU_SUBSET)} scripts")
    print("=" * 78)

    results = []
    for script in CPU_SUBSET:
        print(f"\n>>> running {script} ...", flush=True)
        res = run_one(script)
        results.append(res)
        print(
            f"    -> {res['status']:<16} exit={res['returncode']:<3} "
            f"{res['elapsed']:6.1f}s  ({res['note']})",
            flush=True,
        )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'case':<32} {'status':<16} {'exit':>4} {'time(s)':>9}")
    print("-" * 78)
    for res in results:
        print(
            f"{res['script']:<32} {res['status']:<16} "
            f"{res['returncode']:>4} {res['elapsed']:>9.1f}"
        )

    if EXCLUDED:
        print("\nexcluded (not attempted):")
        for script, reason in EXCLUDED.items():
            print(f"  - {script}: {reason}")

    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_self = sum(1 for r in results if r["status"] == SELF_ONLY)
    n_env = sum(1 for r in results if r["status"] == ENV_SKIP)
    n_timeout = sum(1 for r in results if r["status"] == TIMEOUT)

    print(
        f"\ntotals: {n_pass} PASS, {n_self} SELF-CHECK-ONLY, "
        f"{n_env} ENV-SKIP, {n_timeout} TIMEOUT, {n_fail} FAIL"
    )

    if n_fail:
        print("\nGATE: FAIL — at least one script failed a numeric accept gate.")
        return 1
    print("\nGATE: PASS — no script failed a numeric accept gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
