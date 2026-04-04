"""Run all accuracy validation cases and generate summary.

Each case compares rfx FDTD results against known analytical or published
benchmark values. Cases return exit code 0 on pass, 1 on failure.

Usage:
    python run_all.py
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

cases = [
    ("01_patch_balanis.py",   "Patch antenna resonance (Balanis)",   5.0),
    ("02_waveguide_te10.py",  "WR-90 TE10 cutoff (Pozar)",          1.0),
    ("03_cavity_tm110.py",    "Dielectric cavity TM110 (Pozar)",     2.0),
    ("04_microstrip_z0.py",   "Microstrip Z0 / eps_eff (Hammerstad)", 5.0),
    ("05_coupled_filter.py",  "Coupled filter center freq (Pozar)",  25.0),
]

print("=" * 70)
print("rfx Accuracy Validation Suite")
print("=" * 70)
print(f"Running {len(cases)} benchmark cases against published results\n")

results = []
total_time = 0

for script, name, max_err in cases:
    script_path = os.path.join(SCRIPT_DIR, script)
    print(f"[{script}] {name} (threshold {max_err}%)")
    print("-" * 60)

    t0 = time.time()
    try:
        # Ensure numpy 2.x is available (jax 0.6.2 requires it).
        # Some environments revert numpy between subprocess calls.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "numpy>=2.0"],
            capture_output=True, timeout=60,
        )
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=SCRIPT_DIR,
        )
        elapsed = time.time() - t0
        passed = r.returncode == 0

        # Print last few lines of stdout for context
        stdout_lines = r.stdout.strip().split("\n")
        for line in stdout_lines[-8:]:
            print(f"  {line}")

        if not passed and r.stderr:
            stderr_tail = r.stderr.strip()[-300:]
            print(f"  [stderr] {stderr_tail}")

    except subprocess.TimeoutExpired:
        elapsed = 600
        passed = False
        print(f"  TIMEOUT after {elapsed:.0f}s")
    except Exception as e:
        elapsed = time.time() - t0
        passed = False
        print(f"  ERROR: {e}")

    total_time += elapsed
    status = "PASS" if passed else "FAIL"
    results.append((name, status, max_err, elapsed))
    print(f"  [{status}] {elapsed:.1f}s\n")

# =============================================================================
# Final summary
# =============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Case':<45} {'Status':<8} {'Threshold':<12} {'Time':<8}")
print("-" * 73)
for name, status, max_err, elapsed in results:
    print(f"{name:<45} {status:<8} {max_err:<12.1f} {elapsed:<8.1f}s")

n_pass = sum(1 for _, s, _, _ in results if s == "PASS")
n_total = len(results)
print("-" * 73)
print(f"Total: {n_pass}/{n_total} passed in {total_time:.1f}s")

if n_pass == n_total:
    print("\nAll validation cases PASSED.")
    sys.exit(0)
else:
    print(f"\n{n_total - n_pass} validation case(s) FAILED.")
    sys.exit(1)
