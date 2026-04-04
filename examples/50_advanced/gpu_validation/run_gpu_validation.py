"""GPU Accuracy Validation Runner for Advanced Examples

Runs all 6 GPU-grade validation cases and reports PASS/FAIL summary.
Each case validates an advanced rfx workflow against analytical theory.

Usage:
    python run_gpu_validation.py
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

cases = [
    ("01_patch_opt_validation.py",  "Patch bandwidth optimization"),
    ("02_filter_validation.py",     "WR-90 iris filter (Pozar Ch 8)"),
    ("03_matching_validation.py",   "Broadband matching network (Smith chart)"),
    ("04_coupling_validation.py",   "Array mutual coupling decay"),
    ("05_lens_validation.py",       "Dielectric lens focusing"),
    ("06_matfit_validation.py",     "Debye material characterization"),
]

print("=" * 70)
print("rfx GPU Accuracy Validation Suite (Advanced Examples)")
print("=" * 70)
print(f"Running {len(cases)} validation cases with GPU-grade resolution")
print(f"Each case validates against analytical/textbook theory\n")

results = []
total_time = 0

for script, name in cases:
    script_path = os.path.join(SCRIPT_DIR, script)
    print(f"[{script}] {name}")
    print("-" * 60)

    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per case
            cwd=SCRIPT_DIR,
        )
        elapsed = time.time() - t0
        passed = r.returncode == 0

        # Print last few lines of stdout for context
        stdout_lines = r.stdout.strip().split("\n")
        for line in stdout_lines[-10:]:
            print(f"  {line}")

        if not passed and r.stderr:
            stderr_tail = r.stderr.strip()[-400:]
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
    results.append((name, status, elapsed))
    print(f"  [{status}] {elapsed:.1f}s\n")

# =============================================================================
# Final summary
# =============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Case':<45} {'Status':<8} {'Time':<10}")
print("-" * 63)
for name, status, elapsed in results:
    print(f"{name:<45} {status:<8} {elapsed:<10.1f}s")

n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_total = len(results)
print("-" * 63)
print(f"Total: {n_pass}/{n_total} passed in {total_time:.1f}s")

if n_pass == n_total:
    print("\nAll GPU validation cases PASSED.")
    sys.exit(0)
else:
    print(f"\n{n_total - n_pass} GPU validation case(s) FAILED.")
    sys.exit(1)
