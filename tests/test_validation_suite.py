"""Validation suite test harness.

Runs each of the 9 validation example scripts and checks exit code.
Marked slow since each example runs an FDTD simulation.
"""

import pytest
import subprocess
import sys

EXAMPLES = [
    "examples/14_microstrip_line.py",
    "examples/15_coupled_filter.py",
    "examples/16_dielectric_resonator.py",
    "examples/17_waveguide_coupler.py",
    "examples/18_cavity_filter.py",
    "examples/19_lumped_rlc.py",
    "examples/20_via_throughhole.py",
    "examples/21_curved_patch.py",
    "examples/22_subgridded.py",
]


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES)
def test_validation_example(script):
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stderr[-500:]}"
