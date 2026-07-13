"""End-to-end checks for the materials and far-field tutorials."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIALS_DIR = REPO_ROOT / "examples" / "tutorials"


def _run_tutorial(name: str) -> str:
    path = TUTORIALS_DIR / name
    assert path.exists(), f"missing tutorial: {path}"

    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout


def test_materials_and_dispersion_tutorial_runs():
    """Every material variant runs and the loss advisory changes as taught."""
    output = _run_tutorial("materials_and_dispersion.py")

    assert "Public material names:" in output
    assert "Lossless advisory observed: True" in output
    assert "Lossy advisory observed: False" in output

    peak_text = re.findall(
        r"^.+? peak \|Ez\|:\s*([^\s]+)",
        output,
        flags=re.MULTILINE,
    )
    peaks = [float(value) for value in peak_text]
    assert len(peaks) == 5
    assert all(math.isfinite(value) and value >= 0.0 for value in peaks)


def test_antenna_farfield_pattern_tutorial_runs():
    """The dipole reports its warning, directivity, and a fresh E-plane plot."""
    plot_path = TUTORIALS_DIR / "output" / "short_dipole_e_plane.png"
    plot_path.unlink(missing_ok=True)

    output = _run_tutorial("antenna_farfield_pattern.py")

    assert "Close-box advisory observed: True" in output
    assert "Corrected face spacing:" in output
    assert output.count("[PREFLIGHT] All checks passed") >= 2
    match = re.search(r"Peak directivity:\s*([^\s]+)\s*dBi", output)
    assert match is not None
    peak_directivity_dbi = float(match.group(1))
    assert math.isfinite(peak_directivity_dbi)
    assert abs(peak_directivity_dbi - 1.76) < 0.3
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
