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


def test_ports_and_sparams_101_tutorial_runs():
    """Every port family preflights and the live RLC load changes S11."""
    output = _run_tutorial("ports_and_sparams_101.py")

    assert output.count("[PREFLIGHT] All checks passed") >= 4
    assert "Microstrip port setup ready: True" in output
    assert "Waveguide port setup ready: True" in output
    assert "Coax build-only advisory observed: True" in output
    assert "Coaxial port setup ready: True" in output

    match = re.search(r"RLC changed max \|S11\| by:\s*([^\s]+)", output)
    assert match is not None
    max_change = float(match.group(1))
    assert math.isfinite(max_change) and max_change > 0.0


def test_run_control_and_fields_tutorial_runs():
    """All run controls execute and the final field slice is written."""
    plot_path = TUTORIALS_DIR / "output" / "run_control_ez_slice.png"
    plot_path.unlink(missing_ok=True)

    output = _run_tutorial("run_control_and_fields.py")

    assert "[PREFLIGHT] All checks passed" in output
    assert "Fixed n_steps samples: 120" in output
    assert "Truncation advisory:" in output
    assert "ring-down truncated" in output
    assert "Until-decay truncation advisory observed: False" in output

    match = re.search(r"Until-decay samples:\s*(\d+)", output)
    assert match is not None
    decay_samples = int(match.group(1))
    assert 100 <= decay_samples < 1_200

    assert re.search(r"Probe time_series shape: \(\d+, 1\)", output)
    assert "Final field slice shapes: Ex=(41, 41), Ey=(41, 41), Ez=(41, 41)" in output
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


def test_rcs_scattering_tutorial_runs():
    """The sphere run subtracts its empty reference and reports backscatter."""
    output = _run_tutorial("rcs_scattering.py")

    assert "[PREFLIGHT] All checks passed" in output
    assert "TFSF plane-wave setup ready: True" in output
    assert "Incident-reference subtraction enabled: True" in output
    assert "Deep-null limitation:" in output

    backscatter_match = re.search(r"Backscatter RCS:\s*([^\s]+)\s*m\^2", output)
    area_match = re.search(
        r"Geometric-optics limit pi\*r\^2:\s*([^\s]+)\s*m\^2",
        output,
    )
    assert backscatter_match is not None
    assert area_match is not None

    backscatter = float(backscatter_match.group(1))
    geometric_optics = float(area_match.group(1))
    assert math.isfinite(backscatter) and backscatter > 0.0
    assert math.isfinite(geometric_optics) and geometric_optics > 0.0
    assert 0.1 < backscatter / geometric_optics < 20.0


def test_resonance_harminv_tutorial_runs():
    """The longer cavity record improves the analytic TE101 frequency."""
    output = _run_tutorial("resonance_harminv.py")

    assert "[PREFLIGHT] All checks passed" in output
    assert "Vacuum-cavity loss advisory observed: False" in output
    assert "Short record full mode list:" in output
    assert "Long record full mode list:" in output
    assert output.count("amplitude=") >= 4
    assert "Mode selection: nearest analytic frequency, not strongest amplitude" in output
    assert "Harminv sampling: decimate='auto' is the default" in output

    short_match = re.search(r"Short-record TE101 error:\s*([^%]+)%", output)
    long_match = re.search(r"Long-record TE101 error:\s*([^%]+)%", output)
    assert short_match is not None
    assert long_match is not None

    short_error = float(short_match.group(1))
    long_error = float(long_match.group(1))
    assert math.isfinite(short_error) and short_error < 0.5
    assert math.isfinite(long_error) and long_error < short_error
