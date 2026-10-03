"""End-to-end checks for the materials and far-field tutorials."""

from __future__ import annotations

import math
import re
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_DIR = REPO_ROOT / "examples" / "tutorials"


def _run_tutorial(name: str) -> str:
    path = TUTORIALS_DIR / name
    assert path.exists(), f"missing tutorial: {path}"

    # PYTHONPATH is not optional here.  Running a script BY PATH puts the
    # SCRIPT'S directory on sys.path, not ``cwd``, so ``import rfx`` in the
    # child resolves to whatever ``rfx`` is INSTALLED -- on this pod a path
    # install pointing at a different checkout.  Without this the test runs
    # the tutorial source from THIS tree against SOMEONE ELSE'S rfx, and
    # reports the result as if it were this checkout's.
    #
    # It stayed invisible until a tutorial used a symbol that exists only
    # here: ports_and_sparams_101 imports ``realized_pec_edge_masks`` (#931)
    # and the child raised ImportError against the installed copy.  Every
    # earlier green in this file was measured against the installed rfx.
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH") else str(REPO_ROOT)
    )

    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
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


def _reported_absorber_layers(output: str) -> set[tuple[str, int]]:
    return {
        (face, int(depth))
        for face, depth in re.findall(
            r"face ([xyz]_(?:lo|hi)) is declared (?:cpml|upml) with (\d+) layers",
            output,
        )
    }


def _assert_only_absorber_reports(output: str, minimum: int):
    reports = [line.split("[PREFLIGHT] ", 1)[1]
               for line in output.splitlines() if "[PREFLIGHT] " in line]
    assert len(reports) >= minimum
    assert all(line.startswith("face ") and " is declared cpml with " in line
               for line in reports), reports


def test_antenna_farfield_pattern_tutorial_runs():
    """The dipole reports its warning, directivity, and a fresh E-plane cut:
    the plot, and the cut's samples in degrees (#1271)."""
    plot_path = TUTORIALS_DIR / "output" / "short_dipole_e_plane.png"
    plot_path.unlink(missing_ok=True)
    samples_path = TUTORIALS_DIR / "output" / "short_dipole_e_plane.csv"
    samples_path.unlink(missing_ok=True)

    output = _run_tutorial("antenna_farfield_pattern.py")

    assert "Close-box advisory observed: True" in output
    assert "Corrected face spacing:" in output
    assert _reported_absorber_layers(output) == {
        (face, 6) for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
    }
    # After the explicit corrected report, run() prints its own advisory tier.
    _assert_only_absorber_reports(output.split("Corrected face spacing:", 1)[1], 1)
    match = re.search(r"Peak directivity:\s*([^\s]+)\s*dBi", output)
    assert match is not None
    peak_directivity_dbi = float(match.group(1))
    assert math.isfinite(peak_directivity_dbi)
    assert abs(peak_directivity_dbi - 1.76) < 0.3
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0

    # The cut's samples: one row per polar angle of the tutorial's 73-point
    # theta grid, in degrees, on phi = 0 only. The last column is absolute
    # IEEE gain; for this lossless dipole its peak matches the directivity
    # reported above (broadside, near theta = 90 degrees).
    import numpy as np

    header = samples_path.read_text().splitlines()[0]
    assert header == ("theta_deg,phi_deg,E_theta_mag,E_theta_phase_deg,"
                      "E_phi_mag,E_phi_phase_deg,gain_dBi")
    rows = np.loadtxt(samples_path, delimiter=",", skiprows=1)
    assert rows.shape == (73, 7)
    np.testing.assert_allclose(
        rows[:, 0], np.degrees(np.linspace(0.01, np.pi - 0.01, 73)), rtol=1e-6)
    assert np.all(rows[:, 1] == 0.0)
    # The console rounds to 0.001 dB; CSV rounding is below 1e-6 dB here.
    # #1369 changes this column's old 0 dB peak to about 1.76 dBi.
    assert abs(rows[:, 6].max() - peak_directivity_dbi) < 5.1e-4
    assert 75.0 <= rows[np.argmax(rows[:, 6]), 0] <= 105.0


def test_ports_and_sparams_101_tutorial_runs():
    """Every port family preflights and the live RLC load changes S11."""
    output = _run_tutorial("ports_and_sparams_101.py")

    # The generic-port and microstrip examples retain their four- and
    # three-layer absorbers; both depths must remain visible in the report.
    assert _reported_absorber_layers(output) == {
        (face, depth)
        for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
        for depth in (3, 4)
    }
    # #1138: edge-aware x/y registration removes the sheet-size refusal.
    assert "solved more than 1% off their drawn size" not in output
    assert "Microstrip port setup ready: True" in output
    # The declared foils must BE the realized wall planes, and the gap between
    # them the height the MSL ports were told.  build_microstrip_ports() raises
    # if not; this pins the measured line so a silent plane move is visible.
    assert (
        "Microstrip realized conductor planes along z: [8, 12] "
        "(ground 1.25 mm, trace 2.25 mm, strip-to-ground 1.00 mm"
    ) in output
    assert "Waveguide port setup ready: True" in output
    # The waveguide setup audits are part of what this tutorial teaches, and
    # this small model draws both of them (2.4 far-boundary round trips at the
    # default num_periods, plus the informational E-plane offset). Readiness is
    # report.ok, so those advisories must show WITHOUT flipping it to False.
    assert "record_shorter_than_far_boundary_round_trip" in output
    assert "port_index_mirror_known_e_plane_offset" in output
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

    assert _reported_absorber_layers(output) == {
        (face, 4) for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
    }
    _assert_only_absorber_reports(output, 1)
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
