"""Absorption advisories leave the solved fields unchanged.

Vacuum boxes use unequal absorbing depths and non-integral cell lengths.
Both mesh lanes must report the realized thin faces, including a zero-depth
reflecting edge, without treating a PEC, PMC or periodic face as an absorber.
"""

import re

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def _vacuum(*, graded, token="cpml", depths=(4, 16, 0, 8), z="pec"):
    # A declared periodic axis must be a whole number of cells (#1221 B2
    # refuses an explicit dx that does not divide its period); the other
    # axes stay non-integral on purpose.
    lz = 0.009 if z == "periodic" else 0.0091
    return Simulation(
        freq_max=3e9, domain=(0.0123, 0.0107, lz), dx=1e-3,
        dz_profile=np.linspace(0.0007, 0.00132, 9) if graded else None,
        cpml_layers=16,
        boundary=BoundarySpec(
            x=Boundary(token, token, depths[0], depths[1]),
            y=Boundary(token, token, depths[2], depths[3]),
            z=z,
        ),
    )


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("token", ["cpml", "upml"])
def test_asymmetric_realized_depths(graded, token):
    sim = _vacuum(graded=graded, token=token)
    grid = sim._build_realized_grid()
    assert (grid.pad_x_lo, grid.pad_x_hi, grid.pad_y_lo, grid.pad_y_hi) == (4, 16, 0, 8)
    # The scalar is deliberately silent; reading it instead of the pads must fail.
    assert sim._cpml_layers == 16
    report = sim.preflight()
    found = report.by_code("thin_absorber")
    assert len(found) == 1
    issue = found[0]
    assert issue.severity == "warning"
    assert set(issue.loc.split(", ")) == {"x_lo", "y_lo"}
    named = re.findall(r"face ([xyz]_(?:lo|hi)) is declared (\w+) with (\d+) layers", issue)
    assert named == [("x_lo", token, "4"), ("y_lo", token, "0")]
    assert set(re.findall(r"[xyz]_(?:lo|hi)", issue)) == {"x_lo", "y_lo"}
    assert "0 layers; it is a reflecting edge, not an absorber" in issue
    assert "1 mm cells" in issue
    assert "at least 8" in issue and "16 layers" in issue
    assert report.by_code("conductor_in_thin_absorber") == []


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("token", ["cpml", "upml"])
def test_eight_and_sixteen_layers_are_silent(graded, token):
    sim = _vacuum(graded=graded, token=token, depths=(8, 16, 16, 8), z=token)
    assert sim.preflight().by_code("thin_absorber") == []


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("token", ["pec", "pmc", "periodic"])
def test_nonabsorbing_faces_are_silent_at_zero_depth(graded, token):
    sim = _vacuum(graded=graded, depths=(8, 16, 8, 16), z=token)
    grid = sim._build_realized_grid()
    assert (grid.pad_z_lo, grid.pad_z_hi) == (0, 0)
    assert sim.preflight().by_code("thin_absorber") == []


@pytest.mark.parametrize("depth,rows", [
    (1, {4}), (2, {4}), (3, {4}), (4, {4}),
    (5, {4, 6}), (6, {6}), (7, {6, 8}),
])
def test_measured_reflections_are_quoted_without_interpolation(depth, rows):
    sim = _vacuum(graded=False, depths=(depth, 16, 8, 16))
    issue, = sim.preflight().by_code("thin_absorber")
    # These values come from the leader's measurement table, not the check's
    # constants. Extract the microwave quantities instead of locking prose.
    values = re.findall(
        r"(\d+) layers: (-?\d+) dB at 2 GHz, (-?\d+) dB at 10 GHz, "
        r"(-?\d+ dB|not measured) at 30 GHz, (-?\d+) dB below 1 GHz", issue)
    # No 8-layer record reaches 30 GHz (the direct runs stop at 20 GHz).
    reference = {4: (-20, -23, -17, -6), 6: (-38, -41, -32, -13),
                 8: (-60, -61, None, -21)}
    got = {int(n): tuple(None if r == "not measured" else int(r.split()[0])
                         for r in reflections) for n, *reflections in values}
    assert got == {n: reference[n] for n in rows}
    assert "1 mm cells" in issue
    if depth not in (4, 6):
        assert "nearest measured depths" in issue and "no interpolation" in issue


def test_zero_scalar_still_reports_absorbing_declarations():
    sim = Simulation(freq_max=3e9, domain=(0.0123, 0.0107, 0.0091),
                     dx=1e-3, cpml_layers=0, boundary="cpml")
    issue, = sim.preflight().by_code("thin_absorber")
    assert set(issue.loc.split(", ")) == {f"{ax}_{side}" for ax in "xyz" for side in ("lo", "hi")}
    assert issue.count("with 0 layers") == 6


def test_collapsed_2d_axis_has_no_absorbing_face():
    sim = Simulation(freq_max=3e9, domain=(0.0123, 0.0107, 0.001),
                     dx=1e-3, mode="2d_tmz", cpml_layers=8, boundary="cpml")
    assert sim.preflight().by_code("thin_absorber") == []


def test_realized_pads_are_read_even_when_the_declaration_is_thick(monkeypatch):
    sim = _vacuum(graded=False, depths=(16, 16, 16, 16))
    grid = sim._build_realized_grid()
    # Isolate depth selection on an active absorber axis. Real axis suppression
    # is covered by the public Floquet and waveguide cases below.
    grid.pad_x_lo = 0
    monkeypatch.setattr(sim, "_build_realized_grid", lambda: grid)
    issue, = sim.preflight().by_code("thin_absorber")
    assert issue.loc == "x_lo"
    assert "face x_lo is declared cpml with 0 layers" in issue


def _assert_thin_faces(sim, expected):
    issues = sim.preflight().by_code("thin_absorber")
    if not expected:
        assert issues == []
        return
    issue, = issues
    assert set(issue.loc.split(", ")) == set(expected)
    assert {face: int(depth) for face, depth in re.findall(
        r"face ([xyz]_(?:lo|hi)) is declared cpml with (\d+) layers", issue
    )} == expected
    assert set(re.findall(r"[xyz]_(?:lo|hi)", issue)) == set(expected)


@pytest.mark.parametrize("thin", [False, True], ids=["default", "thin"])
def test_floquet_periodic_faces_are_not_absorbers(thin):
    boundary = (BoundarySpec(x="cpml", y="cpml", z=Boundary("cpml", "cpml", 4, 16))
                if thin else "cpml")
    # The Floquet axes x/y are periodic: whole numbers of cells (11 and 13,
    # unequal on purpose), as #1221 B2 requires for an explicit dx.
    sim = Simulation(freq_max=10e9, domain=(0.011, 0.013, 0.0301),
                     dx=1e-3, boundary=boundary)
    sim.add_floquet_port(position=0.005, axis="z", f0=5e9)
    grid = sim._build_realized_grid()
    assert grid.cpml_axes == "z"
    assert grid.face_layers["x_lo"] == 16 and grid.pad_x_lo == 0
    _assert_thin_faces(sim, {"z_lo": 4} if thin else {})


@pytest.mark.parametrize("thin", [False, True], ids=["default", "thin"])
def test_waveguide_pec_walls_are_not_absorbers(thin):
    boundary = (BoundarySpec(x=Boundary("cpml", "cpml", 4, 16), y="cpml", z="cpml")
                if thin else "cpml")
    sim = Simulation(freq_max=12e9, domain=(0.0803, 0.02286, 0.01016),
                     dx=1e-3, boundary=boundary)
    sim.add_waveguide_port(x_position=0.024, direction="+x",
                          y_range=(0, 0.02286), z_range=(0, 0.01016),
                          f0=10e9, freqs=np.linspace(8e9, 11.5e9, 8))
    grid = sim._build_realized_grid()
    assert grid.cpml_axes == "x"
    assert grid.face_layers["y_lo"] == 16 and grid.pad_y_lo == 0
    _assert_thin_faces(sim, {"x_lo": 4} if thin else {})


@pytest.mark.parametrize("kind,expected", [
    ("normal", {"x_lo": 6}),
    ("bloch", {"x_lo": 6}),
    ("methodB", {"x_lo": 6, "y_lo": 4}),
    ("closed", {"x_lo": 6, "y_lo": 4, "z_lo": 6}),
])
def test_tfsf_names_only_faces_that_absorb_in_the_uniform_run(kind, expected):
    sim = Simulation(
        freq_max=30e9, domain=(0.0483, 0.0427, 0.0381), dx=1e-3,
        cpml_layers=16,
        boundary=BoundarySpec(x=Boundary("cpml", "cpml", 6, 16),
                              y=Boundary("cpml", "cpml", 4, 8),
                              z=Boundary("cpml", "cpml", 6, 16)),
    )
    sim.add_tfsf_source(
        f0=15e9, bandwidth=0.15, margin=3,
        angle_deg=20 if kind in ("bloch", "methodB") else 0,
        method="methodB" if kind == "methodB" else "bloch",
        closed_box=kind == "closed",
    )
    _assert_thin_faces(sim, expected)


def test_nonuniform_tfsf_does_not_inherit_uniform_periodic_override():
    sim = _vacuum(graded=True, depths=(6, 16, 4, 8), z="cpml")
    sim.add_tfsf_source(f0=1.5e9, margin=3)
    _assert_thin_faces(sim, {"x_lo": 6, "y_lo": 4})


@pytest.mark.parametrize("mode", ["3d", "2d_tmz", "2d_tez"])
def test_six_layer_tfsf_keeps_only_its_longitudinal_advisory(mode):
    sim = Simulation(
        freq_max=30e9, domain=(0.0243, 0.0187, 0.0141 if mode == "3d" else 0.001),
        dx=1e-3, cpml_layers=6, mode=mode,
        boundary=BoundarySpec(x=Boundary("cpml", "cpml", 4, 6), y="cpml", z="cpml"),
    )
    sim.add_tfsf_source(f0=15e9, polarization="ey" if mode == "2d_tez" else "ez")
    _assert_thin_faces(sim, {"x_lo": 4, "x_hi": 6})


@pytest.mark.parametrize("error", [AttributeError, KeyError])
def test_unavailable_grid_does_not_guess_depth(monkeypatch, error):
    sim = _vacuum(graded=False)

    def unavailable():
        raise error("grid unavailable")

    monkeypatch.setattr(sim, "_build_realized_grid", unavailable)
    # Call just this check: other preflight checks own their error handling.
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        sim._validate_cfg_thin_absorber(warnings, sim._dx)
    assert caught == []


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_conductor_overlap_is_referenced_once_without_repeating_its_evidence(graded):
    sim = _vacuum(graded=graded, depths=(6, 16, 8, 16))
    # Two conductors reach just the thin x_lo face. Existing per-conductor
    # findings stay separate; their overlap with the depth warning is named once.
    for y in (0.003, 0.006):
        sim.add(Box((0, y, 0.003), (0.001, y + 0.001, 0.004)), material="pec")
    report = sim.preflight()
    issue, = report.by_code("thin_absorber")
    conductor = report.by_code("conductor_in_thin_absorber")
    assert len(conductor) == (0 if graded else 2)
    assert issue.count("is also reported by conductor_in_thin_absorber") == (0 if graded else 1)
    assert "ring-down" not in issue
    assert all("0 cell(s) of clearance" in finding for finding in conductor)
