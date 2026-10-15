"""Consumers take allocated pads from the boundary record."""
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.depths import grid_face_depths, simulation_face_depths
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.grid import Grid
from rfx.runners._distributed_common import _distributed_boundary_layers


@pytest.mark.parametrize("nonuniform", [False, True])
def test_waveguide_preflight_matches_its_grid_pads(nonuniform):
    kwargs = dict(dz_profile=np.full(16, 1e-3)) if nonuniform else {}
    sim = Simulation(freq_max=20e9, domain=(0.032, 0.016, 0.016), dx=1e-3,
                     cpml_layers=4, boundary="cpml", **kwargs)
    sim.add_waveguide_port(0.008, direction="+x", f0=12e9, n_freqs=3,
                           probe_offset=2, ref_offset=1)
    grid = sim._build_realized_grid()
    expected = {face: getattr(grid, f"pad_{face}")
                for face in (f"{a}_{s}" for a in "xyz" for s in ("lo", "hi"))}
    # §7 Addendum 3: diagnostics describe realized waveguide pads, not declared absorbers.
    assert sim._preflight_face_layers() == expected
    assert {face.name: face.realized for face in simulation_face_depths(sim)} == expected
    assert expected["y_lo"] == (4 if nonuniform else 0)


def test_distributed_sizing_uses_nonabsorbing_axis_and_asymmetric_record():
    grid = Grid(freq_max=20e9, domain=(0.064, 0.016, 0.016), dx=1e-3,
                cpml_layers=16, cpml_axes="x", face_layers={"x_lo": 8, "x_hi": 16})
    expected = {face.name: face.realized for face in grid.boundary_depths}
    assert _distributed_boundary_layers(grid, 2) == expected
    assert expected == dict(x_lo=8, x_hi=16, y_lo=0, y_hi=0, z_lo=0, z_hi=0)


def test_wall_override_preserves_the_absorber_on_the_opposite_face():
    grid = Grid(freq_max=20e9, domain=(0.064, 0.016, 0.016), dx=1e-3,
                cpml_layers=16, face_layers={"x_lo": 8, "x_hi": 16})
    depths = {face.name: face.realized for face in grid_face_depths(grid, pec_faces={"x_lo"})}
    assert depths["x_lo"] == 0
    assert depths["x_hi"] == 16


def test_asymmetric_preflight_preserves_declared_depth_on_absorbing_faces():
    sim = Simulation(freq_max=20e9, domain=(0.064, 0.016, 0.016), dx=1e-3,
                     cpml_layers=16, boundary=BoundarySpec(
                         x=Boundary(lo="cpml", hi="cpml", lo_thickness=8, hi_thickness=16),
                         y="pec", z="pmc"))
    assert sim._preflight_face_layers() == dict(x_lo=8, x_hi=16, y_lo=0, y_hi=0, z_lo=0, z_hi=0)


def test_distributed_all_wall_scan_keeps_grid_allocation_budget():
    grid = Grid(freq_max=20e9, domain=(0.064, 0.016, 0.016), dx=1e-3,
                cpml_layers=16)
    walls = {face.name for face in grid.boundary_depths}
    assert _distributed_boundary_layers(grid, 2, pec_faces=walls, cpml_layers=0) == dict.fromkeys(walls, 0)


def test_waveguide_does_not_warn_about_unallocated_transverse_absorbers():
    transverse = Boundary(lo="cpml", hi="cpml", lo_thickness=1, hi_thickness=1)
    sim = Simulation(freq_max=14e9, domain=(0.064, 0.012, 0.002), dx=1e-3,
                     cpml_layers=16, boundary=BoundarySpec(x="cpml", y=transverse, z=transverse))
    sim.add_waveguide_port(0.010, direction="+x", f0=13.5e9, freqs=np.array([13.5e9]),
                           probe_offset=2, ref_offset=1)
    # §7 Addendum 3: the uniform guide allocates no transverse absorbers to warn about.
    report = sim.preflight()
    assert not any(issue.severity == "error" for issue in report), report
    assert "absorber_budget_exceeds_axis" not in {issue.code for issue in report}


@pytest.mark.parametrize("model,nonuniform", [
    (model, graded) for model in ("waveguide", "periodic", "pmc", "asymmetric")
    for graded in (False, True)
] + [("floquet", False), ("tmz", False)])
def test_preflight_face_pads_match_realized_model(model, nonuniform):
    # Floquet and 2-D stepping do not admit graded meshes; test their uniform realization.
    kwargs = dict(dz_profile=np.linspace(0.8e-3, 1.2e-3, 16)) if nonuniform else {}
    boundary = {
        "periodic": BoundarySpec(x="periodic", y="cpml", z="cpml"),
        "pmc": BoundarySpec(x=Boundary("pmc", "cpml"), y="cpml", z="cpml"),
        "asymmetric": BoundarySpec(x=Boundary("cpml", "cpml", 2, 4), y="pec", z="cpml"),
    }.get(model, "cpml")
    sim = Simulation(freq_max=20e9, domain=(0.032, 0.016, 0.016), dx=1e-3,
                     cpml_layers=4, boundary=boundary,
                     mode="2d_tmz" if model == "tmz" else "3d", **kwargs)
    if model == "waveguide":
        sim.add_waveguide_port(0.008, direction="+x", f0=12e9, n_freqs=3,
                              probe_offset=2, ref_offset=1)
    elif model == "floquet":
        sim.add_floquet_port(position=0.005, axis="z", f0=12e9)
    grid = sim._build_realized_grid()
    # §7 Addenda 3/4: advisories use actual pads, including feature-imposed periodic faces.
    assert sim._preflight_face_layers() == {
        f"{axis}_{side}": getattr(grid, f"pad_{axis}_{side}")
        for axis in "xyz" for side in ("lo", "hi")}
