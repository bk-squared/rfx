"""The run's TF/SF topology is also the topology preflight judges."""

from math import pi

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._internal._precision import resolve_coefficient_dtype
from rfx.boundaries import tfsf
from rfx.boundaries.depths import Kind, simulation_face_depths
from rfx.materials.lorentz import lorentz_pole
from rfx.measurement.flux_weights import monitor_periodic_flags
from rfx.sources.tfsf import tfsf_from_declaration


def _uniform_repro():
    return Simulation(freq_max=16e9, domain=(0.2, 0.008, 0.001), dx=1e-3,
                      boundary="cpml", cpml_layers=10, mode="2d_tmz")


def _source(sim, **kwargs):
    sim.add_tfsf_source(f0=9e9, bandwidth=0.9, polarization="ez",
                        direction="+x", **kwargs)
    return sim


@pytest.mark.parametrize("case", ["repro", "normal_3d", "method_b", "closed_box"])
def test_run_and_campaign_flags_match_initialized_stepper(case):
    """Pin 1: red on old open cases; closed-box flags already agree."""
    sim = _uniform_repro() if case == "repro" else Simulation(
        freq_max=16e9, domain=(0.08, 0.016, 0.016), dx=1e-3,
        boundary="cpml", cpml_layers=10)
    _source(sim, angle_deg=30 if case == "method_b" else 0,
            method="methodB" if case == "method_b" else "bloch",
            closed_box=case == "closed_box")
    grid = sim._build_realized_grid()
    cfg, _ = tfsf_from_declaration(
        sim._tfsf, grid, sim._freq_max, bounds_only=True,
        dtype=resolve_coefficient_dtype(sim._precision))
    expected = tfsf.boundary_flags(cfg, grid)[0]
    assert monitor_periodic_flags(sim, grid) == expected
    assert sim._campaign_ctx().periodic == expected
    assert tfsf.run_periodic_flags(sim) == expected


def test_uniform_wrapped_faces_have_no_geometry_in_absorber():
    """Pin 2: red on old code, which diagnoses the allocated y pads."""
    sim = _uniform_repro()
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((0.095, -1.0, -1.0), (0.105, 1.0, 1.0)), material="d")
    _source(sim)
    report = sim.preflight()
    assert report.by_code("geometry_in_absorber") == []
    assert len(report.by_code("tfsf_transverse_periodic")) == 1
    faces = {face.name: face for face in simulation_face_depths(sim)}
    for side in ("lo", "hi"):
        assert (faces[f"y_{side}"].kind, faces[f"y_{side}"].realized) == (Kind.PERIODIC, 0)
        assert (faces[f"x_{side}"].kind, faces[f"x_{side}"].realized) == (Kind.ABSORBER, 10)


def test_uniform_lorentz_slab_has_no_pole_at_wrapped_absorber_face():
    """Pin 3: red on old code for the oracle's uniform Lorentz slab."""
    sim = Simulation(freq_max=18e9, domain=(0.12, 0.005, 0.5e-3),
                     dx=0.5e-3, boundary="cpml", cpml_layers=20, mode="2d_tmz")
    sim.add_material("slab", eps_r=2.0,
                     lorentz_poles=[lorentz_pole(1.0, 2 * pi * 10e9, 0.1 * 2 * pi * 10e9)])
    sim.add(Box((0.06 - 0.008 / 2, -1.0, -1.0),
                (0.06 + 0.008 / 2, 1.0, 1.0)), material="slab")
    sim.add_tfsf_source(f0=10e9, bandwidth=0.7, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    assert sim.preflight().by_code("dispersive_pole_at_absorber_face") == []


def test_graded_tfsf_keeps_declared_y_absorbers():
    """Pin 4: green on old code; red if uniform wrapping leaks to graded meshes."""
    dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
    sim = Simulation(freq_max=8e9, domain=(0.08, 0.006, float(np.sum(dz))),
                     boundary="cpml", cpml_layers=8, dx=1e-3, dz_profile=dz)
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((0.038, -1.0, 0.0012), (0.042, 1.0, 0.0028)), material="d")
    sim.add_tfsf_source(f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3,
                        polarization="ez", direction="+x")
    rows = sim.preflight().by_code("geometry_in_absorber")
    assert len(rows) == 1
    assert "y" in str(rows[0])
    faces = {face.name: face for face in simulation_face_depths(sim)}
    for side in ("lo", "hi"):
        assert (faces[f"y_{side}"].kind, faces[f"y_{side}"].realized) == (Kind.ABSORBER, 8)
    assert tfsf.run_periodic_flags(sim) == sim._periodic_flags()


def test_wrapped_pec_face_one_cell_from_interior_wall_has_no_wall_gap():
    """Pin 5: green on old code; red for zero depth with unwrapped ctx.periodic.

    The run itself refuses this model (a PEC box finite along the wrapped
    axes); the pin holds the reader: a zero depth on a wrapped face is no wall.
    """
    from rfx.boundaries.pec import realized_wall_planes

    sim = Simulation(freq_max=16e9, domain=(0.08, 0.012, 0.012), dx=1e-3,
                     boundary="cpml", cpml_layers=8)
    sim.add(Box((0.038, 0.001, 0.004), (0.042, 0.004, 0.008)), material="pec")
    _source(sim)
    ctx = sim._campaign_ctx()
    assert ctx.error is None
    entry, = [entry for entry in ctx.interior_pec_entries() if entry.kind == "volume"]
    planes = realized_wall_planes(entry.edges(sim._periodic_flags(), tuple(ctx.grid.shape)), 1)
    assert min(planes) - int(ctx.grid.interior[1].start) == 1
    assert sim.preflight().by_code("pec_face_short_of_domain_wall") == []
