"""A 2-D ``mode`` on a mesh that resolves non-uniform.

The non-uniform lane never reads ``mode``: it runs the 3-D update on the
thin box a 2-D model declares. ``2d_tez`` came back with every field zero.
The refusal sits at lane dispatch, on the RESOLVED mesh, so it also covers a
profile that auto-meshing invented and one assigned after construction;
construction itself stays legal (a design-document round trip rebuilds a
simulation from its resolved mode and profiles).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx import Box
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

A, B, DX = 0.020, 0.010, 1e-3
DY = np.full(10, DX)


def _run(sim, component):
    sim.add_source((0.0063, 0.0031, 0.0), component=component,
                   amplitude_kind="field")
    sim.add_probe((0.0137, 0.0069, 0.0), component=component)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(sim.run(n_steps=200, compute_s_params=False)
                          .time_series)[:, 0]


def test_tez_with_a_profile_is_refused_at_run_not_at_construction():
    sim = Simulation(freq_max=30e9, domain=(A, B, DX), dx=DX, boundary="pec",
                     mode="2d_tez", dy_profile=DY)      # constructs
    with pytest.raises(ValueError, match="solves 3-D only"):
        _run(sim, "ey")


def test_tez_on_an_auto_meshed_nonuniform_grid_is_refused_too():
    """No dx= and a thin z feature: auto-meshing invents a dz_profile."""
    sim = Simulation(freq_max=15e9, domain=(0.020, 0.020, 0.004),
                     boundary="pec", mode="2d_tez")
    sim.add_material("slab", eps_r=4.0)
    sim.add(Box((0.0, 0.0, 0.0), (0.020, 0.020, 0.0002)), material="slab")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sim._uses_nonuniform_mesh, \
            "the fixture must reach the auto-meshed non-uniform lane"
    with pytest.raises(ValueError, match="auto-meshing produced one"):
        _run(sim, "ey")


def test_tmz_with_a_profile_and_a_non_pec_z_wall_is_refused():
    sim = Simulation(freq_max=30e9, domain=(A, B, DX), dx=DX, mode="2d_tmz",
                     boundary=BoundarySpec(x="pec", y="pec",
                                           z=Boundary(lo="pmc", hi="pmc")),
                     dy_profile=DY)
    with pytest.raises(ValueError, match="solves 3-D only"):
        _run(sim, "ez")


def test_tmz_between_pec_walls_runs_and_equals_the_three_d_box():
    kw = dict(freq_max=30e9, domain=(A, B, DX), dx=DX, boundary="pec",
              dy_profile=DY)
    two_d = _run(Simulation(mode="2d_tmz", **kw), "ez")
    three_d = _run(Simulation(mode="3d", **kw), "ez")
    assert np.max(np.abs(two_d)) > 0.0
    np.testing.assert_array_equal(two_d, three_d)


def test_the_way_out_named_in_the_message_carries_tez_fields():
    """Two cells between magnetic z walls; one cell holds nothing."""
    def solve(cells):
        sim = Simulation(
            freq_max=30e9, domain=(A, B, cells * DX), dx=DX, mode="3d",
            boundary=BoundarySpec(x="pec", y="pec",
                                  z=Boundary(lo="pmc", hi="pmc")),
            dy_profile=DY)
        sim.add_source((0.0063, 0.0031, (cells - 1) * DX), component="ey",
                       amplitude_kind="field")
        sim.add_probe((0.0137, 0.0069, (cells - 1) * DX), component="ey")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(sim.run(n_steps=200, compute_s_params=False)
                              .time_series)[:, 0]
    assert np.max(np.abs(solve(2))) > 0.0
    assert np.max(np.abs(solve(1))) == 0.0


def test_a_two_d_mode_on_a_uniform_mesh_is_untouched():
    sim = Simulation(freq_max=30e9, domain=(A, B, DX), dx=DX, boundary="pec",
                     mode="2d_tez")
    assert np.max(np.abs(_run(sim, "ey"))) > 0.0
