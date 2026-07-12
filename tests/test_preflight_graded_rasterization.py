"""Preflight advisory for Boxes displaced from a graded-mesh fine band."""

from __future__ import annotations

import numpy as np

from rfx import Box, Simulation


def _issues(sim):
    return sim.preflight()


def _has(issues, substring):
    return any(substring in issue for issue in issues)


def _graded_sim(z_lo: float, z_hi: float) -> Simulation:
    dz = np.array([1.0e-3, 1.0e-3] + [0.25e-3] * 6 + [1.0e-3] * 2)
    sim = Simulation(
        freq_max=10e9,
        domain=(20e-3, 20e-3, float(np.sum(dz))),
        dx=1e-3,
        dz_profile=dz,
        cpml_layers=2,
    )
    sim.add_material("substrate", eps_r=3.5)
    sim.add(Box((5e-3, 5e-3, z_lo), (15e-3, 15e-3, z_hi)),
            material="substrate")
    sim.add_source((10e-3, 10e-3, 1e-3), "ez")
    return sim


def test_shifted_box_warns_with_actual_and_implied_counts():
    sim = _graded_sim(0.6e-3, 2.1e-3)

    issues = _issues(sim)

    assert _has(issues, "rasterizes to 1 z cells (implied 6.0)"), issues
    assert _has(issues, "smooth_grading transition cells may have shifted"), issues


def test_box_pinned_to_actual_fine_band_is_silent():
    sim = _graded_sim(2.0e-3, 3.5e-3)

    assert not _has(_issues(sim), "smooth_grading transition cells may have shifted")


def test_uniform_dz_simulation_skips_check():
    sim = Simulation(
        freq_max=10e9,
        domain=(20e-3, 20e-3, 6e-3),
        dx=1e-3,
        cpml_layers=2,
    )
    sim.add_material("substrate", eps_r=3.5)
    sim.add(Box((5e-3, 5e-3, 0.5e-3), (15e-3, 15e-3, 2.0e-3)),
            material="substrate")
    sim.add_source((10e-3, 10e-3, 1e-3), "ez")

    assert not _has(_issues(sim), "smooth_grading transition cells may have shifted")
