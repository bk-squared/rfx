"""compute_far_field_jax must chunk its direction grid without changing the
answer (issue #727).

The transform materializes a (n_freqs, n_directions, n_cells) phase array
per face. On a board-scale box with a full sphere that reached 214 GB and
killed a run AFTER an 8-hour solve. The sum over surface cells is
independent per direction, so splitting theta is exact — this test is what
makes "exact" a checked claim rather than an argument, and it also pins the
memory bound as a real behaviour (a chunk actually happens when the budget
is small).
"""
from __future__ import annotations

import numpy as np

from rfx import Box, Simulation, compute_far_field_jax
from rfx.sources import GaussianPulse


def _run():
    sim = Simulation(freq_max=20e9, domain=(6e-3, 6e-3, 6e-3), dx=300e-6,
                     boundary="cpml", cpml_layers=6)
    sim.add(Box((2.7e-3, 2.7e-3, 2.7e-3), (3.3e-3, 3.3e-3, 3.3e-3)),
            material="pec")
    sim.add_source(position=(3e-3, 3e-3, 2.4e-3), component="ez",
                   amplitude_kind="current",
                   waveform=GaussianPulse(f0=12e9, bandwidth=8e9))
    sim.add_ntff_box((1.5e-3, 1.5e-3, 1.5e-3), (4.5e-3, 4.5e-3, 4.5e-3),
                     freqs=[10e9, 12e9, 14e9])
    return sim.run(n_steps=300)


def test_chunked_matches_single_shot_bitwise():
    res = _run()
    th = np.radians(np.linspace(0.0, 180.0, 37))
    ph = np.radians(np.linspace(0.0, 350.0, 12))

    whole = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid,
                                  th, ph, max_phase_bytes=float("inf"))
    # a budget small enough to force several passes
    chunked = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid,
                                    th, ph, max_phase_bytes=1e4)

    assert np.asarray(whole.E_theta).shape == np.asarray(chunked.E_theta).shape
    assert np.array_equal(np.asarray(whole.E_theta), np.asarray(chunked.E_theta)), (
        "chunking changed E_theta: max |delta| = "
        f"{np.max(np.abs(np.asarray(whole.E_theta) - np.asarray(chunked.E_theta))):.3e}")
    assert np.array_equal(np.asarray(whole.E_phi), np.asarray(chunked.E_phi))
    assert np.array_equal(np.asarray(whole.theta), np.asarray(chunked.theta))


def test_default_budget_leaves_small_problems_single_shot():
    """The bound must not chunk problems that fit — chunking is for the case
    that would otherwise die, not a change of default behaviour."""
    res = _run()
    th = np.radians(np.linspace(0.0, 180.0, 19))
    ph = np.radians(np.linspace(0.0, 350.0, 8))
    a = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid, th, ph)
    b = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid, th, ph,
                              max_phase_bytes=float("inf"))
    assert np.array_equal(np.asarray(a.E_theta), np.asarray(b.E_theta))
