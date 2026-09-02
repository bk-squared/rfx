"""Far-field on an in-plane graded mesh (#743).

Before this fix, `compute_far_field*` read the in-plane spacing as
`grid.dx` / `grid.dy` — the BOUNDARY cell sizes on a NonUniformGrid — so a
mesh graded in x or y had both its surface elements and its face
coordinates computed with the wrong spacing, and returned a pattern with
no warning.

Two properties are pinned:

1. a uniform-valued in-plane profile must reproduce the plain uniform
   grid's pattern (the profile is the identity there, so any difference is
   the plumbing, not physics — the #  "uniform-valued profile tests
   plumbing" lesson);
2. a genuinely graded in-plane mesh must integrate with the LOCAL cell
   sizes: the total radiated power over the closed box must match the
   uniform run's within the discretization envelope, whereas using the
   boundary cell everywhere (the old behaviour) mis-weights every face.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation, compute_far_field_jax
from rfx.sources import GaussianPulse

NF = [30e9]


def _sim(dx_profile=None, dy_profile=None):
    """A lambda/2-clean radiator: 30 GHz (lambda = 10 mm), 22 mm domain, a
    1.5 mm PEC scatterer at the centre and the NTFF box 5.75 mm away, so the
    transform's own validity condition is satisfied (checked: preflight is
    clean on this fixture). The first attempt at this test used a 9 mm
    domain at 12 GHz, where preflight said the box sits inside lambda/4 and
    the pattern is corrupted — comparing two corrupted patterns measures
    nothing."""
    kw = {}
    if dx_profile is not None:
        kw["dx_profile"] = dx_profile
    if dy_profile is not None:
        kw["dy_profile"] = dy_profile
    sim = Simulation(freq_max=40e9, domain=(22e-3, 22e-3, 22e-3), dx=250e-6,
                     boundary="cpml", cpml_layers=6, **kw)
    sim.add(Box((10.25e-3, 10.25e-3, 10.25e-3),
                (11.75e-3, 11.75e-3, 11.75e-3)), material="pec")
    sim.add_source(position=(11e-3, 11e-3, 9.5e-3), component="ez",
                   amplitude_kind="current",
                   waveform=GaussianPulse(f0=30e9, bandwidth=0.5))
    sim.add_ntff_box((4.5e-3, 4.5e-3, 4.5e-3), (17.5e-3, 17.5e-3, 17.5e-3),
                     freqs=NF)
    return sim


def _pattern(res):
    th = np.radians(np.linspace(0.0, 180.0, 25))
    ph = np.radians(np.linspace(0.0, 350.0, 12))
    ff = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid, th, ph)
    U = (np.abs(np.asarray(ff.E_theta)) ** 2
         + np.abs(np.asarray(ff.E_phi)) ** 2)
    dth, dph = np.gradient(th), np.gradient(ph)
    p_rad = np.sum(U * np.sin(th)[None, :, None] * dth[None, :, None]
                   * dph[None, None, :], axis=(1, 2))
    return U, p_rad


@pytest.mark.slow
def test_uniform_valued_inplane_profile_matches_the_plain_uniform_grid():
    """The profile is the identity here; a difference would be plumbing."""
    prof = np.full(88, 250e-6)  # 22 mm / 250 um — the identity profile
    u_res = _sim().run(n_steps=400)
    g_res = _sim(dx_profile=prof, dy_profile=prof).run(n_steps=400)
    Uu, Pu = _pattern(u_res)
    Ug, Pg = _pattern(g_res)
    rel = abs(float(Pg[0] - Pu[0])) / max(float(Pu[0]), 1e-30)
    assert rel < 5e-3, (
        f"uniform-valued in-plane profile changed the radiated power by "
        f"{100 * rel:.3f}% — the in-plane spacing is not being read per cell")
    denom = max(float(np.max(Uu)), 1e-30)
    assert float(np.max(np.abs(Ug - Uu))) / denom < 5e-3


@pytest.mark.slow
def test_graded_inplane_mesh_integrates_with_local_cell_sizes():
    """A graded mesh must not be integrated with the boundary cell size.

    The centre band is refined and the shoulders coarsened at constant total
    length, so the boundary cell — which the pre-#743 code used for x and y
    everywhere — is NOT the mean cell. The test computes the pattern BOTH
    ways from the same run data (hiding dx_arr/dy_arr reproduces the old
    behaviour exactly) and asserts the local-cell integration is the one
    that agrees with the uniform mesh. That comparison is the discriminator;
    the absolute envelope is secondary and stated from measurement.
    """
    prof = np.concatenate([np.full(30, 250e-6), np.full(8, 312.5e-6),
                           np.full(16, 125e-6), np.full(8, 312.5e-6),
                           np.full(30, 250e-6)])
    assert abs(prof.sum() - 22e-3) < 1e-12, prof.sum()

    u_res = _sim().run(n_steps=600)
    g_res = _sim(dx_profile=prof, dy_profile=prof).run(n_steps=600)
    _, Pu = _pattern(u_res)
    _, Pg = _pattern(g_res)

    class _NoInPlaneArrays:
        """The pre-#743 view of the grid: in-plane spacing as a scalar."""

        def __init__(self, grid):
            self._g = grid

        def __getattr__(self, name):
            if name in ("dx_arr", "dy_arr"):
                raise AttributeError(name)
            return getattr(self._g, name)

    th = np.radians(np.linspace(0.0, 180.0, 25))
    ph = np.radians(np.linspace(0.0, 350.0, 12))
    ff_old = compute_far_field_jax(g_res.ntff_data, g_res.ntff_box,
                                   _NoInPlaneArrays(g_res.grid), th, ph)
    U_old = (np.abs(np.asarray(ff_old.E_theta)) ** 2
             + np.abs(np.asarray(ff_old.E_phi)) ** 2)
    dth, dph = np.gradient(th), np.gradient(ph)
    P_old = np.sum(U_old * np.sin(th)[None, :, None] * dth[None, :, None]
                   * dph[None, None, :], axis=(1, 2))

    err_new = abs(float(Pg[0] - Pu[0])) / float(Pu[0])
    err_old = abs(float(P_old[0] - Pu[0])) / float(Pu[0])
    assert err_new < err_old / 2, (
        "integrating with the local cell sizes must agree with the uniform "
        f"mesh better than the boundary-cell convention does: new "
        f"{100 * err_new:.1f}% vs old {100 * err_old:.1f}%")
    assert err_new < 0.05, (
        f"graded mesh radiates {100 * err_new:.1f}% differently from the "
        "uniform mesh (measured 2.1% when this envelope was set)")
