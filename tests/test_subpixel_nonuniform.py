"""Subpixel smoothing on the non-uniform mesh path.

Pins the fix for the silent-drop bug where ``Simulation.run(
subpixel_smoothing=True)`` was a no-op on any sim with a
``dz_profile`` / ``dx_profile`` / ``dy_profile`` (see research note
``2026-04-19_v175_crossval05_followups.md`` §Critical finding).

These tests assert that the parameter now (a) propagates to the
non-uniform runner without error, (b) actually changes the simulation
output, and (c) produces a smoothed-eps array whose interface voxels
take values between background and bulk.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.sources.sources import GaussianPulse


def _build_sim(*, n_z: int = 12) -> Simulation:
    """Tiny non-uniform-z domain with a half-cell-shifted dielectric block.

    The block boundary at z = 0.65·dz_min straddles a Yee voxel — exactly
    the configuration where subpixel smoothing is supposed to matter.
    """
    dx = 1e-3
    dz_profile = np.full(n_z, dx)
    dz_profile[n_z // 2] = dx * 0.5  # introduce a fine cell

    sim = Simulation(
        freq_max=20e9,
        domain=(8 * dx, 8 * dx, 0),
        dx=dx,
        dz_profile=dz_profile,
        boundary=BoundarySpec.uniform("pec"),
        cpml_layers=0,
    )
    sim.add_material("diel", eps_r=4.0, sigma=0.0)
    # Block fills cells [4..8] in z but offset by half a cell so the lower
    # face crosses a Yee voxel.
    z_lo = (n_z // 2) * dx + 0.5 * (dx * 0.5)
    z_hi = (n_z - 1) * dx
    sim.add(Box((0.0, 0.0, z_lo),
                (8 * dx, 8 * dx, z_hi)), material="diel")
    sim.add_source(
        position=(4 * dx, 4 * dx, 1.5 * dx),
        component="ex",
        waveform=GaussianPulse(f0=10e9, bandwidth=0.8),
    )
    sim.add_probe(
        position=(4 * dx, 4 * dx, 6 * dx),
        component="ex",
    )
    return sim


def test_subpixel_propagates_through_nu_runner():
    """The subpixel_smoothing flag reaches the non-uniform runner — the
    sim accepts the kwarg and runs without errors."""
    sim = _build_sim()
    res = sim.run(n_steps=64, subpixel_smoothing=True)
    ts = np.asarray(res.time_series).ravel()
    assert np.isfinite(ts).all()
    assert np.any(np.abs(ts) > 0)


def test_subpixel_changes_nu_result():
    """Subpixel-on vs subpixel-off MUST produce different time series on
    a non-uniform mesh with a fractional-cell dielectric. Pre-fix this
    test would have failed because subpixel was silently dropped."""
    sim_off = _build_sim()
    sim_on = _build_sim()

    res_off = sim_off.run(n_steps=128, subpixel_smoothing=False)
    res_on = sim_on.run(n_steps=128, subpixel_smoothing=True)

    ts_off = np.asarray(res_off.time_series).ravel()
    ts_on = np.asarray(res_on.time_series).ravel()

    # Same shape, both finite
    assert ts_off.shape == ts_on.shape
    assert np.isfinite(ts_off).all()
    assert np.isfinite(ts_on).all()

    # Must NOT be identical — if subpixel were still silently dropped,
    # the two runs would return bit-for-bit the same time series.
    diff = np.max(np.abs(ts_off - ts_on))
    assert diff > 1e-6, (
        f"subpixel_smoothing=True did not change the NU result "
        f"(max|diff|={diff:.2e}); kernel branch likely still bypassed."
    )


def test_compute_smoothed_eps_nonuniform_interface_values():
    """Direct unit test of the smoother: interface voxels get an ε in
    (background, bulk) — i.e. the per-axis fill-fraction → Kottke
    averaging is wired up correctly."""
    from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform
    from rfx.geometry.csg import Box as CsgBox
    from rfx.nonuniform import make_nonuniform_grid

    dx = 1e-3
    dz_profile = np.full(8, dx)
    grid = make_nonuniform_grid(
        domain_xy=(8 * dx, 8 * dx),
        dz_profile=dz_profile,
        dx=dx,
        cpml_layers=0,
    )
    eps_bulk = 4.0
    # Box covers half of the z-axis with the lower face at z = 4.3·dx,
    # placing the boundary inside a Yee voxel.
    box = CsgBox((0.0, 0.0, 4.3 * dx),
                 (8 * dx, 8 * dx, 8 * dx))

    eps_ex, eps_ey, eps_ez = compute_smoothed_eps_nonuniform(
        grid, [(box, eps_bulk)], background_eps=1.0,
    )
    eps_ez_np = np.asarray(eps_ez)

    # Bulk cells (well inside box) should hit eps_bulk; outside cells stay
    # at background. The boundary-z slice should sit strictly between.
    assert eps_ez_np[4, 4, 7] == eps_bulk          # well inside (k=7)
    assert eps_ez_np[4, 4, 1] == 1.0               # well outside (k=1)
    boundary = eps_ez_np[4, 4, 4]                  # straddles z=4.3·dx
    assert 1.0 < boundary < eps_bulk, (
        f"interface voxel ε should be between 1.0 and {eps_bulk}, "
        f"got {boundary}"
    )
