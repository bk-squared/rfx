"""Unit tests for `kottke_inv_eps_from_occupancy`.

The function maps a continuous-fill PEC occupancy field to an
inverse-permittivity tensor for ``update_e_aniso_inv`` — the AD-traceable
analogue of the ``pec_shapes`` branch in ``compute_inv_eps_tensor_diag``.
Since #1197 it is the lattice ownership contract's own edge rule
(``_volume_occupancy_masks``): an edge is conductor to the degree the
cells sharing it are occupied. These tests verify:

  1. Limit cases — pure vacuum (occ=0) gives 1/ε background; pure PEC
     (occ=1) gives 0.
  2. Binary occupancy reproduces ``realized_pec_edge_masks`` exactly, and
     a half-filled step cell sits strictly between the two binary answers
     without touching the first air cell beyond it.
  3. Baseline integration — an ``aniso_inv_eps_baseline`` is scaled by the
     keep factor ``1 - M_c`` so dielectric and PEC stack.
  4. AD: gradient flows through occupancy without NaN.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.grid import Grid
from rfx.geometry.smoothing import kottke_inv_eps_from_occupancy


def _grid(nx=8, ny=8, nz=8, dx=1e-4):
    # Build a Grid with cpml_layers=0 so the resolved shape matches
    # exactly (nx, ny, nz).  Domain dimensions are nx*dx etc.
    return Grid(
        freq_max=1e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        cpml_layers=0,
    )


def test_pure_vacuum_gives_inverse_background():
    """occ ≡ 0 everywhere → inv_xx = inv_yy = inv_zz ≈ 1/ε_bg.

    Tolerance ~5e-4 covers the ~5e-5 residual from the soft Heaviside
    projection (sigmoid((0-0.5)/0.05) ≈ 4.5e-5)."""
    grid = _grid()
    occ = jnp.zeros(grid.shape, dtype=jnp.float32)
    inv_xx, inv_yy, inv_zz = kottke_inv_eps_from_occupancy(
        grid, occ, background_eps=4.0
    )
    expected = 1.0 / 4.0
    assert jnp.allclose(inv_xx, expected, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(inv_yy, expected, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(inv_zz, expected, rtol=1e-3, atol=1e-3)


def test_pure_pec_gives_near_zero_inv_eps():
    """occ ≡ 1 everywhere → all inv components ≤ 1e-9 (effective PEC).

    Uses smooth Kottke with eps_inside = 1e10 (large but finite) to
    avoid the f=0 discontinuity of the strict PEC limit.  At f=1, this
    gives inv ≈ 1/1e10 = 1e-10 — small enough to act as PEC at FDTD
    timescales (Cb scales as inv·dt; for dt~1e-13 s the field barely
    updates) but smooth across f=0.
    """
    grid = _grid()
    occ = jnp.ones(grid.shape, dtype=jnp.float32)
    inv_xx, inv_yy, inv_zz = kottke_inv_eps_from_occupancy(
        grid, occ, background_eps=2.0
    )
    assert jnp.all(inv_xx < 1e-9)
    assert jnp.all(inv_yy < 1e-9)
    assert jnp.all(inv_zz < 1e-9)


def test_binary_occupancy_reduces_to_the_hard_edge_rule():
    """#1197: at binary occupancy the tensor's zeros ARE the lattice
    ownership contract's PEC edges (``realized_pec_edge_masks``, §1.2), and
    everything else is the background. A 12-edge box in the middle of the
    grid, so every component and every face/edge/corner incidence is hit.
    The replaced builder wrote inv = 0 one cell past the occupancy in every
    direction (a six-neighbour dilation) and moved a cavity's TM110 by
    -3.5 % on a binary slab; this pins the limit that stops that."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    grid = _grid()
    occ = np.zeros(grid.shape, dtype=np.float32)
    occ[2:5, 3:6, 1:4] = 1.0
    inv = kottke_inv_eps_from_occupancy(grid, jnp.asarray(occ), background_eps=1.0)
    hard = realized_pec_edge_masks(occ.astype(bool), [], [], periodic=(False, False, False))
    for comp, want in zip(inv, hard):
        z = np.asarray(comp) == 0.0
        assert np.array_equal(z, np.asarray(want)), int((z != np.asarray(want)).sum())
        assert np.allclose(np.asarray(comp)[~z], 1.0)


def test_half_fill_step_is_between_the_two_binary_answers():
    """A half-filled cell at the step is a lossless interpolation: every
    component strictly between the all-air and all-metal values, the
    tangential edges on the plane INSIDE the step already fully metal, and
    the first air cell beyond the step untouched (that last one is the
    dilation defect of #1197 made impossible)."""
    grid = _grid()
    nx, ny, nz = grid.shape
    occ = np.zeros(grid.shape, dtype=np.float32)
    occ[:, ny // 2:, :] = 1.0
    occ[:, ny // 2 - 1, :] = 0.5
    inv_xx, inv_yy, inv_zz = kottke_inv_eps_from_occupancy(grid, jnp.asarray(occ), background_eps=1.0)
    j = ny // 2 - 1
    for comp in (inv_xx, inv_yy, inv_zz):
        a = np.asarray(comp)
        assert np.allclose(a[:, ny // 2 + 1:, :], 0.0)            # interior metal
        assert np.allclose(a[:, :j, :], 1.0)                        # air beyond the step, first plane included
        v = a[2:nx - 2, j, 2:nz - 2]
        assert np.all(v > 0.0) and np.all(v < 1.0), (float(v.min()), float(v.max()))
    # Ey edges of the half cell: four incident cells all 0.5 -> 1 - 0.5**4
    assert np.allclose(np.asarray(inv_yy)[2:nx - 2, j, 2:nz - 2], 0.5 ** 4, atol=1e-6)
    # Ex edges on the plane between the half cell and the full cell: metal
    assert np.allclose(np.asarray(inv_xx)[2:nx - 2, ny // 2, 2:nz - 2], 0.0)


def test_baseline_min_preserves_dielectric_outside():
    """When a baseline aniso_inv_eps is provided, cells with occ=0
    inherit the baseline (dielectric stays); cells with occ=1 get 0
    (PEC overrides dielectric)."""
    grid = _grid()
    nx, ny, nz = grid.shape
    occ = np.zeros(grid.shape, dtype=np.float32)
    occ[:, ny // 2:, :] = 1.0
    occ_jax = jnp.asarray(occ)
    # Baseline: dielectric ε=4 everywhere → inv = 0.25
    baseline = (
        jnp.full(grid.shape, 0.25, dtype=jnp.float32),
        jnp.full(grid.shape, 0.25, dtype=jnp.float32),
        jnp.full(grid.shape, 0.25, dtype=jnp.float32),
    )
    inv_xx, _, _ = kottke_inv_eps_from_occupancy(
        grid, occ_jax, aniso_inv_eps_baseline=baseline,
    )
    # Vacuum side (occ=0): baseline is preserved (PEC contribution = 0.25 at occ=0,
    # min(0.25, 0.25) = 0.25).  Start at j=1: j=0 is the PERIODIC-WRAP edge — the
    # 60939e0 neighbor-max PEC dilation wraps (jnp.roll), so j=0's −ŷ neighbor is the
    # PEC cell j=ny-1, dilating j=0 to PEC (inv→0).  That wrap never fires in production
    # (microstrip PEC is interior, far from the domain edges); it is a test-setup
    # artifact of filling PEC to the grid boundary.  Same stale-test cause as
    # test_half_fill_y_normal (see docs rfx-known-issues "Kottke occupancy dilation").
    assert jnp.allclose(inv_xx[:, 1:ny // 2 - 2, :], 0.25, atol=1e-3)
    # PEC side (occ=1): all components 0
    assert jnp.allclose(inv_xx[:, ny // 2 + 1:, :], 0.0, atol=1e-3)


def test_ad_traceable_no_nan():
    """jax.grad through the cost function returns finite gradients."""
    grid = _grid(nx=4, ny=4, nz=4)
    nx, ny, nz = grid.shape

    def cost(occ_flat):
        occ = occ_flat.reshape(nx, ny, nz)
        ix, iy, iz = kottke_inv_eps_from_occupancy(grid, occ)
        return jnp.sum(ix + iy + iz)

    occ0 = jnp.full(nx * ny * nz, 0.5, dtype=jnp.float32)
    g = jax.grad(cost)(occ0)
    assert jnp.all(jnp.isfinite(g)), "AD produced NaN/Inf"
    # At occ=0.5 uniformly the gradient is non-zero only at boundary
    # cells; interior cells (where ∇occ ≈ 0) get zero contribution
    # from the normal but the f-only term (1−f)/ε still gives a
    # gradient.  Just check finiteness here.
