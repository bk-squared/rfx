"""Unit tests for `kottke_inv_eps_from_occupancy`.

The new function maps a continuous-fill PEC occupancy field to a
Stage 2 Kottke inv-eps tensor — the AD-traceable analogue of the
``pec_shapes`` branch in ``compute_inv_eps_tensor_diag``.  These tests
verify:

  1. Limit cases — pure vacuum (occ=0) gives 1/ε background; pure PEC
     (occ=1) gives 0.
  2. Half-fill with a clear normal direction gives the Kottke
     PEC-limit signature (inv_par = 0, inv_perp = 0.5/ε on the
     perpendicular axis).
  3. Baseline integration — when an aniso_inv_eps_baseline is
     provided, the PEC contribution is taken via elementwise min so
     dielectric and PEC stack correctly.
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


def test_half_fill_y_normal_kottke_signature():
    """Step in occupancy along ŷ:
    - cells with occ=1 → all inv components = 0 (PEC)
    - cells with occ=0 → 1/ε_bg (vacuum)
    - boundary cell (occ=0.5, adjacent to PEC) → PEC-dilated (60939e0) →
        inv_yy (perpendicular) ≈ 0  (hard PEC mirror)
        inv_xx, inv_zz (parallel) = 0  (any f > 0 zeros parallel)"""
    grid = _grid()
    nx, ny, nz = grid.shape
    occ = np.zeros(grid.shape, dtype=np.float32)
    # Sharp half-step at j=4: occ=1 for j>=4, occ=0 for j<4.
    occ[:, ny // 2:, :] = 1.0
    # Smooth the half-step over one cell so the central-difference
    # gradient resolves a clear ŷ-direction normal at the interface.
    # Use occ=0.5 at the boundary cell (j=ny//2 - 1) to mimic a
    # 1-cell-wide sigmoid edge.
    occ[:, ny // 2 - 1, :] = 0.5
    occ_jax = jnp.asarray(occ)
    inv_xx, inv_yy, inv_zz = kottke_inv_eps_from_occupancy(
        grid, occ_jax, background_eps=1.0
    )
    # Cells with occ=1 (interior PEC, j ≥ ny//2): all components 0.
    interior = inv_xx[:, ny // 2 + 1:ny - 1, :]
    assert jnp.allclose(interior, 0.0), (
        f"interior PEC inv_xx should be 0, got max |{float(jnp.max(jnp.abs(interior))):.3e}|"
    )
    # Cells with occ=0 (deep vacuum, j ≤ ny//2 - 3): all components 1.
    deep_vacuum = inv_xx[:, 1:ny // 2 - 2, :]
    assert jnp.allclose(deep_vacuum, 1.0, atol=1e-3), (
        f"deep-vacuum inv_xx should be 1.0; got mean {float(jnp.mean(deep_vacuum)):.4f}"
    )
    # Boundary cell (occ=0.5):
    #   inv_par = 0 (any f > 0 zeros it),
    #   inv_perp (along ŷ) = (1 − 0.5)/ε = 0.5
    bnd_xx = inv_xx[2:nx - 2, ny // 2 - 1, 2:nz - 2]
    bnd_yy = inv_yy[2:nx - 2, ny // 2 - 1, 2:nz - 2]
    bnd_zz = inv_zz[2:nx - 2, ny // 2 - 1, 2:nz - 2]
    assert jnp.allclose(bnd_xx, 0.0, atol=1e-2), (
        f"boundary inv_xx (parallel) should be 0; got max {float(jnp.max(jnp.abs(bnd_xx))):.3e}"
    )
    assert jnp.allclose(bnd_zz, 0.0, atol=1e-2), (
        f"boundary inv_zz (parallel) should be 0; got max {float(jnp.max(jnp.abs(bnd_zz))):.3e}"
    )
    # POST-DILATION contract (commit 60939e0 "1-cell PEC dilation via
    # neighbor-max"): the occ=0.5 boundary cell adjacent to a PEC cell
    # (occ=1) is neighbor-max-dilated, so interior_mask = sigmoid((1−0.5)/
    # smooth_width) ≈ 1 and the Kottke output is ×(1−1) = 0 → inv_yy → 0
    # (hard PEC mirror).  This is the AD-smooth analogue of the binary
    # apply_pec_mask `pec_mask & (roll | roll)` rule that the production
    # compute_msl_s_matrix path uses for Box(material="pec"); it is
    # VESSL-validated by witness |s21| 0.27 → 0.77 (the wave now reflects
    # cleanly off the open-stub end instead of leaking).  The pre-dilation
    # 0.25 expectation (from ancestor commit ef6f570, authored ~6 min before
    # 60939e0) is stale — see docs rfx-known-issues "Kottke occupancy dilation".
    assert jnp.allclose(bnd_yy, 0.0, atol=1e-2), (
        f"boundary inv_yy (perp) should be ~0 (post-60939e0 PEC dilation); "
        f"got mean {float(jnp.mean(bnd_yy)):.4f}"
    )


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
