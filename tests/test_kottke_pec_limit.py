"""Stage 2 Step 1 — unit tests for inv-eps tensor with PEC limit.

Reference: stage2_ca_cb_derivation.md (local memory).

These tests pin down the mathematical contract derived from
Farjadpour 2006 Eq. (1) + Kottke 2008 Eq. (22-23) restricted to the
isotropic-on-both-sides case with PEC limit (ε_inside → ∞).

Scope:
1. ``_kottke_inv_eps_diag(f, eps_in, eps_out, n_x, n_y, n_z, is_pec)``
   returns the diagonal of the Kottke (ε̄⁻¹)_lab tensor.
2. ``compute_inv_eps_tensor_diag(grid, dielectric_shapes, pec_shapes,
   background_eps)`` is the public API that runs Kottke smoothing on
   Yee-staggered positions for a list of shapes (PEC and dielectric).

Stage 1 code is unchanged — these tests only exercise the new Stage 2
helpers. No regression of existing Stage 1 tests is expected.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import Grid
from rfx.geometry.csg import Box


# -----------------------------------------------------------------------------
# Direct unit tests on _kottke_inv_eps_diag
# -----------------------------------------------------------------------------


def test_pec_half_space_y_normal_f05():
    """PEC half-space, wall normal n̂ = ŷ, fill fraction f=0.5, ε_out=1.

    Per derivation §4: inv_perp = (1−f)/ε_out = 0.5; inv_par = 0.
    Diagonal:
      inv_xx = n_x²·inv_perp + (1−n_x²)·inv_par = 0·0.5 + 1·0 = 0
      inv_yy = n_y²·inv_perp + (1−n_y²)·inv_par = 1·0.5 + 0·0 = 0.5
      inv_zz = n_z²·inv_perp + (1−n_z²)·inv_par = 0·0.5 + 1·0 = 0
    """
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    f = jnp.array(0.5, dtype=jnp.float32)
    eps_outside = jnp.array(1.0, dtype=jnp.float32)
    n_x, n_y, n_z = (jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))

    inv_xx, inv_yy, inv_zz = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside, n_x, n_y, n_z, is_pec=True,
    )
    assert float(inv_xx) == pytest.approx(0.0, abs=1e-6)
    assert float(inv_yy) == pytest.approx(0.5, abs=1e-6)
    assert float(inv_zz) == pytest.approx(0.0, abs=1e-6)


def test_pec_fully_interior():
    """f=1 (Yee point fully inside PEC): all inv components = 0.

    Cell is entirely PEC; E is frozen in all directions (the Ca/Cb
    update with inv=0 gives Ca=1, Cb=0 → E^{n+1}=E^n=0 if init zero).
    """
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    f = jnp.array(1.0, dtype=jnp.float32)
    eps_outside = jnp.array(1.0, dtype=jnp.float32)
    n_x, n_y, n_z = (jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))

    inv_xx, inv_yy, inv_zz = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside, n_x, n_y, n_z, is_pec=True,
    )
    assert float(inv_xx) == 0.0
    assert float(inv_yy) == 0.0
    assert float(inv_zz) == 0.0


def test_pec_fully_exterior():
    """f=0 (no PEC in cell): all inv components = 1/ε_out.

    Cell is pure background dielectric. Stage 2 must not introduce a
    spurious bias when there is no PEC anywhere in the smoothing voxel.
    """
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    f = jnp.array(0.0, dtype=jnp.float32)
    eps_outside = jnp.array(1.0, dtype=jnp.float32)
    n_x, n_y, n_z = (jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))

    inv_xx, inv_yy, inv_zz = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside, n_x, n_y, n_z, is_pec=True,
    )
    assert float(inv_xx) == pytest.approx(1.0, abs=1e-6)
    assert float(inv_yy) == pytest.approx(1.0, abs=1e-6)
    assert float(inv_zz) == pytest.approx(1.0, abs=1e-6)


def test_pec_tilted_30deg_f05():
    """Tilted PEC wall, n̂ = (sin30°, cos30°, 0), f=0.5, ε_out=1.

    inv_perp = 0.5; inv_par = 0.
    inv_xx = sin²(30°)·0.5 = 0.25·0.5 = 0.125
    inv_yy = cos²(30°)·0.5 = 0.75·0.5 = 0.375
    inv_zz = 0·0.5 = 0
    """
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    f = jnp.array(0.5, dtype=jnp.float32)
    eps_outside = jnp.array(1.0, dtype=jnp.float32)
    sin30 = 0.5
    cos30 = float(np.sqrt(3) / 2)
    n_x, n_y, n_z = (jnp.array(sin30), jnp.array(cos30), jnp.array(0.0))

    inv_xx, inv_yy, inv_zz = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside, n_x, n_y, n_z, is_pec=True,
    )
    assert float(inv_xx) == pytest.approx(0.125, abs=1e-5)
    assert float(inv_yy) == pytest.approx(0.375, abs=1e-5)
    assert float(inv_zz) == pytest.approx(0.0, abs=1e-6)


def test_dielectric_branch_matches_farjadpour_eq1():
    """Non-PEC branch must reproduce Farjadpour 2006 Eq. (1).

    For ε₁=4, ε₂=1, f=0.5, n̂=ŷ:
      ⟨ε⁻¹⟩ = 0.5/4 + 0.5/1 = 0.625
      ⟨ε⟩⁻¹ = 1/(0.5·4 + 0.5·1) = 1/2.5 = 0.4
    Diagonal:
      inv_xx = 0·0.625 + 1·0.4 = 0.4
      inv_yy = 1·0.625 + 0·0.4 = 0.625
      inv_zz = 0·0.625 + 1·0.4 = 0.4

    This is the load-bearing check that the dielectric path is
    consistent with the existing rfx Kottke implementation.
    """
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    f = jnp.array(0.5, dtype=jnp.float32)
    eps_in = jnp.array(4.0, dtype=jnp.float32)
    eps_out = jnp.array(1.0, dtype=jnp.float32)
    n_x, n_y, n_z = (jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))

    inv_xx, inv_yy, inv_zz = _kottke_inv_eps_diag(
        f, eps_in, eps_out, n_x, n_y, n_z, is_pec=False,
    )
    assert float(inv_xx) == pytest.approx(0.4, abs=1e-5)
    assert float(inv_yy) == pytest.approx(0.625, abs=1e-5)
    assert float(inv_zz) == pytest.approx(0.4, abs=1e-5)


def test_pec_branch_no_nan_at_f0_with_inf_eps_inside():
    """Edge-case stability: f=0 with eps_inside=∞ must not produce
    NaN. The PEC branch handles this by using `(1-f)/eps_out` directly
    for inv_perp (skipping the f/eps_inside term that would be
    indeterminate as 0/∞)."""
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    f = jnp.array(0.0, dtype=jnp.float32)
    eps_outside = jnp.array(1.0, dtype=jnp.float32)
    n_x, n_y, n_z = (jnp.array(0.5), jnp.array(0.5),
                     jnp.array(float(np.sqrt(0.5))))

    inv_xx, inv_yy, inv_zz = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside, n_x, n_y, n_z, is_pec=True,
    )
    assert np.isfinite(float(inv_xx))
    assert np.isfinite(float(inv_yy))
    assert np.isfinite(float(inv_zz))


# -----------------------------------------------------------------------------
# Integration: compute_inv_eps_tensor_diag on WR-90 PEC half-space
# -----------------------------------------------------------------------------


def test_compute_inv_eps_tensor_diag_wr90_perpendicular_boundary_cell():
    """WR-90 dx=1mm with PEC half-space at y > 22.86 mm.

    The perpendicular boundary cell is at the **Ey** position
    (i, j+0.5, k). For j=22 (Ey y=22.5 mm):
      sdf = 22.86 − 22.5 = 0.36 mm (outside box, positive)
      f = clip(0.5 − sdf/dx, 0, 1) = clip(0.5 − 0.36, 0, 1) = 0.14
      inv_perp = (1−f)/ε_out = 0.86/1 = 0.86
      n̂ ≈ ±ŷ, n_y² = 1 → inv_yy = 0.86

    For j=23 (Ey y=23.5 mm, fully inside PEC):
      f = 1.0 → inv_yy = 0.

    For all Yee components tangential to the wall (Ex, Ez): inv = 0.
    """
    from rfx.geometry.smoothing import compute_inv_eps_tensor_diag

    grid = Grid(
        freq_max=10e9,
        domain=(0.06, 0.025, 0.012),
        dx=0.001,
        cpml_layers=0,
    )
    pec_box = Box((-1.0, 0.02286, -1.0), (1.0, 1.0, 1.0))
    inv_xx, inv_yy, inv_zz = compute_inv_eps_tensor_diag(
        grid,
        dielectric_shapes=[],
        pec_shapes=[pec_box],
        background_eps=1.0,
    )

    i_mid = grid.shape[0] // 2
    k_mid = grid.shape[2] // 2

    # Perpendicular boundary cell at Ey(j=22, y=22.5mm).
    inv_yy_perp_boundary = float(inv_yy[i_mid, 22, k_mid])
    assert inv_yy_perp_boundary == pytest.approx(0.86, abs=0.05), (
        f"expected inv_yy ≈ 0.86 at Ey(j=22) perpendicular boundary; "
        f"got {inv_yy_perp_boundary:.4f}"
    )

    # Tangential at the same logical row: Ex(j=23) is inside-PEC region
    # (sdf=−0.14, f=0.64), but n̂=ŷ so n_x²=0 → inv_xx = inv_par = 0.
    inv_xx_tangential = float(inv_xx[i_mid, 23, k_mid])
    assert inv_xx_tangential == pytest.approx(0.0, abs=1e-5), (
        f"expected inv_xx ≈ 0 (Ex tangential to PEC); "
        f"got {inv_xx_tangential:.4f}"
    )
    inv_zz_tangential = float(inv_zz[i_mid, 23, k_mid])
    assert inv_zz_tangential == pytest.approx(0.0, abs=1e-5), (
        f"expected inv_zz ≈ 0 (Ez tangential to PEC); "
        f"got {inv_zz_tangential:.4f}"
    )

    # Fully-PEC region: Ey(j=23, y=23.5mm) is inside the half-space.
    inv_yy_fully_pec = float(inv_yy[i_mid, 23, k_mid])
    assert inv_yy_fully_pec == pytest.approx(0.0, abs=1e-5), (
        f"expected inv_yy ≈ 0 inside PEC; got {inv_yy_fully_pec:.4f}"
    )

    # Pure-vacuum interior: Ex(j=10, y=10mm) is far from the wall.
    inv_xx_vacuum = float(inv_xx[i_mid, 10, k_mid])
    assert inv_xx_vacuum == pytest.approx(1.0, abs=1e-5), (
        f"expected inv_xx ≈ 1/ε_out=1 in vacuum; got {inv_xx_vacuum:.4f}"
    )


def test_compute_inv_eps_tensor_diag_no_pec_matches_smoothed_eps_inverse():
    """Without any PEC shape, ``compute_inv_eps_tensor_diag`` should be
    the elementwise inverse of ``compute_smoothed_eps`` (or of the
    background-eps array when no shapes are present).

    This pins the dielectric path's bit-stable bridge: Stage 2 inverts
    the existing Kottke output exactly when there's no PEC content.
    """
    from rfx.geometry.smoothing import (
        compute_inv_eps_tensor_diag, compute_smoothed_eps,
    )

    grid = Grid(
        freq_max=10e9,
        domain=(0.04, 0.04, 0.04),
        dx=0.002,
        cpml_layers=0,
    )
    diel_box = Box((0.012, 0.012, 0.012), (0.028, 0.028, 0.028))
    shapes = [(diel_box, 4.0)]

    eps_ex, eps_ey, eps_ez = compute_smoothed_eps(grid, shapes, background_eps=1.0)
    inv_xx, inv_yy, inv_zz = compute_inv_eps_tensor_diag(
        grid, dielectric_shapes=shapes, pec_shapes=[], background_eps=1.0,
    )

    np.testing.assert_allclose(np.asarray(inv_xx), 1.0 / np.asarray(eps_ex),
                                rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(inv_yy), 1.0 / np.asarray(eps_ey),
                                rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(inv_zz), 1.0 / np.asarray(eps_ez),
                                rtol=1e-6, atol=1e-9)
