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


# -----------------------------------------------------------------------------
# Post-review hardening tests (HIGH/MEDIUM findings from 2026-05-01 PR review)
# -----------------------------------------------------------------------------


def test_two_non_overlapping_pec_walls_per_component_min_correct():
    """Code-review HIGH: ``jnp.minimum`` PEC union is correct *only*
    when the PEC bodies are non-overlapping (precondition documented in
    the docstring). This regression test pins the precondition's
    correctness at cells *clearly assigned to one wall*: the y_hi wall
    at probe location is far from the z_hi wall, and vice versa.

    Note on corner cells: the min-merge intentionally collapses to 0
    where two walls' boundary effects coincide on the same Yee point
    — that matches the union-SDF + nearest-normal pattern for corners
    (the closer wall's tangential-suppression dominates), as
    documented in stage2_ca_cb_derivation.md §8 and Kottke 2008 §III
    on corner singularities.
    """
    from rfx.geometry.smoothing import compute_inv_eps_tensor_diag

    grid = Grid(
        freq_max=10e9,
        domain=(0.06, 0.025, 0.012),
        dx=0.001,
        cpml_layers=0,
    )
    pec_y = Box((-1.0, 0.02286, -1.0), (1.0, 1.0, 1.0))
    pec_z = Box((-1.0, -1.0, 0.01016), (1.0, 1.0, 1.0))
    inv_xx, inv_yy, inv_zz = compute_inv_eps_tensor_diag(
        grid,
        dielectric_shapes=[],
        pec_shapes=[pec_y, pec_z],
        background_eps=1.0,
    )

    inv_xx_np = np.asarray(inv_xx)
    inv_yy_np = np.asarray(inv_yy)
    inv_zz_np = np.asarray(inv_zz)

    # Interior cell, well within bulk vacuum region.
    i_mid = grid.shape[0] // 2
    assert inv_xx_np[i_mid, 5, 5] == pytest.approx(1.0, abs=1e-5)
    assert inv_yy_np[i_mid, 5, 5] == pytest.approx(1.0, abs=1e-5)
    assert inv_zz_np[i_mid, 5, 5] == pytest.approx(1.0, abs=1e-5)

    # y_hi perpendicular boundary at Ey(j=22, k=5).
    # Ey is at (i, j+0.5, k) → (·, 22.5mm, 5mm). The y-wall is the
    # closer wall (0.36mm from Ey, vs 5.16mm to z-wall). Expected:
    #   pec_y: sdf=+0.36, f=0.14, inv_perp=(1−0.14)/1=0.86,
    #          n_y=±1 → inv_yy = 1·0.86 + 0·0 = 0.86.
    #   pec_z: sdf=+5.16 ≫ dx/2 → f=0 → inv_yy = 1 (vacuum).
    #   min(0.86, 1) = 0.86 ✓
    assert inv_yy_np[i_mid, 22, 5] == pytest.approx(0.86, abs=0.05), (
        f"y_hi perpendicular boundary cell expected 0.86, got "
        f"{inv_yy_np[i_mid, 22, 5]:.4f}"
    )

    # z_hi perpendicular boundary at Ez(j=5, k=10).
    # Ez is at (i, j, k+0.5) → (·, 5mm, 10.5mm). The z-wall is closer
    # (0.34mm from Ez, vs 17.86mm to y-wall). Expected:
    #   pec_z: sdf=−0.34 (inside), f=0.84, inv_perp=(1−0.84)/1=0.16,
    #          n_z=±1 → inv_zz = 1·0.16 + 0·0 = 0.16.
    #   pec_y: sdf=+17.86 ≫ dx/2 → f=0 → inv_zz = 1 (vacuum).
    #   min(0.16, 1) = 0.16 ✓
    assert inv_zz_np[i_mid, 5, 10] == pytest.approx(0.16, abs=0.05), (
        f"z_hi perpendicular boundary cell expected 0.16, got "
        f"{inv_zz_np[i_mid, 5, 10]:.4f}"
    )

    # Tangential at Ex(j=23, k=5): inside y-wall, far from z-wall.
    # pec_y: sdf=−0.14 (inside), f=0.64, n_y=±1 → inv_xx = 0·inv_perp
    #        + 1·inv_par = 0 (tangential).
    # pec_z: f=0 → inv_xx = 1 (vacuum).
    # min(0, 1) = 0 ✓
    assert inv_xx_np[i_mid, 23, 5] == pytest.approx(0.0, abs=1e-5)

    # Tangential at Ex(j=5, k=10): far from y-wall, near z-wall.
    # pec_y: f=0 → inv_xx = 1.
    # pec_z: sdf=+0.16 (outside), f=0.34, n_z=±1 → inv_xx = 0·inv_perp
    #        + 1·inv_par = 0 (tangential).
    # min(1, 0) = 0 ✓
    assert inv_xx_np[i_mid, 5, 10] == pytest.approx(0.0, abs=1e-5)

    # Deep inside both PECs (j=24, k=11): all inv components = 0.
    assert inv_xx_np[i_mid, 24, 11] == pytest.approx(0.0, abs=1e-5)
    assert inv_yy_np[i_mid, 24, 11] == pytest.approx(0.0, abs=1e-5)
    assert inv_zz_np[i_mid, 24, 11] == pytest.approx(0.0, abs=1e-5)


def test_kottke_inv_eps_diag_jit_compatible():
    """Code-review MEDIUM: ``is_pec`` is used as a static-argument
    branch. JIT compilation must be possible with `static_argnames`.
    Step 2 (``update_e_aniso_inv``) will be the first JIT'd consumer;
    catching the static-arg issue here avoids a hard-to-triangulate
    compile error in Step 2."""
    import jax
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    fn_jit = jax.jit(_kottke_inv_eps_diag, static_argnames=("is_pec",))

    f = jnp.array(0.5, dtype=jnp.float32)
    eps_outside = jnp.array(1.0, dtype=jnp.float32)
    n_x, n_y, n_z = (jnp.array(0.0, dtype=jnp.float32),
                     jnp.array(1.0, dtype=jnp.float32),
                     jnp.array(0.0, dtype=jnp.float32))

    # PEC branch under JIT.
    inv_xx, inv_yy, inv_zz = fn_jit(
        f, jnp.inf, eps_outside, n_x, n_y, n_z, is_pec=True,
    )
    assert float(inv_yy) == pytest.approx(0.5, abs=1e-6)

    # Dielectric branch under JIT (different static path).
    inv_xx, inv_yy, inv_zz = fn_jit(
        f, jnp.array(4.0), eps_outside, n_x, n_y, n_z, is_pec=False,
    )
    assert float(inv_yy) == pytest.approx(0.625, abs=1e-5)


@pytest.mark.parametrize("f_val", [1e-6, 1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-6])
def test_kottke_inv_eps_diag_pec_float32_precision_sweep(f_val):
    """Code-review MEDIUM: float32 precision behavior across the f
    range, especially near 0 and 1 where the PEC branch transitions.
    Verify monotonicity, finiteness, and the limit values:
      f → 0+:  inv_perp → 1/ε_out  (PEC barely present)
      f → 1−:  inv_perp → 0        (PEC nearly fills cell)
    """
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    f = jnp.array(f_val, dtype=jnp.float32)
    eps_outside = jnp.array(1.0, dtype=jnp.float32)
    n_x, n_y, n_z = (jnp.array(0.0, dtype=jnp.float32),
                     jnp.array(1.0, dtype=jnp.float32),
                     jnp.array(0.0, dtype=jnp.float32))

    inv_xx, inv_yy, inv_zz = _kottke_inv_eps_diag(
        f, jnp.inf, eps_outside, n_x, n_y, n_z, is_pec=True,
    )

    # Finiteness across the sweep.
    assert np.isfinite(float(inv_xx))
    assert np.isfinite(float(inv_yy))
    assert np.isfinite(float(inv_zz))

    # Tangential components stay at 0 (n_y=1 puts perp on y axis).
    assert float(inv_xx) == pytest.approx(0.0, abs=1e-6)
    assert float(inv_zz) == pytest.approx(0.0, abs=1e-6)

    # Perpendicular component matches (1-f)/ε_out within float32 tol.
    expected = (1.0 - f_val) / 1.0
    # float32 ULP at f≈1 is ~1e-7; allow 5 ULP for chained ops.
    assert float(inv_yy) == pytest.approx(expected, abs=5e-6)


def test_kottke_inv_eps_diag_pec_monotone_in_fill_fraction():
    """Companion to the precision sweep: inv_perp must be monotonically
    *decreasing* in f (more PEC ⇒ less perpendicular conductivity).
    Catches a sign flip or a clamp regression."""
    from rfx.geometry.smoothing import _kottke_inv_eps_diag

    eps_outside = jnp.array(1.0, dtype=jnp.float32)
    n_x, n_y, n_z = (jnp.array(0.0, dtype=jnp.float32),
                     jnp.array(1.0, dtype=jnp.float32),
                     jnp.array(0.0, dtype=jnp.float32))
    fs = np.linspace(0.0, 1.0, 21)
    inv_yy_vals = []
    for f_val in fs:
        f = jnp.array(f_val, dtype=jnp.float32)
        _, inv_yy, _ = _kottke_inv_eps_diag(
            f, jnp.inf, eps_outside, n_x, n_y, n_z, is_pec=True,
        )
        inv_yy_vals.append(float(inv_yy))
    inv_yy_arr = np.array(inv_yy_vals)
    # Monotone non-increasing.
    diffs = np.diff(inv_yy_arr)
    assert np.all(diffs <= 1e-6), (
        f"inv_yy not monotone non-increasing in f: diffs={diffs}"
    )


def test_compute_inv_eps_tensor_diag_returns_float32():
    """Code-review MEDIUM: dtype of returned arrays must be float32
    regardless of input ε dtype (which can be float64 if user passes
    Python floats). Pins the explicit cast in
    ``compute_inv_eps_tensor_diag``."""
    from rfx.geometry.smoothing import compute_inv_eps_tensor_diag

    grid = Grid(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.002,
        cpml_layers=0,
    )
    pec = Box((-1.0, 0.011, -1.0), (1.0, 1.0, 1.0))
    inv_xx, inv_yy, inv_zz = compute_inv_eps_tensor_diag(
        grid, dielectric_shapes=[], pec_shapes=[pec], background_eps=1.0,
    )
    assert inv_xx.dtype == jnp.float32
    assert inv_yy.dtype == jnp.float32
    assert inv_zz.dtype == jnp.float32
