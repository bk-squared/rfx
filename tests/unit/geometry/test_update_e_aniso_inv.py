"""Stage 2 Step 2 — unit tests for ``update_e_aniso_inv``.

The Stage 2 design replaces ``update_e_aniso(eps_ex, eps_ey, eps_ez)``
(divide-by-eps form) with ``update_e_aniso_inv(inv_xx, inv_yy, inv_zz)``
(multiply-by-inv-eps form). The new form is numerically stable in the
PEC limit (inv = 0): the original eps form would produce eps = ∞ which
is a NaN trap; the inv form gives Ca = 1, Cb = 0, freezing E cleanly.

Reference: stage2_ca_cb_derivation.md §5 (Ca/Cb in inv_eps form).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import init_state, init_materials, EPS_0
from rfx.stepping.yee import update_h, update_e, update_e_aniso




def _make_state_with_seeded_field(shape, seed=0):
    """Initialize an FDTDState and seed Ez with a small Gaussian-ish
    pattern so that subsequent updates produce non-trivial differences."""
    state = init_state(shape)
    rng = np.random.default_rng(seed)
    ez_seed = rng.normal(size=shape).astype(np.float32) * 0.01
    return state._replace(ez=jnp.asarray(ez_seed))


def _make_materials_constant(shape, eps_r=1.0, sigma=0.0):
    """Materials filled with constant ε_r and σ across the whole grid."""
    materials = init_materials(shape)
    return materials._replace(
        eps_r=jnp.full(shape, eps_r, dtype=jnp.float32),
        sigma=jnp.full(shape, sigma, dtype=jnp.float32),
    )




def test_update_e_aniso_inv_vacuum_matches_update_e():
    """With inv_eps = (1,1,1) and σ=0 (vacuum), ``update_e_aniso_inv``
    must produce the same field as the scalar ``update_e`` to within
    float32 ULP. Pins the bit-stable bridge: legacy callers that
    redirect through Stage 2 see no observable change in vacuum.

    The update_e and update_e_aniso_inv paths differ in arithmetic ordering, so a few ULP of difference is expected and harmless."""
    from rfx.stepping.yee import update_e_aniso_inv

    shape = (16, 12, 10)
    dx = 0.001
    dt = dx / (3e8 * np.sqrt(3.0)) * 0.99

    state0 = _make_state_with_seeded_field(shape, seed=42)
    state0 = update_h(state0, _make_materials_constant(shape), dt, dx, high_wall=(False, False, False))

    materials = _make_materials_constant(shape, eps_r=1.0, sigma=0.0)
    ones = jnp.ones(shape, dtype=jnp.float32)

    out_legacy = update_e(state0, materials, dt, dx, high_wall=(False, False, False))
    out_inv = update_e_aniso_inv(state0, materials, ones, ones, ones, dt, dx, high_wall=(False, False, False))

    np.testing.assert_allclose(
        np.asarray(out_legacy.ex), np.asarray(out_inv.ex),
        rtol=5e-6, atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(out_legacy.ey), np.asarray(out_inv.ey),
        rtol=5e-6, atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(out_legacy.ez), np.asarray(out_inv.ez),
        rtol=5e-6, atol=1e-7,
    )




def test_update_e_aniso_inv_dielectric_matches_update_e_aniso():
    """For a dielectric ε=4 (inv = 0.25), the new ``_inv`` form and the
    legacy ``update_e_aniso(eps=4)`` form must agree to float32 ULP.

    This is the dielectric-path bridge: with no PEC content,
    Stage 2's update is bit/ULP equivalent to Stage 1's update. The
    only place they may differ is float arithmetic ordering — within
    a few ULP per cell, well under any physical metric."""
    from rfx.stepping.yee import update_e_aniso_inv

    shape = (16, 12, 10)
    dx = 0.001
    dt = dx / (3e8 * np.sqrt(3.0)) * 0.99
    eps_r = 4.0

    state0 = _make_state_with_seeded_field(shape, seed=7)
    state0 = update_h(state0, _make_materials_constant(shape), dt, dx, high_wall=(False, False, False))

    materials = _make_materials_constant(shape, eps_r=1.0, sigma=0.0)

    eps_aniso = (
        jnp.full(shape, eps_r, dtype=jnp.float32),
        jnp.full(shape, eps_r, dtype=jnp.float32),
        jnp.full(shape, eps_r, dtype=jnp.float32),
    )
    inv_aniso = (
        jnp.full(shape, 1.0 / eps_r, dtype=jnp.float32),
        jnp.full(shape, 1.0 / eps_r, dtype=jnp.float32),
        jnp.full(shape, 1.0 / eps_r, dtype=jnp.float32),
    )

    out_legacy = update_e_aniso(
        state0, materials, eps_aniso[0], eps_aniso[1], eps_aniso[2], dt, dx,
     high_wall=(False, False, False))
    out_inv = update_e_aniso_inv(
        state0, materials, inv_aniso[0], inv_aniso[1], inv_aniso[2], dt, dx,
     high_wall=(False, False, False))

    np.testing.assert_allclose(
        np.asarray(out_legacy.ex), np.asarray(out_inv.ex),
        rtol=5e-6, atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(out_legacy.ey), np.asarray(out_inv.ey),
        rtol=5e-6, atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(out_legacy.ez), np.asarray(out_inv.ez),
        rtol=5e-6, atol=1e-9,
    )




def test_update_e_aniso_inv_pec_tangential_frozen():
    """Setting inv_eps = 0 on a component must freeze that component
    regardless of curl(H), σ, or J. This is the load-bearing physics
    claim of Stage 2 — tangential E at PEC is auto-zeroed via Ca=1,
    Cb=0, NOT via a separate apply_pec_faces / apply_conformal_pec
    pass."""
    from rfx.stepping.yee import update_e_aniso_inv

    shape = (16, 12, 10)
    dx = 0.001
    dt = dx / (3e8 * np.sqrt(3.0)) * 0.99

    state0 = _make_state_with_seeded_field(shape, seed=3)
    state0 = update_h(state0, _make_materials_constant(shape), dt, dx, high_wall=(False, False, False))

    assert any(np.any(np.asarray(field) != 0) for field in (state0.hx, state0.hy, state0.hz)) and np.any(np.diff(np.asarray(state0.hz), axis=1)[:, :, 1:] - np.diff(np.asarray(state0.hy), axis=2)[:, 1:, :]), "the seeded field must produce nonzero H and a nontrivial curl for the electric update"
    rng = np.random.default_rng(11)
    ex_seed = rng.normal(size=shape).astype(np.float32) * 0.05
    assert np.any(ex_seed != 0), "the Ex seed must be nonzero so freezing cannot be confused with zeroing"
    state0 = state0._replace(ex=jnp.asarray(ex_seed))

    materials = _make_materials_constant(shape, eps_r=1.0, sigma=0.0)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    ones = jnp.ones(shape, dtype=jnp.float32)

    out = update_e_aniso_inv(state0, materials, zeros, ones, ones, dt, dx, high_wall=(False, False, False))

    np.testing.assert_allclose(
        np.asarray(out.ex), ex_seed, rtol=0, atol=1e-7,
        err_msg="inv_xx=0 should freeze Ex; got non-trivial change",
    )


def test_update_e_aniso_inv_pec_tangential_with_sigma_still_frozen():
    """Even with σ > 0, inv_xx = 0 must freeze Ex (the σ·dt·inv_xx /
    (2·ε₀) factor is 0 regardless of σ when inv_xx = 0). This pins
    the physics that "PEC has no loss" emerges naturally from the
    inv-eps form: σ → ∞ is *not* needed and is in fact unstable
    (see derivation §6)."""
    from rfx.stepping.yee import update_e_aniso_inv

    shape = (8, 8, 8)
    dx = 0.001
    dt = dx / (3e8 * np.sqrt(3.0)) * 0.99

    state0 = _make_state_with_seeded_field(shape, seed=23)
    state0 = update_h(state0, _make_materials_constant(shape), dt, dx, high_wall=(False, False, False))

    rng = np.random.default_rng(99)
    ex_seed = rng.normal(size=shape).astype(np.float32) * 0.1
    state0 = state0._replace(ex=jnp.asarray(ex_seed))

    LOSSY_CONDUCTIVITY_S_PER_METRE = 100.0
    materials = _make_materials_constant(shape, eps_r=1.0, sigma=LOSSY_CONDUCTIVITY_S_PER_METRE)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    ones = jnp.ones(shape, dtype=jnp.float32)

    out = update_e_aniso_inv(state0, materials, zeros, ones, ones, dt, dx, high_wall=(False, False, False))

    np.testing.assert_allclose(
        np.asarray(out.ex), ex_seed, rtol=0, atol=1e-7,
        err_msg="inv_xx=0 with σ>0 should still freeze Ex",
    )




def test_update_e_aniso_inv_partial_pec_perpendicular_finite():
    """When inv_eps takes a fractional value (boundary cell with
    partial PEC), the update must produce finite values — no NaN, no
    inf. This is the PEC-boundary perpendicular regime.

    Setup: inv_eps = 0.36 (matches the WR-90 dx=1mm boundary cell from
    Stage 2 step 1) on Ey only, with vacuum Ex and Ez."""
    from rfx.stepping.yee import update_e_aniso_inv

    shape = (8, 8, 8)
    dx = 0.001
    dt = dx / (3e8 * np.sqrt(3.0)) * 0.99

    state0 = _make_state_with_seeded_field(shape, seed=55)
    state0 = update_h(state0, _make_materials_constant(shape), dt, dx, high_wall=(False, False, False))

    materials = _make_materials_constant(shape, eps_r=1.0, sigma=0.0)
    ones = jnp.ones(shape, dtype=jnp.float32)
    inv_yy = jnp.full(shape, 0.36, dtype=jnp.float32)

    out = update_e_aniso_inv(state0, materials, ones, inv_yy, ones, dt, dx, high_wall=(False, False, False))

    assert np.all(np.isfinite(np.asarray(out.ex)))
    assert np.all(np.isfinite(np.asarray(out.ey)))
    assert np.all(np.isfinite(np.asarray(out.ez)))




def test_update_e_aniso_inv_jit_compatible():
    """``update_e_aniso_inv`` must be JIT-compilable (it will run inside
    the FDTD scan body in Stage 2 step 3). No Python-level conditionals
    on traced arrays."""
    import jax
    from rfx.stepping.yee import update_e_aniso_inv

    shape = (8, 8, 8)
    dx = 0.001
    dt = dx / (3e8 * np.sqrt(3.0)) * 0.99

    fn_jit = jax.jit(update_e_aniso_inv, static_argnames=('high_wall',))

    state0 = _make_state_with_seeded_field(shape, seed=1)
    materials = _make_materials_constant(shape, eps_r=2.0, sigma=0.5)
    inv_xx = jnp.full(shape, 0.5, dtype=jnp.float32)
    inv_yy = jnp.full(shape, 0.5, dtype=jnp.float32)
    inv_zz = jnp.full(shape, 0.5, dtype=jnp.float32)

    out = fn_jit(state0, materials, inv_xx, inv_yy, inv_zz, dt, dx,
                 high_wall=(False,) * 3)
    assert np.all(np.isfinite(np.asarray(out.ex)))
    assert np.all(np.isfinite(np.asarray(out.ey)))
    assert np.all(np.isfinite(np.asarray(out.ez)))
