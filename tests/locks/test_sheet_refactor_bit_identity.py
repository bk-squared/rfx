"""Bit-identity gates for the #677 Stage-1 code-motion refactors.

Two helpers were factored out so the surface-impedance sheet operator can
share them with the validated kernels instead of carrying gated copies:

* ``rfx.boundaries.pec.tangential_edge_masks`` — the thin-sheet neighbor
  rule, out of ``apply_pec_mask``;
* ``rfx.core.yee.curl_h`` / ``curl_h_nu`` — the curl(H) stencils, out of
  ``update_e`` / ``update_e_nu``.

These tests pin the refactored functions BYTE-EXACTLY to an inline copy of
the pre-refactor expressions (development-methodology refactor rule: a
code-motion refactor is gated by bit identity, not tolerance). If either
helper's algebra drifts, the drift shows up here before it shows up as a
subtle sheet-vs-kernel stencil mismatch in physics.

#689 moved one of the two references, and it is worth being explicit about
what moved and what did not. ``tangential_edge_masks`` used ``jnp.roll`` on
every axis; the wrap is correct only where cell 0 and cell n-1 really are
neighbours, so it now applies on a PERIODIC axis or a length-1 (2-D) axis
and a zero pad applies otherwise. The gate below is rewritten, not
loosened, and it got teeth rather than losing them:

  * the ``jnp.roll`` reference is kept and now pins the PERIODIC branch
    byte-exactly, so that branch is still literally the pre-#689 rule;
  * a second reference built from ``_shift_bwd`` / ``_shift_fwd`` pins the
    non-periodic branch byte-exactly;
  * an interior-slice comparison pins that the change is confined to the
    two boundary slices of each axis — which is why no committed physics
    number moves.

The boundary behaviour itself (the defect, both wrap-keeping guards, and
the 2-D end-to-end witness) is pinned in
``tests/unit/boundaries/test_pec_mask_boundary_convention.py``.
"""

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "hand-derived (inline copy of the pre-refactor expressions)",
    "commit": "13de212",
    "date": "2026-08-20",
    "run_id": "local",
    "host": "JAX cpu float32 (os / jax version not recorded in #678)",
    "pinned_until": "2027-02-16",
}

import numpy as np
import jax.numpy as jnp

from rfx.core.yee import (
    MaterialArrays,
    _diff_bwd_o,
    _shift_bwd,
    _shift_fwd,
    curl_h,
    curl_h_nu,
    init_state,
    update_e,
    update_e_nu,
    EPS_0,
)
from rfx.boundaries.pec import apply_pec_mask, tangential_edge_masks

_SHAPE = (9, 8, 7)


def _rand_state(rng):
    st = init_state(_SHAPE)
    f = lambda: jnp.asarray(rng.standard_normal(_SHAPE).astype(np.float32))
    return st._replace(ex=f(), ey=f(), ez=f(), hx=f(), hy=f(), hz=f())


def _rand_materials(rng):
    return MaterialArrays(
        eps_r=jnp.asarray(1.0 + rng.random(_SHAPE).astype(np.float32) * 3),
        sigma=jnp.asarray(rng.random(_SHAPE).astype(np.float32) * 5),
        mu_r=jnp.ones(_SHAPE, jnp.float32),
    )


def test_tangential_edge_masks_bit_identity_with_inline_rule():
    """Periodic branch == the pre-#677/#689 inline roll rule, exact booleans.

    ``_SHAPE`` is (9,8,7) with a random 30%-fill mask, so index 0 and index
    n-1 are populated on all three axes — the fixture DOES exercise the
    boundary, which is what made this gate the one that encoded the defect.
    """
    rng = np.random.default_rng(677)
    mask = jnp.asarray(rng.random(_SHAPE) < 0.3)

    # Inline copy of the pre-refactor apply_pec_mask expressions.
    ref_ex = mask & (jnp.roll(mask, 1, axis=0) | jnp.roll(mask, -1, axis=0))
    ref_ey = mask & (jnp.roll(mask, 1, axis=1) | jnp.roll(mask, -1, axis=1))
    ref_ez = mask & (jnp.roll(mask, 1, axis=2) | jnp.roll(mask, -1, axis=2))

    got_ex, got_ey, got_ez = tangential_edge_masks(mask, (True, True, True))
    assert np.array_equal(np.asarray(got_ex), np.asarray(ref_ex))
    assert np.array_equal(np.asarray(got_ey), np.asarray(ref_ey))
    assert np.array_equal(np.asarray(got_ez), np.asarray(ref_ez))


def test_tangential_edge_masks_bit_identity_with_inline_zero_pad_rule():
    """Non-periodic branch == an inline zero-pad rule, exact booleans (#689)."""
    rng = np.random.default_rng(677)
    mask = jnp.asarray(rng.random(_SHAPE) < 0.3)

    ref = [mask & (_shift_bwd(mask, ax) | _shift_fwd(mask, ax))
           for ax in range(3)]
    got = tangential_edge_masks(mask)          # default: non-periodic
    for ax in range(3):
        assert np.array_equal(np.asarray(got[ax]), np.asarray(ref[ax])), ax


def test_tangential_edge_masks_interior_bit_identity_with_the_pre_689_rule():
    """The #689 change is confined to the two boundary slices of each axis,
    which is the 'no committed number moves' gate."""
    rng = np.random.default_rng(677)
    mask = jnp.asarray(rng.random(_SHAPE) < 0.3)
    got = tangential_edge_masks(mask)
    for ax in range(3):
        old = mask & (jnp.roll(mask, 1, axis=ax) | jnp.roll(mask, -1, axis=ax))
        sl = [slice(None)] * 3
        sl[ax] = slice(1, -1)
        sl = tuple(sl)
        assert np.array_equal(np.asarray(got[ax][sl]),
                              np.asarray(old[sl])), ax
        # ... and it is NOT a no-op overall on this fixture, so the gate has
        # teeth: index 0 / n-1 do differ.
        assert not np.array_equal(np.asarray(got[ax]), np.asarray(old)), ax


def test_apply_pec_mask_uses_shared_rule_bit_identity():
    """apply_pec_mask output == hand-applied shared-rule zeroing, byte-exact."""
    rng = np.random.default_rng(678)
    st = _rand_state(rng)
    mask = jnp.asarray(rng.random(_SHAPE) < 0.3)
    mex, mey, mez = tangential_edge_masks(mask)
    out = apply_pec_mask(st, mask)
    ref_ex = st.ex * (1.0 - mex.astype(st.ex.dtype))
    ref_ey = st.ey * (1.0 - mey.astype(st.ey.dtype))
    ref_ez = st.ez * (1.0 - mez.astype(st.ez.dtype))
    assert np.asarray(out.ex).tobytes() == np.asarray(ref_ex).tobytes()
    assert np.asarray(out.ey).tobytes() == np.asarray(ref_ey).tobytes()
    assert np.asarray(out.ez).tobytes() == np.asarray(ref_ez).tobytes()


def test_curl_h_bit_identity_with_inline_stencil():
    """curl_h == the pre-#677 inline update_e stencil, byte-exact, for
    stencil orders 2/4 x periodic on/off, plus the complex Bloch path."""
    rng = np.random.default_rng(679)
    st = _rand_state(rng)
    dx = 1e-3
    cases = [
        (2, (False, False, False), None, jnp.float32),
        (2, (True, False, True), None, jnp.float32),
        (4, (False, False, False), None, jnp.float32),
        (4, (True, False, True), None, jnp.float32),
        (2, (True, True, True),
         tuple(complex(np.exp(-1j * 0.3 * (i + 1))) for i in range(3)),
         jnp.complex64),
    ]
    for so, per, bloch, cdt in cases:
        hx = st.hx.astype(cdt)
        hy = st.hy.astype(cdt)
        hz = st.hz.astype(cdt)
        ref_x = (_diff_bwd_o(hz, 1, per, so, bloch) / dx
                 - _diff_bwd_o(hy, 2, per, so, bloch) / dx)
        ref_y = (_diff_bwd_o(hx, 2, per, so, bloch) / dx
                 - _diff_bwd_o(hz, 0, per, so, bloch) / dx)
        ref_z = (_diff_bwd_o(hy, 0, per, so, bloch) / dx
                 - _diff_bwd_o(hx, 1, per, so, bloch) / dx)
        got = curl_h(hx, hy, hz, dx, per, so, bloch)
        for g, r in zip(got, (ref_x, ref_y, ref_z)):
            assert np.asarray(g).tobytes() == np.asarray(r).tobytes(), (
                so, per, bloch)


def test_curl_h_nu_bit_identity_with_inline_stencil():
    rng = np.random.default_rng(680)
    st = _rand_state(rng)
    inv_dx, inv_dy, inv_dz = (
        jnp.asarray(rng.random(n).astype(np.float32) + 0.5) for n in _SHAPE)
    hx, hy, hz = st.hx, st.hy, st.hz
    ref_x = ((hz - _shift_bwd(hz, 1)) * inv_dy[None, :, None]
             - (hy - _shift_bwd(hy, 2)) * inv_dz[None, None, :])
    ref_y = ((hx - _shift_bwd(hx, 2)) * inv_dz[None, None, :]
             - (hz - _shift_bwd(hz, 0)) * inv_dx[:, None, None])
    ref_z = ((hy - _shift_bwd(hy, 0)) * inv_dx[:, None, None]
             - (hx - _shift_bwd(hx, 1)) * inv_dy[None, :, None])
    got = curl_h_nu(hx, hy, hz, inv_dx, inv_dy, inv_dz)
    for g, r in zip(got, (ref_x, ref_y, ref_z)):
        assert np.asarray(g).tobytes() == np.asarray(r).tobytes()


def test_update_kernels_bit_identity_via_curl_helpers():
    """The refactored kernels equal a jit'd hand-run of coeffs + shared
    curls to a few float32 ULP. Two *different* jitted graphs are not
    byte-comparable (XLA fuses ca*E+cb*curl differently per graph), so this
    functional check is tolerance-based; the byte-exact pre==post refactor
    gate was run as a SHA-256 digest comparison of update_e / update_e_nu /
    apply_pec_mask outputs on fixed random fixtures before and after the
    #677 code motion — all seven digests unchanged (recorded in the #677 PR
    body). The helper-level tests above stay byte-exact (eager vs eager)."""
    import jax

    rng = np.random.default_rng(681)
    st = _rand_state(rng)
    mats = _rand_materials(rng)
    dt, dx = 1e-12, 1e-3
    per = (False, False, False)

    @jax.jit
    def ref_uniform(st, mats):
        eps = mats.eps_r * EPS_0
        loss = mats.sigma * dt / (2.0 * eps)
        ca = (1.0 - loss) / (1.0 + loss)
        cb = (dt / eps) / (1.0 + loss)
        cx, cy, cz = curl_h(st.hx, st.hy, st.hz, dx, per, 2, None)
        return (ca * st.ex + cb * cx,
                ca * st.ey + cb * cy,
                ca * st.ez + cb * cz)

    ref = ref_uniform(st, mats)
    got = update_e(st, mats, dt, dx, periodic=per)
    for g, r in zip((got.ex, got.ey, got.ez), ref):
        r_np = np.asarray(r)
        np.testing.assert_allclose(np.asarray(g), r_np, rtol=1e-5,
                                   atol=1e-5 * np.abs(r_np).max())

    inv = [jnp.asarray(rng.random(n).astype(np.float32) + 0.5) for n in _SHAPE]

    @jax.jit
    def ref_nu(st, mats, inv_dx, inv_dy, inv_dz):
        eps = mats.eps_r * EPS_0
        loss = mats.sigma * dt / (2.0 * eps)
        ca = (1.0 - loss) / (1.0 + loss)
        cb = (dt / eps) / (1.0 + loss)
        cx, cy, cz = curl_h_nu(st.hx, st.hy, st.hz, inv_dx, inv_dy, inv_dz)
        return (ca * st.ex + cb * cx,
                ca * st.ey + cb * cy,
                ca * st.ez + cb * cz)

    ref = ref_nu(st, mats, *inv)
    got = jax.jit(update_e_nu)(st, mats, dt, *inv)
    for g, r in zip((got.ex, got.ey, got.ez), ref):
        r_np = np.asarray(r)
        np.testing.assert_allclose(np.asarray(g), r_np, rtol=1e-5,
                                   atol=1e-5 * np.abs(r_np).max())
