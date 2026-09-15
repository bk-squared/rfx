"""Bit-identity gates for the #677 Stage-1 code-motion refactors.

Two helpers were factored out so the surface-impedance sheet operator can
share them with the validated kernels instead of carrying gated copies:

* ``rfx.boundaries.pec`` — the conductor edge rule, out of
  ``apply_pec_mask``;
* ``rfx.core.yee.curl_h`` / ``curl_h_nu`` — the curl(H) stencils, out of
  ``update_e`` / ``update_e_nu``.

These tests pin the refactored functions BYTE-EXACTLY to an inline copy of
the expressions they replaced (development-methodology refactor rule: a
code-motion refactor is gated by bit identity, not tolerance). If either
helper's algebra drifts, the drift shows up here before it shows up as a
subtle sheet-vs-kernel stencil mismatch in physics.

The conductor half of this lock was rewritten twice, and it is worth being
explicit about what moved.

* #689 replaced ``jnp.roll`` on every axis with a wrap only where cell 0
  and cell n-1 really are neighbours (a PERIODIC axis or a length-1 2-D
  axis) and a zero pad otherwise.
* #931 replaced the rule itself. ``tangential_edge_masks`` — masked cell
  AND a masked neighbour along the component's own axis — was a SHEET
  classification applied to volumes, so a body's far face was never a wall.
  It is deleted; :func:`rfx.boundaries.pec.realized_pec_edge_masks` is the
  one realization, and the volume branch is "an edge is PEC iff it is
  incident to an occupied cell" (four cells, backward shifts along the two
  axes transverse to the component).

The gate below follows the rule, not the old expression, and keeps its
teeth on the part that did NOT change — the #689 boundary convention:

  * the ``jnp.roll`` reference pins the PERIODIC branch byte-exactly;
  * a second reference built from ``_shift_bwd`` pins the non-periodic
    branch byte-exactly;
  * an interior-slice comparison pins that the periodic/non-periodic
    difference is confined to the boundary slices of each axis.

The boundary behaviour itself (both wrap-keeping guards and the 2-D
end-to-end witness) is pinned in
``tests/unit/boundaries/test_pec_mask_boundary_convention.py``; the
realization contract itself is
``tests/contracts/test_lattice_ownership_contract.py``.
"""

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "hand-derived (inline copy of the refactored expressions)",
    "commit": "f31ab907",
    "date": "2026-09-07",
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
    curl_h,
    curl_h_nu,
    init_state,
    update_e,
    update_e_nu,
    EPS_0,
)
from rfx.boundaries.pec import apply_pec_mask, realized_pec_edge_masks

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


def _inline_volume_rule(mask, shift):
    """The §1.2 rule spelled out: the four cells incident to each edge.

    ``shift(arr, ax)`` is the BACKWARD neighbour along ``ax`` under the
    caller's chosen boundary convention. Written as an explicit four-term
    OR (not the chained form the implementation uses) so the two spellings
    are genuinely independent.
    """
    out = []
    for c in range(3):
        t1, t2 = [t for t in range(3) if t != c]
        out.append(mask | shift(mask, t1) | shift(mask, t2)
                   | shift(shift(mask, t1), t2))
    return out


def test_realized_edges_bit_identity_with_inline_periodic_rule():
    """Periodic branch == an inline roll spelling of §1.2, exact booleans.

    ``_SHAPE`` is (9,8,7) with a random 30%-fill mask, so index 0 and index
    n-1 are populated on all three axes — the fixture DOES exercise the
    boundary, which is what made this gate the one that encoded the #689
    defect.
    """
    rng = np.random.default_rng(677)
    mask = jnp.asarray(rng.random(_SHAPE) < 0.3)

    ref = _inline_volume_rule(mask, lambda a, ax: jnp.roll(a, 1, axis=ax))
    got = realized_pec_edge_masks(mask, periodic=(True, True, True))
    for ax in range(3):
        assert np.array_equal(np.asarray(got[ax]), np.asarray(ref[ax])), ax


def test_realized_edges_bit_identity_with_inline_zero_pad_rule():
    """Non-periodic branch == an inline zero-pad spelling of §1.2 (#689)."""
    rng = np.random.default_rng(677)
    mask = jnp.asarray(rng.random(_SHAPE) < 0.3)

    ref = _inline_volume_rule(mask, _shift_bwd)
    got = realized_pec_edge_masks(mask)        # default: non-periodic
    for ax in range(3):
        assert np.array_equal(np.asarray(got[ax]), np.asarray(ref[ax])), ax


def test_realized_edges_interior_is_convention_independent():
    """The #689 wrap-vs-pad choice is confined to the boundary slices of the
    two axes TRANSVERSE to each component — the 'no interior number moves'
    gate."""
    rng = np.random.default_rng(677)
    mask = jnp.asarray(rng.random(_SHAPE) < 0.3)
    pad = realized_pec_edge_masks(mask)
    wrap = realized_pec_edge_masks(mask, periodic=(True, True, True))
    for c in range(3):
        sl = [slice(None)] * 3
        for t in range(3):
            if t != c:
                sl[t] = slice(1, None)
        sl = tuple(sl)
        assert np.array_equal(np.asarray(pad[c][sl]),
                              np.asarray(wrap[c][sl])), c
        # ... and it is NOT a no-op overall on this fixture, so the gate has
        # teeth: the transverse index-0 slices do differ.
        assert not np.array_equal(np.asarray(pad[c]), np.asarray(wrap[c])), c


def test_apply_pec_mask_uses_shared_rule_bit_identity():
    """apply_pec_mask output == hand-applied shared-rule zeroing, byte-exact."""
    rng = np.random.default_rng(678)
    st = _rand_state(rng)
    mask = jnp.asarray(rng.random(_SHAPE) < 0.3)
    mex, mey, mez = realized_pec_edge_masks(mask)
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
