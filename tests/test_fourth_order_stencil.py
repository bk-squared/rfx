"""Contract + bit-identity tests for the (2,4) fourth-order-in-space stencil
option on the core ``update_e`` / ``update_h`` kernels (``rfx/core/yee.py``).

Provenance: the A1 spike (``docs/research_notes/20260627_memory_efficiency_techniques_exploration.md``
+ ``docs/research_notes/experiments/a1_fourth_order_spike/``). The 4th-order
option cuts cells-per-WAVELENGTH ~2.5x for the same dispersion error (a SMOOTH-
PROPAGATION / electrically-large lever). Gate-2 established it gives NO benefit
at PEC / geometric features (it reverts to 2nd order in the 1-cell boundary
ribbon and PEC staircases at 2nd order regardless), and does NOT replace
NU/subgridding for geometric multi-scale. ``stencil_order=2`` is the byte-
identical default; this PR adds the kernel-level option only (threading through
the Simulation API + CPML-4th + per-path fences are follow-ups).

NOTE: the 4th-order convergence is verified on COARSE meshes only — in the
production float32 path the 4th-order truncation error drops below the ~1e-6
float32 round-off floor at fine meshes, so a fine-mesh slope would saturate.
This test deliberately does NOT enable jax x64 (process-global flip would turn
every pytest-split shard red — see feedback_jax_x64_module_level_tests).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import (
    EPS_0, MU_0, MaterialArrays, _diff_bwd_o, _diff_fwd_o, _shift_bwd,
    _shift_fwd, init_state, update_e, update_h,
)


def test_order2_helpers_byte_identical():
    """order=2 difference helpers == the original inline staggered diff, exactly."""
    arr = jax.random.normal(jax.random.PRNGKey(1), (10, 7, 6), dtype=jnp.float32)
    for ax in range(3):
        for per in (True, False):
            pe = tuple(per if i == ax else False for i in range(3))
            ref_f = (jnp.roll(arr, -1, ax) if per else _shift_fwd(arr, ax)) - arr
            ref_b = arr - (jnp.roll(arr, 1, ax) if per else _shift_bwd(arr, ax))
            assert jnp.array_equal(_diff_fwd_o(arr, ax, pe, 2), ref_f)
            assert jnp.array_equal(_diff_bwd_o(arr, ax, pe, 2), ref_b)


def _rand_state_mats(shape=(8, 7, 6)):
    ks = jax.random.split(jax.random.PRNGKey(0), 6)
    st = init_state(shape)._replace(
        ex=jax.random.normal(ks[0], shape), ey=jax.random.normal(ks[1], shape),
        ez=jax.random.normal(ks[2], shape), hx=jax.random.normal(ks[3], shape),
        hy=jax.random.normal(ks[4], shape), hz=jax.random.normal(ks[5], shape))
    mats = MaterialArrays(eps_r=jnp.full(shape, 2.0), sigma=jnp.zeros(shape),
                          mu_r=jnp.ones(shape))
    return st, mats


def test_update_order2_state_byte_identical():
    """BIT-IDENTITY GATE: stencil_order=2 update == explicit 2nd-order curl."""
    st, mats = _rand_state_mats()
    dt, dx, per = 1e-13, 1e-3, (False, True, False)

    def bwd(a, ax):
        return jnp.roll(a, 1, ax) if per[ax] else _shift_bwd(a, ax)

    def fwd(a, ax):
        return jnp.roll(a, -1, ax) if per[ax] else _shift_fwd(a, ax)

    e2 = update_e(st, mats, dt, dx, per, stencil_order=2)
    h2 = update_h(st, mats, dt, dx, per, stencil_order=2)

    # E update reference (sigma=0 -> ca=1, cb=dt/eps). ey's curl_y spans AXIS 0
    # (via bwd(hz, 0)), so asserting .ex AND .ey covers all three diff axes.
    hx, hy, hz = (st.hx.astype(jnp.float32), st.hy.astype(jnp.float32),
                  st.hz.astype(jnp.float32))
    cb = dt / (mats.eps_r * EPS_0)
    ex_ref = (st.ex.astype(jnp.float32)
              + cb * ((hz - bwd(hz, 1)) / dx - (hy - bwd(hy, 2)) / dx)).astype(st.ex.dtype)
    ey_ref = (st.ey.astype(jnp.float32)
              + cb * ((hx - bwd(hx, 2)) / dx - (hz - bwd(hz, 0)) / dx)).astype(st.ey.dtype)
    assert jnp.array_equal(e2.ex, ex_ref)
    assert jnp.array_equal(e2.ey, ey_ref)

    # H update reference; hy's curl_y spans AXIS 0 (via fwd(ez, 0)).
    ex, ey, ez = (st.ex.astype(jnp.float32), st.ey.astype(jnp.float32),
                  st.ez.astype(jnp.float32))
    cm = dt / (mats.mu_r * MU_0)
    hx_ref = (st.hx.astype(jnp.float32)
              - cm * ((fwd(ez, 1) - ez) / dx - (fwd(ey, 2) - ey) / dx)).astype(st.hx.dtype)
    hy_ref = (st.hy.astype(jnp.float32)
              - cm * ((fwd(ex, 2) - ex) / dx - (fwd(ez, 0) - ez) / dx)).astype(st.hy.dtype)
    assert jnp.array_equal(h2.hx, hx_ref)
    assert jnp.array_equal(h2.hy, hy_ref)


def test_stencil_convergence_order():
    """CONTRACT: the staggered first-difference is O(h^2) at order=2 and O(h^4)
    at order=4 (coarse meshes, float32-safe — no x64 flip)."""
    for order, expect in ((2, 2.0), (4, 4.0)):
        errs, hs = [], []
        for n in (8, 12, 16, 24, 32):
            x = np.linspace(0, 1, n, endpoint=False)
            h = 1.0 / n
            a = jnp.asarray(np.sin(2 * np.pi * x).astype(np.float32)).reshape(n, 1, 1)
            d = np.asarray(_diff_fwd_o(a, 0, (True, False, False), order)[:, 0, 0]) / h
            exact = 2 * np.pi * np.cos(2 * np.pi * (x + h / 2))
            errs.append(float(np.max(np.abs(d.astype(np.float64) - exact))))
            hs.append(h)
        slope = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
        assert abs(slope - expect) < 0.4, f"order {order}: slope {slope:.2f} != {expect}"


def test_order4_is_more_accurate_at_coarse_mesh():
    """The whole point: order=4 at a COARSE mesh beats order=2 at a finer one."""
    def deriv_err(order, n):
        x = np.linspace(0, 1, n, endpoint=False)
        h = 1.0 / n
        a = jnp.asarray(np.sin(2 * np.pi * x).astype(np.float32)).reshape(n, 1, 1)
        d = np.asarray(_diff_fwd_o(a, 0, (True, False, False), order)[:, 0, 0]) / h
        return float(np.max(np.abs(d.astype(np.float64) - 2 * np.pi * np.cos(2 * np.pi * (x + h / 2)))))
    assert deriv_err(4, 16) < deriv_err(2, 48)  # 3x coarser, still more accurate


def test_order4_nonperiodic_stable():
    """Multi-step non-periodic order=4 stays bounded for a SMOOTH (band-limited)
    field — guards the 2-cell boundary ribbon. A RANDOM IC is deliberately NOT
    used: its near-Nyquist content transiently amplifies the max ~150x for BOTH
    order 2 and order 4 (a generic Yee group-velocity artifact, not an order-4
    instability), which would mask a real boundary blow-up. A smooth Gaussian
    bump (peak 1.0) must stay bounded < 2.0 over 500 steps."""
    nx, ny, nz = 24, 22, 20
    st = init_state((nx, ny, nz))
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    bump = np.exp(-(((gx - nx / 2) / 4) ** 2 + ((gy - ny / 2) / 4) ** 2
                    + ((gz - nz / 2) / 4) ** 2)).astype(np.float32)
    st = st._replace(ez=jnp.asarray(bump))
    mats = MaterialArrays(eps_r=jnp.ones((nx, ny, nz)), sigma=jnp.zeros((nx, ny, nz)),
                          mu_r=jnp.ones((nx, ny, nz)))
    dx, c = 1e-3, 299_792_458.0
    dt = 0.5 * dx / (c * np.sqrt(3))
    per = (False, False, False)
    peak = 0.0
    for _ in range(500):
        st = update_h(st, mats, dt, dx, per, stencil_order=4)
        st = update_e(st, mats, dt, dx, per, stencil_order=4)
        peak = max(peak, float(jnp.max(jnp.abs(st.ez))))
    assert np.isfinite(peak) and peak < 2.0, f"order4 not bounded: peak={peak}"


def test_invalid_stencil_order_raises():
    """Only 2 and 4 are valid — order=3/6 must raise, not silently run order=4."""
    st, mats = _rand_state_mats()
    for bad in (1, 3, 6):
        with pytest.raises(ValueError):
            update_e(st, mats, 1e-13, 1e-3, (False, False, False), stencil_order=bad)
        with pytest.raises(ValueError):
            update_h(st, mats, 1e-13, 1e-3, (False, False, False), stencil_order=bad)


def test_order4_ad_finite_and_matches_fd():
    """order=4 stays AD-traceable (plain jax.grad, no custom_vjp) and the grad
    matches central finite-difference."""
    st, mats = _rand_state_mats()
    dt, dx, per = 1e-13, 1e-3, (False, True, False)

    def loss(scale):
        m2 = mats._replace(eps_r=mats.eps_r * scale)
        return jnp.sum(update_e(st, m2, dt, dx, per, stencil_order=4).ex ** 2)

    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(g) and abs(g) > 0
    fd = float((loss(1.0 + 1e-3) - loss(1.0 - 1e-3)) / 2e-3)
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-2, f"AD {g} vs FD {fd}"
