"""Bit-identity / correctness gates for the #404 Phase-B complex+Bloch yee path.

The oblique-periodic Bloch field-transformation adds an OPT-IN complex path to
``update_e``/``update_h`` (a per-axis ``bloch`` phase on the periodic rolls).
The real path MUST stay byte-identical and the complex path MUST reduce to the
real path at zero transverse wavenumber (bloch phase = 1).  These gates pin that
contract so a future edit cannot silently perturb the real solver.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.core.yee import (
    FDTDState, MaterialArrays, update_e, update_h, EPS_0, MU_0,
)

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)
SHAPE = (16, 12, 10)


def _state(dtype):
    k = jax.random.split(jax.random.PRNGKey(404), 6)
    r = lambda i: jax.random.normal(k[i], SHAPE, dtype=jnp.float32).astype(dtype)
    return FDTDState(ex=r(0), ey=r(1), ez=r(2), hx=r(3), hy=r(4), hz=r(5),
                     step=jnp.array(0, jnp.int32))


def _materials():
    k = jax.random.split(jax.random.PRNGKey(99), 2)
    eps_r = 1.0 + 3.0 * jax.random.uniform(k[0], SHAPE, dtype=jnp.float32)
    sigma = 0.01 * jax.random.uniform(k[1], SHAPE, dtype=jnp.float32)
    return MaterialArrays(eps_r=eps_r, sigma=sigma,
                          mu_r=jnp.ones(SHAPE, jnp.float32))


DX = 0.002
DT = 0.5 * DX / C0


@pytest.mark.parametrize("periodic", [(False, False, False),
                                      (False, True, True),
                                      (True, True, True)])
@pytest.mark.parametrize("order", [2, 4])
def test_real_path_unaffected_by_new_signature(periodic, order):
    """bloch=None (default) leaves the real float32 path byte-identical: the
    added parameter and complex-aware dtype must not perturb the default call."""
    mat = _materials()
    st_a = _state(jnp.float32)
    st_b = _state(jnp.float32)
    for _ in range(6):
        st_a = update_h(st_a, mat, DT, DX, periodic=periodic, stencil_order=order)
        st_a = update_e(st_a, mat, DT, DX, periodic=periodic, stencil_order=order)
        # explicit bloch=None must be identical to omitting it
        st_b = update_h(st_b, mat, DT, DX, periodic=periodic, stencil_order=order,
                        bloch=None)
        st_b = update_e(st_b, mat, DT, DX, periodic=periodic, stencil_order=order,
                        bloch=None)
    for f in ("ex", "ey", "ez", "hx", "hy", "hz"):
        assert np.array_equal(np.asarray(getattr(st_a, f)),
                              np.asarray(getattr(st_b, f)))


@pytest.mark.parametrize("periodic", [(False, True, True), (True, True, True)])
def test_complex_zero_bloch_reduces_to_real(periodic):
    """Complex fields with a unit Bloch phase (k=0) reproduce the real path in
    their real part (the complex path is a strict superset at zero wavenumber)."""
    mat = _materials()
    st_r = _state(jnp.float32)
    st_c = _state(jnp.complex64)
    bloch = (1.0 + 0j, 1.0 + 0j, 1.0 + 0j)
    for _ in range(6):
        st_r = update_h(st_r, mat, DT, DX, periodic=periodic)
        st_r = update_e(st_r, mat, DT, DX, periodic=periodic)
        st_c = update_h(st_c, mat, DT, DX, periodic=periodic, bloch=bloch)
        st_c = update_e(st_c, mat, DT, DX, periodic=periodic, bloch=bloch)
    for f in ("ex", "ey", "ez", "hx", "hy", "hz"):
        r = np.asarray(getattr(st_r, f))
        c = np.asarray(getattr(st_c, f))
        # ~1e-4 (not bit-level): complex64 real-channel arithmetic fuses/rounds
        # differently from float32 (FMA/op-ordering), so the numerical reduction
        # drifts ~1e-5 rel over 6 steps. The BYTE-identity gate is the real-path
        # test above; this checks the complex path REDUCES to the real physics.
        scale = max(float(np.max(np.abs(r))), 1.0)
        assert np.allclose(r, c.real, rtol=2e-4, atol=1e-4 * scale), f
        assert np.max(np.abs(c.imag)) < 1e-4 * scale, f"{f}: unit bloch leaked into imag"


@pytest.mark.parametrize("fn", [update_e, update_h])
def test_bloch_requires_order2(fn):
    """A non-trivial Bloch phase with a 4th-order stencil is fail-loud (the wide
    stencil's far rolls need their own phase, not implemented)."""
    mat = _materials()
    st = _state(jnp.complex64)
    with pytest.raises(ValueError, match="stencil_order=2"):
        fn(st, mat, DT, DX, periodic=(False, True, True), stencil_order=4,
           bloch=(1.0 + 0j, 1.0 + 0.5j, 1.0 + 0j))


def _kernel_prop_angle(theta_deg, ny=16, nx=600):
    """Injected Poynting angle from the REAL update_e/h complex+Bloch path.

    CPML-FREE, soft-source, measured in a clean interior window ahead of the
    source before any x-boundary reflection — the trustworthy injection-angle
    witness for #404 (the in-rfx CPML-terminated path contaminates steep angles
    via imperfect absorption; see the Phase-B session findings)."""
    from rfx.core.yee import init_state, init_materials
    shape = (nx, ny, 1)
    mat = init_materials(shape)
    f0 = 5e9
    k0 = 2 * np.pi * f0 / C0
    dx = 0.002
    dt = 0.5 * dx / C0
    tau = 1.0 / (np.pi * f0 * 0.3)
    t0 = 5 * tau
    kY = -k0 * np.sin(np.radians(theta_deg))
    bloch = (1.0 + 0j, complex(np.exp(-1j * kY * dx)), 1.0 + 0j)
    st = init_state(shape, field_dtype=jnp.complex64)
    yph = np.exp(-1j * kY * np.arange(ny) * dx)
    src_x = 60
    xl, xh = src_x + 60, src_x + 220
    Sx = Sy = 0.0
    for step in range(700):
        t = step * dt
        st = update_h(st, mat, dt, dx, periodic=(False, True, True), bloch=bloch)
        st = update_e(st, mat, dt, dx, periodic=(False, True, True), bloch=bloch)
        src = np.exp(-1j * 2 * np.pi * f0 * (t - t0)) * np.exp(-((t - t0) / tau) ** 2)
        st = st._replace(ez=st.ez.at[src_x, :, 0].add(complex(src)))
        if step >= 700 // 3:
            ez = np.asarray(st.ez[xl:xh, :, 0]) * yph[None, :]
            hx = np.asarray(st.hx[xl:xh, :, 0]) * yph[None, :]
            hy = np.asarray(st.hy[xl:xh, :, 0]) * yph[None, :]
            Sx += float(np.sum(-ez.real * hy.real))
            Sy += float(np.sum(ez.real * hx.real))
    # y-uniformity witness (Snell k_y conservation ⇒ envelope P uniform in y)
    ezw = np.abs(np.asarray(st.ez[xl:xh, :, 0]))
    ystd = float(ezw.std(axis=1).mean() / max(ezw.mean(), 1e-30))
    return np.degrees(np.arctan2(Sy, Sx)), ystd


def test_oblique_injection_angle_tracks_request():
    """#404 core gate: the injected wave propagates at the REQUESTED oblique angle
    (the original bug injected ~0.4x, e.g. 60deg->25deg). Normal is the known-good
    (theta~0, Sx>0); 30/45/60 track within a numerical-dispersion margin (the ~5deg
    deficit at 60deg is dx=lambda/30 dispersion, not the gross under-tilt bug)."""
    a0, ys0 = _kernel_prop_angle(0.0)
    assert abs(a0) < 5.0, f"normal incidence must flow along +x, got {a0:.1f}"
    for req, lo, hi in [(30.0, 27.0, 33.0), (45.0, 41.0, 49.0), (60.0, 51.0, 62.0)]:
        a, ys = _kernel_prop_angle(req)
        assert lo < a < hi, f"{req}deg injected angle {a:.1f} outside [{lo},{hi}]"
        assert ys < 1e-3, f"{req}deg envelope not y-uniform (yStd={ys:.1e})"


def test_bloch_path_is_differentiable():
    """AD gate (#404 new numerics path — hard constraint): jax.grad w.r.t. a
    material permittivity flows FINITE and NONZERO through the complex+Bloch
    update_e/h path, so oblique-periodic scattering is usable for inverse design.
    """
    from rfx.core.yee import init_state, init_materials
    nx, ny = 80, 16
    shape = (nx, ny, 1)
    dx = 0.002
    dt = 0.5 * dx / C0
    f0 = 5e9
    k0 = 2 * np.pi * f0 / C0
    kY = -k0 * np.sin(np.radians(45.0))
    bloch = (1.0 + 0j, complex(np.exp(-1j * kY * dx)), 1.0 + 0j)
    per = (False, True, True)
    tau = 1.0 / (np.pi * f0 * 0.15)
    t0 = 5 * tau

    def obj(eps_slab):
        mat = init_materials(shape)
        mat = mat._replace(eps_r=mat.eps_r.at[nx // 2:, :, :].set(eps_slab))
        st = init_state(shape, field_dtype=jnp.complex64)

        def body(st, step):
            t = step * dt
            st = update_h(st, mat, dt, dx, periodic=per, bloch=bloch)
            st = update_e(st, mat, dt, dx, periodic=per, bloch=bloch)
            src = jnp.exp(-1j * 2 * jnp.pi * f0 * (t - t0)) * jnp.exp(-((t - t0) / tau) ** 2)
            st = st._replace(ez=st.ez.at[20, :, 0].add(src.astype(jnp.complex64)))
            return st, None

        st, _ = jax.lax.scan(body, st, jnp.arange(300))
        return jnp.sum(jnp.abs(st.ez[nx // 2 - 10:nx // 2, :, 0]) ** 2).real

    g = jax.grad(obj)(4.0)
    assert np.isfinite(g), f"grad not finite: {g}"
    assert abs(float(g)) > 0.0, "grad is zero through the complex Bloch path"


def test_oblique_injection_angle_domain_size_invariant():
    """The fix is domain-size INVARIANT (the broken plain-periodic baseline swung
    46.9->57.6deg with transverse size). A correct Bloch BC gives the same angle
    for commensurate and non-commensurate transverse periods."""
    angles = [_kernel_prop_angle(60.0, ny=ny)[0] for ny in (12, 16, 24, 40)]
    spread = max(angles) - min(angles)
    assert spread < 2.0, f"60deg angle varies {spread:.1f}deg across ny (should be ~0): {angles}"
