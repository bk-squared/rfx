"""Radiated-power sphere integral: periodic phi weights (#2618).

A Hertzian dipole has the closed-form directivity D = 1.5 (1.7609 dBi) on any
full-turn phi grid. phi is periodic, so the azimuthal weights must sum to 2*pi
for any spacing, and a sample repeated at the 0/2*pi seam is one direction.
Expected values here are closed forms or formed in the test.
"""

import sys

import jax.numpy as jnp
import numpy as np
import pytest

import rfx.measurement._radiated_power as radiated_power_module
from rfx.farfield import FarFieldResult, directivity
from rfx.measurement._radiated_power import integrate_radiated_power

D_DIPOLE_DBI = 10 * np.log10(1.5)
THETA = np.linspace(0, np.pi, 181)

_EVEN_PHI = [
    pytest.param(np.linspace(0, 2 * np.pi, 13), id="13-endpoint"),
    pytest.param(np.linspace(0, 2 * np.pi, 13, endpoint=False), id="13-open"),
    pytest.param(np.linspace(0, 2 * np.pi, 49), id="49-endpoint"),
    pytest.param(np.linspace(0, 2 * np.pi, 49, endpoint=False), id="49-open"),
]
_ALL_PHI = _EVEN_PHI + [pytest.param(np.array([0.0, 1.0, 2.0, 5.0]), id="uneven-0-1-2-5")]


def _z_dipole(theta, phi):
    # U = sin^2(theta).
    e_theta = np.sin(theta)[None, :, None] * np.ones((1, 1, len(phi)), complex)
    return FarFieldResult(e_theta, np.zeros_like(e_theta), theta, phi, np.array([3e9]))


def _x_dipole(theta, phi):
    # U = cos^2(theta) cos^2(phi) + sin^2(phi) = 1 - sin^2(theta) cos^2(phi).
    e_theta = (np.cos(theta)[:, None] * np.cos(phi)[None, :])[None].astype(complex)
    e_phi = np.broadcast_to(-np.sin(phi)[None, None, :], e_theta.shape).astype(complex)
    return FarFieldResult(e_theta, e_phi, theta, phi, np.array([3e9]))


@pytest.mark.parametrize("phi", _ALL_PHI)
def test_z_dipole_directivity(phi):
    # The theta rule's error on 181 samples is h^4/80 relative (5e-9 dB); the
    # phi weights sum to 2*pi, so phi adds rounding only.
    d = directivity(_z_dipole(THETA, phi))[0]
    assert abs(d - D_DIPOLE_DBI) <= 1e-6, d


@pytest.mark.parametrize("phi", _EVEN_PHI)
def test_x_dipole_directivity(phi):
    # The theta rule's error is led by the h^2/6 term of the 2*pi*sin(theta)
    # part, 1.7e-4 dB on 181 samples.
    d = directivity(_x_dipole(THETA, phi))[0]
    assert abs(d - D_DIPOLE_DBI) <= 1e-3, d


@pytest.mark.parametrize("phi", _ALL_PHI)
def test_phi_weight_sum_through_public_path(phi):
    # theta = pi/2 has sin = 1 and an np.gradient cell of exactly 0.25, so an
    # intensity of 4 there (0 elsewhere) makes each summand the phi weight.
    theta = np.array([0.0, np.pi / 2 - 0.25, np.pi / 2, np.pi / 2 + 0.25, np.pi])
    intensity = np.zeros((1, len(theta), len(phi)))
    intensity[0, 2, :] = 4.0
    total = integrate_radiated_power(intensity, theta, phi)[0]
    assert abs(total - 2 * np.pi) <= 8 * np.spacing(2 * np.pi), total - 2 * np.pi


def _test_periodic_weights(phi):
    # Periodic trapezoid weights on the unique samples; with a seam repeat,
    # half the first interval and half the interval up to phi[0] + 2*pi.
    # Formed here in float64.
    seam = phi[-1] - phi[0] >= 2 * np.pi - 1e-9
    u = phi[:-1] if seam else phi
    m = len(u)
    w = np.empty(m)
    for i in range(m):
        nxt = u[i + 1] if i + 1 < m else u[0] + 2 * np.pi
        prv = u[i - 1] if i > 0 else u[m - 1] - 2 * np.pi
        w[i] = (nxt - prv) / 2
    if seam:
        w = np.append(w, (phi[0] + 2 * np.pi - phi[-2]) / 2)
        w[0] = (phi[1] - phi[0]) / 2
    return w


def test_r1_seam_weights_equal_closed_trapezoid():
    phi = np.linspace(0, 2 * np.pi, 73)
    expected = np.empty(73)
    expected[1:-1] = 0.5 * (phi[2:] - phi[:-2])
    expected[0] = 0.5 * (phi[1] - phi[0])
    expected[-1] = 0.5 * (phi[72] - phi[71])
    weights = radiated_power_module._phi_weights(phi)
    assert weights.dtype == np.float64
    ulps = np.abs(weights - expected) / np.spacing(expected)
    print(f"r1 max ulp gap per index: {ulps.max():.1f} at {int(np.argmax(ulps))}; "
          f"ends {ulps[0]:.1f}, {ulps[-1]:.1f}", file=sys.stderr)
    assert np.all(ulps <= 2), ulps


def _f32_steps(a, b):
    ia = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    ib = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


@pytest.mark.parametrize("pattern", ["z-dipole", "x-dipole"])
def test_r2_objective_grid_moves_within_rounding_bound(pattern):
    theta = np.linspace(0.0, np.pi, 37)
    phi = np.linspace(0.0, 2.0 * np.pi, 73)
    st, cp = np.sin(theta)[:, None], np.cos(phi)[None, :]
    u_np = (st ** 2 * np.ones_like(cp)) if pattern == "z-dipole" else (1 - st ** 2 * cp ** 2)
    intensity = jnp.asarray(u_np[None], dtype=jnp.float32)

    p_new = integrate_radiated_power(intensity, theta, phi, jax_trapezoid=True)
    sin_theta = jnp.asarray(np.sin(theta), dtype=intensity.dtype)
    integrand = intensity * sin_theta[None, :, None]
    y = jnp.trapezoid(integrand, theta, axis=1)
    p_base = jnp.trapezoid(y, phi, axis=1)
    assert p_new.dtype == jnp.float32 and p_base.dtype == jnp.float32

    y = np.abs(np.asarray(y, np.float64)[0])
    w = _test_periodic_weights(phi)
    dx = np.diff(phi.astype(np.float32)).astype(np.float64)
    w_trap = np.zeros(73)
    w_trap[:-1] += dx / 2
    w_trap[1:] += dx / 2
    u = 2.0 ** -24

    def gamma(n):
        return n * u / (1 - n * u)

    bound = np.sum((np.abs(w - w_trap) + (u + gamma(73) * (1 + u)) * np.abs(w)
                    + gamma(74) * np.abs(w_trap)) * y)
    delta = abs(float(p_new[0]) - float(p_base[0]))
    steps = int(_f32_steps(p_new[0], p_base[0]))
    print(f"r2 {pattern}: P_new={float(p_new[0])!r} P_base={float(p_base[0])!r} "
          f"|dP|={delta:.3e} B={bound:.3e} |dP|/B={delta / bound:.4f} "
          f"float32 steps={steps}", file=sys.stderr)
    assert delta <= bound


def test_r3_both_paths_read_the_one_weight_function(monkeypatch):
    theta = np.linspace(0.0, np.pi, 37)
    phi = np.linspace(0.0, 2.0 * np.pi, 73)
    u_np = (np.sin(theta)[:, None] ** 2 * np.ones((1, 73)))[None]
    before_np = integrate_radiated_power(u_np, theta, phi)
    before_jax = integrate_radiated_power(jnp.asarray(u_np, jnp.float32), theta, phi,
                                          jax_trapezoid=True)
    original = radiated_power_module._phi_weights
    monkeypatch.setattr(radiated_power_module, "_phi_weights",
                        lambda p: 3.0 * original(p))
    after_np = integrate_radiated_power(u_np, theta, phi)
    after_jax = integrate_radiated_power(jnp.asarray(u_np, jnp.float32), theta, phi,
                                         jax_trapezoid=True)
    np.testing.assert_allclose(after_np, 3.0 * before_np, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(after_jax), 3.0 * np.asarray(before_jax), rtol=1e-6)


def _float32_built_seam_grid(n):
    # Built in float32 and carried in float64, as compute_far_field stores
    # it: the last sample is float32(2*pi), 1.7e-7 past 2*pi.
    phi = np.asarray(jnp.linspace(0, 2 * jnp.pi, n), dtype=np.float64)
    assert phi.dtype == np.float64 and 0 < phi[-1] - 2 * np.pi < 1e-6
    return phi


@pytest.mark.parametrize("n", [73, 13])
def test_float32_built_seam_grid_weight_sum(n):
    phi = _float32_built_seam_grid(n)
    theta = np.array([0.0, np.pi / 2 - 0.25, np.pi / 2, np.pi / 2 + 0.25, np.pi])
    intensity = np.zeros((1, len(theta), n))
    intensity[0, 2, :] = 4.0
    total = integrate_radiated_power(intensity, theta, phi)[0]
    print(f"float32-built n={n}: sum - 2*pi = {total - 2 * np.pi:.3e} "
          f"({(total - 2 * np.pi) / np.spacing(2 * np.pi):.1f} ulp)", file=sys.stderr)
    assert abs(total - 2 * np.pi) <= 8 * np.spacing(2 * np.pi), total - 2 * np.pi


@pytest.mark.parametrize("n", [73, 13])
def test_float32_built_seam_grid_takes_the_seam_branch(n):
    phi = _float32_built_seam_grid(n)
    weights = radiated_power_module._phi_weights(phi)
    assert len(weights) == n
    assert weights[-1] == 0.5 * (phi[0] + 2 * np.pi - phi[-2])


@pytest.mark.parametrize("n", [73, 13])
def test_float32_built_seam_grid_z_dipole(n):
    d = directivity(_z_dipole(THETA, _float32_built_seam_grid(n)))[0]
    assert abs(d - D_DIPOLE_DBI) <= 1e-6, d


def test_phi_span_beyond_full_turn_is_refused():
    phi = np.linspace(0, 2 * np.pi + 1e-5, 73)
    with pytest.raises(ValueError, match="full-sphere coverage"):
        directivity(_z_dipole(THETA, phi))


_PHI24 = np.linspace(0, 2 * np.pi, 24, endpoint=False)


@pytest.mark.parametrize("jax_trapezoid", [False, True], ids=["numpy", "jax"])
def test_theta_stopping_short_of_poles_is_accepted(jax_trapezoid):
    # 0.01 rad short of each pole, inside the 0.02 rad pole tolerance on both
    # paths (refused by the earlier half-step rule).
    theta = np.linspace(0.01, np.pi - 0.01, 1801)
    u = (np.sin(theta)[:, None] ** 2 * np.ones((1, len(_PHI24))))[None]
    if jax_trapezoid:
        u = jnp.asarray(u, jnp.float32)
    p = np.asarray(integrate_radiated_power(u, theta, _PHI24, jax_trapezoid=jax_trapezoid))
    assert np.all(np.isfinite(p)) and np.all(p > 0)


@pytest.mark.parametrize("jax_trapezoid", [False, True], ids=["numpy", "jax"])
def test_theta_uncovered_beyond_tolerance_is_refused(jax_trapezoid):
    # Uncovered polar angle beyond the end sample's own cell = 0.05 rad.
    # NumPy cell edge: theta[0] - h/2; trapezoid cell edge: theta[0].
    n = 1801
    if jax_trapezoid:
        a = 0.05
    else:
        a = (0.05 + np.pi / (2 * (n - 1))) / (1 + 1 / (n - 1))
    theta = np.linspace(a, np.pi - a, n)
    if not jax_trapezoid:
        h = theta[1] - theta[0]
        assert abs(theta[0] - h / 2 - 0.05) < 1e-12
    u = (np.sin(theta)[:, None] ** 2 * np.ones((1, len(_PHI24))))[None]
    if jax_trapezoid:
        u = jnp.asarray(u, jnp.float32)
    with pytest.raises(ValueError, match="full-sphere coverage") as exc:
        integrate_radiated_power(u, theta, _PHI24, jax_trapezoid=jax_trapezoid)
    assert "0.02 rad" in str(exc.value)
