"""CPU evidence for #1829; fake mps dispatch does not measure Apple GPU kernels."""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from rfx import adjoint
from rfx import _ringdown_jax as ringdown
from rfx._precision import HIGHEST
from rfx.core import jax_utils as lb
from rfx.sources.coaxial_port import _solve_two_port_from_wave_amplitudes_jnp
from rfx.sparams._common import msl_solve_s_from_waves


def _system(n, k, condition, dtype):
    rng = np.random.default_rng(1829 + 10 * n + k)
    z = rng.normal(size=(3, n, n)) + 1j * rng.normal(size=(3, n, n))
    u, _, vh = np.linalg.svd(z)
    singular_values = np.geomspace(1., 1. / condition, n)
    a = ((u * singular_values) @ vh).astype(dtype)
    b = (rng.normal(size=(3, n, k)) + 1j * rng.normal(size=(3, n, k))).astype(dtype)
    return jnp.asarray(a), jnp.asarray(b)


def _ratio(actual, expected, condition, eps):
    actual, expected = np.asarray(actual), np.asarray(expected)
    relative = np.max(np.abs(actual - expected), axis=(-2, -1)) / np.max(
        np.abs(expected), axis=(-2, -1))
    return float(np.max(relative / (condition * eps)))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128], ids=["c64", "c128"])
@pytest.mark.parametrize("condition", [10., 1.e4], ids=["well", "ill"])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("k", [1, 3])
def test_a_block_forward_error(dtype, condition, n, k):
    with jax.enable_x64() if dtype == np.complex128 else nullcontext():
        a, b = _system(n, k, condition, dtype)
        expected = jnp.linalg.solve(a, b)
        actual = lb._block_solve(a, b)
        cond = np.linalg.cond(np.asarray(a).astype(np.complex128))
        eps = np.finfo(np.float64 if dtype == np.complex128 else np.float32).eps
        ratio = _ratio(actual, expected, cond, eps)
        print(f"A {np.dtype(dtype).name} n={n} k={k} target_cond={condition:g} "
              f"cond_max={cond.max():.9g} ratio={ratio:.9g}")
        assert actual.dtype == expected.dtype
        assert ratio <= 8., f"forward-error ratio {ratio} exceeds 8"


def _native_block_reference(a, b):
    """Independent real-system construction; no backend helper builds the oracle."""
    a, b = np.asarray(a), np.asarray(b)
    matrix = np.concatenate((np.concatenate((a.real, -a.imag), axis=-1),
                             np.concatenate((a.imag, a.real), axis=-1)), axis=-2)
    rhs = np.concatenate((b.real, b.imag), axis=-2)
    y = jnp.linalg.solve(jnp.asarray(matrix), jnp.asarray(rhs))
    n = a.shape[-1]
    return (y[..., :n, :] + 1j * y[..., n:, :]).astype(a.dtype)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("operation", ["solve", "inv", "upper", "lower"])
def test_b_dispatch_bit_equal(monkeypatch, backend, operation):
    a, b = _system(4, 3, 10., np.complex64)
    if operation == "inv":
        rhs = jnp.eye(4, dtype=a.dtype)
        native = jnp.linalg.inv(a)
        run = lambda: lb.complex_inv(a)
    elif operation in ("upper", "lower"):
        lower = operation == "lower"
        a = jnp.tril(a[0]) if lower else jnp.triu(a[0])
        rhs = b[0]
        native = jsp_linalg.solve_triangular(a, rhs, lower=lower)
        run = lambda: lb.complex_solve_triangular(a, rhs, lower=lower)
    else:
        rhs = b
        native = jnp.linalg.solve(a, rhs)
        run = lambda: lb.complex_solve(a, rhs)
    expected = _native_block_reference(a, rhs) if backend == "mps" else native
    if backend == "mps":
        monkeypatch.setattr(jax, "default_backend", lambda: "mps")
        assert np.array_equal(lb._block_solve(a, rhs), expected)
    assert np.array_equal(run(), expected)
    print(f"B {backend} {operation} bit_equal=True")


@pytest.mark.parametrize("complex_operand", ["neither", "a", "b", "both"])
def test_real_and_mixed_operands(monkeypatch, complex_operand):
    a = jnp.asarray([[3., 1.], [0.5, 2.]], dtype=jnp.float32)
    b = jnp.asarray([[1.], [2.]], dtype=jnp.float32)
    if complex_operand in ("a", "both"):
        a = a + 0.25j
    if complex_operand in ("b", "both"):
        b = b + 0.5j
    expected = jnp.linalg.solve(a, b)
    monkeypatch.setattr(jax, "default_backend", lambda: "mps")
    np.testing.assert_allclose(lb.complex_solve(a, b), expected, rtol=8 * 10 * 2**-23)
    if complex_operand == "neither":
        assert np.array_equal(lb._block_solve(a, b), expected)
        assert np.array_equal(lb.complex_inv(a), jnp.linalg.inv(a))
        assert np.array_equal(lb.complex_solve_triangular(jnp.triu(a), b, lower=False),
                              jsp_linalg.solve_triangular(jnp.triu(a), b, lower=False))


def _site_case(site):
    a, b = _system(2, 2, 10., np.complex64)
    if site == "msl":
        waves_a = [[a[:, j, d] for j in range(2)] for d in range(2)]
        waves_b = [[b[:, j, d] for j in range(2)] for d in range(2)]
        expected = jnp.moveaxis(jnp.swapaxes(jnp.linalg.solve(
            jnp.swapaxes(a, -1, -2), jnp.swapaxes(b, -1, -2)), -1, -2), 0, -1)
        return lambda: msl_solve_s_from_waves(waves_a, waves_b)[0], expected, a
    if site == "coax":
        inc, out = jnp.moveaxis(a, 0, -1), jnp.moveaxis(b, 0, -1)
        expected = jnp.moveaxis(jnp.matmul(b, jnp.linalg.inv(a), precision=HIGHEST), 0, -1)
        return lambda: _solve_two_port_from_wave_amplitudes_jnp(inc, out).s_params, expected, a
    if site == "adjoint":
        freqs = jnp.asarray([0.071, 0.113], dtype=jnp.float32)
        dt, length = 1., 128
        target = b[0]
        t = (jnp.arange(length, dtype=freqs.dtype) + 1) * dt
        carrier = jnp.exp(2j * jnp.pi * t[:, None] * freqs)
        # Independent primitive/basis, with the same evaluation order as the site.
        def primitive(k):
            env = jnp.where((k >= 0) & (k < length - 1),
                            jnp.sin(jnp.pi * (k + 1) / length)**4, 0.)
            return env[..., None] * jnp.exp(2j * jnp.pi * ((k + 1) * dt)[..., None] * freqs)
        n = jnp.arange(length, dtype=freqs.dtype)
        basis = primitive(n) - primitive(n - 1)
        transform = jnp.conj(carrier).T * dt
        a = jnp.matmul(transform, basis, precision=HIGHEST)
        b = jnp.matmul(transform, jnp.conj(basis), precision=HIGHEST)
        cross = jnp.linalg.solve(jnp.conj(a), jnp.conj(b))
        rhs = target - jnp.matmul(b, jnp.linalg.solve(jnp.conj(a), jnp.conj(target)),
                                  precision=HIGHEST)
        schur = a - jnp.matmul(b, cross, precision=HIGHEST)
        expected = jnp.linalg.solve(schur, rhs)
        return lambda: adjoint._wavelet_coefficients(freqs, dt, length, target), expected, jnp.stack((a, schur))
    rng = np.random.default_rng(1829)
    a = jnp.asarray(rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3)), dtype=jnp.complex64)
    b = jnp.asarray(rng.normal(size=(7, 2)) + 1j * rng.normal(size=(7, 2)), dtype=jnp.complex64)
    mask = jnp.asarray([1., 1., 0.], dtype=jnp.float32)
    masked = a * mask.astype(a.dtype)[None, :]
    norms = jnp.where(mask > 0, jnp.sqrt(jnp.sum(jnp.abs(masked)**2, axis=0)), 1.)
    augmented = jnp.concatenate((masked / norms.astype(a.dtype)[None, :],
                                 jnp.diag((1. - mask).astype(a.dtype))), axis=0)
    with jax.default_matmul_precision("highest"):
        q, r = jnp.linalg.qr(augmented, mode="reduced")
    rhs = jnp.concatenate((b, jnp.zeros((3, 2), dtype=b.dtype)), axis=0)
    x = jsp_linalg.solve_triangular(r, jnp.matmul(jnp.conj(q).T, rhs, precision=HIGHEST), lower=False)
    expected = (x / norms.astype(x.dtype)[:, None]) * mask.astype(x.dtype)[:, None]
    return lambda: ringdown.lstsq_masked(a, b, mask), expected, r


@pytest.mark.parametrize("site", ["msl", "adjoint", "coax", "ringdown"])
def test_c_site_wiring(monkeypatch, site):
    run, expected, matrices = _site_case(site)
    assert np.array_equal(run(), expected), f"{site}: native CPU bits changed"
    cond = float(np.max(np.linalg.cond(np.asarray(matrices))))
    monkeypatch.setattr(jax, "default_backend", lambda: "mps")

    def reject_complex(native):
        def guarded(*args, **kwargs):
            assert not any(jnp.iscomplexobj(arg) for arg in args), "mps complex kernel reached"
            return native(*args, **kwargs)
        return guarded

    monkeypatch.setattr(jnp.linalg, "solve", reject_complex(jnp.linalg.solve))
    monkeypatch.setattr(jnp.linalg, "inv", reject_complex(jnp.linalg.inv))
    monkeypatch.setattr(jsp_linalg, "solve_triangular", reject_complex(jsp_linalg.solve_triangular))
    actual = run()
    relative = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))) /
                     np.max(np.abs(np.asarray(expected))))
    ratio = relative / (cond * np.finfo(np.float32).eps)
    print(f"C {site} native_bit_equal=True cond={cond:.9g} rel={relative:.9g} ratio={ratio:.9g}")
    assert ratio <= 8., f"{site}: forward-error ratio {ratio} exceeds 8"


def test_jit_and_jvp(monkeypatch):
    a, b = _system(2, 1, 10., np.complex64)
    da = jnp.full_like(a, 0.01 + 0.02j)
    db = jnp.full_like(b, -0.02 + 0.01j)
    value = jnp.linalg.solve(a, b)
    tangent = jnp.linalg.solve(a, db - jnp.matmul(da, value, precision=HIGHEST))
    monkeypatch.setattr(jax, "default_backend", lambda: "mps")
    actual, derivative = jax.jit(lambda x, y: jax.jvp(lb.complex_solve, (x, y), (da, db)))(a, b)
    np.testing.assert_allclose(actual, value, rtol=8 * 10 * 2**-23)
    np.testing.assert_allclose(derivative, tangent, rtol=8 * 10 * 2**-23)


def test_block_preserves_complex_matrix_dtype():
    with jax.enable_x64():
        a, b = _system(2, 1, 10., np.complex64)
        b = b.astype(jnp.complex128)
        expected = jnp.linalg.solve(a, b.astype(a.dtype))
        actual = lb._block_solve(a, b)
        assert actual.dtype == a.dtype
        cond = np.linalg.cond(np.asarray(a).astype(np.complex128))
        assert _ratio(actual, expected, cond, np.finfo(np.float32).eps) <= 8.


def test_block_solve_batched_matrix_vector_rhs_matches_native():
    rng = np.random.default_rng(1829)
    a = jnp.asarray(rng.normal(size=(5, 3, 3)) + 1j * rng.normal(size=(5, 3, 3)) + 4 * np.eye(3),
                    dtype=jnp.complex64)
    b = jnp.asarray(rng.normal(size=3) + 1j * rng.normal(size=3), dtype=jnp.complex64)
    expected = jnp.linalg.solve(a, b)
    actual = lb._block_solve(a, b)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=8 * 10 * 2**-23 * float(jnp.max(jnp.abs(expected))))


@pytest.mark.parametrize("lower", [False, True])
def test_mps_triangular_solve_reads_only_its_triangle(monkeypatch, lower):
    # The native op ignores the other triangle; a full matrix must give the
    # same answer and a zero gradient there.
    rng = np.random.default_rng(1830)
    full = jnp.asarray(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) + 5 * np.eye(4),
                       dtype=jnp.complex64)
    b = jnp.asarray(rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)), dtype=jnp.complex64)
    expected = jsp_linalg.solve_triangular(full, b, lower=lower)
    monkeypatch.setattr(jax, "default_backend", lambda: "mps")
    actual = lb.complex_solve_triangular(full, b, lower=lower)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=8 * 10 * 2**-23 * float(jnp.max(jnp.abs(expected))))
    grad = jax.grad(lambda r: jnp.sum(jnp.abs(lb.complex_solve_triangular(r, b, lower=lower)) ** 2))(full)
    ignored = jnp.triu(grad, 1) if lower else jnp.tril(grad, -1)
    assert float(jnp.max(jnp.abs(ignored))) == 0.0
