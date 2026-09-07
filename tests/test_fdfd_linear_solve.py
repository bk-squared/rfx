"""rfx.fdfd.linear_solve: a host-factorised sparse solve that JAX
differentiates in both modes through the implicit-function rule."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps

from rfx.fdfd.linear_solve import sparse_matvec, sparse_solve
from tests._x64_compat import enable_x64


def _system(n=40, seed=1):
    rng = np.random.default_rng(seed)
    m = (sps.random(n, n, density=0.15, random_state=seed, dtype=float) + sps.eye(n) * 5).tocoo()
    data = m.data + 0.3j * rng.standard_normal(len(m.data))
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return m.row, m.col, jnp.asarray(data), jnp.asarray(b)


def _fd4(f, x0, direction, d):
    """Fourth-order central difference of a scalar function along a direction."""
    return (-f(x0 + 2 * d * direction) + 8 * f(x0 + d * direction)
            - 8 * f(x0 - d * direction) + f(x0 - 2 * d * direction)) / (12 * d)


def test_solve_residual_and_matvec_consistency():
    with enable_x64():
        rows, cols, data, b = _system()
        x = sparse_solve(data, rows, cols, b)
        assert float(jnp.linalg.norm(sparse_matvec(data, rows, cols, x) - b)) < 1e-12
        dense = sps.csc_matrix((np.asarray(data), (rows, cols)), shape=(len(b), len(b)))
        assert np.linalg.norm(dense @ np.asarray(x) - np.asarray(b)) < 1e-12


def test_reverse_mode_matches_finite_differences():
    with enable_x64():
        rows, cols, data, b = _system()

        def loss(d, rhs):
            return jnp.sum(jnp.abs(sparse_solve(d, rows, cols, rhs)) ** 2)

        g_data, g_b = jax.grad(loss, argnums=(0, 1))(data, b)
        rng = np.random.default_rng(7)
        # JAX convention for real L of complex z: grad = dL/dRe(z) - 1j dL/dIm(z)
        for i in rng.choice(len(rows), 3, replace=False):
            e = jnp.zeros(len(rows), jnp.complex128).at[i].set(1.0)
            d_re = _fd4(lambda z: loss(z, b), data, e, 1e-4)
            d_im = _fd4(lambda z: loss(z, b), data, 1j * e, 1e-4)
            assert abs(float(jnp.real(g_data[i])) - float(d_re)) < 1e-7 * max(1.0, abs(float(d_re)))
            assert abs(-float(jnp.imag(g_data[i])) - float(d_im)) < 1e-7 * max(1.0, abs(float(d_im)))
        e = jnp.zeros(len(b), jnp.complex128).at[3].set(1.0)
        d_re = _fd4(lambda z: loss(data, z), b, e, 1e-4)
        assert abs(float(jnp.real(g_b[3])) - float(d_re)) < 1e-7 * max(1.0, abs(float(d_re)))


def test_forward_mode_matches_reverse_mode_and_jit():
    with enable_x64():
        rows, cols, data, b = _system()

        def loss(d):
            return jnp.sum(jnp.abs(sparse_solve(d, rows, cols, b)) ** 2)

        rng = np.random.default_rng(3)
        t = jnp.asarray(rng.standard_normal(len(rows)) + 1j * rng.standard_normal(len(rows)))
        _, jvp_val = jax.jvp(loss, (data,), (t,))
        g = jax.grad(loss)(data)
        # directional derivative of a real function along complex direction t
        directional = jnp.sum(jnp.real(g) * jnp.real(t) - jnp.imag(g) * jnp.imag(t))
        assert abs(float(jvp_val) - float(directional)) < 1e-9 * abs(float(directional))
        fd = _fd4(loss, data, t, 1e-4)
        assert abs(float(jvp_val) - float(fd)) < 1e-6 * abs(float(fd))
        assert abs(float(jax.jit(loss)(data)) - float(loss(data))) < 1e-12


def test_requires_x64():
    rows, cols, data, b = _system()
    if jax.config.read("jax_enable_x64"):
        pytest.skip("x64 is enabled globally in this environment")
    with pytest.raises(RuntimeError, match="x64"):
        sparse_solve(data, rows, cols, b)


# ---------------------------------------------------------------------------
# Multi-right-hand-side (n, m) mode: one factorisation, m excitations.
# ---------------------------------------------------------------------------

def _block_system(n=40, seed=1, m=3):
    rows, cols, data, _ = _system(n, seed)
    rng = np.random.default_rng(seed + 100)
    b = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    return rows, cols, data, jnp.asarray(b)


def _block_loss(rows, cols):
    # Column weights so the objective is not symmetric across right-hand sides.
    def loss(d, rhs):
        x = sparse_solve(d, rows, cols, rhs)
        w = jnp.arange(1, rhs.shape[1] + 1, dtype=jnp.float64)
        return jnp.sum(w * jnp.sum(jnp.abs(x) ** 2, axis=0))
    return loss


def test_block_solve_equals_separate_column_solves():
    with enable_x64():
        rows, cols, data, b = _block_system()
        x = sparse_solve(data, rows, cols, b)
        assert x.shape == b.shape
        sep = jnp.stack([sparse_solve(data, rows, cols, b[:, j]) for j in range(b.shape[1])], axis=1)
        # measured 1.4e-17 (same LU, same column solves); gate 1e-12
        assert float(jnp.max(jnp.abs(x - sep))) < 1e-12
        assert float(jnp.linalg.norm(sparse_matvec(data, rows, cols, x) - b)) < 1e-12
        with pytest.raises(ValueError, match=r"\(n, m\)"):
            sparse_solve(data, rows, cols, b[:, :, None])


def test_block_reverse_mode_matches_finite_differences():
    with enable_x64():
        rows, cols, data, b = _block_system()
        loss = _block_loss(rows, cols)
        g_data, g_b = jax.grad(loss, argnums=(0, 1))(data, b)
        assert g_b.shape == b.shape
        rng = np.random.default_rng(7)
        # measured worst relative disagreement 2.7e-11 (data), 1.7e-10 (b) at h=1e-4,
        # where FD4 moves a ~17-magnitude objective by ~1e-5 >> LU noise; gate 1e-7
        for i in rng.choice(len(rows), 4, replace=False):
            e = jnp.zeros(len(rows), jnp.complex128).at[i].set(1.0)
            d_re = float(_fd4(lambda z: loss(z, b), data, e, 1e-4))
            d_im = float(_fd4(lambda z: loss(z, b), data, 1j * e, 1e-4))
            assert abs(float(jnp.real(g_data[i])) - d_re) < 1e-7 * max(1.0, abs(d_re))
            assert abs(-float(jnp.imag(g_data[i])) - d_im) < 1e-7 * max(1.0, abs(d_im))
        e = jnp.zeros(b.shape, jnp.complex128).at[3, 1].set(1.0)
        d_re = float(_fd4(lambda z: loss(data, z), b, e, 1e-4))
        d_im = float(_fd4(lambda z: loss(data, z), b, 1j * e, 1e-4))
        assert abs(float(jnp.real(g_b[3, 1])) - d_re) < 1e-7 * max(1.0, abs(d_re))
        assert abs(-float(jnp.imag(g_b[3, 1])) - d_im) < 1e-7 * max(1.0, abs(d_im))


def test_block_forward_mode_matches_reverse_mode_fd_and_jit():
    with enable_x64():
        rows, cols, data, b = _block_system()
        loss = _block_loss(rows, cols)
        rng = np.random.default_rng(3)
        t = jnp.asarray(rng.standard_normal(len(rows)) + 1j * rng.standard_normal(len(rows)))
        tb = jnp.asarray(rng.standard_normal(b.shape) + 1j * rng.standard_normal(b.shape))
        _, jvp_val = jax.jvp(loss, (data, b), (t, tb))
        g_data, g_b = jax.grad(loss, argnums=(0, 1))(data, b)
        directional = (jnp.sum(jnp.real(g_data) * jnp.real(t) - jnp.imag(g_data) * jnp.imag(t))
                       + jnp.sum(jnp.real(g_b) * jnp.real(tb) - jnp.imag(g_b) * jnp.imag(tb)))
        # measured 0.0 relative (same linearisation), gate 1e-9
        assert abs(float(jvp_val) - float(directional)) < 1e-9 * abs(float(directional))
        fd = _fd4(lambda s: loss(data + s * t, b + s * tb), jnp.float64(0.0), 1.0, 1e-4)
        # measured 4.5e-13 relative vs FD4, gate 1e-6
        assert abs(float(jvp_val) - float(fd)) < 1e-6 * abs(float(fd))
        # jvp of the block primal equals the stacked jvps of the column solves (measured 8e-17)
        _, dx = jax.jvp(lambda d, r: sparse_solve(d, rows, cols, r), (data, b), (t, tb))
        _, dsep = jax.jvp(
            lambda d, r: jnp.stack([sparse_solve(d, rows, cols, r[:, j]) for j in range(b.shape[1])], axis=1),
            (data, b), (t, tb))
        assert float(jnp.max(jnp.abs(dx - dsep))) < 1e-12
        # jit: value and gradient identical to eager (measured 0.0)
        assert abs(float(jax.jit(loss)(data, b)) - float(loss(data, b))) < 1e-12
        gj = jax.jit(jax.grad(loss))(data, b)
        assert float(jnp.max(jnp.abs(gj - g_data))) < 1e-12
