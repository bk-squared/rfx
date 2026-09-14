"""rfx.fdfd.linear_solve: a host-factorised sparse solve that JAX
differentiates in both modes through the implicit-function rule."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sps

import rfx.fdfd.linear_solve as ls
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


# ---------------------------------------------------------------------------
# LU cache key: the pattern belongs in it, not only the entry values.
# ---------------------------------------------------------------------------

def test_lu_cache_distinguishes_two_patterns_with_identical_data():
    """Nothing but the pattern ties the entry values to the matrix they
    belong to, so byte-identical ``data`` on a different pattern used to
    hand the second problem the first one's factors. The key now includes
    ``(n, rows, cols)``.

    Here the second problem is the transpose, i.e. the same data with
    ``rows``/``cols`` swapped. Measured: residuals 2.4e-16 and 3.0e-16
    relative (gate 1e-12) and the two solutions differing by 0.37 x max|x1|,
    so a stale cache hit cannot hide -- with the old data-only key x2 came
    back bit-identical to x1 (verified by monkeypatching the digest) and its
    relative residual was 0.41."""
    with enable_x64():
        rows, cols, data, b = _system()
        ls.clear_factor_cache()
        x1 = sparse_solve(data, rows, cols, b)
        x2 = sparse_solve(data, cols, rows, b)          # A^T: same data bytes
        assert len(ls._FACTOR_CACHE) == 2               # two problems, two factors
        nb = float(jnp.linalg.norm(b))
        r1 = float(jnp.linalg.norm(sparse_matvec(data, rows, cols, x1) - b)) / nb
        r2 = float(jnp.linalg.norm(sparse_matvec(data, cols, rows, x2) - b)) / nb
        assert r1 < 1e-12 and r2 < 1e-12, (r1, r2)
        assert float(jnp.max(jnp.abs(x1 - x2))) > 0.1 * float(jnp.max(jnp.abs(x1)))
        # repeating either one is a cache hit, not a third factorisation
        sparse_solve(data, rows, cols, b)
        sparse_solve(data, cols, rows, b)
        assert len(ls._FACTOR_CACHE) == 2


# ---------------------------------------------------------------------------
# SuperLU column ordering (permc_spec): same answer, same gradient, own cache
# entry. Timings and fill live in the module docstring of linear_solve.
# ---------------------------------------------------------------------------

def test_permc_spec_gives_the_same_solution_and_its_own_cache_entry():
    """Every accepted ordering solves the same system to the LU noise floor,
    an unknown one is rejected, and the LU cache is keyed by the ordering --
    except that ``None`` and ``"COLAMD"`` are the same ordering and share one
    entry (scipy: identical ``perm_c``, ``perm_r`` and LU nnz).
    Measured max |x - x_COLAMD| / max |x| on the random 40 x 40 system:
    0.0 (None, i.e. the same ordering by another name), 3.4e-16 (NATURAL),
    4.1e-16 (MMD_ATA), 2.5e-16 (MMD_AT_PLUS_A); gate 1e-12."""
    with enable_x64():
        rows, cols, data, b = _system()
        ls.clear_factor_cache()
        ref = sparse_solve(data, rows, cols, b, permc_spec="COLAMD")
        assert len(ls._FACTOR_CACHE) == 1
        sparse_solve(data, rows, cols, b, permc_spec=None)
        assert len(ls._FACTOR_CACHE) == 1        # None is COLAMD: one entry, not two
        for spec in ls.PERMC_SPECS:
            x = sparse_solve(data, rows, cols, b, permc_spec=spec)
            assert float(jnp.linalg.norm(sparse_matvec(data, rows, cols, x) - b)) < 1e-12
            assert float(jnp.max(jnp.abs(x - ref))) < 1e-12 * float(jnp.max(jnp.abs(ref)))
        # one entry per DISTINCT ordering (4 of them), all inside the LRU cap
        assert len(ls._FACTOR_CACHE) == 4 <= ls._FACTOR_CACHE_SIZE
        n_before = len(ls._FACTOR_CACHE)
        sparse_solve(data, rows, cols, b, permc_spec="MMD_AT_PLUS_A")
        assert len(ls._FACTOR_CACHE) == n_before
        with pytest.raises(ValueError, match="permc_spec"):
            sparse_solve(data, rows, cols, b, permc_spec="AMD")
        assert ls.get_default_permc_spec() is None   # SuperLU's default (COLAMD)


def test_default_permc_spec_scope_is_restored_and_validated():
    """``default_permc_spec`` is the hook for solvers that do not forward the
    keyword (hplane.solve, yee3d.solve, ports3d.s_matrix). It must actually
    change the ordering a bare ``sparse_solve`` uses, restore the previous
    default on exit and on an exception, nest, and reject a bad name. Being
    a ContextVar rather than a plain module attribute, it is also private to
    the thread / task that set it."""
    with enable_x64():
        rows, cols, data, b = _system()
        assert ls.get_default_permc_spec() is None
        with ls.default_permc_spec("MMD_AT_PLUS_A"):
            assert ls.get_default_permc_spec() == "MMD_AT_PLUS_A"
            ls.clear_factor_cache()
            x = sparse_solve(data, rows, cols, b)               # no keyword
            assert len(ls._FACTOR_CACHE) == 1
            # the same call names COLAMD explicitly -> a second, different LU
            sparse_solve(data, rows, cols, b, permc_spec="COLAMD")
            assert len(ls._FACTOR_CACHE) == 2
            with ls.default_permc_spec("NATURAL"):
                assert ls.get_default_permc_spec() == "NATURAL"
            assert ls.get_default_permc_spec() == "MMD_AT_PLUS_A"
        assert ls.get_default_permc_spec() is None
        # measured 2.5e-16 relative between the scoped default and COLAMD
        assert float(jnp.max(jnp.abs(x - sparse_solve(data, rows, cols, b)))) < \
            1e-12 * float(jnp.max(jnp.abs(x)))
        with pytest.raises(ValueError, match="permc_spec"):
            with ls.default_permc_spec("AMD"):
                pass
        assert ls.get_default_permc_spec() is None
        with pytest.raises(ZeroDivisionError):
            with ls.default_permc_spec("NATURAL"):
                1 / 0
        assert ls.get_default_permc_spec() is None


# ---------------------------------------------------------------------------
# A self-contained parametric PDE system, so the ordering gate does not need
# another module: 2-D Helmholtz, 5-point stencil, graded absorbing ring,
# two dielectric slabs as the differentiable parameters.
# ---------------------------------------------------------------------------

H2D_N = 60                # 60 x 60 = 3600 unknowns, the hplane refinement-2 size
H2D_K0H = 0.376           # k0 * h; 1-norm condition estimate 1.3e4 at THETA0
H2D_NABS = 6              # absorbing ring thickness, cells
H2D_SIGMA = 0.1           # peak Im(eps) of the ring
H2D_THETA0 = (2.7, 1.9)   # slab permittivities (the parameters)


def _h2d_pattern(ng=H2D_N):
    """Static 5-point pattern; ``sizes`` lets the data be built by blocks."""
    idx = np.arange(ng * ng).reshape(ng, ng)
    rows, cols = [idx.ravel()], [idx.ravel()]
    i, j = np.meshgrid(np.arange(ng), np.arange(ng), indexing="ij")
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii, jj = i + di, j + dj
        ok = (ii >= 0) & (ii < ng) & (jj >= 0) & (jj < ng)
        rows.append(idx[i[ok], j[ok]])
        cols.append(idx[ii[ok], jj[ok]])
    return (np.concatenate(rows), np.concatenate(cols),
            tuple(int(r.size) for r in rows))


H2D_ROWS, H2D_COLS, H2D_SIZES = _h2d_pattern()


def _h2d_masks(ng=H2D_N):
    e = np.arange(ng)
    d = np.minimum(np.minimum(e[:, None], (ng - 1 - e)[:, None]),
                   np.minimum(e[None, :], (ng - 1 - e)[None, :])).astype(float)
    ramp = np.clip((H2D_NABS - d) / H2D_NABS, 0.0, 1.0) ** 2
    slab_a = np.zeros((ng, ng))
    slab_a[24:30, 10:50] = 1.0
    slab_b = np.zeros((ng, ng))
    slab_b[40:46, 10:50] = 1.0
    b = np.zeros(ng * ng, np.complex128)
    b[12 * ng + ng // 2] = 1.0
    return ramp, slab_a, slab_b, b


H2D_RAMP, H2D_SLAB_A, H2D_SLAB_B, H2D_B = _h2d_masks()


def _h2d_data(theta):
    """COO values of ``h^2 (nabla^2 + k0^2 eps)`` for slab permittivities ``theta``."""
    eps = (jnp.ones((H2D_N, H2D_N), jnp.complex128)
           + (theta[0] - 1.0) * jnp.asarray(H2D_SLAB_A)
           + (theta[1] - 1.0) * jnp.asarray(H2D_SLAB_B)
           - 1j * H2D_SIGMA * jnp.asarray(H2D_RAMP))
    diag = H2D_K0H ** 2 * eps.reshape(-1) - 4.0
    return jnp.concatenate([diag] + [jnp.ones(s, jnp.complex128) for s in H2D_SIZES[1:]])


def _h2d_field(theta, spec):
    return sparse_solve(_h2d_data(jnp.asarray(theta)), H2D_ROWS, H2D_COLS,
                        jnp.asarray(H2D_B), permc_spec=spec)


def _h2d_obj(theta, spec=None):
    """Power on an output line: a real scalar of the complex field."""
    x = _h2d_field(theta, spec).reshape(H2D_N, H2D_N)
    return jnp.sum(jnp.abs(x[52, 10:50]) ** 2)


def test_helmholtz_value_field_and_gradient_are_independent_of_the_ordering():
    """The column ordering is a fill heuristic, not physics. On a
    self-contained 2-D Helmholtz system (N = 3600, 1-norm condition estimate
    1.3e4, |x|max 2.092, objective 4.963437, gradient (6.4749, 6.5345))
    all four distinct orderings must give the same field, the same objective
    and the same d(objective)/d(slab eps), and each gradient must be a real
    gradient.

    Measured against COLAMD -- field 3.6e-14, objective 5.1e-14, gradient
    3.9e-14 relative (worst over NATURAL / MMD_ATA / MMD_AT_PLUS_A); gate
    1e-9, the LU noise floor. AD vs FD4 at h = 2e-3, which moves the
    objective by 2.6e-3 (six orders above that floor): 3.8e-9 and 2.9e-8
    relative on the two parameters; gate 1e-6, where FD4 truncation (2.3e-6
    and 1.8e-5 at h = 1e-2) already dominates. One LU serves the forward and
    the adjoint solve: the cache holds exactly one entry after
    value_and_grad."""
    with enable_x64():
        ls.clear_factor_cache()
        v_ref, g_ref = jax.value_and_grad(_h2d_obj)(H2D_THETA0, "COLAMD")
        v_ref = float(v_ref)
        g_ref = np.array([float(g) for g in g_ref])
        assert len(ls._FACTOR_CACHE) == 1, "forward and adjoint must share one LU"
        x_ref = np.asarray(_h2d_field(H2D_THETA0, "COLAMD"))
        assert v_ref > 1.0 and np.all(np.abs(g_ref) > 1.0), (v_ref, g_ref)
        for spec in ("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"):
            ls.clear_factor_cache()
            v, g = jax.value_and_grad(_h2d_obj)(H2D_THETA0, spec)
            g = np.array([float(z) for z in g])
            assert len(ls._FACTOR_CACHE) == 1
            assert abs(float(v) - v_ref) < 1e-9 * abs(v_ref), (spec, v, v_ref)
            assert np.all(np.abs(g - g_ref) < 1e-9 * np.abs(g_ref)), (spec, g, g_ref)
            x = np.asarray(_h2d_field(H2D_THETA0, spec))
            assert np.max(np.abs(x - x_ref)) < 1e-9 * np.max(np.abs(x_ref)), spec
        for spec in ("COLAMD", "MMD_AT_PLUS_A"):
            for i in (0, 1):
                def shifted(t, i=i, spec=spec):
                    th = tuple(w + t * (j == i) for j, w in enumerate(H2D_THETA0))
                    return float(_h2d_obj(th, spec))
                h = 2e-3
                fd = float(_fd4(shifted, 0.0, 1.0, h))
                assert abs(shifted(h) - v_ref) > 1e-4 * abs(v_ref)   # FD moves it
                assert abs(g_ref[i] - fd) < 1e-6 * abs(fd), (spec, i, g_ref[i], fd)


# ---------------------------------------------------------------------------
# The same gate on the production matrix (cond ~1e12), reached through the
# scoped default because hplane.solve does not forward permc_spec.
# ---------------------------------------------------------------------------

A_WR90 = 22.86e-3
F0 = 10e9


def _two_iris():
    """The hplane two-iris gate model at refinement 2 (N = 3619)."""
    from rfx.fdfd import hplane
    return hplane.build(hplane.HPlaneSpec(a=A_WR90, base_cells=24, refinement=2,
                                          apertures_cells=(12, 10), cavities_cells=(20,),
                                          thickness_cells=1, margin_cells=8))


def _with_ordering(spec, fn):
    """Run ``fn`` with ``spec`` as the scoped default ordering, so a solver
    that does not thread ``permc_spec`` through (hplane, yee3d) is still
    exercised on both orderings. The cache is cleared on both sides so the
    factorisation really happens under ``spec``."""
    with ls.default_permc_spec(spec):
        ls.clear_factor_cache()
        try:
            return fn()
        finally:
            ls.clear_factor_cache()


def test_gradient_is_independent_of_the_lu_ordering():
    """Same claim as the Helmholtz test above, on the real thing:
    d|S11|^2/dwidth of the hplane two-iris model (refinement 2, N = 3619,
    cond ~1e12, widths off the nominal nodes) must be the same under COLAMD
    and MMD_AT_PLUS_A. Measured (|S11|^2 = 0.96083, gradient (80.27, 52.05)):
    field 6.0e-13 relative, |S11|^2 4.6e-14, gradient 2.9e-12 and 1.7e-12
    on the two width groups; gate 1e-9
    relative, which is the LU noise floor of this system. Both gradients
    also match FD4 to 3.0e-7 relative at a step of 0.02 h, which moves the
    objective by 7.6e-4 / 5.0e-4 -- six orders above that noise floor, so
    FD4 truncation dominates there (measured 3.0e-7 and 1.0e-7)."""
    from rfx.fdfd import hplane
    with enable_x64():
        m = _two_iris()
        ws0 = (12.3 * m.h, 10.6 * m.h)

        def obj(ws):
            return jnp.abs(hplane.solve(m, F0, apertures=ws)[0]) ** 2

        out = {spec: _with_ordering(spec, lambda: jax.value_and_grad(obj)(ws0))
               for spec in ("COLAMD", "MMD_AT_PLUS_A")}
        v_c, g_c = float(out["COLAMD"][0]), np.array([float(g) for g in out["COLAMD"][1]])
        v_m, g_m = float(out["MMD_AT_PLUS_A"][0]), np.array([float(g) for g in out["MMD_AT_PLUS_A"][1]])
        assert abs(v_c) > 0.9                                  # a real objective, not ~0
        assert abs(v_c - v_m) < 1e-9 * abs(v_c)
        assert np.all(np.abs(g_c) > 1.0)                       # gradients well away from zero
        assert np.all(np.abs(g_c - g_m) < 1e-9 * np.abs(g_c)), (g_c, g_m)
        # the two fields agree too, not only the scalar read off them
        data, rhs = hplane.assemble(m, F0, apertures=ws0)
        xs = [np.asarray(_with_ordering(s, lambda: sparse_solve(data, m.rows, m.cols, rhs)))
              for s in ("COLAMD", "MMD_AT_PLUS_A")]
        assert np.max(np.abs(xs[0] - xs[1])) < 1e-9 * np.max(np.abs(xs[0]))
        # and each is a real gradient: FD4 under the same ordering
        for spec, g in (("COLAMD", g_c), ("MMD_AT_PLUS_A", g_m)):
            for i in (0, 1):
                def shifted(t, i=i):
                    return float(obj(tuple(w + t * (j == i) for j, w in enumerate(ws0))))
                fd = float(_with_ordering(spec, lambda: _fd4(shifted, 0.0, 1.0, 0.02 * m.h)))
                assert abs(float(g[i]) - fd) < 1e-5 * abs(fd), (spec, i, g[i], fd)
                assert abs(fd) * 0.02 * m.h > 1e-5             # FD moves the objective


# ---------------------------------------------------------------------------
# memory: the process must not grow with the number of solves
#
# scipy's SuperLU keeps its whole factor when the object is deallocated on a
# thread other than the one that built it, and under ``jit`` XLA runs the
# ``pure_callback`` host solves on its own worker pool while
# ``clear_factor_cache()`` runs on the caller's thread. Before
# ``linear_solve``'s factor thread, a jitted THREE-solve program lost 2.54 of
# its three factors per call: at N = 12739 (the spiral fixture) RSS went
# 1.319 / 1.962 / 2.608 / 3.250 GB over four solves at the same theta with
# the cache emptied and gc forced between them. The gates below run the same
# shape of program on a self-contained system small enough for the fast lane.
# Full series and the diagnosis: validation/fdfd/memory_probe.{py,json}.

LAP_K = 20              # 20^3 = 8000 unknowns, 5.36e4 stored entries; measured
                        # 3877216 LU nonzeros, 0.084 GB and 0.25 s per splu here


def _lap3(k=LAP_K):
    """Complex 7-point Laplacian on a k x k x k grid, as a COO pattern.

    Chosen because it FILLS: the point of the test is a factor big enough
    that losing one is visible in RSS above the ~0.01 GB noise of a Python
    process, without paying for a real 3-D FDFD model.
    """
    n = k ** 3
    idx = np.arange(n).reshape(k, k, k)
    rows = [idx.ravel()]
    cols = [idx.ravel()]
    vals = [np.full(n, 6.0 + 0.01j)]
    for ax in range(3):
        a = np.take(idx, np.arange(k - 1), axis=ax).ravel()
        b = np.take(idx, np.arange(1, k), axis=ax).ravel()
        rows += [a, b]
        cols += [b, a]
        vals += [np.full(a.size, -1.0 + 0j)] * 2
    return (np.concatenate(rows), np.concatenate(cols),
            jnp.asarray(np.concatenate(vals)), n)


def _three_solve_program(rows, cols, n):
    """A jitted program with THREE independent sparse_solves -- the shape of
    ``spiral.solve_spiral``'s dut/open/short, and the shape that made the
    leak visible (one callback alone only plateaus)."""
    rhs = jnp.ones((n, 2), jnp.complex128)

    def three(d):
        return sum(sparse_solve(d * (1.0 + 1e-3 * j), rows, cols, rhs).sum()
                   for j in range(3))

    return jax.jit(three)


def test_repeated_jitted_solves_do_not_grow_the_resident_set():
    """Six jitted three-solve calls: growth after the second solve must be
    under 15 % of the first solve's increment.

    The calls are driven from a WORKER thread and the cache is emptied from
    this one, which is the design loop's situation made deterministic: under
    ``jit`` XLA decides for itself whether to run a ``pure_callback`` on the
    calling thread or on its pool, and it only moves off the caller's thread
    above some program size (measured here: the same program runs on the
    main thread at 13824 unknowns and on a pool thread at 17576). Driving it
    from a worker pins the build thread without needing a system big enough
    to make XLA do it, so the gate keeps its teeth in the fast lane.

    Threshold from the measurement: after the fix the growth over solves
    2..6 is 0.000 GB here, and 0.016 % of the first increment at N = 12739
    (0.00017 GB on 1.021 GB, validation/fdfd/memory_probe.json); with the
    release path this module used before, the same loop here grows
    0.21 GB per solve -- 0.842 GB over solves 2..6 against a 0.292 GB first
    increment, a ratio of 2.89. 15 % sits two orders of magnitude above the
    fixed behaviour and 19x below the broken one, and clears the ~0.01 GB
    RSS jitter of a Python process on this machine.
    """
    import gc
    from concurrent.futures import ThreadPoolExecutor
    psutil = pytest.importorskip("psutil")
    proc = psutil.Process()

    def rss_gb():
        return proc.memory_info().rss / 2.0 ** 30

    with enable_x64():
        rows, cols, data, n = _lap3()
        f = _three_solve_program(rows, cols, n)
        ls.clear_factor_cache()
        gc.collect()
        series = []
        with ThreadPoolExecutor(max_workers=1) as ex:
            base = rss_gb()
            for _ in range(6):
                ex.submit(lambda: jax.block_until_ready(f(data))).result()
                ls.clear_factor_cache()          # from THIS thread, as a design loop does
                gc.collect()
                series.append(rss_gb())
    first = series[0] - base
    later = series[-1] - series[1]
    # the test only means something if the factors are big enough to see
    assert first > 0.05, (first, series)
    assert later < 0.15 * first, (first, later, series)


def test_every_factor_is_built_and_dropped_on_one_thread():
    """The invariant the fix rests on, guarded structurally as well as by
    RSS: all three host callbacks of one jitted program factorise on the
    SAME thread, it is not the caller's, and ``clear_factor_cache()``
    empties the cache on that same thread."""
    import threading
    seen: list[int] = []
    orig = ls._factor

    def spy(*a):
        seen.append(threading.get_ident())
        return orig(*a)

    with enable_x64():
        rows, cols, data, n = _lap3(8)              # tiny: this is not a timing test
        f = _three_solve_program(rows, cols, n)
        ls._factor = spy
        try:
            jax.block_until_ready(f(data))
        finally:
            ls._factor = orig
        assert len(seen) == 3, seen                 # three fixtures, three factorisations
        assert len(set(seen)) == 1, seen
        assert seen[0] != threading.get_ident()
        assert ls._on_factor_thread(threading.get_ident) == seen[0]
        ls.clear_factor_cache()
        assert len(ls._FACTOR_CACHE) == 0


def test_factor_cache_size_knob_bounds_the_live_factors():
    """``factor_cache_size`` is the documented knob for peak memory: the
    cache never holds more than it says, shrinking drops the excess at once,
    and the default is restored by this test."""
    assert ls.factor_cache_size() == 4               # documented default
    with enable_x64():
        rows, cols, data, n = _lap3(8)
        f = _three_solve_program(rows, cols, n)
        try:
            ls.clear_factor_cache()
            ls.factor_cache_size(2)
            assert ls.factor_cache_size() == 2
            jax.block_until_ready(f(data))           # three distinct matrices
            assert len(ls._FACTOR_CACHE) == 2
            ls.factor_cache_size(4)
            jax.block_until_ready(f(data * 1.7))     # three more
            assert len(ls._FACTOR_CACHE) == 4
            ls.factor_cache_size(1)                  # shrinking trims immediately
            assert len(ls._FACTOR_CACHE) == 1
            ls.factor_cache_size(0)
            assert len(ls._FACTOR_CACHE) == 0
            with pytest.raises(ValueError, match=">= 0"):
                ls.factor_cache_size(-1)
        finally:
            ls.factor_cache_size(4)
            ls.clear_factor_cache()


def test_gradient_still_shares_one_factorisation_with_the_adjoint():
    """The factor thread must not have broken what the cache is FOR: a
    reverse-mode pass re-uses the forward solve's factorisation, so a
    gradient costs one ``splu``, not two."""
    import scipy.sparse.linalg as spl
    calls = []
    orig = spl.splu

    def counted(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    with enable_x64():
        rows, cols, data, b = _system()

        def loss(d):
            return jnp.sum(jnp.abs(sparse_solve(d, rows, cols, b)) ** 2)

        ls.clear_factor_cache()
        ls.spl.splu = counted
        try:
            g = jax.grad(loss)(data)
            jax.block_until_ready(g)
        finally:
            ls.spl.splu = orig
        assert len(calls) == 1, len(calls)
        assert float(jnp.linalg.norm(g)) > 0.0
