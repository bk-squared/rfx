"""rfx.fdfd.hplane: referee equality, port transparency, and the gradients
with respect to frequency, permittivity and continuous aperture width."""
from __future__ import annotations

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.fdfd import HPlaneSpec, build, solve
from tests._x64_compat import enable_x64

A_WR90 = 22.86e-3
F0 = 10e9


def _referee():
    path = pathlib.Path(__file__).resolve().parents[3] / "validation/crossval/comparators/fdfd_hplane.py"
    spec = importlib.util.spec_from_file_location("fdfd_hplane_referee", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fd4(f, x0, d):
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def _two_iris():
    return build(HPlaneSpec(a=A_WR90, base_cells=24, refinement=2, apertures_cells=(12, 10),
                            cavities_cells=(20,), thickness_cells=1, margin_cells=8))


def test_empty_guide_is_transparent():
    with enable_x64():
        s11, s21 = solve(build(HPlaneSpec(a=A_WR90, base_cells=24, refinement=2)), F0)
        assert abs(complex(s11)) < 1e-12
        assert abs(abs(complex(s21)) - 1.0) < 1e-12


def test_matches_independent_referee_node_for_node():
    """Same grid, same PEC staircase: the two solves differ only by the LU
    roundoff of a cond ~1e12 system (the referee documents ~1e-9 spread
    across orderings, #884)."""
    ref = _referee()
    with enable_x64():
        s11, s21 = solve(_two_iris(), F0)
    r11, r21, _ = ref.solve(A_WR90, F0, 24, 2, (12, 10), (20,), 1, 8)
    assert abs(complex(s11) - r11) < 1e-8
    assert abs(complex(s21) - r21) < 1e-8
    assert abs(abs(complex(s11)) ** 2 + abs(complex(s21)) ** 2 - 1.0) < 1e-8


def test_gradients_wrt_eps_and_freq_match_finite_differences():
    with enable_x64():
        m = _two_iris()
        eps0 = jnp.ones(m.shape, jnp.complex128)

        def r(freq, eps):
            return jnp.abs(solve(m, freq, eps)[0]) ** 2

        g_eps = jax.grad(r, argnums=1)(F0, eps0)
        g_f = jax.grad(r, argnums=0)(F0, eps0)
        # FD steps are chosen above the ~1e-9 LU noise floor of the objective.
        i, j = 11, 42
        assert not m.metal[i, j]
        fd = _fd4(lambda t: r(F0, eps0.at[i, j].add(t)), 0.0, 1e-2)
        assert abs(float(jnp.real(g_eps[i, j])) - float(fd)) < 1e-3 * abs(float(fd))
        fd_f = _fd4(lambda t: r(F0 + t, eps0), 0.0, 1e5)
        assert abs(float(g_f) - float(fd_f)) < 1e-3 * abs(float(fd_f))
        # metal nodes carry no permittivity sensitivity
        assert float(jnp.max(jnp.abs(g_eps[m.metal]))) == 0.0


def test_continuous_aperture_is_exact_at_nominal_and_smooth_between():
    with enable_x64():
        m24 = build(HPlaneSpec(a=A_WR90, base_cells=48, apertures_cells=(24,), margin_cells=16))
        m26 = build(HPlaneSpec(a=A_WR90, base_cells=48, apertures_cells=(26,), margin_cells=16))
        r24 = abs(complex(solve(m24, F0)[0])) ** 2
        r26 = abs(complex(solve(m26, F0)[0])) ** 2
        vals = [abs(complex(solve(m24, F0, apertures=(c * m24.h,))[0])) ** 2
                for c in (24, 24.5, 25, 25.5, 26)]
        # nominal width: the grid is uniform, so the override is the Dirichlet solve
        assert abs(vals[0] - r24) < 1e-9
        # 26 cells on the 24-nominal model is a STRETCHED grid: same physics,
        # different discretisation, so agreement is to discretisation error
        assert abs(vals[-1] - r26) < 2e-2 * r26
        assert all(a > b for a, b in zip(vals, vals[1:]))
        # port transparency does not depend on the stretch
        e11, e21 = solve(m24, F0, apertures=(25.3 * m24.h,), pec=False)
        assert abs(complex(e11)) < 1e-12
        assert abs(abs(complex(e21)) - 1.0) < 1e-12


def test_width_gradient_is_smooth_across_nodes_and_eigenvalue_degeneracies():
    """Body-fitted grid: no kink where an edge would cross a node of the
    nominal grid, and no nan where the stretched transverse operator has an
    exactly degenerate eigenvalue pair (measured at 34.5 fine cells here)."""
    with enable_x64():
        m = build(HPlaneSpec(a=A_WR90, base_cells=16, refinement=4, apertures_cells=(8,),
                             margin_cells=6))

        def r(w):
            return jnp.abs(solve(m, F0, apertures=(w,))[0]) ** 2

        cells = np.arange(33.5, 35.01, 0.25)
        grads = np.array([float(jax.grad(r)(c * m.h)) for c in cells])
        assert np.all(np.isfinite(grads))
        for c, g in zip(cells, grads):
            fd = _fd4(r, c * m.h, 0.02 * m.h)
            assert abs(g - float(fd)) < 1e-5 * abs(float(fd)), (c, g, float(fd))
        # smooth: neighbouring slopes differ by well under 1 % (the cut-cell
        # edge this replaced jumped ~50 % at the node crossing at 34.0)
        assert np.max(np.abs(np.diff(grads))) < 1e-2 * np.max(np.abs(grads))


def test_width_gradient_converges_with_refinement():
    with enable_x64():
        w = 8.29 * A_WR90 / 16
        grads = []
        for ref in (1, 2, 4):
            m = build(HPlaneSpec(a=A_WR90, base_cells=16, refinement=ref, apertures_cells=(8,),
                                 margin_cells=6))
            grads.append(float(jax.grad(lambda w, m=m: jnp.abs(solve(m, F0, apertures=(w,))[0]) ** 2)(w)))
        d1, d2 = abs(grads[1] - grads[0]), abs(grads[2] - grads[1])
        assert d2 < 0.5 * d1, grads


def test_aperture_width_gradient_matches_finite_differences():
    with enable_x64():
        m = _two_iris()

        def r(ws):
            return jnp.abs(solve(m, F0, apertures=ws)[0]) ** 2

        ws0 = (12.3 * m.h, 10.6 * m.h)          # edges between nodes
        g = jax.grad(r)(ws0)
        for i in range(2):
            def r_i(t, i=i):
                return r(tuple(w + (t if j == i else 0.0) for j, w in enumerate(ws0)))
            fd = _fd4(r_i, 0.0, 0.02 * m.h)
            assert abs(float(g[i]) - float(fd)) < 1e-5 * abs(float(fd))
        _, jv = jax.jvp(r, (ws0,), ((1.0, -0.5),))
        assert abs(float(jv) - float(g[0] - 0.5 * g[1])) < 1e-9 * abs(float(jv))
        assert abs(float(jax.jit(r)(ws0)) - float(r(ws0))) < 1e-12


def test_aperture_count_is_checked():
    with enable_x64():
        m = _two_iris()
        with pytest.raises(ValueError, match="apertures"):
            solve(m, F0, apertures=(1e-3,))
