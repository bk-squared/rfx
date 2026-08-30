"""M0 falsifier battery (F-M0-a/b/c of the pre-declaration).

Run:
  PYTHONPATH=<worktree> .venv/bin/python -m pytest \
      validation/research/portgrid/test_portgrid_m0.py -o addopts="" -q
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.research.portgrid.certificate import (
    EPS0,
    MU0,
    build_region_matrices,
    certify_region,
    classical_cfl_dt,
    dt_max_certificate,
)
from validation.research.portgrid.operators import (
    adjoint_residual,
    build_interface_operators,
    pullback_residual,
    supply_rate_residual,
)

RATIOS = (3, 5, 7, 4)  # odd lane + the 2-D paper's own r=4 fixture ratio
EDGE_COUNTS = (1, 2, 5, 8)
WINDOW_EXACT = 1e-13   # pre-declared F-M0-a/b window


# ---------------------------------------------------------------- F-M0-a
@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("m", EDGE_COUNTS)
@pytest.mark.parametrize("ell", (1.0, 1e-3))
def test_p_norm_adjoint_pair(r, m, ell):
    ops = build_interface_operators(r, m, ell)
    assert adjoint_residual(ops) <= WINDOW_EXACT
    rng = np.random.default_rng(20260829 + r * 100 + m)
    assert pullback_residual(ops, rng) <= WINDOW_EXACT


def test_odd_restriction_flag():
    with pytest.raises(ValueError):
        build_interface_operators(4, 2, require_odd=True)
    build_interface_operators(5, 2, require_odd=True)  # must not raise


def test_jax_vjp_matches_p_adjoint():
    """The jax.vjp pullback of applied T_c2f equals P_c . T_f2c . P_f^{-1}."""
    import jax
    import jax.numpy as jnp

    from tests._x64_compat import enable_x64

    with enable_x64():
        for r, m in ((3, 4), (5, 2), (7, 1), (4, 5)):
            ops = build_interface_operators(r, m, 1e-3)
            t_c2f = jnp.asarray(ops["T_c2f"])

            def apply_c2f(x):
                return t_c2f @ x

            rng = np.random.default_rng(77 + r)
            x0 = jnp.asarray(rng.standard_normal(m))
            w = jnp.asarray(rng.standard_normal(m * r))
            _, vjp = jax.vjp(apply_c2f, x0)
            (xbar,) = vjp(w)
            expected = ops["P_c"] * (ops["T_f2c"] @ (np.asarray(w) / ops["P_f"]))
            res = np.max(np.abs(np.asarray(xbar) - expected)) / max(np.max(np.abs(expected)), 1e-300)
            assert res <= WINDOW_EXACT, (r, m, res)


# ---------------------------------------------------------------- F-M0-b
@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("m", (1, 3, 8))
def test_interconnect_supply_rate_is_zero(r, m):
    ops = build_interface_operators(r, m, 1e-3)
    rng = np.random.default_rng(42 + r * 10 + m)
    worst = max(supply_rate_residual(ops, rng) for _ in range(64))
    assert worst <= WINDOW_EXACT


# ---------------------------------------------------------------- F-M0-c
def _uniform_region(nx, ny, dx, dy, dt, eps=EPS0, mu=MU0, sigma=0.0):
    return build_region_matrices(
        nx, ny, dx, dy, dt,
        eps_x=np.full((nx, ny + 1), eps),
        eps_y=np.full((nx + 1, ny), eps),
        sigma_x=np.full((nx, ny + 1), sigma),
        sigma_y=np.full((nx + 1, ny), sigma),
        mu=np.full((nx, ny), mu),
    )


@pytest.mark.parametrize("nx,ny,dx,dy", [(4, 3, 1e-3, 2e-3), (8, 6, 1e-3, 1e-3)])
def test_certificate_uniform_region_vs_classical_cfl(nx, ny, dx, dy):
    m0 = _uniform_region(nx, ny, dx, dy, dt=1e-12)
    dt_cert = dt_max_certificate(m0)
    dt_cfl = classical_cfl_dt(dx, dy, EPS0, MU0)
    # Paper: classical CFL (47) is sufficient for (29a) -> certificate can only be looser.
    assert dt_cert >= dt_cfl * (1.0 - 1e-12), (dt_cert, dt_cfl)

    # Crossing check: PD strictly below the certified bound, indefinite above it.
    below = certify_region(_uniform_region(nx, ny, dx, dy, dt=0.99 * dt_cert))
    above = certify_region(_uniform_region(nx, ny, dx, dy, dt=1.01 * dt_cert))
    assert below["R_positive_definite"], below
    assert not above["R_positive_definite"], above
    assert below["R_symmetry_residual"] <= WINDOW_EXACT
    assert below["B_LLTB_residual"] <= WINDOW_EXACT
    assert below["LTB_structure_residual"] <= WINDOW_EXACT
    assert below["dissipative"]


def test_certificate_random_materials_paper_class():
    """Random eps in [eps0, 3 eps0], sigma in [0, 50 uS/m] (paper Sec. VI-A class)."""
    rng = np.random.default_rng(20260829)
    nx, ny, dx, dy = 6, 5, 1e-3, 1.5e-3
    eps_x = EPS0 * (1.0 + 2.0 * rng.random((nx, ny + 1)))
    eps_y = EPS0 * (1.0 + 2.0 * rng.random((nx + 1, ny)))
    sig_x = 50e-6 * rng.random((nx, ny + 1))
    sig_y = 50e-6 * rng.random((nx + 1, ny))
    mu = np.full((nx, ny), MU0)
    m0 = build_region_matrices(nx, ny, dx, dy, 1e-12, eps_x, eps_y, sig_x, sig_y, mu)
    dt_cert = dt_max_certificate(m0)
    # Sufficient condition with the smallest material values (3-D paper eq. (35) analog).
    dt_cfl_suff = classical_cfl_dt(dx, dy, min(eps_x.min(), eps_y.min()), mu.min())
    assert dt_cert >= dt_cfl_suff * (1.0 - 1e-12)

    cert = certify_region(
        build_region_matrices(nx, ny, dx, dy, 0.99 * dt_cert, eps_x, eps_y, sig_x, sig_y, mu)
    )
    assert cert["dissipative"], cert
    assert cert["FFt_eigmin"] >= 0.0  # sigma >= 0 -> PSD exactly (diagonal blocks)


def test_certificate_flags_negative_conductivity():
    nx, ny, dx, dy = 4, 4, 1e-3, 1e-3
    sig_x = np.zeros((nx, ny + 1))
    sig_x[2, 2] = -1e-6  # one active edge
    m0 = build_region_matrices(
        nx, ny, dx, dy, 1e-13,
        np.full((nx, ny + 1), EPS0), np.full((nx + 1, ny), EPS0),
        sig_x, np.zeros((nx + 1, ny)), np.full((nx, ny), MU0),
    )
    cert = certify_region(m0)
    assert cert["FFt_eigmin"] < 0.0
    assert not cert["dissipative"]


def test_certificate_r_indefinite_when_cfl_violated_lossy():
    """A lossy region above its dt bound must still be flagged (proof-distrust gate)."""
    nx, ny, dx, dy = 4, 3, 1e-3, 1e-3
    m0 = _uniform_region(nx, ny, dx, dy, dt=1e-12, sigma=0.05)
    dt_cert = dt_max_certificate(m0)
    bad = certify_region(_uniform_region(nx, ny, dx, dy, dt=1.05 * dt_cert, sigma=0.05))
    assert not bad["R_positive_definite"]


# ------------------------------------------- F-M0-c, two-sided (review finding)
@pytest.mark.parametrize("nx,ny,dx,dy", [(4, 3, 1e-3, 2e-3), (8, 6, 1e-3, 1e-3),
                                         (5, 5, 2e-3, 5e-4)])
def test_region_dual_cells_tile_the_area_exactly(nx, ny, dx, dy):
    """Two-sided structural gate on the secondary (dual) edge lengths.

    Every other F-M0-c assertion is ONE-SIDED — they all read
    ``dt_cert >= <classical CFL>`` ("the certificate can only be looser"), so a
    mutation that merely GROWS dt_cert leaves the whole M0 battery green.  The
    reviewer's mutation is exactly of that kind: setting
    ``certificate.py`` ``d_lpy[:, 0] = dy`` (dropping the half-length of the
    boundary secondary edge, paper Sec. II-A) enlarges R's Ex diagonal, raises
    dt_cert, and fires nothing.

    The gate here is geometric, not numerical-comparison: the Yee dual cells
    TILE the region exactly once, so for each E-field family the sum of
    (primary edge length x its dual edge length) equals the region area
    EXACTLY.  The boundary column of dual lengths must be dy/2 (resp. dx/2)
    for that partition to close:
        dy/2 + (ny - 1) dy + dy/2 = ny dy.
    Any boundary factor other than 1/2 breaks the identity in one direction or
    the other, so this fires for dy/2 -> dy AND for dy/2 -> dy/4.

    R's diagonal exposes the products directly (eq. (17)):
    Ex rows carry d_lx*d_lpy*eps/dt, Ey rows d_ly*d_lpx*eps/dt, Hz rows
    d_a*mu/dt; the off-diagonal curl blocks do not touch the diagonal.
    """
    dt = 1e-12
    m0 = _uniform_region(nx, ny, dx, dy, dt)
    n_ex, n_ey = nx * (ny + 1), (nx + 1) * ny
    d = np.diag(m0.R)
    area = (nx * dx) * (ny * dy)
    sums = {
        "Ex primal-x x dual-y": float(d[:n_ex].sum()) * dt / EPS0,
        "Ey primal-y x dual-x": float(d[n_ex:n_ex + n_ey].sum()) * dt / EPS0,
        "Hz cell areas": float(d[n_ex + n_ey:].sum()) * dt / MU0,
    }
    for name, val in sums.items():
        assert abs(val - area) <= 1e-12 * area, (name, val, area)

    # Localised form of the same identity: the boundary dual edge is EXACTLY
    # half the interior one (ratio 1/2, two-sided), read off R's diagonal.
    ex = d[:n_ex].reshape((ny + 1, nx))      # index j*nx + i
    ey = d[n_ex:n_ex + n_ey].reshape((ny, nx + 1))
    assert ny >= 3 and nx >= 3
    for edge, interior in ((ex[0], ex[1]), (ex[-1], ex[-2])):
        assert np.allclose(edge, 0.5 * interior, rtol=1e-13, atol=0.0)
    for edge, interior in ((ey[:, 0], ey[:, 1]), (ey[:, -1], ey[:, -2])):
        assert np.allclose(edge, 0.5 * interior, rtol=1e-13, atol=0.0)
