"""Unit tests for the MSL (microstrip line) port primitive.

These tests exercise the geometry/material setup and the closed-form
3-probe de-embedding math without running an FDTD scan.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.sources.msl_port import (
    MSLPort,
    _msl_yz_cells,
    compute_s21,
    extract_msl_s_params,
    msl_forward_amplitude,
    msl_probe_x_coords,
    setup_msl_port,
)


def _empty_materials(grid: Grid) -> MaterialArrays:
    shape = grid.shape
    return MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Test 1: cross-section coverage
# ---------------------------------------------------------------------------


def test_msl_yz_cells_covers_full_cross_section():
    grid = Grid(freq_max=10e9, domain=(5e-3, 3e-3, 1e-3),
                dx=1e-4, cpml_layers=0)
    port = MSLPort(
        feed_x=1e-3,
        y_lo=1e-3, y_hi=2e-3,
        z_lo=0.0, z_hi=2.54e-4,
        direction="+x",
        impedance=50.0,
        excitation=None,
    )
    cells = _msl_yz_cells(grid, port)
    assert len(cells) > 0

    # Constant feed-plane i.
    i_set = {c[0] for c in cells}
    assert len(i_set) == 1
    i_feed = next(iter(i_set))
    expected_i, _, _ = grid.position_to_index((1e-3, 1e-3, 0.0))
    assert i_feed == expected_i

    # j range matches the y endpoints (inclusive).
    j_set = sorted({c[1] for c in cells})
    j_lo_expected = grid.position_to_index((1e-3, 1e-3, 0.0))[1]
    j_hi_expected = grid.position_to_index((1e-3, 2e-3, 0.0))[1]
    assert j_set[0] == j_lo_expected
    assert j_set[-1] == j_hi_expected
    assert j_set == list(range(j_set[0], j_set[-1] + 1))

    # k range covers z_lo..z_hi.
    k_set = sorted({c[2] for c in cells})
    k_lo_expected = grid.position_to_index((1e-3, 1e-3, 0.0))[2]
    k_hi_expected = grid.position_to_index((1e-3, 1e-3, 2.54e-4))[2]
    assert k_set[0] == k_lo_expected
    assert k_set[-1] == k_hi_expected
    # Total cell count = N_y * N_z.
    assert len(cells) == len(j_set) * len(k_set)


# ---------------------------------------------------------------------------
# Test 2: total resistance distribution
# ---------------------------------------------------------------------------


def test_setup_msl_port_total_resistance():
    dx = dy = 1e-4
    dz = 5e-5
    # Build a uniform grid; we override dz via a profile so we can test
    # axis-aware sigma computation.
    grid = Grid(freq_max=10e9, domain=(2e-3, 2e-3, 5e-4),
                dx=dx, cpml_layers=0)
    # Inject a synthetic dz_profile so the formula uses dz != dx.
    grid.dz_profile = np.full(grid.nz, dz, dtype=float)
    grid.dy_profile = np.full(grid.ny, dy, dtype=float)
    grid.dx_profile = np.full(grid.nx, dx, dtype=float)

    Z0 = 50.0
    # Width = 0.3 mm → 3 y-cells (with rounding).
    # Height = 0.2 mm → 4 z-cells (with rounding from dz=5e-5).
    width = 3 * dy
    height = 4 * dz
    port = MSLPort(
        feed_x=1e-3,
        y_lo=1e-3, y_hi=1e-3 + width,
        z_lo=0.0, z_hi=height,
        direction="+x",
        impedance=Z0,
        excitation=None,
    )

    cells = _msl_yz_cells(grid, port)
    n_y = len({c[1] for c in cells})
    n_z = len({c[2] for c in cells})

    materials = _empty_materials(grid)
    out = setup_msl_port(grid, port, materials)
    expected_sigma = (n_z * dz) / (Z0 * n_y * dx * dy)

    sigma_arr = np.asarray(out.sigma)
    for (i, j, k) in cells:
        assert sigma_arr[i, j, k] == pytest.approx(expected_sigma, rel=1e-6)

    # Cells outside the cross-section remain zero.
    total_zero = sigma_arr.size - len(cells)
    nonzero_count = int(np.count_nonzero(sigma_arr))
    assert nonzero_count == len(cells)
    assert total_zero > 0


# ---------------------------------------------------------------------------
# Test 3: 3-probe de-embedding for a pure forward wave
# ---------------------------------------------------------------------------


def test_3probe_deembedding_pure_forward():
    beta = 200.0  # rad/m
    Delta = 0.5e-3
    x1 = 5e-3
    Z0 = 50.0
    V_plus = 1.0
    I_plus = V_plus / Z0

    V1 = np.array([V_plus * np.exp(-1j * beta * x1)])
    V2 = np.array([V_plus * np.exp(-1j * beta * (x1 + Delta))])
    V3 = np.array([V_plus * np.exp(-1j * beta * (x1 + 2 * Delta))])
    I1 = np.array([I_plus * np.exp(-1j * beta * x1)])

    s11, z0, q = extract_msl_s_params(V1, V2, V3, I1)
    assert abs(s11[0]) < 1e-3
    assert abs(z0[0].real - 50.0) / 50.0 < 1e-3
    # q should equal exp(-jβΔ).
    assert abs(q[0] - np.exp(-1j * beta * Delta)) < 1e-3


# ---------------------------------------------------------------------------
# Test 4: 3-probe de-embedding with reflection
# ---------------------------------------------------------------------------


def test_3probe_deembedding_with_reflection():
    beta = 200.0
    Delta = 0.5e-3
    Z0 = 50.0

    # Set up wave amplitudes AT the probe-1 reference plane directly so
    # the extractor's reported S11 is meant to equal these values.
    alpha1 = 1.0 + 0.0j
    s11_at_probe1 = 0.3 * np.exp(1j * 0.5)
    gamma1 = s11_at_probe1 * alpha1

    def _v_at(dx_off):
        return alpha1 * np.exp(-1j * beta * dx_off) + gamma1 * np.exp(+1j * beta * dx_off)

    V1 = np.array([_v_at(0.0)])
    V2 = np.array([_v_at(Delta)])
    V3 = np.array([_v_at(2 * Delta)])
    I1 = np.array([(alpha1 - gamma1) / Z0])  # I = (α − γ)/Z0 at probe 1

    s11, z0, q = extract_msl_s_params(V1, V2, V3, I1)
    err_mag = abs(abs(s11[0]) - abs(s11_at_probe1))
    err_phase = abs(np.angle(s11[0]) - np.angle(s11_at_probe1))
    assert err_mag < 1e-3, f"|S11| error {err_mag}"
    assert err_phase < 1e-3, f"phase error {err_phase}"
    assert abs(z0[0].real - Z0) / Z0 < 1e-3


# ---------------------------------------------------------------------------
# Test 5: probe positions are downstream of feed
# ---------------------------------------------------------------------------


def test_msl_probe_positions_are_downstream():
    grid = Grid(freq_max=10e9, domain=(10e-3, 4e-3, 1e-3),
                dx=1e-4, cpml_layers=0)
    feed_x = 2e-3
    port_pos = MSLPort(
        feed_x=feed_x, y_lo=1e-3, y_hi=2e-3,
        z_lo=0.0, z_hi=2.54e-4,
        direction="+x", impedance=50.0, excitation=None,
    )
    x1, x2, x3 = msl_probe_x_coords(grid, port_pos, n_offset_cells=5,
                                    n_spacing_cells=3)
    assert x1 > feed_x
    assert x2 > x1
    assert x3 > x2

    port_neg = MSLPort(
        feed_x=feed_x, y_lo=1e-3, y_hi=2e-3,
        z_lo=0.0, z_hi=2.54e-4,
        direction="-x", impedance=50.0, excitation=None,
    )
    x1n, x2n, x3n = msl_probe_x_coords(grid, port_neg, n_offset_cells=5,
                                       n_spacing_cells=3)
    assert x1n < feed_x
    assert x2n < x1n
    assert x3n < x2n


# ---------------------------------------------------------------------------
# Test 6: add_msl_port API surface
# ---------------------------------------------------------------------------


def test_add_msl_port_api():
    sim = Simulation(freq_max=10e9, domain=(5e-3, 3e-3, 1e-3))
    out = sim.add_msl_port(
        position=(1e-3, 1.5e-3, 0.0),
        width=1e-3,
        height=2.54e-4,
        direction="+x",
        impedance=50.0,
    )
    assert out is sim
    assert len(sim._msl_ports) == 1
    entry = sim._msl_ports[0]
    assert entry.direction == "+x"
    assert entry.impedance == 50.0
    assert entry.width == 1e-3
    assert entry.height == 2.54e-4
    assert entry.name == "msl_0"

    # Invalid direction.
    with pytest.raises(ValueError):
        sim.add_msl_port(position=(2e-3, 1.5e-3, 0.0),
                         width=1e-3, height=2.54e-4, direction="+y")
    # Invalid width.
    with pytest.raises(ValueError):
        sim.add_msl_port(position=(2e-3, 1.5e-3, 0.0),
                         width=-1e-3, height=2.54e-4)
    # Invalid height.
    with pytest.raises(ValueError):
        sim.add_msl_port(position=(2e-3, 1.5e-3, 0.0),
                         width=1e-3, height=0.0)
    # Invalid impedance.
    with pytest.raises(ValueError):
        sim.add_msl_port(position=(2e-3, 1.5e-3, 0.0),
                         width=1e-3, height=2.54e-4, impedance=0.0)
    # n_probe_offset too small.
    with pytest.raises(ValueError):
        sim.add_msl_port(position=(2e-3, 1.5e-3, 0.0),
                         width=1e-3, height=2.54e-4, n_probe_offset=2)


# ---------------------------------------------------------------------------
# Test 7: msl_forward_amplitude + compute_s21 sanity
# ---------------------------------------------------------------------------


def test_compute_s21_round_trip():
    beta = 200.0
    Delta = 0.5e-3
    x1 = 5e-3
    V_plus = 1.0
    V1 = np.array([V_plus * np.exp(-1j * beta * x1)])
    V2 = np.array([V_plus * np.exp(-1j * beta * (x1 + Delta))])
    V3 = np.array([V_plus * np.exp(-1j * beta * (x1 + 2 * Delta))])
    alpha_drv, _ = msl_forward_amplitude(V1, V2, V3)
    # Passive port sees half the forward amplitude (lossy line).
    V1p = 0.5 * V1
    V2p = 0.5 * V2
    V3p = 0.5 * V3
    alpha_pas, _ = msl_forward_amplitude(V1p, V2p, V3p)
    s21 = compute_s21(alpha_pas, alpha_drv)
    assert abs(s21[0] - 0.5) < 1e-3
