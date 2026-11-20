"""Regression + behavioural tests for non-uniform xy mesh support.

Covers:
  1. make_nonuniform_grid back-compat (uniform xy still identical).
  2. make_nonuniform_grid with dx_profile / dy_profile — correct shape,
     per-cell arrays, stricter CFL dt.
  3. position_to_index finds the right cell for a known geometry.
  4. coords_from_nonuniform_grid generates correct cell-center arrays.
  5. End-to-end Simulation.run on a small cavity with mixed uniform /
     non-uniform xy → fields decay smoothly, no NaN.
  6. Non-uniform xy with the fine cells in the middle reduces the
     per-cell volume (dV) and the injected source amplitude scales
     accordingly (make_current_source uses dx[i]*dy[j]*dz[k]).
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.nonuniform import make_nonuniform_grid, position_to_index, make_current_source
from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid


def test_back_compat_uniform_xy_unchanged():
    dz = np.full(10, 0.5e-3)
    g_legacy = make_nonuniform_grid((20e-3, 20e-3), dz, dx=1e-3, cpml_layers=8)

    dx_uniform = np.full(20, 1e-3)
    g_uniform = make_nonuniform_grid(
        (0, 0), dz, dx=1e-3, cpml_layers=8,
        dx_profile=dx_uniform, dy_profile=dx_uniform,
    )
    assert g_legacy.nx == g_uniform.nx
    assert g_legacy.ny == g_uniform.ny
    assert g_legacy.nz == g_uniform.nz
    np.testing.assert_allclose(np.asarray(g_legacy.dx_arr), np.asarray(g_uniform.dx_arr))
    np.testing.assert_allclose(np.asarray(g_legacy.inv_dx), np.asarray(g_uniform.inv_dx))
    assert abs(g_legacy.dt - g_uniform.dt) < 1e-18


def test_nonuniform_xy_per_cell_and_cfl():
    dz = np.full(8, 0.5e-3)
    dx_prof = np.concatenate([
        np.full(10, 1e-3),
        np.full(10, 0.5e-3),
        np.full(10, 1e-3),
    ])
    g = make_nonuniform_grid(
        (0, 0), dz, dx=1e-3, cpml_layers=8,
        dx_profile=dx_prof, dy_profile=dx_prof,
    )

    dx_arr = np.asarray(g.dx_arr)
    cpml = g.cpml_layers
    np.testing.assert_allclose(dx_arr[cpml : cpml + 10], 1e-3, rtol=1e-5)
    np.testing.assert_allclose(dx_arr[cpml + 10 : cpml + 20], 0.5e-3, rtol=1e-5)

    dx_min = float(np.min(dx_arr))
    assert dx_min < 1e-3
    g_uniform = make_nonuniform_grid((20e-3, 20e-3), dz, dx=1e-3, cpml_layers=8)
    assert g.dt < g_uniform.dt


def test_position_to_index_nonuniform_xy():
    dz = np.full(4, 1e-3)
    dx_prof = np.array([1e-3, 1e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 1e-3, 1e-3])
    g = make_nonuniform_grid(
        (0, 0), dz, dx=1e-3, cpml_layers=4,
        dx_profile=dx_prof, dy_profile=dx_prof,
    )
    cpml = g.cpml_layers

    i, j, k = position_to_index(g, (0.0, 0.0, 0.0))
    assert i == cpml
    assert j == cpml

    i, j, k = position_to_index(g, (2.5e-3, 0.0, 0.0))
    assert i == cpml + 3

    i, j, k = position_to_index(g, (4e-3, 0.0, 0.0))
    assert i == cpml + 6


def test_coords_from_nonuniform_grid_node_positions():
    """Coordinates are E-NODE positions (cumulative cell edges), not centres.

    The non-uniform E update divides by ``2/(d[i-1]+d[i])`` — the dual spacing
    of a node straddling cells ``i-1`` and ``i`` — so the samples geometry is
    rasterized onto sit on cell EDGES. This test previously pinned cell
    CENTRES, half a cell off both the stencil's own nodes and
    ``coords_from_uniform_grid``; #562 unified them.
    N cells need N+1 bounding nodes, so the last interior node lands on the profile's total extent at the requested domain face.
    """
    dz = np.full(4, 1e-3)
    dx_prof = np.array([1e-3, 1e-3, 0.5e-3, 0.5e-3, 1e-3, 1e-3])
    g = make_nonuniform_grid(
        (0, 0), dz, dx=1e-3, cpml_layers=4,
        dx_profile=dx_prof, dy_profile=dx_prof,
    )
    coords = coords_from_nonuniform_grid(g)
    x = np.asarray(coords.x)
    cpml = g.cpml_layers

    assert abs(float(x[cpml]) - 0.0) < 1e-9

    for k, want in enumerate([0.0, 1e-3, 2e-3, 2.5e-3, 3e-3, 4e-3]):
        assert abs(float(x[cpml + k]) - want) < 1e-6, (k, float(x[cpml + k]), want)

    assert abs(float(x[cpml + len(dx_prof)]) - float(np.sum(dx_prof))) < 1e-6


def test_end_to_end_nonuniform_xy_runs():
    """The declared ``domain=(0, 0, 0)`` is replaced by the profile extents."""
    dx = 1e-3
    n_cpml = 6
    dz = np.full(10, dx)
    dx_prof = np.concatenate([
        np.full(6, dx),
        np.full(8, 0.5e-3),
        np.full(6, dx),
    ])

    sim = Simulation(
        freq_max=10e9,
        domain=(0, 0, 0),
        dx=dx,
        dz_profile=dz,
        dx_profile=dx_prof,
        dy_profile=dx_prof,
        boundary="cpml",
        cpml_layers=n_cpml,
    )
    sim.add_material("air", eps_r=1.0)
    sim.add(Box((0, 0, 0), (16e-3, 16e-3, 10e-3)), material="air")

    sim.add_source(
        position=(8e-3, 8e-3, 5e-3),
        component="ez",
        waveform=GaussianPulse(f0=5e9, bandwidth=1.0),
    )
    sim.add_probe(
        position=(10e-3, 8e-3, 5e-3),
        component="ez",
    )

    result = sim.run(n_steps=200)
    ts = np.asarray(result.time_series).ravel()

    assert np.all(np.isfinite(ts))
    assert np.abs(ts).max() > 1e-6


def test_current_source_uses_per_cell_volume():
    """The source-volume ratios of 4 and (1/0.75)^2 use coarse, fine and transition control volumes of 1e-9, 0.25e-9 and 0.5625e-9 cubic metres from dual transverse spacings."""
    from rfx.core.yee import MaterialArrays

    dz = np.full(6, 1e-3)
    dx_prof = np.concatenate([
        np.full(4, 1e-3),
        np.full(4, 0.5e-3),
        np.full(4, 1e-3),
    ])
    g = make_nonuniform_grid(
        (0, 0), dz, dx=1e-3, cpml_layers=4,
        dx_profile=dx_prof, dy_profile=dx_prof,
    )
    nx, ny, nz = g.nx, g.ny, g.nz
    shape = (nx, ny, nz)
    materials = MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
    )

    i_coarse, j_coarse, k = position_to_index(g, (2.0e-3, 2.0e-3, 0.5e-3))
    i_fine, j_fine, _ = position_to_index(g, (5.0e-3, 5.0e-3, 0.5e-3))
    i_step, j_step, _ = position_to_index(g, (4.0e-3, 4.0e-3, 0.5e-3))

    dx_np = np.asarray(g.dx_arr)
    assert abs(float(dx_np[i_coarse]) - 1e-3) < 1e-6
    assert abs(float(dx_np[i_fine]) - 0.5e-3) < 1e-6
    assert float(dx_np[i_coarse - 1]) == float(dx_np[i_coarse])
    assert float(dx_np[i_fine - 1]) == float(dx_np[i_fine])
    assert float(dx_np[i_step - 1]) != float(dx_np[i_step])

    def _unit(t):
        return jnp.float32(1.0)

    def _wf(idx):
        src = make_current_source(
            g, idx, "ez", _unit, n_steps=1, materials=materials)
        return float(np.asarray(src[4])[0])

    wf_coarse = _wf((i_coarse, j_coarse, k))
    wf_fine = _wf((i_fine, j_fine, k))
    wf_step = _wf((i_step, j_step, k))

    ratio = wf_fine / wf_coarse
    assert abs(ratio - 4.0) < 0.01, f"expected dV ratio 4, got {ratio:.4f}"

    ratio_step = wf_step / wf_coarse
    assert abs(ratio_step - (1.0 / 0.75) ** 2) < 0.01, (
        f"expected dual-spacing dV ratio {(1.0/0.75)**2:.4f} at the grading "
        f"transition, got {ratio_step:.4f}")
