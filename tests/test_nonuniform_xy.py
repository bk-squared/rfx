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


# ---------------------------------------------------------------------
# 1. Back-compat: uniform xy path still produces identical output
# ---------------------------------------------------------------------
def test_back_compat_uniform_xy_unchanged():
    dz = np.full(10, 0.5e-3)
    g_legacy = make_nonuniform_grid((20e-3, 20e-3), dz, dx=1e-3, cpml_layers=8)

    # Same call with an explicit uniform dx_profile of the same length
    # should produce an identical grid (modulo floating point).
    dx_uniform = np.full(20, 1e-3)  # 20 * 1mm = 20mm domain
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


# ---------------------------------------------------------------------
# 2. Non-uniform xy profiles produce per-cell arrays and stricter dt
# ---------------------------------------------------------------------
def test_nonuniform_xy_per_cell_and_cfl():
    dz = np.full(8, 0.5e-3)
    # Coarse at edges (1mm), fine in the middle (0.5mm)
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
    # The first 10 interior cells should be 1mm, next 10 should be 0.5mm
    np.testing.assert_allclose(dx_arr[cpml : cpml + 10], 1e-3, rtol=1e-5)
    np.testing.assert_allclose(dx_arr[cpml + 10 : cpml + 20], 0.5e-3, rtol=1e-5)

    # CFL dt must reflect the smallest cell size
    dx_min = float(np.min(dx_arr))
    assert dx_min < 1e-3  # we have 0.5mm cells
    # A uniform-1mm grid's dt would be larger; check that dt is stricter
    g_uniform = make_nonuniform_grid((20e-3, 20e-3), dz, dx=1e-3, cpml_layers=8)
    assert g.dt < g_uniform.dt


# ---------------------------------------------------------------------
# 3. position_to_index on non-uniform xy lands in the correct cell
# ---------------------------------------------------------------------
def test_position_to_index_nonuniform_xy():
    dz = np.full(4, 1e-3)
    # Interior x profile: [1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1] mm
    # Cumulative x (left edges, starting at 0): 0, 1, 2, 2.5, 3, 3.5, 4, 5, 6
    dx_prof = np.array([1e-3, 1e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 1e-3, 1e-3])
    g = make_nonuniform_grid(
        (0, 0), dz, dx=1e-3, cpml_layers=4,
        dx_profile=dx_prof, dy_profile=dx_prof,
    )
    cpml = g.cpml_layers

    # x=0 → first interior cell (index cpml)
    i, j, k = position_to_index(g, (0.0, 0.0, 0.0))
    assert i == cpml
    assert j == cpml

    # x=2.5mm falls at the boundary between cell 2 and cell 3 (edge @2.5)
    # → position_to_index snaps to nearest edge
    i, j, k = position_to_index(g, (2.5e-3, 0.0, 0.0))
    assert i == cpml + 3  # cell index 3 (left-edge at 2.5mm)

    # x=4mm is at edge 6 → cell index 6
    i, j, k = position_to_index(g, (4e-3, 0.0, 0.0))
    assert i == cpml + 6


# ---------------------------------------------------------------------
# 4. coords_from_nonuniform_grid E-node positions (#562: was cell centres)
# ---------------------------------------------------------------------
def test_coords_from_nonuniform_grid_node_positions():
    """Coordinates are E-NODE positions (cumulative cell edges), not centres.

    The non-uniform E update divides by ``2/(d[i-1]+d[i])`` — the dual spacing
    of a node straddling cells ``i-1`` and ``i`` — so the samples geometry is
    rasterized onto sit on cell EDGES. This test previously pinned cell
    CENTRES, half a cell off both the stencil's own nodes and
    ``coords_from_uniform_grid``; #562 unified them.
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

    # The first interior node IS the interior origin (user x = 0).
    assert abs(float(x[cpml]) - 0.0) < 1e-9

    # Then each node sits on the running sum of the profile's cell sizes:
    # 1mm, 2mm, 2.5mm (after the first 0.5mm cell), 3mm, 4mm.
    for k, want in enumerate([0.0, 1e-3, 2e-3, 2.5e-3, 3e-3, 4e-3]):
        assert abs(float(x[cpml + k]) - want) < 1e-6, (k, float(x[cpml + k]), want)

    # N cells need N+1 bounding nodes, so the last interior node lands exactly
    # on the profile's total extent — the requested domain face.
    assert abs(float(x[cpml + len(dx_prof)]) - float(np.sum(dx_prof))) < 1e-6


# ---------------------------------------------------------------------
# 5. End-to-end Simulation with mixed uniform z + non-uniform xy
# ---------------------------------------------------------------------
def test_end_to_end_nonuniform_xy_runs():
    # Small air-filled cavity so the test is fast.
    # Substrate dimensions are kept dummy — we only want to confirm
    # the non-uniform-xy code path runs without NaN and probes yield
    # a non-zero field.
    dx = 1e-3
    n_cpml = 6
    dz = np.full(10, dx)   # uniform z, 10mm
    dx_prof = np.concatenate([
        np.full(6, dx),
        np.full(8, 0.5e-3),
        np.full(6, dx),
    ])  # 6 + 4 + 6 = 16mm in x; 20 interior cells

    sim = Simulation(
        freq_max=10e9,
        domain=(0, 0, 0),      # will be overridden by profiles
        dx=dx,
        dz_profile=dz,
        dx_profile=dx_prof,
        dy_profile=dx_prof,
        boundary="cpml",
        cpml_layers=n_cpml,
    )
    sim.add_material("air", eps_r=1.0)
    sim.add(Box((0, 0, 0), (16e-3, 16e-3, 10e-3)), material="air")

    # Source at the geometric center (within the fine-mesh region)
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

    # Must be finite everywhere
    assert np.all(np.isfinite(ts))
    # Probe must have observed a field (non-trivial amplitude)
    assert np.abs(ts).max() > 1e-6


# ---------------------------------------------------------------------
# 6. make_current_source uses per-cell dV for non-uniform xy
# ---------------------------------------------------------------------
def test_current_source_uses_per_cell_volume():
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

    # Two positions, both on LOCALLY UNIFORM nodes: one inside the 1mm run,
    # one inside the 0.5mm run. The original spelling probed 4.25mm, which
    # ties between nodes 4.0 and 4.5 and resolves to node 4 — the grading
    # TRANSITION itself, where the primal cell (0.5mm) and the dual spacing
    # (0.75mm) differ. It expected 4, the primal-only answer, and issue #672
    # showed the control volume takes the DUAL spacing on the two axes
    # transverse to the component. The transition node is now covered
    # explicitly below with the value the dual rule gives.
    i_coarse, j_coarse, k = position_to_index(g, (2.0e-3, 2.0e-3, 0.5e-3))
    i_fine, j_fine, _ = position_to_index(g, (5.0e-3, 5.0e-3, 0.5e-3))
    i_step, j_step, _ = position_to_index(g, (4.0e-3, 4.0e-3, 0.5e-3))

    dx_np = np.asarray(g.dx_arr)
    # Guarantee we really did land in different cell sizes, and that the two
    # main probes sit where primal == dual (adjacent cells equal).
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

    # cb is identical (same eps, same dt). The only difference is 1/dV.
    # Both probes sit on locally uniform nodes, so dual == primal there:
    # dV_fine = 0.5mm * 0.5mm * 1mm = 0.25e-9
    # dV_coarse = 1mm * 1mm * 1mm = 1e-9
    # ratio = 4
    ratio = wf_fine / wf_coarse
    assert abs(ratio - 4.0) < 0.01, f"expected dV ratio 4, got {ratio:.4f}"

    # On the transition node both TRANSVERSE axes are graded 1mm/0.5mm, so
    # dV_step = 0.75mm * 0.75mm * 1mm = 0.5625e-9 and the ratio against the
    # coarse node is (1/0.75)^2 = 1.7778 — not the 4 the primal-only rule
    # would give, and not the 1 a naive "same cell size" reading would
    # (issue #672).
    ratio_step = wf_step / wf_coarse
    assert abs(ratio_step - (1.0 / 0.75) ** 2) < 0.01, (
        f"expected dual-spacing dV ratio {(1.0/0.75)**2:.4f} at the grading "
        f"transition, got {ratio_step:.4f}")
