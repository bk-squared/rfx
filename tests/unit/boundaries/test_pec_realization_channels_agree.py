"""One body, every PEC channel, one answer — or a written exemption (#931 §1.7).

The inventory's second finding for this group was that the repo realizes PEC
through several independent channels and nothing cross-checked them on the
same body:

  1. the binary edge rule            ``realized_pec_edge_masks``
  2. the soft (AD) relaxation        ``apply_pec_occupancy``
  3. the solver lanes                ``run()`` / ``forward()``
  4. the reporting surfaces          ``fidelity_report``, ``conductor_mask``
  5. Dey-Mittra conformal weights    ``pec_shapes`` -> SDF fill fractions
  6. Kottke inverse-permittivity     fractional ``inv_eps`` tensor

§1.7 makes (1) the only source and (2)-(4) its consumers. (5) and (6) are
fenced OUT by §1.8 — they are subpixel models with their own interior
selection and their reconciliation is a named follow-up, not an omission.
This file asserts the first four agree on one hand-checkable body and states
the exemption for the last two in a test that fails if they are quietly
folded in without a decision.

Fixture: a 20 mm PEC cube at dx = 1 mm, ``boundary="pec"`` so there is no
CPML pad and node ``i`` reads as ``i`` mm. Body drawn 6 -> 14 mm on x and y,
8 -> 12 mm on z: 8 x 8 x 4 cells, walls on node planes 6..14 / 6..14 / 8..12.
Every number below is read off those corners.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.pec import (
    apply_pec_edges, apply_pec_occupancy, realized_pec_edge_masks,
    realized_wall_planes,
)
from rfx.core.yee import init_state
from rfx.geometry.rasterize_grid import (
    cell_centres_from_nodes, classify_pec_entry, coords_from_uniform_grid,
)

DX = 1e-3
DOM = (0.020, 0.020, 0.020)
LO, HI = (0.006, 0.006, 0.008), (0.014, 0.014, 0.012)
WALLS = [list(range(6, 15)), list(range(6, 15)), list(range(8, 13))]


def _sim():
    sim = Simulation(freq_max=10e9, domain=DOM, dx=DX, boundary="pec")
    sim.add(Box(LO, HI), material="pec")
    return sim


def _cells_and_edges(sim):
    grid = sim._build_grid()
    co = coords_from_uniform_grid(grid)
    cells, sheet, wire = classify_pec_entry(
        Box(LO, HI), co, cell_centres_from_nodes(co), name="pec")
    assert sheet is None and wire is None
    return grid, cells, realized_pec_edge_masks(cells)


def test_channel_1_the_binary_rule_puts_the_walls_on_the_drawn_faces():
    """The reference every other channel is compared against."""
    _grid, cells, edges = _cells_and_edges(_sim())
    assert int(np.asarray(cells).sum()) == 8 * 8 * 4
    assert [realized_wall_planes(edges, a) for a in range(3)] == WALLS


def test_channel_2_the_soft_path_is_bit_identical_at_binary_occupancy():
    """§1.6: the noisy-OR relaxation collapses to the four-cell OR."""
    _grid, cells, edges = _cells_and_edges(_sim())
    shape = tuple(np.asarray(cells).shape)
    st = init_state(shape)._replace(
        ex=jnp.ones(shape, jnp.float32), ey=jnp.ones(shape, jnp.float32),
        ez=jnp.ones(shape, jnp.float32))
    hard = apply_pec_edges(st, edges)
    soft = apply_pec_occupancy(st, jnp.asarray(cells, jnp.float32))
    for c in ("ex", "ey", "ez"):
        np.testing.assert_array_equal(np.asarray(getattr(hard, c)),
                                      np.asarray(getattr(soft, c)), err_msg=c)


def test_channel_3_the_solver_lane_zeroes_exactly_those_edges():
    """The consumer check the group had none of: what one run actually
    zeroes is the edge set the rule realizes — nothing more (a port or a
    source releasing a wall would show up here) and nothing less."""
    sim = _sim()
    sim.add_source((0.003, 0.010, 0.010), "ez", waveform=GaussianPulse(f0=5e9),
                   amplitude_kind="field")
    sim.add_probe((0.005, 0.010, 0.010), "ez")   # outside the body
    res = sim.run(n_steps=40, skip_preflight=True)
    _grid, _cells, edges = _cells_and_edges(sim)
    for c, m in zip(("ex", "ey", "ez"), edges):
        f = np.asarray(getattr(res.state, c))
        m = np.asarray(m, bool)
        assert not np.any(f[m]), f"{c}: a realized PEC edge is not zero"
        # ... and the body is not over-realized: some edge just outside the
        # body carries field, so "all zero" cannot pass vacuously.
        assert np.any(np.abs(f[~m]) > 0.0), c


def test_channel_4_the_reporting_surfaces_read_the_same_realization():
    """``fidelity_report`` prints the realized wall planes and
    ``conductor_mask`` the occupied cells; both must be the rule's own
    answer, not a re-derivation."""
    sim = _sim()
    grid, cells, _edges = _cells_and_edges(sim)
    pads = grid.axis_pads

    rep = sim.fidelity_report(print_report=False)
    (row,) = [it for it in rep if it["entity"].startswith("geometry[0]")]
    planes = row["realized_wall_planes"]
    for a, name in enumerate(("x", "y", "z")):
        got = [k - pads[a] for k in planes[name]["planes"]]
        assert got == WALLS[a], (name, got)
    assert "one-cell" not in row["realization"]

    cm = np.asarray(sim.conductor_mask(), bool)
    np.testing.assert_array_equal(cm, np.asarray(cells, bool))


def test_the_subpixel_channels_are_fenced_out_in_writing():
    """§1.8, stated as a measurement so folding them in becomes a decision.

    Dey-Mittra conformal weights and the Kottke inverse-permittivity tensor
    never form an edge set: they are continuous fill fractions with their
    own interior selection, and the contract explicitly leaves them
    unchanged. ``_assemble_materials`` still hands the DECLARED shapes out
    as ``pec_shapes`` (they are the user's own Boxes, not a realization),
    and the conformal path is what turns them into weights.

    The fence is measurable on a body whose face sits mid-cell: the binary
    rule owns a cell or does not, while the SDF weight is fractional there.
    That difference is the thing §1.8 declines to reconcile in this change;
    if a future edit derives the weights from ``realized_pec_edge_masks``,
    this test is where the decision gets recorded.
    """
    from rfx.geometry.conformal import compute_conformal_weights

    sim = _sim()
    grid = sim._build_grid()
    out = sim._assemble_materials(grid)
    # the shapes are handed on unchanged — a declaration, not a realization
    assert [type(s).__name__ for s in out[4]] == ["Box"]
    assert out[4][0].corner_lo == LO and out[4][0].corner_hi == HI

    # a face half a cell off the node line: binary owns whole cells, the
    # SDF weight does not.
    off = Box((0.0065, 0.006, 0.008), (0.014, 0.014, 0.012))
    co = coords_from_uniform_grid(grid)
    cells = classify_pec_entry(off, co, cell_centres_from_nodes(co),
                               name="pec")[0]
    binary = np.asarray(cells, bool)
    assert set(np.unique(binary.astype(float))) <= {0.0, 1.0}
    w = compute_conformal_weights(grid, [off])
    n_frac = sum(int(((a > 1e-6) & (a < 1.0 - 1e-6)).sum())
                 for a in (np.asarray(c, dtype=float) for c in w))
    assert n_frac > 0, (
        "fixture guard: the conformal channel must be fractional somewhere, "
        "otherwise the fence below is vacuous")
    assert n_frac == 64, n_frac   # 32 Ey + 32 Ez on the mid-cell x face
