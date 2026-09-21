"""#931 lattice ownership contract — the pre-declared battery (design note §5).

Normative source: ``docs/design_notes/20260906_plan_realign_lattice_ownership.md``.
One sentence covers every conductor: **an E component is PEC iff its own
location is inside the closed conductor region**, evaluated for a 3-D
(volume), 2-D (sheet) and 1-D (wire) region on the Yee lattice.

Index conventions (§1.1): node ``i`` is the LOWER corner of primal cell
``i``; ``Ex[i,j,k]`` sits at ``(x_{i+1/2}, y_j, z_k)`` and cyclically. A
Box drawn ``z_a -> z_b`` on node planes occupies cells ``a .. b-1`` and
must realize tangential walls at BOTH ``z_a`` and ``z_b``.

The slab battery goes through ``apply_pec_mask`` (present in the old code
too) so that the far-face assertions FAIL on the pre-#931 neighbour rule
rather than erroring at import: on the old rule a 1-cell slab is a single
plane at ``lo`` and its far face at ``hi`` is never a wall.

Sections that need the rasterizer side of the contract (sheet
declarations through ``add_thin_conductor``, the sub-cell refusal, the
``two_plane`` grep) are marked in their docstrings; they stay red until
that stage lands.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.pec import apply_pec_mask
from rfx.core.yee import init_state

RFX_ROOT = Path(__file__).resolve().parents[2]

SHAPE = (7, 8, 9)
LO = (2, 2, 2)          # first occupied cell per axis
HI = (5, 6, 7)          # one past the last occupied cell per axis


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ones_state(shape):
    ones = jnp.ones(shape, jnp.float32)
    return init_state(shape)._replace(ex=ones, ey=ones, ez=ones)


def _zeroed(state):
    """Boolean per-component 'this edge was zeroed' masks from a ones state."""
    return tuple(np.asarray(getattr(state, c)) == 0.0 for c in ("ex", "ey", "ez"))


def _box_cells(shape, lo, hi):
    m = np.zeros(shape, dtype=bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


def _slab_cells(shape, axis, k0, t, foot_lo=(1, 1, 1), foot_hi=None):
    """Cells of a slab ``t`` cells thick along ``axis`` starting at cell k0."""
    lo = list(foot_lo)
    hi = list(foot_hi) if foot_hi is not None else [s - 1 for s in shape]
    lo[axis] = k0
    hi[axis] = k0 + t
    return _box_cells(shape, lo, hi)


def _expected_volume_edges(shape, lo, hi):
    """Closed-region reading of the contract for a node-aligned Box.

    Written directly from §1.1 (where each component sits), NOT from the
    §1.2 formula, so the test is an independent statement of the rule.
    Ex[i,j,k] at (x_{i+1/2}, y_j, z_k) is inside the closed box iff
    lo_x <= i < hi_x (its centre), lo_y <= j <= hi_y, lo_z <= k <= hi_z.
    """
    ex = np.zeros(shape, dtype=bool)
    ey = np.zeros(shape, dtype=bool)
    ez = np.zeros(shape, dtype=bool)
    ex[lo[0]:hi[0], lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
    ey[lo[0]:hi[0] + 1, lo[1]:hi[1], lo[2]:hi[2] + 1] = True
    ez[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2]] = True
    return ex, ey, ez


def _interior_slab_case(axis, t):
    lo = [2, 2, 2]
    hi = [5, 6, 7]
    lo[axis] = 3
    hi[axis] = 3 + t
    return tuple(lo), tuple(hi)


# ---------------------------------------------------------------------------
# §5 slab battery — volume: walls at lo AND hi, no live normal edge inside
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("t", [1, 2, 3])
def test_slab_realizes_both_faces_and_shorts_its_interior(axis, t):
    lo, hi = _interior_slab_case(axis, t)
    cells = _box_cells(SHAPE, lo, hi)
    got = _zeroed(apply_pec_mask(_ones_state(SHAPE), jnp.asarray(cells)))
    tangential = [c for c in range(3) if c != axis]
    k_lo, k_hi = lo[axis], hi[axis]
    for c in tangential:
        m = got[c]
        wall_lo = np.take(m, k_lo, axis=axis)
        wall_hi = np.take(m, k_hi, axis=axis)
        assert wall_lo.any(), f"axis={axis} t={t}: no wall at lo plane {k_lo} for E{'xyz'[c]}"
        assert wall_hi.any(), (
            f"axis={axis} t={t}: FAR FACE MISSING — no wall at hi plane "
            f"{k_hi} for E{'xyz'[c]} (old one-plane-per-cell rule)")
        # the two faces carry the same footprint
        np.testing.assert_array_equal(wall_lo, wall_hi)
        # nothing outside the closed extent
        outside = [k for k in range(SHAPE[axis]) if k < k_lo or k > k_hi]
        for k in outside:
            assert not np.take(m, k, axis=axis).any(), (axis, t, c, k)
    # normal component: every edge inside the closed slab (cells and the
    # lateral rim nodes) is shorted, none outside
    n = got[axis]
    exp_n = _expected_volume_edges(SHAPE, lo, hi)[axis]
    for k in range(SHAPE[axis]):
        plane = np.take(n, k, axis=axis)
        if k_lo <= k < k_hi:
            assert np.all(plane[np.take(cells, k, axis=axis)]), \
                f"live normal edge inside slab k={k}"
            np.testing.assert_array_equal(plane, np.take(exp_n, k, axis=axis))
        else:
            assert not plane.any(), (axis, t, k)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("t", [1, 2, 3])
def test_slab_realized_thickness_equals_drawn(axis, t):
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    lo, hi = _interior_slab_case(axis, t)
    masks = realized_pec_edge_masks(jnp.asarray(_box_cells(SHAPE, lo, hi)))
    planes = realized_wall_planes(masks, axis)
    assert planes == list(range(lo[axis], hi[axis] + 1)), (axis, t, planes)
    assert planes[-1] - planes[0] == t


def test_one_cell_box_is_a_filled_slab_with_no_flag():
    """A 1-cell PEC Box realizes two faces on every axis — no ``two_plane``."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    lo, hi = (3, 3, 3), (4, 4, 4)
    got = realized_pec_edge_masks(jnp.asarray(_box_cells(SHAPE, lo, hi)))
    exp = _expected_volume_edges(SHAPE, lo, hi)
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(got[c]), exp[c], err_msg="xyz"[c])
    # 4 edges per component: the 12 edges of one cube
    assert [int(np.asarray(g).sum()) for g in got] == [4, 4, 4]


# ---------------------------------------------------------------------------
# §5 footprint battery — a patch Box realizes its drawn rectangle (closed)
# ---------------------------------------------------------------------------

def test_volume_box_realizes_its_closed_extent_exactly():
    from rfx.boundaries.pec import realized_pec_edge_masks
    got = realized_pec_edge_masks(jnp.asarray(_box_cells(SHAPE, LO, HI)))
    exp = _expected_volume_edges(SHAPE, LO, HI)
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(got[c]), exp[c], err_msg="xyz"[c])


def test_volume_box_through_the_api_has_walls_on_both_drawn_planes():
    """Node-aligned Box via ``sim.add(..., material='pec')`` — cell set is
    the same under node and centre sampling, so this holds with either
    rasterizer; the wall planes are what the contract adds."""
    from rfx import Box, Simulation
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    dx = 1e-3
    sim = Simulation(freq_max=10e9, domain=(0.010, 0.010, 0.010),
                     boundary="pec", dx=dx)
    lo = (0.002, 0.003, 0.004)
    hi = (0.006, 0.007, 0.005)      # z: 1 cell thick
    sim.add(Box(lo, hi), material="pec")
    grid = sim._build_grid()
    pec_mask = sim._assemble_materials(grid)[3]
    assert pec_mask is not None
    masks = realized_pec_edge_masks(pec_mask)
    i_lo = grid.position_to_index(lo)
    i_hi = grid.position_to_index(hi)
    for ax in range(3):
        assert realized_wall_planes(masks, ax) == list(range(i_lo[ax], i_hi[ax] + 1)), ax
    exp = _expected_volume_edges(grid.shape, i_lo, i_hi)
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(masks[c]), exp[c], err_msg="xyz"[c])


# ---------------------------------------------------------------------------
# §5 sheet battery — one plane, live normal edge, closed footprint
# ---------------------------------------------------------------------------

def _sheet(shape, axis, plane, foot_lo, foot_hi):
    """SheetSpec with a closed node footprint [foot_lo, foot_hi] in-plane."""
    from rfx.boundaries.pec import SheetSpec
    f = np.zeros(shape, dtype=bool)
    sl = [slice(foot_lo[a], foot_hi[a] + 1) for a in range(3)]
    sl[axis] = slice(plane, plane + 1)
    f[tuple(sl)] = True
    return SheetSpec(normal_axis=axis, plane=plane, footprint=jnp.asarray(f))


def _expected_sheet_edges(shape, axis, plane, foot_lo, foot_hi):
    """§1.3 read directly: E_t (t != axis) at plane, both end nodes in F."""
    out = [np.zeros(shape, dtype=bool) for _ in range(3)]
    for t in range(3):
        if t == axis:
            continue
        sl = [slice(foot_lo[a], foot_hi[a] + 1) for a in range(3)]
        sl[axis] = slice(plane, plane + 1)
        sl[t] = slice(foot_lo[t], foot_hi[t])          # edges i .. i+1, last node excluded
        out[t][tuple(sl)] = True
    return tuple(out)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sheet_realizes_one_plane_and_leaves_the_normal_edge_live(axis):
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    plane = 4
    foot_lo, foot_hi = (1, 2, 1), (4, 5, 6)
    spec = _sheet(SHAPE, axis, plane, foot_lo, foot_hi)
    got = realized_pec_edge_masks(None, sheets=[spec])
    exp = _expected_sheet_edges(SHAPE, axis, plane, foot_lo, foot_hi)
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(got[c]), exp[c], err_msg="xyz"[c])
    assert not bool(jnp.any(got[axis])), "sheet-normal E must stay live"
    assert realized_wall_planes(got, axis) == [plane]


def test_sheet_with_a_cell_mask_of_none_or_zeros_is_the_same():
    from rfx.boundaries.pec import realized_pec_edge_masks
    spec = _sheet(SHAPE, 2, 3, (1, 1, 0), (5, 5, 0))
    a = realized_pec_edge_masks(None, sheets=[spec])
    b = realized_pec_edge_masks(jnp.zeros(SHAPE, bool), sheets=[spec])
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(a[c]), np.asarray(b[c]))


def test_abutting_sheets_on_one_plane_union_before_the_edge_rule():
    """Two footprints sharing a node row realize seamlessly — no slit."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    left = _sheet(SHAPE, 2, 3, (1, 1, 0), (3, 5, 0))
    right = _sheet(SHAPE, 2, 3, (3, 1, 0), (5, 5, 0))
    whole = _sheet(SHAPE, 2, 3, (1, 1, 0), (5, 5, 0))
    two = realized_pec_edge_masks(None, sheets=[left, right])
    one = realized_pec_edge_masks(None, sheets=[whole])
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(two[c]), np.asarray(one[c]))


def test_sheets_on_adjacent_planes_keep_a_live_normal_edge_between_them():
    """#690 semantics: two films one cell apart, the gap edge stays live."""
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    a = _sheet(SHAPE, 2, 3, (1, 1, 0), (5, 5, 0))
    b = _sheet(SHAPE, 2, 4, (1, 1, 0), (5, 5, 0))
    got = realized_pec_edge_masks(None, sheets=[a, b])
    assert realized_wall_planes(got, 2) == [3, 4]
    assert not bool(jnp.any(got[2]))


def test_sheet_footprint_is_the_closed_rectangle():
    """The hi row of the footprint carries a wall edge (closed sampling)."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    foot_lo, foot_hi = (1, 2, 0), (4, 5, 0)
    got = realized_pec_edge_masks(None, sheets=[_sheet(SHAPE, 2, 3, foot_lo, foot_hi)])
    ex, ey, _ = (np.asarray(g) for g in got)
    assert ex[1:4, 5, 3].all()          # hi row j=5 carries Ex edges
    assert ey[4, 2:5, 3].all()          # hi column i=4 carries Ey edges
    assert not ex[4, :, 3].any()        # no edge leaves the footprint
    assert not ey[:, 5, 3].any()


def test_sheet_spec_validates_its_layer():
    from rfx.boundaries.pec import SheetSpec
    f = np.zeros(SHAPE, dtype=bool)
    f[1:3, 1:3, 2] = True
    f[1:3, 1:3, 4] = True       # a second layer
    with pytest.raises(ValueError):
        SheetSpec(normal_axis=2, plane=2, footprint=jnp.asarray(f))


# ---------------------------------------------------------------------------
# §5 mirror / axis-permutation invariance
# ---------------------------------------------------------------------------

def _random_body(rng, shape):
    m = np.zeros(shape, dtype=bool)
    for _ in range(3):
        lo = [rng.integers(1, s - 2) for s in shape]
        hi = [rng.integers(l + 1, s - 1) for l, s in zip(lo, shape)]
        m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_realized_edges_commute_with_axis_permutation(perm):
    from rfx.boundaries.pec import realized_pec_edge_masks
    rng = np.random.default_rng(931)
    cells = _random_body(rng, (8, 8, 8))
    base = [np.asarray(m) for m in realized_pec_edge_masks(jnp.asarray(cells))]
    permuted = realized_pec_edge_masks(jnp.asarray(np.transpose(cells, perm)))
    for new_axis, old_axis in enumerate(perm):
        np.testing.assert_array_equal(np.asarray(permuted[new_axis]),
                                      np.transpose(base[old_axis], perm))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_realized_edges_commute_with_mirror_of_an_interior_body(axis):
    """Mirroring cells i <-> n-1-i maps the along-axis component index
    i -> n-1-i and the other two (node-plane) components i -> n-i."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    rng = np.random.default_rng(7)
    cells = _random_body(rng, (8, 8, 8))
    base = [np.asarray(m) for m in realized_pec_edge_masks(jnp.asarray(cells))]
    mirrored = [np.asarray(m) for m in
                realized_pec_edge_masks(jnp.asarray(np.flip(cells, axis)))]
    for c in range(3):
        if c == axis:
            np.testing.assert_array_equal(mirrored[c], np.flip(base[c], axis))
        else:
            flipped = np.flip(base[c], axis)            # plane i -> n-1-i
            shifted = np.roll(flipped, 1, axis=axis)   # -> n-i
            # plane n (index n) has no array entry; interior bodies never
            # touch it, so roll wrap carries only zeros
            np.testing.assert_array_equal(mirrored[c], shifted)


# ---------------------------------------------------------------------------
# §5 soft == hard at binary occupancy
# ---------------------------------------------------------------------------

def _soft_hard_battery():
    n = 8
    shape = (n, n, n)
    cases = []
    for axis in range(3):
        for t in (1, 2, 3):
            for k0 in (0, 2, n - t):          # face 0, interior, face n-1
                cases.append((f"slab ax={axis} t={t} k0={k0}",
                              _slab_cells(shape, axis, k0, t, (1, 1, 1), (n - 1, n - 1, n - 1)),
                              (False, False, False)))
    # a seam-straddling body on a periodic axis (cells n-1 and 0)
    for axis in range(3):
        m = np.zeros(shape, dtype=bool)
        sl = [slice(2, 5)] * 3
        sl[axis] = slice(0, 1)
        m[tuple(sl)] = True
        sl[axis] = slice(n - 1, n)
        m[tuple(sl)] = True
        per = [False, False, False]
        per[axis] = True
        cases.append((f"seam ax={axis}", m, tuple(per)))
    rng = np.random.default_rng(3)
    cases.append(("random", rng.random(shape) < 0.35, (False, False, False)))
    cases.append(("random-periodic-y", rng.random(shape) < 0.35, (False, True, False)))
    cases.append(("2d-lane", _slab_cells((9, 9, 1), 0, 3, 2, (0, 2, 0), (9, 7, 1)),
                  (False, False, True)))
    return cases


@pytest.mark.parametrize("case", _soft_hard_battery(), ids=lambda c: c[0])
def test_soft_occupancy_is_bit_identical_to_hard_mask_at_binary(case):
    from rfx.boundaries.pec import apply_pec_occupancy
    name, cells, periodic = case
    st = _ones_state(cells.shape)
    hard = apply_pec_mask(st, jnp.asarray(cells), periodic)
    soft = apply_pec_occupancy(st, jnp.asarray(cells, dtype=jnp.float32), periodic)
    for c in ("ex", "ey", "ez"):
        np.testing.assert_array_equal(np.asarray(getattr(hard, c)),
                                      np.asarray(getattr(soft, c)), err_msg=f"{name} {c}")


def test_soft_path_ors_sheet_masks_in_statically():
    from rfx.boundaries.pec import apply_pec_occupancy, realized_pec_edge_masks
    spec = _sheet(SHAPE, 2, 3, (1, 1, 0), (5, 5, 0))
    sheet_masks = realized_pec_edge_masks(None, sheets=[spec])
    cells = _box_cells(SHAPE, (1, 1, 5), (3, 3, 7))
    st = _ones_state(SHAPE)
    hard = apply_pec_mask(st, jnp.asarray(cells), sheets=[spec])
    soft = apply_pec_occupancy(st, jnp.asarray(cells, jnp.float32),
                               sheet_edge_masks=sheet_masks)
    for c in ("ex", "ey", "ez"):
        np.testing.assert_array_equal(np.asarray(getattr(hard, c)),
                                      np.asarray(getattr(soft, c)), err_msg=c)


def test_a_binary_zero_occupancy_override_keeps_the_ports_edge_clearing():
    """§1.6 through the override ENTRY POINT, not just the operator.

    ``pec_occupancy_override=zeros`` has to be the hard path exactly. It was
    not: the soft lane rebuilt its static sheet/wire masks from the original
    declarations, while the masks it had been handed were already port-cleared
    (§1.9 — a port releases the one component it drives, at its own cells). A
    50 ohm Ex port standing on a PEC sheet therefore read a False Ex entry in
    the masks passed in and a True one in the reconstruction, so the override
    re-shorted the port. Measured on a 12 mm board at dx = 2 mm, 60 steps: an
    off-sheet Ez probe moved 5.7245574 -> 12.5340872 when the all-zero
    override was added.

    Falsifier: drop the intersection in ``rfx/simulation.py`` and the two
    ``forward()`` calls below stop agreeing while the port's own Ex probe
    stays at zero either way — which is why the off-sheet probe is the one
    that carries the claim.
    """
    import jax.numpy as _jnp
    from rfx import Box, Simulation

    dx = 2e-3
    port = (6e-3, 6e-3, 6e-3)

    def _build():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(freq_max=15e9, domain=(12e-3, 12e-3, 12e-3),
                             dx=dx, boundary="pec")
            sim.add(Box((2e-3, 2e-3, 6e-3), (10e-3, 10e-3, 6e-3)),
                    material="pec")
            sim.add_port(port, "ex", impedance=50.0)
            sim.add_probe(position=(6e-3, 6e-3, 8e-3), component="ez")
        return sim

    grid = _build()._build_grid()
    zeros = _jnp.zeros(grid.shape, dtype=_jnp.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = _build().forward(n_steps=60)
        over = _build().forward(n_steps=60, pec_occupancy_override=zeros)

    a = np.asarray(plain.time_series)[:, 0]
    b = np.asarray(over.time_series)[:, 0]
    assert float(np.max(np.abs(a))) > 0, "the control read zero — nothing ran"
    np.testing.assert_array_equal(
        a, b, err_msg="a binary-zero pec_occupancy_override is not the hard "
                      "path: the port's edge clearing was undone")


def test_soft_occupancy_is_differentiable_and_noisy_or():
    import jax
    from rfx.boundaries.pec import apply_pec_occupancy
    st = _ones_state((5, 5, 5))

    def f(occ):
        return jnp.sum(apply_pec_occupancy(st, occ).ex)

    occ = 0.5 * jnp.ones((5, 5, 5), jnp.float32)
    g = jax.grad(f)(occ)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0
    # one interior cell at 0.5: the four Ex edges it touches read 1 - (1-0.5) = 0.5
    occ1 = jnp.zeros((5, 5, 5), jnp.float32).at[2, 2, 2].set(0.5)
    out = apply_pec_occupancy(st, occ1).ex
    assert float(out[2, 2, 2]) == pytest.approx(0.5)
    assert float(out[2, 3, 3]) == pytest.approx(0.5)
    assert float(out[2, 2, 4]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# wires (§1.4) and the helpers (§1.7 / §1.9)
# ---------------------------------------------------------------------------

def test_wire_path_realizes_the_axis_aligned_lattice_edges():
    from rfx.boundaries.pec import WireSpec, realized_pec_edge_masks, wire_path_edge_masks
    nodes = [(1, 1, 1), (4, 1, 1), (4, 3, 1), (4, 3, 5)]
    edges = wire_path_edge_masks(nodes, SHAPE)
    ex, ey, ez = (np.asarray(e) for e in edges)
    assert ex[1:4, 1, 1].all() and ex.sum() == 3
    assert ey[4, 1:3, 1].all() and ey.sum() == 2
    assert ez[4, 3, 1:5].all() and ez.sum() == 4
    got = realized_pec_edge_masks(None, wires=[WireSpec(edges=edges)])
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(got[c]), (ex, ey, ez)[c])


def test_wire_path_refuses_a_diagonal_segment():
    from rfx.boundaries.pec import wire_path_edge_masks
    with pytest.raises(ValueError, match="axis-aligned"):
        wire_path_edge_masks([(1, 1, 1), (3, 2, 1)], SHAPE)


def test_edge_is_pec_and_clear_edges():
    from rfx.boundaries.pec import clear_edges, edge_is_pec, realized_pec_edge_masks
    masks = realized_pec_edge_masks(jnp.asarray(_box_cells(SHAPE, LO, HI)))
    assert edge_is_pec(masks, "ez", 3, 3, 3)
    assert edge_is_pec(masks, 0, 2, 2, 2)
    assert not edge_is_pec(masks, "ex", 5, 2, 2)       # x edge past the hi face
    cleared = clear_edges(masks, [(3, 3, 3)])
    for c in ("ex", "ey", "ez"):
        assert not edge_is_pec(cleared, c, 3, 3, 3)
    assert edge_is_pec(cleared, "ez", 3, 3, 4)
    as_mask = np.zeros(SHAPE, dtype=bool)
    as_mask[3, 3, 3] = True
    cleared2 = clear_edges(masks, jnp.asarray(as_mask))
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(cleared[c]), np.asarray(cleared2[c]))


def test_realized_wall_planes_region_and_column():
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    masks = realized_pec_edge_masks(jnp.asarray(_box_cells(SHAPE, LO, HI)))
    assert realized_wall_planes(masks, 2) == [2, 3, 4, 5, 6, 7]
    assert realized_wall_planes(masks, 2, ij=(3, 3)) == [2, 3, 4, 5, 6, 7]
    assert realized_wall_planes(masks, 2, ij=(0, 0)) == []
    # hi-rim node (i=5, j=6): its incident tangential edges are at i-1 / j-1
    assert realized_wall_planes(masks, 2, ij=(5, 6)) == [2, 3, 4, 5, 6, 7]
    region = (slice(0, 2), slice(None), slice(None))
    assert realized_wall_planes(masks, 2, region=region) == []


# ---------------------------------------------------------------------------
# rasterizer-facing contract (needs the geometry stage; red until it lands)
# ---------------------------------------------------------------------------

def _patch_sim(**thin_kwargs):
    from rfx import Box, Simulation
    sim = Simulation(freq_max=10e9, domain=(0.010, 0.010, 0.006),
                     boundary="pec", dx=1e-3)
    sim.add_thin_conductor(Box((0.002, 0.003, 0.003), (0.006, 0.007, 0.003)),
                           **thin_kwargs)
    return sim


def test_pec_thin_conductor_is_a_sheet_not_a_cell():
    """A zero-thickness PEC Box via ``add_thin_conductor`` owns no cell and
    realizes exactly its closed drawn footprint on one plane (§1.3)."""
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    sim = _patch_sim()
    grid = sim._build_grid()
    pec_sheets: list = []
    pec_mask = sim._assemble_materials(grid, pec_sheets=pec_sheets)[3]
    assert pec_mask is None or not bool(jnp.any(pec_mask))
    assert len(pec_sheets) == 1
    spec = pec_sheets[0]
    assert spec.normal_axis == 2
    i_lo = grid.position_to_index((0.002, 0.003, 0.003))
    i_hi = grid.position_to_index((0.006, 0.007, 0.003))
    assert spec.plane == i_lo[2]
    masks = realized_pec_edge_masks(pec_mask, sheets=pec_sheets)
    exp = _expected_sheet_edges(grid.shape, 2, i_lo[2], i_lo, i_hi)
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(masks[c]), exp[c], err_msg="xyz"[c])
    assert realized_wall_planes(masks, 2) == [i_lo[2]]


def test_g4_pec_sheet_and_f0_sheet_share_footprint_and_edge_set():
    """G4 by construction: the same Box declared PEC and declared lossy
    (``surface_impedance_f0``) gives one footprint and one edge set."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx
    pec_sheets: list = []
    sim_pec = _patch_sim()
    sim_pec._assemble_materials(sim_pec._build_grid(), pec_sheets=pec_sheets)
    f0_specs: list = []
    sim_f0 = _patch_sim(sigma_bulk=1e4, thickness=35e-6, surface_impedance_f0=5e9)
    sim_f0._assemble_materials(sim_f0._build_grid(), sheet_specs=f0_specs)
    assert len(pec_sheets) == 1 and len(f0_specs) == 1
    np.testing.assert_array_equal(np.asarray(pec_sheets[0].footprint),
                                  np.asarray(f0_specs[0].mask))
    pec_edges = realized_pec_edge_masks(None, sheets=pec_sheets)
    ctx = build_sheet_impedance_ctx(f0_specs)
    for got, exp in zip((ctx.mask_ex, ctx.mask_ey, ctx.mask_ez), pec_edges):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(exp))


def test_zero_thickness_pec_box_via_add_is_a_sheet_declaration():
    """§1.5 (amended): a PEC Box with EXACTLY ONE zero-extent axis passed to
    ``sim.add`` is a SHEET — zero thickness is a statement of intent, not an
    inference — and realizes exactly what ``add_thin_conductor`` realizes for
    the same Box. This is what keeps the documented five-line patch workflow
    and ``first-patch.mdx`` valid."""
    from rfx import Box, Simulation
    from rfx.boundaries.pec import realized_pec_edge_masks

    def _sim(via_add):
        sim = Simulation(freq_max=10e9, domain=(0.010, 0.010, 0.006),
                         boundary="pec", dx=1e-3)
        box = Box((0.002, 0.003, 0.003), (0.006, 0.007, 0.003))
        if via_add:
            sim.add(box, material="pec")
        else:
            sim.add_thin_conductor(box)
        return sim

    out = []
    for via_add in (True, False):
        sim = _sim(via_add)
        sheets: list = []
        pec_mask = sim._assemble_materials(sim._build_grid(),
                                           pec_sheets=sheets)[3]
        assert pec_mask is None or not bool(jnp.any(pec_mask)), via_add
        assert len(sheets) == 1, via_add
        out.append((sheets[0].normal_axis, sheets[0].plane,
                    realized_pec_edge_masks(None, sheets=sheets)))
    assert out[0][0] == out[1][0] and out[0][1] == out[1][1]
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(out[0][2][c]),
                                      np.asarray(out[1][2][c]))


@pytest.mark.parametrize("hi", [(0.006, 0.003, 0.003), (0.002, 0.003, 0.003)])
def test_pec_box_with_two_or_three_zero_axes_is_refused(hi):
    """§1.5: a line or a point is not a conductor; the message names
    ``PolylineWire``."""
    from rfx import Box, Simulation
    sim = Simulation(freq_max=10e9, domain=(0.010, 0.010, 0.006),
                     boundary="pec", dx=1e-3)
    sim.add(Box((0.002, 0.003, 0.003), hi), material="pec")
    with pytest.raises(ValueError, match="PolylineWire"):
        sim._assemble_materials(sim._build_grid())


def test_sub_cell_non_box_sheet_is_realized_from_its_mid_plane_cross_section():
    """§1.3: a non-Box shape declared through ``add_thin_conductor`` — a
    Cylinder pad one cell tall — is a sheet whose footprint is its
    cross-section at its OWN mid-plane, placed on the nearest node plane.
    How many node planes its thickness straddles never enters."""
    from rfx import Cylinder, Simulation
    sim = Simulation(freq_max=10e9, domain=(0.010, 0.010, 0.006),
                     boundary="pec", dx=1e-3)
    sim.add_thin_conductor(
        Cylinder(center=(0.005, 0.005, 0.0035), radius=0.002, height=1e-3,
                 axis="z"))
    grid = sim._build_grid()
    sheets: list = []
    pec_mask = sim._assemble_materials(grid, pec_sheets=sheets)[3]
    assert pec_mask is None or not bool(jnp.any(pec_mask))
    assert len(sheets) == 1
    spec = sheets[0]
    assert spec.normal_axis == 2
    # mid-plane 3.5 mm, an exact half-cell tie -> the LOWER node plane
    assert spec.plane == grid.position_to_index((0.005, 0.005, 0.003))[2]
    fp = np.asarray(spec.footprint)
    # the disc's own cross-section, on ONE layer
    assert fp.sum() == fp[:, :, spec.plane].sum() > 0
    ci, cj = grid.position_to_index((0.005, 0.005, 0.003))[:2]
    assert fp[ci, cj, spec.plane]
    assert not fp[ci + 3, cj, spec.plane]      # 3 mm out, radius is 2 mm


def test_sub_cell_pec_box_via_add_is_refused():
    from rfx import Box, Simulation
    sim = Simulation(freq_max=10e9, domain=(0.010, 0.010, 0.006),
                     boundary="pec", dx=1e-3)
    sim.add(Box((0.002, 0.003, 0.003), (0.006, 0.007, 0.0034)), material="pec")
    with pytest.raises(ValueError, match="add_thin_conductor"):
        sim._assemble_materials(sim._build_grid())


def test_conformal_weights_do_not_replace_the_realized_edges():
    """§1.7 / §1.8: Dey-Mittra conformal is a subpixel UPDATE-COEFFICIENT
    model, not a second realization — a run that carries conformal weights
    still applies ``(Mx, My, Mz)`` in its step body.

    ``apply_conformal_pec`` zeroes only edges whose weight is exactly 0, and
    NO edge of a one-cell PEC slab is fully covered (both its faces sit on
    the slab's own boundary, so w = 1/2 there — measured below). While the
    waveguide S-matrix lane folded interior PEC into sigma=1e10 the
    conductor survived that gap; with the fold deleted, a step body that
    chose one branch OR the other dropped it outright (measured: conformal
    PEC-short min|S11| 0.2296 against a 0.99 gate, restored to 0.9942).
    Falsifier: restore the ``elif`` in ``rfx/simulation.py``'s step body and
    the field on the slab is nonzero.
    """
    import jax.numpy as _jnp
    from rfx import Box
    from rfx.core.yee import init_materials
    from rfx.geometry.conformal import (
        clamp_conformal_weights, compute_conformal_weights_sdf)
    from rfx.grid import Grid
    from rfx.simulation import make_source, run as run_simulation
    from rfx.sources.sources import GaussianPulse
    from rfx.boundaries.pec import realized_pec_edge_masks

    grid = Grid(freq_max=10e9, domain=(0.024, 0.012, 0.012), dx=0.003)
    shape = grid.shape
    slab = Box((0.012, 0.0, 0.0), (0.015, 0.012, 0.012))   # one cell thick
    k = grid.position_to_index((0.012, 0.006, 0.006))[0]
    cells = np.zeros(shape, bool)
    cells[k, :, :] = True
    w = clamp_conformal_weights(
        *compute_conformal_weights_sdf(grid, [slab]), 0.1)
    # The premise, measured: the conformal path alone zeroes NOTHING on a
    # one-cell slab — no edge of it has weight 0.
    for c in range(3):
        assert int(np.count_nonzero(np.asarray(w[c]) == 0.0)) == 0, c
    # ... while the realized edge set shorts both of its faces.
    edges = realized_pec_edge_masks(_jnp.asarray(cells))
    assert bool(np.asarray(edges[1])[k].all()) and bool(np.asarray(edges[1])[k + 1].all())

    src = make_source(grid, (0.018, 0.006, 0.006), "ey",
                      GaussianPulse(f0=6e9, bandwidth=0.8), 40)
    res = run_simulation(
        grid, init_materials(shape), 40, sources=[src], boundary="pec",
        pec_mask=_jnp.asarray(cells), conformal_weights=w)
    st = res.state
    for plane in (k, k + 1):
        assert float(np.abs(np.asarray(st.ey)[plane]).max()) == 0.0, plane
        assert float(np.abs(np.asarray(st.ez)[plane]).max()) == 0.0, plane
    # ... and the run is not trivially zero everywhere
    assert float(np.abs(np.asarray(st.ey)[k + 3]).max()) > 0.0


def test_two_plane_is_gone_from_the_package():
    """§1.5: no ``two_plane`` and no per-entry realization knob in rfx/.
    The reference-plane helper ``refplane_zc_two_plane`` is unrelated."""
    out = subprocess.run(
        ["grep", "-rnI", "--exclude-dir=__pycache__", "two_plane", str(RFX_ROOT / "rfx")],
        capture_output=True, text=True, check=False).stdout.splitlines()
    hits = [h for h in out if "refplane_zc_two_plane" not in h]
    assert hits == [], "\n".join(hits)
    knob = re.compile(r"\brealization\s*=")
    bad = []
    # rfx/preflight/ is globbed alongside rfx/api/ because #980 Phase 3 moves
    # the preflight bodies -- including every realized-PEC reader this scan
    # exists for -- out of rfx/api/_preflight.py into that package. A scan
    # that stayed pinned to rfx/api/ would keep passing while the code it is
    # meant to cover walked out from under it.
    scanned = [*(RFX_ROOT / "rfx" / "api").glob("*.py"),
               *(RFX_ROOT / "rfx" / "preflight").glob("*.py")]
    for p in scanned:
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if knob.search(line):
                bad.append(f"{p}:{n}: {line.strip()}")
    assert bad == [], "\n".join(bad)


# ---------------------------------------------------------------------------
# review fixes (2026-09-07) — design note §6
# ---------------------------------------------------------------------------

def test_clear_edges_releases_only_the_named_component():
    """§1.9 corrected: a port releases the ONE edge it drives.

    The three-component form releases the two edges TANGENTIAL to the port
    at its foot, which wherever the foot stands on a conductor's node plane
    ARE that conductor's wall. A ground plane under an MSL feed, the top
    face of a body under a probe feed: both lost wall edges to it.
    """
    from rfx.boundaries.pec import clear_edges, edge_is_pec, realized_pec_edge_masks
    masks = realized_pec_edge_masks(jnp.asarray(_box_cells(SHAPE, LO, HI)))
    cleared = clear_edges(masks, [(3, 3, 3)], component="ez")
    assert not edge_is_pec(cleared, "ez", 3, 3, 3)
    assert edge_is_pec(cleared, "ex", 3, 3, 3)
    assert edge_is_pec(cleared, "ey", 3, 3, 3)
    as_mask = np.zeros(SHAPE, dtype=bool)
    as_mask[3, 3, 3] = True
    cleared_m = clear_edges(masks, jnp.asarray(as_mask), component=2)
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(cleared[c]),
                                      np.asarray(cleared_m[c]))


def test_edges_are_pec_matches_edge_is_pec():
    from rfx.boundaries.pec import edge_is_pec, edges_are_pec, realized_pec_edge_masks
    masks = realized_pec_edge_masks(jnp.asarray(_box_cells(SHAPE, LO, HI)))
    cells = [(3, 3, 3), (5, 2, 2), (0, 0, 0), (2, 2, 2)]
    assert (edges_are_pec(masks, "ex", cells)
            == [edge_is_pec(masks, "ex", *c) for c in cells])


def test_realized_wall_planes_column_follows_the_periodic_wrap():
    """§1.7 / #689: node 0's backward incident edge is stored at n-1."""
    from rfx.boundaries.pec import (
        SheetSpec, realized_pec_edge_masks, realized_wall_planes,
    )
    fp = np.zeros(SHAPE, dtype=bool)
    fp[SHAPE[0] - 1, 3, 3] = True
    fp[0, 3, 3] = True
    sheet = SheetSpec(normal_axis=2, plane=3, footprint=jnp.asarray(fp))
    periodic = (True, False, False)
    masks = realized_pec_edge_masks(None, sheets=[sheet], periodic=periodic)
    # the seam edge Ex[n-1, 3, 3] joins node n-1 to node 0 and IS realized
    assert bool(np.asarray(masks[0])[SHAPE[0] - 1, 3, 3])
    assert realized_wall_planes(masks, 2, ij=(SHAPE[0] - 1, 3)) == [3]
    assert realized_wall_planes(masks, 2, ij=(0, 3)) == []
    assert realized_wall_planes(masks, 2, ij=(0, 3), periodic=periodic) == [3]


def test_wire_path_has_no_periodic_argument():
    """A path is a list of edges the caller named, not a neighbour rule."""
    import inspect

    from rfx.boundaries.pec import wire_path_edge_masks
    assert "periodic" not in inspect.signature(wire_path_edge_masks).parameters


def test_subcell_refusal_covers_non_box_shapes():
    """§1.5 is on the DRAWN extent of any shape, not just a Box.

    A 0.3-cell Cylinder pad used to realize as a ONE-CELL slab with two
    faces at one z and raise "ZERO cells" at another 0.3 cell away —
    raster-dependent thickness, which §1.5 exists to forbid.
    """
    from rfx.geometry import Cylinder
    from rfx.geometry.rasterize_grid import (
        cell_sizes_from_uniform_grid, centres_from_uniform_grid,
        classify_pec_entry, coords_from_uniform_grid,
    )
    from rfx.grid import Grid

    g = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3)
    coords = coords_from_uniform_grid(g)
    centres = centres_from_uniform_grid(g)
    sizes = cell_sizes_from_uniform_grid(g)
    for z_c in (8.5e-3, 8.2e-3):
        pad = Cylinder(center=(0.0, 0.0, z_c), radius=3e-3,
                       height=0.3e-3, axis="z")
        with pytest.raises(ValueError, match="thinner than one cell"):
            classify_pec_entry(pad, coords, centres, sizes, name="pad")


def test_sheet_plane_outside_the_node_line_is_refused():
    """§1.3: "nearest" means within half a cell, not "clamp onto the end"."""
    from rfx.geometry.csg import Box
    from rfx.geometry.rasterize_grid import (
        cell_sizes_from_uniform_grid, coords_from_uniform_grid,
        sheet_spec_from_shape,
    )
    from rfx.grid import Grid

    g = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3)
    coords = coords_from_uniform_grid(g)
    sizes = cell_sizes_from_uniform_grid(g)
    z_out = float(np.asarray(coords.z)[-1]) + 4e-3
    box = Box(corner_lo=(-5e-3, -5e-3, z_out), corner_hi=(5e-3, 5e-3, z_out))
    with pytest.raises(ValueError, match="nearest node line"):
        sheet_spec_from_shape(box, coords, sizes, normal_axis=2, name="foil")


def test_cell_centres_use_the_actual_node_line():
    """A caller-supplied axis with a fractional origin gets node + d/2."""
    from rfx.geometry.rasterize_grid import GridCoords, cell_centres_from_nodes
    nodes = 0.25 + np.arange(10.0)
    c = GridCoords(x=nodes, y=nodes, z=nodes, shape=(10, 10, 10))
    got = cell_centres_from_nodes(c)
    np.testing.assert_allclose(np.asarray(got.x) - nodes, 0.5)


def test_stackup_foils_sit_on_the_dielectric_faces():
    """§4 rule 2: a foil sheet is ON its laminate face, not half a foil off."""
    from rfx.pcb import Stackup
    shapes = Stackup.standard_2layer().to_shapes()
    faces = set()
    for (box, mat) in shapes:
        if box.corner_lo[2] != box.corner_hi[2]:
            faces.add(round(box.corner_lo[2], 15))
            faces.add(round(box.corner_hi[2], 15))
    for (box, mat) in shapes:
        if box.corner_lo[2] == box.corner_hi[2]:
            assert round(box.corner_lo[2], 15) in faces


# ---------------------------------------------------------------------------
# the two spellings of a sheet, the domain-BC fence, and the report
# ---------------------------------------------------------------------------

def test_add_thin_conductor_takes_both_sheet_spellings_and_lands_on_one_plane():
    """§1.3: a zero-extent Box and a ONE-CELL Box declare the same sheet.

    Three oracle fixtures spell the same operator two ways —
    ``tests/oracle/test_sheet_film_rta_analytic.py`` hands
    ``add_thin_conductor`` a Box one cell thick, while
    ``test_sheet_perturbation_q.py`` and ``test_leontovich_alpha_oracle.py``
    hand it a zero-extent Box — and nothing said which was normative, so
    "what does a one-cell Box mean to ``add_thin_conductor``" was an open
    question with two live answers.

    The contract settles it without a new rule: the sheet's plane is the
    node plane NEAREST the shape's mid-plane, tie to the LOWER plane. A
    Box from node ``k`` to node ``k+1`` has its mid-plane exactly half a
    cell above node ``k``, which is the tie — so it lands on ``k``, the
    same plane the zero-extent Box at ``k`` lands on. One cell is the
    ceiling: ``refuse_thick`` rejects anything above it ("not a sheet;
    use add() for a volume").
    """
    from rfx.geometry.csg import Box
    from rfx.geometry.rasterize_grid import (
        cell_sizes_from_uniform_grid, coords_from_uniform_grid,
        sheet_spec_from_shape,
    )
    from rfx.grid import Grid

    g = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3)
    coords = coords_from_uniform_grid(g)
    sizes = cell_sizes_from_uniform_grid(g)
    z_k = float(np.asarray(coords.z)[6])
    d = 1e-3

    zero = sheet_spec_from_shape(
        Box(corner_lo=(2e-3, 2e-3, z_k), corner_hi=(8e-3, 8e-3, z_k)),
        coords, sizes, normal_axis=2, name="zero", refuse_thick=True)
    one_cell = sheet_spec_from_shape(
        Box(corner_lo=(2e-3, 2e-3, z_k), corner_hi=(8e-3, 8e-3, z_k + d)),
        coords, sizes, normal_axis=2, name="one_cell", refuse_thick=True)

    assert zero.plane == 6, zero.plane
    assert one_cell.plane == zero.plane, (
        "a one-cell-thick Box handed to add_thin_conductor must land on the "
        "same node plane as the zero-extent Box at its lo face (half-cell "
        f"tie -> lower plane): got {one_cell.plane} vs {zero.plane}")
    np.testing.assert_array_equal(np.asarray(one_cell.footprint),
                                  np.asarray(zero.footprint))

    with pytest.raises(ValueError, match="not a sheet"):
        sheet_spec_from_shape(
            Box(corner_lo=(2e-3, 2e-3, z_k), corner_hi=(8e-3, 8e-3, z_k + 2 * d)),
            coords, sizes, normal_axis=2, name="two_cell", refuse_thick=True)


def test_domain_boundary_pec_is_not_a_conductor_body():
    """§1.8 scope fence, in code rather than prose.

    ``BoundarySpec`` faces / ``apply_pec`` / ``apply_pec_faces`` zero the
    tangential E on the face plane and are NOT bodies: they own no cell,
    no sheet and no wire, and ``realized_pec_edge_masks`` never produces
    them. cv09, cv10, cv14, cv24, ``adi_solver_demo``, ``hello_world`` and
    ``resonance_harminv`` are the controls that must not move, and
    fourteen files in the oracle/lock group rest on the domain rule, so a
    well-meaning unification of the two would move every one of them
    silently. This is the tripwire.
    """
    from rfx.boundaries.pec import apply_pec_faces, realized_pec_edge_masks

    shape = (6, 6, 6)
    empty = np.zeros(shape, dtype=bool)
    with pytest.raises(ValueError, match="no cell mask, sheets or wires"):
        realized_pec_edge_masks(None)
    masks = realized_pec_edge_masks(jnp.asarray(empty))
    for c in range(3):
        assert not np.asarray(masks[c]).any(), (
            "a domain PEC face must not appear in the realized conductor "
            "edge set — it is a boundary condition, not a body")

    st = init_state(shape)
    st = st._replace(ex=jnp.ones(shape), ey=jnp.ones(shape), ez=jnp.ones(shape))
    out = apply_pec_faces(st, {"z_lo"})
    assert float(np.asarray(out.ex)[3, 3, 0]) == 0.0
    assert float(np.asarray(out.ey)[3, 3, 0]) == 0.0
    assert float(np.asarray(out.ez)[3, 3, 0]) == 1.0   # normal at the face
    assert float(np.asarray(out.ex)[3, 3, 1]) == 1.0   # one plane only


def test_fidelity_report_realized_extent_comes_from_the_wall_planes():
    """§3: the drawn-vs-realized table is read off the realized edges.

    Before #931 ``fidelity_report`` derived its realization string from the
    CELL census, which is why the committed example snapshot carries
    realization strings phrased in the old rule and ``sheet-own-cell-live``
    findings whose remedy was ``two_plane=True``. A 1-cell PEC Box must now
    report as a volume with walls on BOTH bounding planes, and a declared
    sheet must report ONE plane and no cells.
    """
    from rfx import Box, Simulation
    from rfx.fidelity import fidelity_report

    dx = 1e-3
    sim = Simulation(freq_max=15e9, domain=(20e-3, 20e-3, 20e-3), dx=dx,
                     boundary="pec")
    sim.add(Box((4e-3, 4e-3, 10e-3), (16e-3, 16e-3, 11e-3)), material="pec")
    sim.add_thin_conductor(Box((4e-3, 4e-3, 5e-3), (16e-3, 16e-3, 5e-3)),
                           sigma_bulk=5.8e7, thickness=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = fidelity_report(sim, print_report=False)
    plate = next(it for it in rep if it["entity"].startswith("geometry["))
    foil = next(it for it in rep if it["entity"].startswith("thin_conductor["))

    assert "one-cell PEC volume" in plate["realization"], plate["realization"]
    planes = plate["realized_wall_planes"]["z"]["planes"]
    assert planes == [10, 11], planes
    z_um = plate["realized_wall_planes"]["z"]["planes_um"]
    assert abs((z_um[-1] - z_um[0]) - 1e3) < 1e-6, z_um

    assert foil["realization"].startswith("PEC sheet on node plane z"), \
        foil["realization"]
    assert foil["realized_plane"]["index"] == 5, foil["realized_plane"]
    assert "realized_wall_planes" not in foil, (
        "a sheet owns no cell, so it has no volume wall-plane row")


def test_degenerate_sheet_and_wire_declarations_are_refused():
    """The sheet/wire analogue of the empty-PolylineWire refusal.

    ``tests/interop/test_interop_value_validation.py`` already refuses an
    empty point list because a schema-valid document in which the conductor
    is simply not there is the worst possible outcome. The same failure is
    available one layer down: a footprint that rasterizes to zero nodes, and
    a path with fewer than two nodes.
    """
    from rfx.boundaries.pec import wire_path_edge_masks
    from rfx.geometry.csg import Box
    from rfx.geometry.rasterize_grid import (
        cell_sizes_from_uniform_grid, coords_from_uniform_grid,
        sheet_spec_from_shape,
    )
    from rfx.grid import Grid

    g = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3)
    coords = coords_from_uniform_grid(g)
    sizes = cell_sizes_from_uniform_grid(g)
    z_k = float(np.asarray(coords.z)[6])
    # a footprint entirely between two node lines on both in-plane axes
    with pytest.raises(ValueError, match="ZERO nodes"):
        sheet_spec_from_shape(
            Box(corner_lo=(2.2e-3, 2.2e-3, z_k), corner_hi=(2.8e-3, 2.8e-3, z_k)),
            coords, sizes, normal_axis=2, name="vanished")

    with pytest.raises(ValueError, match="at least two nodes"):
        wire_path_edge_masks([(1, 1, 1)], (6, 6, 6))


# ---------------------------------------------------------------------------
# the sigma-fill fence (§1.8)
# ---------------------------------------------------------------------------

def test_a_sigma_fill_conductor_is_not_a_pec_body():
    """§1.8: the raw high-sigma ``rasterize`` path is OUT of the contract.

    Three oracle modules (``tests/oracle/test_rcs.py``,
    ``test_oblique_rcs_specular.py``, ``test_rcs_mie_fixture.py``), cv16,
    cv21 and the coax stamp model metal as a LOSSY VOLUME: they call
    ``rasterize(grid, [(shape, eps, 1e7)])`` and hand the resulting
    ``sigma`` array straight to the kernel. Fields decay INSIDE such a
    cell; nothing is zeroed on an edge. That is a different operator from
    PEC realization, and the two must not drift into each other — if a
    future change routed sigma fills through ``realized_pec_edge_masks``,
    every RCS gate in the tree would move without a word.

    Two independent differences, measured on the same 12 mm sphere at
    dx = 2 mm, so the fence is checkable rather than asserted:

    * the sigma path produces NO realized PEC edges at all (it owns no
      edge, only cell conductivity);
    * the two paths do not even select the same cells — ``rasterize``
      samples a shape at NODES while a PEC volume is sampled at cell
      CENTRES (§1.1), giving 910 vs 912 cells here. Equal counts would be
      a coincidence, not a contract.

    The threshold that separates the two is
    ``Simulation._PEC_SIGMA_THRESHOLD`` (1e6 S/m) applied in
    ``rfx/api/_compile.py`` to a MATERIAL on a geometry entry. A raw
    ``rasterize`` call never passes through it.
    """
    from rfx import Simulation
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.geometry.csg import Sphere, rasterize
    from rfx.grid import Grid

    domain, dx = (0.06, 0.06, 0.06), 2e-3
    sphere = Sphere(center=(0.03, 0.03, 0.03), radius=0.012)

    grid = Grid(freq_max=10e9, domain=domain, dx=dx, cpml_layers=0)
    _eps, sigma = rasterize(grid, [(sphere, 1.0, 1e7)])
    sigma_cells = np.asarray(sigma) > 1e6
    assert sigma_cells.any()

    sim = Simulation(freq_max=10e9, domain=domain, dx=dx, boundary="pec")
    sim.add(sphere, material="pec")
    sheets, wires = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g2 = sim._build_grid()
        _m, _d, _l, pec, *_rest = sim._assemble_materials(
            g2, pec_sheets=sheets, pec_wires=wires)
    volume_cells = np.asarray(pec, dtype=bool)

    assert not np.array_equal(volume_cells, sigma_cells), (
        "the sigma-fill and PEC-volume cell selections coincided; they are "
        "sampled differently (nodes vs centres) and equating them would "
        "hide the fence")
    assert int(sigma_cells.sum()) == 910 and int(volume_cells.sum()) == 912, (
        int(sigma_cells.sum()), int(volume_cells.sum()))

    # the sigma array alone realizes no PEC edge
    empty = realized_pec_edge_masks(jnp.zeros(g2.shape, dtype=bool))
    for c in range(3):
        assert not np.asarray(empty[c]).any()
    edges = realized_pec_edge_masks(pec, sheets=sheets, wires=wires)
    assert all(np.asarray(m).any() for m in edges)


# ---------------------------------------------------------------------------
# the Kottke Stage-2 fence (§1.8)
# ---------------------------------------------------------------------------

def test_kottke_owns_its_own_volume_the_ownership_rule_does_not_overwrite_it():
    """§1.8: ``subpixel_smoothing="kottke_pec"`` is fenced OUT of the contract.

    Stage 2 gives a partially filled edge a FRACTIONAL inverse permittivity;
    that fraction IS the model. The §1.2 ownership rule calls the same edge's
    cell occupied and would hard-zero it, which throws the subpixel result
    away and puts the staircase back. Measured on a PEC sphere of radius
    2.1 mm at dx = 1 mm: 16 edges per component carry a positive Kottke
    inverse permittivity and are claimed by the volume rule, among them
    ``Ex[4,3,4]`` at ``inv = 0.0329``, which the unfenced step body reads as
    exactly 0 after 60 steps.

    The fence keeps the two terms the tensor cannot supply itself: the edges
    the tensor DID freeze (the defense-in-depth re-zero) and the sheets and
    wires, which own no cell and so never enter
    ``compute_inv_eps_tensor_diag`` at all. Falsifier: drop the
    ``kottke_fenced_edge_masks`` call in ``rfx/simulation.py`` and the probed
    edge reads zero while its inverse permittivity stays 0.0329.
    """
    import rfx.simulation as _sm
    from rfx import Simulation, Sphere
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.smoothing import compute_inv_eps_tensor_diag

    dx, cell = 1e-3, (4, 3, 4)

    def _build():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(
                freq_max=30e9, domain=(10e-3, 10e-3, 10e-3), dx=dx,
                boundary=BoundarySpec(
                    x=Boundary(lo="pec", hi="pec"),
                    y=Boundary(lo="pec", hi="pec"),
                    z=Boundary(lo="pec", hi="pec")))
            sim.add(Sphere((5e-3, 5e-3, 5e-3), 2.1e-3), material="pec")
            sim.add_source((2e-3, 2e-3, 2e-3), "ex", amplitude_kind="field")
        return sim

    sim = _build()
    grid = sim._build_grid()
    inv = compute_inv_eps_tensor_diag(
        grid,
        dielectric_shapes=[(e.shape, sim._resolve_material(e.material_name).eps_r)
                           for e in sim._geometry],
        pec_shapes=[e.shape for e in sim._geometry], background_eps=1.0)

    # the premise, measured: this edge is a FRACTIONAL Kottke edge, not a
    # frozen one — so anything that zeroes it is overwriting the model.
    inv_xx = float(np.asarray(inv[0])[cell])
    assert 0.03 < inv_xx < 0.04, inv_xx

    applied = {}
    _orig = _sm.apply_pec_edges

    def _record(state, edge_masks):
        applied["masks"] = edge_masks
        return _orig(state, edge_masks)

    _sm.apply_pec_edges = _record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _build().run(n_steps=60, subpixel_smoothing="kottke_pec")
    finally:
        _sm.apply_pec_edges = _orig

    masks = applied.get("masks")
    assert masks is not None, "the kottke lane applied no edge masks at all"
    for c in range(3):
        m = np.asarray(masks[c], dtype=bool)
        overwritten = m & (np.asarray(inv[c]) > 1e-9)
        assert not overwritten.any(), (
            f"component {c}: {int(overwritten.sum())} edges with a POSITIVE "
            "Kottke inverse permittivity were hard-zeroed by the ownership "
            "rule (§1.8 fences the kottke_pec path out of the contract)")

    ex = float(np.asarray(result.state.ex)[cell])
    assert abs(ex) > 1e-4, (
        f"Ex{cell} = {ex:g} on the kottke_pec lane: a fractional Kottke edge "
        f"(inv = {inv_xx:.6g}) was frozen by the volume rule")


def test_the_kottke_fence_keeps_sheets_and_wires_the_tensor_cannot_see():
    """§1.8 fence, other half: a sheet owns no cell, so it contributes
    nothing to the inverse-permittivity tensor. Restricting the applied set
    to ``inv < 1e-9`` alone would delete every declared sheet and wire on the
    Kottke lane; the fence unions their edges back in.
    """
    from rfx.boundaries.pec import (
        SheetSpec, kottke_fenced_edge_masks, realized_pec_edge_masks)

    shape = (6, 6, 6)
    cells = np.zeros(shape, dtype=bool)
    cells[2:4, 2:4, 2:4] = True

    footprint = np.zeros(shape, dtype=bool)
    footprint[1:5, 1:5, 5] = True
    sheet = SheetSpec(normal_axis=2, plane=5,
                      footprint=jnp.asarray(footprint), name="foil")
    sheets = (sheet,)

    edges = realized_pec_edge_masks(jnp.asarray(cells), sheets=sheets)
    # a tensor that froze NOTHING: the volume term must vanish entirely and
    # the sheet term must survive in full.
    open_tensor = tuple(jnp.ones(shape, dtype=jnp.float32) for _ in range(3))
    fenced = kottke_fenced_edge_masks(edges, open_tensor, sheets=sheets)
    sheet_only = realized_pec_edge_masks(None, sheets=sheets)
    for c in range(3):
        assert np.array_equal(np.asarray(fenced[c]), np.asarray(sheet_only[c])), c
    assert np.asarray(fenced[0]).any(), "the sheet was fenced away with the volume"

    # a tensor that froze EVERYTHING gives the realized set back unchanged.
    closed = tuple(jnp.zeros(shape, dtype=jnp.float32) for _ in range(3))
    kept = kottke_fenced_edge_masks(edges, closed, sheets=sheets)
    for c in range(3):
        assert np.array_equal(np.asarray(kept[c]), np.asarray(edges[c])), c


# ---------------------------------------------------------------------------
# §1.9 — one source, every consumer: the collectors at the assemblers
#
# A sheet owns no cell, so it cannot ride out of ``_assemble_materials`` in
# ``pec_mask``: only a caller that passes ``pec_sheets=`` / ``pec_wires=``
# receives it at all. A collector-less caller therefore holds a model with
# the conductor MISSING and cannot tell that from a model that never had
# one. The tests below pin both halves of the fix: every solve entry point
# collects AND applies the sheet, and the assembler refuses a caller that
# would have dropped it.
# ---------------------------------------------------------------------------

SHEET_PLANE_Z = 5e-3
SHEET_PROBE = (5e-3, 5e-3, SHEET_PLANE_Z)      # an Ex edge INSIDE the sheet
CONTROL_PROBE = (5e-3, 5e-3, 4e-3)             # the same Ex edge one plane below


def _sheet_and_volume_sim():
    """One PEC volume and one PEC sheet, with an Ex probe on each.

    Probe 0 sits on a tangential edge of the sheet, probe 1 one node plane
    below it. Probe 1 is the falsifier: without it, "probe 0 read zero"
    is also what a simulation that never ran would report.
    """
    from rfx import Box, Simulation
    sim = Simulation(freq_max=15e9, domain=(10e-3, 10e-3, 8e-3), dx=1e-3,
                     boundary="pec")
    sim.add(Box((2e-3, 2e-3, 2e-3), (4e-3, 4e-3, 3e-3)), material="pec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(Box((3e-3, 2e-3, SHEET_PLANE_Z),
                                   (8e-3, 8e-3, SHEET_PLANE_Z)))
    sim.add_source((5e-3, 5e-3, 3e-3), "ez", amplitude_kind="field")
    sim.add_probe(SHEET_PROBE, "ex")
    sim.add_probe(CONTROL_PROBE, "ex")
    return sim


def _run_entry(sim, entry):
    """Peak |Ex| at (sheet edge, control edge) for one solve entry point."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if entry == "run":
            ts = np.asarray(sim.run(n_steps=60, skip_preflight=True,
                                    compute_s_params=False).time_series)
        elif entry == "forward":
            ts = np.asarray(sim.forward(n_steps=60,
                                        skip_preflight=True).time_series)
        elif entry == "vmap_material_sweep":
            from rfx.vmap_sweep import vmap_material_sweep
            ts = np.asarray(vmap_material_sweep(
                sim, "eps_r", [1.0, 1.0], n_steps=60).time_series)[0]
        elif entry == "optimize":
            # optimize()'s step IS sim.forward() under eps_override /
            # pec_mask_override — the override carries VOLUME cells only,
            # so a sheet survives it only if forward() re-collects.
            import jax.numpy as _jnp
            from rfx.optimize import DesignRegion, optimize
            region = DesignRegion(corner_lo=(6e-3, 6e-3, 6e-3),
                                  corner_hi=(8e-3, 8e-3, 7e-3),
                                  eps_range=(1.0, 4.0))
            peaks = []
            for probe in (0, 1):
                out = optimize(
                    _sheet_and_volume_sim(), region,
                    lambda r, i=probe: _jnp.max(_jnp.abs(r.time_series[:, i])),
                    n_iters=1, n_steps=60, verbose=False, skip_preflight=True)
                peaks.append(float(out.loss_history[0]))
            return tuple(peaks)
        else:  # pragma: no cover - guarded by the parametrization
            raise AssertionError(entry)
    return (float(np.max(np.abs(ts[:, 0]))), float(np.max(np.abs(ts[:, 1]))))


def test_the_probe_sits_on_a_realized_sheet_edge_and_the_control_does_not():
    """The premise of the entry-point battery, checked against §1.7.

    If probe 0 were not on a realized sheet edge the "reads zero" assertion
    below would pass for a model with no conductor at all.
    """
    from rfx.boundaries.pec import edge_is_pec, realized_pec_edge_masks
    sim = _sheet_and_volume_sim()
    grid = sim._build_grid()
    sheets: list = []
    wires: list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _m, _d, _l, pec, *_rest = sim._assemble_materials(
            grid, pec_sheets=sheets, pec_wires=wires)
    assert len(sheets) == 1 and sheets[0].normal_axis == 2
    assert pec is not None and int(np.asarray(pec).sum()) > 0, (
        "the volume must be in the cell mask; only the sheet is not")
    edges = realized_pec_edge_masks(pec, sheets=tuple(sheets),
                                    wires=tuple(wires),
                                    periodic=sim._periodic_flags())
    assert edge_is_pec(edges, "ex", *grid.position_to_index(SHEET_PROBE))
    assert not edge_is_pec(edges, "ex", *grid.position_to_index(CONTROL_PROBE))


@pytest.mark.parametrize(
    "entry", ["run", "forward", "vmap_material_sweep", "optimize"])
def test_every_solve_entry_point_applies_a_declared_sheet(entry):
    """§1.9: a sheet reaches the solve on every lane, or the lane refuses.

    Measured on FIELDS, not masks: the sheet's tangential Ex is exactly
    zero for every step, while the same component one node plane away is
    not. A lane that assembles without collectors sees ``pec_mask``
    without the sheet and both probes ring.
    """
    on_sheet, control = _run_entry(_sheet_and_volume_sim(), entry)
    assert on_sheet == 0.0, (
        f"{entry}: tangential E on the sheet plane is {on_sheet}, so the "
        "declared PEC sheet was dropped on this lane")
    assert control > 1e-3, (
        f"{entry}: the control edge reads {control} — the model did not "
        "run, so 'the sheet edge is zero' proves nothing")


def test_assembling_without_collectors_is_refused_and_names_the_caller():
    """The guard that stops a future collector-less caller appearing.

    Passing empty collectors and dropping them is still allowed — that is
    an explicit "I read cells only" at the call site.
    """
    sim = _sheet_and_volume_sim()
    grid = sim._build_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="pec_sheets/pec_wires") as exc:
            sim._assemble_materials(grid)
    msg = str(exc.value)
    assert "1 PEC sheet(s)" in msg, msg
    assert "test_lattice_ownership_contract.py" in msg, (
        "the refusal must name the caller, not just the assembler: " + msg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])


def test_the_non_uniform_assembler_refuses_the_same_way():
    """Both assemblers share the guard (``runners/nonuniform.py``)."""
    from rfx.runners.nonuniform import assemble_materials_nu
    sim = _sheet_and_volume_sim()
    sim._dz_profile = np.full(8, 1e-3)
    grid = sim._build_nonuniform_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="non-uniform lane"):
            assemble_materials_nu(sim, grid)


def test_conductor_mask_counts_a_sub_cell_wire():
    """§1.4 through the read-only accessor: a filament owns no cell either,
    so a footprint built from cells and sheet planes alone reports metal-free
    exactly where the wire runs."""
    from rfx import PolylineWire, Simulation
    sim = Simulation(freq_max=15e9, domain=(10e-3, 10e-3, 8e-3), dx=1e-3,
                     boundary="pec")
    sim.add(PolylineWire(((3e-3, 5e-3, 4e-3), (7e-3, 5e-3, 4e-3)),
                         radius=0.2e-3), material="pec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cond = np.asarray(sim.conductor_mask(), dtype=bool)
    # nodes 3..7 on the path, one row, one plane
    assert cond.sum() == 5, int(cond.sum())
    assert np.array_equal(np.flatnonzero(cond.any(axis=(1, 2))),
                          np.arange(3, 8))


def test_validate_subgrid_refuses_the_sheet_the_runner_refuses():
    """A read-only consumer that must agree with its runner: the SBP-SAT
    lane raises on a sheet, so the validator cannot call the same model
    supported."""
    sim = _sheet_and_volume_sim()
    sim.add_refinement(z_range=(2e-3, 6e-3), ratio=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.validate_subgrid()
    codes = [i.code for i in report.issues]
    assert "subgrid_pec_sheet_or_wire_unsupported" in codes, codes
    assert not report.supported


def test_the_material_only_coaxial_lanes_refuse_a_sheet():
    """``_build_materials`` drops ``pec_mask`` by construction, so its three
    S-parameter callers cannot realize a sheet. Refuse instead of solving
    geometry the model did not declare."""
    sim = _sheet_and_volume_sim()
    grid = sim._build_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError, match="coaxial S-parameter"):
            sim._build_materials(grid)


def test_a_lane_that_drops_volume_pec_does_not_recommend_drawing_a_volume():
    """A refusal is a claim about what works, and this one was false.

    ``_build_materials`` discards the cell mask by construction, so a declared
    PEC VOLUME is as absent from the coaxial S-parameter lanes as a sheet is —
    measured, a one-cell PEC Box through it returns eps_r == 1 and sigma == 0
    everywhere. The #931 refusal nevertheless told the user to redraw the
    sheet as a volume. The drop is the same for all three kinds, so the
    refusal is now the same for all three, and the remedy names what this
    path actually supports: a sigma fill, or a lane that realizes the
    declaration.
    """
    from rfx import Box, Simulation

    sim = Simulation(freq_max=15e9, domain=(12e-3, 12e-3, 12e-3), dx=1e-3,
                     boundary="pec")
    sim.add(Box((4e-3, 4e-3, 4e-3), (8e-3, 8e-3, 5e-3)), material="pec")
    grid = sim._build_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError) as excinfo:
            sim._build_materials(grid)
    msg = str(excinfo.value)
    assert "PEC volume" in msg, msg
    assert "sigma fill" in msg or "sigma=" in msg, msg
    assert "does NOT help" in msg, (
        "the refusal must say that redrawing as a volume is not a remedy "
        "on this path")

    # a dielectric-only model still builds — the refusal is about CONDUCTORS
    clean = Simulation(freq_max=15e9, domain=(12e-3, 12e-3, 12e-3), dx=1e-3,
                       boundary="pec")
    clean.add_material("sub", eps_r=4.0)
    clean.add(Box((0, 0, 0), (12e-3, 12e-3, 4e-3)), material="sub")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        materials, _d, _l = clean._build_materials(clean._build_grid())
    assert float(np.max(np.asarray(materials.eps_r))) == pytest.approx(4.0)


def test_the_permittivity_viewer_draws_a_sheet_it_cannot_see_in_eps():
    """A sheet writes no eps, so an eps-only cross-section of a clad board
    is bare laminate. The viewer collects the sheets, picks the plane that
    carries the metal, and overlays the realized footprint."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from rfx import Box, Simulation
    from rfx.visualize import plot_geometry_2d_slice

    def _board(with_metal):
        sim = Simulation(freq_max=15e9, domain=(10e-3, 10e-3, 8e-3), dx=1e-3,
                         boundary="pec")
        sim.add_material("sub", eps_r=4.0)
        sim.add(Box((0, 0, 0), (10e-3, 10e-3, 3e-3)), material="sub")
        if with_metal:
            sim.add_thin_conductor(Box((3e-3, 2e-3, 3e-3), (8e-3, 8e-3, 3e-3)))
        return sim

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for axis in (0, 1, 2):
            ax = plot_geometry_2d_slice(_board(True), axis=axis).axes[0]
            assert len(ax.images) == 2, (axis, len(ax.images))
            assert ax.get_legend() is not None, axis
        # control: no metal, no overlay and no legend to explain
        bare = plot_geometry_2d_slice(_board(False), axis=1).axes[0]
        assert len(bare.images) == 1
        assert bare.get_legend() is None
