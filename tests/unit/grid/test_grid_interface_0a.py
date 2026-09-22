"""The uniform grid answers the per-cell metric interface (step 0a).

A uniform ``Grid`` has one cell size everywhere, so every array the interface
returns is constant and the dual spacing equals the primal cell exactly. That
is the whole point of putting the interface on this class too: a consumer can
hold either grid, ask ``cells``/``duals``/``node_of``, and never branch on the
grid's type -- and on the uniform lane it must get the same numbers it gets
today, to the bit, or the migration of step 0b cannot be judged by bit
identity.

The judge this file carries is judge (iii) of the design note
(``docs/design_notes/20260922_nu_grid_core_predeclaration.md``): on a uniform
``Grid``, ``cells`` is constant and ``node_of(i)`` equals the closed form
``coords_from_uniform_grid`` uses, bit for bit.

One wording to be exact about. The note writes that closed form as
``i * dx``. The form in the code is ``(i - pad) * dx``
(``_uniform_axis_nodes``, ``rfx/geometry/rasterize_grid.py:76-81``): physical
coordinates are measured from the first INTERIOR node, not from the first
padded one, so ``i * dx`` is the literal answer only on an axis with no
absorber padding. Both readings are tested below -- the pad-free grid against
the bare ``i * dx``, the padded grid against ``(i - pad_lo) * dx`` -- and the
accessor is written to the second, because that is the one that makes
``node_of`` and ``index_of`` inverses and that reproduces the coordinate
spine.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx.grid import Grid
from rfx.geometry.rasterize_grid import coords_from_uniform_grid

AXES = ("x", "y", "z")


def _padded() -> Grid:
    """An ordinary absorbing box: 8 CPML cells on all six faces."""
    return Grid(freq_max=10e9, domain=(50e-3, 20e-3, 10e-3), dx=1e-3,
                cpml_layers=8)


def _pad_free() -> Grid:
    """PEC on all six faces, so every axis has pad 0 and node 0 is x = 0."""
    return Grid(freq_max=10e9, domain=(12e-3, 9e-3, 6e-3), dx=1e-3,
                cpml_layers=8,
                pec_faces={"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"})


def _asymmetric() -> Grid:
    """PEC on the lo faces only -- pad_lo and pad_hi differ on every axis."""
    return Grid(freq_max=10e9, domain=(20e-3, 20e-3, 20e-3), dx=1e-3,
                cpml_layers=6, pec_faces={"x_lo", "y_lo", "z_lo"})


def _two_d() -> Grid:
    return Grid(freq_max=10e9, domain=(20e-3, 20e-3, 1e-3), dx=1e-3,
                cpml_layers=8, mode="2d_tmz")


GRIDS = {"padded": _padded, "pad_free": _pad_free, "asymmetric": _asymmetric}


def _extent(grid: Grid, axis: str) -> tuple[int, int]:
    ax = AXES.index(axis)
    return (grid.shape[ax], grid.axis_pads[ax])


@pytest.mark.parametrize("name", sorted(GRIDS))
@pytest.mark.parametrize("axis", AXES)
def test_cells_is_constant_and_spans_the_padded_axis(name, axis):
    grid = GRIDS[name]()
    n, _ = _extent(grid, axis)
    cells = grid.cells(axis)
    assert cells.dtype == np.float64
    assert cells.shape == (n,)
    assert np.all(cells == cells[0])
    assert float(cells[0]) == grid.dx
    assert grid.is_constant(axis) is True
    assert grid.is_traced(axis) is False


@pytest.mark.parametrize("name", sorted(GRIDS))
@pytest.mark.parametrize("axis", AXES)
def test_dual_equals_primal_bit_for_bit(name, axis):
    """``dual[k] = (d[k-1]+d[k])/2`` collapses to ``d[k]`` on a constant axis.

    This is what guarantees no uniform-mesh number can move when a consumer
    switches from the primal spelling to the dual one in 0b.
    """
    grid = GRIDS[name]()
    assert np.array_equal(grid.duals(axis), grid.cells(axis))


@pytest.mark.parametrize("name", sorted(GRIDS))
@pytest.mark.parametrize("axis", AXES)
def test_node_of_reproduces_the_coordinate_spine(name, axis):
    """Judge (iii), against the array the rasterizer actually places on."""
    grid = GRIDS[name]()
    n, pad_lo = _extent(grid, axis)
    spine = np.asarray(getattr(coords_from_uniform_grid(grid), axis))
    for i in range(n):
        assert grid.node_of(axis, i) == float(spine[i]), i
        assert grid.node_of(axis, i) == (float(i) - pad_lo) * grid.dx, i
    assert grid.node_of(axis, pad_lo) == 0.0


@pytest.mark.parametrize("axis", AXES)
def test_node_of_is_the_bare_closed_form_when_there_is_no_pad(axis):
    """The note's ``i * dx``, literally, on an axis whose pad is 0."""
    grid = _pad_free()
    n, pad_lo = _extent(grid, axis)
    assert pad_lo == 0
    for i in range(n):
        assert grid.node_of(axis, i) == i * grid.dx, i


@pytest.mark.parametrize("name", sorted(GRIDS))
@pytest.mark.parametrize("axis", AXES)
def test_index_of_inverts_node_of(name, axis):
    grid = GRIDS[name]()
    n, _ = _extent(grid, axis)
    for i in range(n):
        assert grid.index_of(axis, grid.node_of(axis, i)) == i, i


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_index_of_agrees_with_position_to_index(name):
    """The accessor must be the per-axis spelling of the existing method,
    so that a consumer moved onto it in 0b lands on the same cell."""
    grid = GRIDS[name]()
    nx, ny, nz = grid.shape
    for i in range(nx):
        for j in (0, ny // 2, ny - 1):
            for k in (0, nz - 1):
                pos = (grid.node_of("x", i), grid.node_of("y", j),
                       grid.node_of("z", k))
                expected = grid.position_to_index(pos)
                got = tuple(grid.index_of(a, p) for a, p in zip(AXES, pos))
                assert got == expected, pos


@pytest.mark.parametrize("name", sorted(GRIDS))
@pytest.mark.parametrize("axis", AXES)
def test_boundary_cell_is_the_cell_size_on_both_faces(name, axis):
    grid = GRIDS[name]()
    assert grid.boundary_cell(axis, "lo") == grid.dx
    assert grid.boundary_cell(axis, "hi") == grid.dx


def test_two_d_mode_keeps_the_single_z_cell_addressable():
    """2-D mode holds one z cell and ``position_to_index`` answers 0 there;
    the accessor must not disagree with it."""
    grid = _two_d()
    assert grid.nz == 1
    assert grid.cells("z").shape == (1,)
    assert grid.node_of("z", 0) == 0.0
    for z in (0.0, 5e-3, -5e-3):
        assert grid.index_of("z", z) == 0
        assert grid.index_of("z", z) == grid.position_to_index((0.0, 0.0, z))[2]


@pytest.mark.parametrize("spelling,expected", [
    ("x", 0), ("y", 1), ("z", 2), ("X", 0), ("Z", 2), (0, 0), (1, 1), (2, 2),
])
def test_axis_accepts_both_spellings(spelling, expected):
    grid = _padded()
    assert np.array_equal(grid.cells(spelling), grid.cells(AXES[expected]))


@pytest.mark.parametrize("bad", ["w", "xy", "", 3, -1, None, 1.0, True])
def test_a_bad_axis_is_refused(bad):
    """There is no default axis: a consumer that does not know which axis it
    is asking about is the bug this interface exists to remove."""
    grid = _padded()
    with pytest.raises(ValueError):
        grid.cells(bad)


def test_a_bad_side_is_refused():
    grid = _padded()
    with pytest.raises(ValueError):
        grid.boundary_cell("x", "left")


def test_an_out_of_grid_index_and_coordinate_are_refused():
    grid = _padded()
    with pytest.raises(IndexError):
        grid.node_of("x", grid.nx)
    with pytest.raises(ValueError):
        grid.index_of("x", 1e6)


# ---------------------------------------------------------------------------
# Mutations -- the judges must be able to fail
# ---------------------------------------------------------------------------

def test_mutation_cumsum_instead_of_the_closed_form_breaks_node_identity():
    """(b) Revive the pre-#807 node line with every helper call kept.

    Deriving a constant axis's nodes by cumulative sum instead of by the
    closed form is arithmetically the same answer and a DIFFERENT float: the
    running sum rounds at every step. That one-bit gap is what put the
    non-uniform lane's rasterization one plane away from the uniform lane's
    at node-aligned faces (#807), so the judge above has to be bitwise, and
    this test proves it is by showing the two lines are not equal.
    """
    grid = _padded()
    closed = np.asarray(coords_from_uniform_grid(grid).x)
    cells = grid.cells("x")
    edges = np.insert(np.cumsum(cells), 0, 0.0)
    cumsum_line = edges[:-1] - edges[grid.pad_x_lo]

    np.testing.assert_allclose(cumsum_line, closed, rtol=0, atol=1e-15)
    assert not np.array_equal(cumsum_line, closed), (
        "the cumulative sum landed on exactly the closed form's bits for "
        "every node of this grid, so the bitwise judge above cannot see the "
        "#807 defect on this fixture. Pick a longer axis or a dx whose "
        "running sum rounds."
    )


@pytest.mark.parametrize("name", sorted(GRIDS))
@pytest.mark.parametrize("axis", AXES)
def test_index_of_refuses_a_coordinate_outside_the_domain(name, axis):
    """The uniform half of the shared refusal contract.

    The non-uniform accessor was made to match this in the same change; its
    counterpart lives in tests/unit/nonuniform/test_grid_interface_0a.py.
    """
    grid = GRIDS[name]()
    n, pad_lo = _extent(grid, axis)
    for outside in (-1e6, 1e6):
        with pytest.raises(ValueError):
            grid.index_of(axis, outside)
    assert grid.index_of(axis, grid.node_of(axis, 0)) == 0
    assert grid.index_of(axis, grid.node_of(axis, n - 1)) == n - 1
