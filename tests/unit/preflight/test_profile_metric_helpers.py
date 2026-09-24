"""Preflight's profile-side metric helpers answer what the grid answers.

Preflight runs before a grid is necessarily built, so several checks would
otherwise pay for a full ``_build_nonuniform_grid()`` to read one cell.
``rfx/preflight/_common.py`` spells three of the grid interface's questions on
the declared profile instead: the cell at a face, the cell containing a
coordinate, and whether an axis is constant.

That is only safe if the two agree. They do by construction --
``make_nonuniform_grid`` pins a profile's first and last cells as the boundary
cells and pads each face with copies of them -- and this module pins it
against a real ``NonUniformGrid`` rather than restating the construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx.auto_config import smooth_grading
from rfx.nonuniform import make_nonuniform_grid
from rfx.preflight._common import (
    profile_boundary_cell,
    profile_cell_at,
    profile_node_at,
    profile_span_is_uniform,
)

_A, _B = 0.02286, 0.01016
_DX_COARSE, _DX_FINE = 1.5e-3, 0.75e-3
_PEC = {"y_lo", "y_hi", "z_lo", "z_hi"}


def _band_profile():
    """coarse | fine | coarse, the shape the WR-90 sparam fixture uses."""
    raw = np.concatenate([
        np.full(20, _DX_COARSE), np.full(54, _DX_FINE), np.full(20, _DX_COARSE),
    ])
    return smooth_grading(raw, max_ratio=1.3)


def _asymmetric_profile():
    """Ends that DIFFER, which is what separates the two faces."""
    return np.concatenate([
        np.full(8, _DX_COARSE), np.full(10, _DX_FINE), np.full(8, 0.5 * _DX_COARSE),
    ])


def test_the_face_cell_matches_the_grids_boundary_cell_on_x():
    """x pins both its ends to the boundary scalar, so the two faces agree
    there; the helper must still answer from the profile, not the scalar."""
    profile = _band_profile()
    grid = make_nonuniform_grid(
        (_A, _B), np.full(10, 1.016e-3), float(profile[0]),
        cpml_layers=8, dx_profile=profile, pec_faces=_PEC)
    for side in ("lo", "hi"):
        assert profile_boundary_cell(grid.dx, profile, side) == \
            grid.boundary_cell("x", side)


def test_the_face_cell_matches_the_grids_boundary_cell_on_z():
    """z is where the two faces can differ. ``make_nonuniform_grid`` pins
    ``dx_profile[0] == dx_profile[-1] == dx`` for x and ties y's two ends to
    each other, but puts no such constraint on the z profile -- so an
    asymmetric profile is a z one, and z is exactly the axis whose absorber
    thickness this PR stopped measuring from the wrong end."""
    profile = _asymmetric_profile()
    grid = make_nonuniform_grid(
        (_A, _B), profile, _DX_COARSE, cpml_layers=8, pec_faces={"y_lo", "y_hi"})
    assert grid.boundary_cell("z", "lo") != grid.boundary_cell("z", "hi")
    for side in ("lo", "hi"):
        assert profile_boundary_cell(_DX_COARSE, profile, side) == \
            grid.boundary_cell("z", side)


def test_the_two_faces_differ_where_the_profile_says_so():
    """Otherwise reading the leading entry for both faces would be harmless
    and the check that replaced it would be untestable."""
    profile = _asymmetric_profile()
    lo = profile_boundary_cell(_DX_COARSE, profile, "lo")
    hi = profile_boundary_cell(_DX_COARSE, profile, "hi")
    assert lo == pytest.approx(_DX_COARSE)
    assert hi == pytest.approx(0.5 * _DX_COARSE)
    assert lo != hi


def test_the_cell_at_a_coordinate_matches_the_grids_cells():
    """The cell beside the node the grid puts a coordinate on:
    ``cells[index_of(coord)]``, and the one below it for ``toward="lo"``.

    Probed at 0.3 and 0.7 of every interior cell, so the nearest node is
    sometimes the cell's lower node and sometimes its upper one. The
    grid-side answer comes from the grid's own ``index_of``."""
    profile = _band_profile()
    grid = make_nonuniform_grid(
        (_A, _B), np.full(10, 1.016e-3), float(profile[0]),
        cpml_layers=8, dx_profile=profile, pec_faces=_PEC)
    edges = np.concatenate([[0.0], np.cumsum(profile)])
    cells = grid.cells("x")
    for k in range(1, profile.size - 1):
        for frac in (0.3, 0.7):
            x = float(edges[k] + frac * profile[k])
            i = grid.index_of("x", x)
            assert profile_node_at(grid.dx, profile, x) == \
                pytest.approx(grid.node_of("x", i), abs=1e-15), (k, frac)
            assert profile_cell_at(grid.dx, profile, x) == \
                float(cells[i]), (k, frac)
            assert profile_cell_at(grid.dx, profile, x, toward="lo") == \
                float(cells[i - 1]), (k, frac)


def test_a_span_inside_one_zone_is_uniform_and_one_crossing_a_ramp_is_not():
    """The predicate a cell COUNT rests on: across a ramp the count depends
    on where you start, so the caller refuses instead of answering."""
    profile = _band_profile()
    edges = np.concatenate([[0.0], np.cumsum(profile)])
    ramp = next(i for i in range(1, profile.size) if profile[i] != profile[i - 1])
    x_ramp = float(edges[ramp])

    assert profile_span_is_uniform(_DX_COARSE, profile, 0.002, 0.004) is True
    assert profile_span_is_uniform(
        _DX_COARSE, profile, x_ramp - 0.002, 0.004) is False
    # no profile at all is one zone by definition
    assert profile_span_is_uniform(_DX_COARSE, None, 0.0, 1.0) is True


# ---------------------------------------------------------------------------
# Spans that end on a node, and spans that leave the declared profile
# ---------------------------------------------------------------------------

def _two_zone():
    return np.array([_DX_COARSE] * 10 + [_DX_FINE] * 10 + [_DX_COARSE] * 10,
                    float)


def test_a_span_ending_on_a_node_stops_at_the_cell_below_it():
    """A span that closes exactly on a node does not reach the cell above it.

    Ten fine cells end at that node; the span covering them lies wholly
    inside one zone. Counting the coarse cell on the far side made it read as
    mixed, which refuses a port whose probes never leave the fine zone.
    """
    prof = _two_zone()
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    node = float(edges[20])
    span = float(np.sum(prof[10:20]))
    assert profile_span_is_uniform(_DX_COARSE, prof, node, -span) is True
    # one cell further back is still inside the zone; one cell further
    # forward crosses into the coarse zone and is not
    assert profile_span_is_uniform(
        _DX_COARSE, prof, node, -(span - _DX_FINE)) is True
    assert profile_span_is_uniform(
        _DX_COARSE, prof, node, _DX_COARSE) is True
    assert profile_span_is_uniform(
        _DX_COARSE, prof, node - _DX_FINE, 2 * _DX_COARSE) is False


def test_a_span_that_left_the_profile_is_not_called_uniform():
    """Beyond the declared profile the cells are the absorber pad's, which
    this function was not given. A guard that passes only because the span
    ran off the board is not being told anything, so it gets False."""
    prof = _two_zone()
    total = float(np.sum(prof))
    assert profile_span_is_uniform(_DX_COARSE, prof, total + 1e-3, 1e-3) is False
    assert profile_span_is_uniform(_DX_COARSE, prof, -2e-3, 1e-3) is False
    # a span still touching the profile is judged on the cells it does cover
    assert profile_span_is_uniform(_DX_COARSE, prof, total - _DX_COARSE,
                                   2 * _DX_COARSE) is True


def test_the_cell_at_a_node_depends_on_which_way_you_look():
    """A node has a cell on each side; ``toward`` picks one. Every coordinate
    whose nearest node is this one gets the same two answers, including a
    coordinate a few 1e-18 m off it, which is where a literal lands."""
    prof = _two_zone()
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    node = float(edges[20])
    for x in (node, node - 1e-18, node + 1e-18,
              node - 0.3 * _DX_FINE, node + 0.3 * _DX_COARSE):
        assert profile_cell_at(_DX_COARSE, prof, x, toward="lo") == \
            pytest.approx(_DX_FINE)
        assert profile_cell_at(_DX_COARSE, prof, x, toward="hi") == \
            pytest.approx(_DX_COARSE)
    with pytest.raises(ValueError):
        profile_cell_at(_DX_COARSE, prof, node, toward="up")


def test_a_span_that_touches_a_node_by_float_noise_does_not_cross_it():
    """A span closing 1e-17 m past a zone's last node, which is how far the
    grid's coordinate spine and a mesher's cumulative sum disagree, stays in
    the zone; one reaching a real fraction of a cell past it does not."""
    prof = _two_zone()
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    start, end = float(edges[10]), float(edges[20])
    assert profile_span_is_uniform(_DX_COARSE, prof, start,
                                   end - start + 1e-17) is True
    assert profile_span_is_uniform(_DX_COARSE, prof, end,
                                   -(end - start) - 1e-17) is True
    assert profile_span_is_uniform(_DX_COARSE, prof, start,
                                   end - start + 0.01 * _DX_COARSE) is False


def test_cells_equal_to_the_fourteenth_digit_are_one_size():
    """``np.diff`` of evenly spaced node coordinates gives cells that differ
    in the last digits; they are one zone. A real step is not."""
    nodes = np.linspace(0.0, 90 * _DX_FINE, 91)
    prof = np.diff(nodes)
    assert len(set(prof.tolist())) > 1
    assert profile_span_is_uniform(_DX_COARSE, prof, 0.0, 80 * _DX_FINE)
    stepped = np.concatenate([prof, [_DX_FINE * (1 + 1e-6)]])
    assert not profile_span_is_uniform(_DX_COARSE, stepped, 0.0,
                                       float(np.sum(stepped)))
