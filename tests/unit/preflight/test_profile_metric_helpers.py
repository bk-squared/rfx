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
    """The cell CONTAINING a coordinate, which is not the cell at the nearest
    node: halfway between two nodes those are two different cells. The
    coordinate probed is each interior cell's own midpoint, and the grid-side
    answer is that cell's entry in the padded array."""
    profile = _band_profile()
    grid = make_nonuniform_grid(
        (_A, _B), np.full(10, 1.016e-3), float(profile[0]),
        cpml_layers=8, dx_profile=profile, pec_faces=_PEC)
    edges = np.concatenate([[0.0], np.cumsum(profile)])
    for k in range(profile.size):
        midpoint = float(0.5 * (edges[k] + edges[k + 1]))
        assert profile_cell_at(grid.dx, profile, midpoint) == \
            float(grid.cells("x")[grid.pad_x_lo + k]), k


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
