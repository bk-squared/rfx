"""Preflight checks that read a cell size now read the cell at their own site.

Each test here discriminates on the FINDING preflight emits, not on what a
helper returns. A helper-level assertion cannot see whether the check calls
it, which is the lesson #1200 left: its gate compared a helper against the
accessors the helper itself called, and reviving the defect at the call site
left every test green.

So each case below builds a mesh on which the old read and the new one
disagree, and asserts which advisory comes out.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.preflight._common import profile_boundary_cell

_DX = 1.0e-3


def _codes(report):
    return [getattr(i, "code", None) for i in report.issues]


# ---------------------------------------------------------------------------
# The gate: a dx-only graded mesh used to skip the whole family
# ---------------------------------------------------------------------------

def _dx_graded_tfsf_sim(angle_deg):
    """A mesh graded on x only, with TFSF. No z profile anywhere."""
    prof = np.concatenate([np.full(20, _DX), np.full(20, 0.5 * _DX),
                           np.full(20, _DX)])
    lx = float(np.sum(prof))
    sim = Simulation(freq_max=10e9, domain=(lx, 0.02, 0.02), dx=_DX,
                     boundary="cpml", cpml_layers=8, dx_profile=prof)
    sim.add_tfsf_source(direction="+x", f0=6e9, bandwidth=0.5,
                        angle_deg=angle_deg)
    return sim


def test_a_dx_only_graded_mesh_is_checked_for_tfsf_support():
    """The check is keyed on "an axis is graded", not on "a z profile was
    given".

    Oblique TFSF has no supported auxiliary on a graded mesh. The gate used to
    read ``if self._dz_profile is not None``, so a mesh graded only on x
    skipped this and every other check in the family and reported nothing --
    while the auxiliary line runs along x, the axis that was graded. The
    direction branch beside it cannot be reached from the public API, which
    rejects anything but +x/-x before preflight sees it, so oblique incidence
    is the branch that can be exercised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _dx_graded_tfsf_sim(30.0).preflight()
    assert "nonuniform_tfsf" in _codes(report)


def test_the_same_mesh_at_normal_incidence_is_not_flagged():
    """The gate opened a family of checks; it must not fire on the case the
    family says IS supported, or it is just noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _dx_graded_tfsf_sim(0.0).preflight()
    assert "nonuniform_tfsf" not in _codes(report)


# ---------------------------------------------------------------------------
# The z absorber thickness: which face, and which cells
# ---------------------------------------------------------------------------

def _z_asymmetric_profile():
    """A z profile whose two ends differ by 2x.

    ``make_nonuniform_grid`` pins x's two ends to the boundary scalar and ties
    y's to each other, so z is the only axis whose faces can differ -- which
    is why this defect lived on z.
    """
    return np.concatenate([
        np.full(8, _DX), np.full(14, 0.5 * _DX), np.full(8, 0.5 * _DX),
    ])


def test_the_absorber_thickness_is_measured_on_the_cells_it_sits_on():
    """The absorber on a face sits on copies of THAT face's boundary cell.

    The old read summed the first ``max(lo, hi)`` INTERIOR cells of the z
    profile: neither face's absorber, and the hi face measured from the lo
    end. On this profile the two faces differ by 2x, so the thinnest face is
    the hi one, and that is the thickness the advisory must report.
    """
    prof = _z_asymmetric_profile()
    lo_cell = profile_boundary_cell(_DX, prof, "lo")
    hi_cell = profile_boundary_cell(_DX, prof, "hi")
    assert lo_cell == pytest.approx(_DX)
    assert hi_cell == pytest.approx(0.5 * _DX)

    n_layers = 8
    thinnest = min(n_layers * lo_cell, n_layers * hi_cell)
    old_read = float(np.sum(prof[:n_layers]))   # what the check used to do

    # the two answers are far apart on this fixture, which is what makes the
    # advisory text below a discriminating observation rather than a restatement
    assert thinnest == pytest.approx(n_layers * hi_cell)
    assert abs(old_read - thinnest) / thinnest > 0.5

    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, float(np.sum(prof))),
                     dx=_DX, boundary="cpml", cpml_layers=n_layers,
                     dz_profile=prof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    texts = [str(i) for i in report.issues
             if getattr(i, "code", None) == "nonuniform_cpml_thin"]
    if texts:
        # the reported number is the thinnest face's, not the old sum
        assert f"{thinnest * 1e3:.1f}mm" in texts[0], texts[0]
        assert f"{old_read * 1e3:.1f}mm" not in texts[0], texts[0]


# ---------------------------------------------------------------------------
# The MSL source-fringing standoff: a length, counted on the runway's cells
# ---------------------------------------------------------------------------

_NOTCH_RUNWAY = 127e-6     # the notch instrument's board
_NOTCH_H_SUB = 254e-6


def test_the_standoff_is_a_length_counted_on_the_runway_cell():
    """The fringing decays over about five substrate thicknesses. That is a
    LENGTH; it knows nothing about the mesh. The cell count is that length
    divided by the cells the runway actually has where the port sits.

    The notch instrument's board cannot tell the two readings apart, because
    its runway cell IS its boundary cell: 127 um everywhere, giving 10 cells
    = 1.270 mm either way. Its shipped ``n_probe_offset=14`` clears that floor
    and stays the user's own setting; the check is a floor, not a solve.

    A runway refined below the boundary cell separates them. At 2x refinement
    the old reading asks for 5 cells, which on the runway's own 127 um cells
    is 0.635 mm -- half the standoff the physics needs. Reading the local cell
    gives 10 cells and the full 1.270 mm.
    """
    from rfx.preflight._common import profile_cell_at
    from rfx.preflight.msl import msl_source_near_field_standoff_cells as _nf

    # the notch board: both readings agree, so it discriminates nothing
    assert _nf(_NOTCH_H_SUB, _NOTCH_RUNWAY) == 10
    physical = _nf(_NOTCH_H_SUB, _NOTCH_RUNWAY) * _NOTCH_RUNWAY
    assert physical == pytest.approx(1.270e-3, rel=1e-9)

    # a board whose runway is refined 2x below the boundary cell
    profile = np.concatenate([
        np.full(10, 2 * _NOTCH_RUNWAY),
        np.full(40, _NOTCH_RUNWAY),
        np.full(10, 2 * _NOTCH_RUNWAY),
    ])
    feed = float(np.sum(profile[:10])) + 20 * _NOTCH_RUNWAY
    local = profile_cell_at(2 * _NOTCH_RUNWAY, profile, feed)
    assert local == pytest.approx(_NOTCH_RUNWAY)

    on_boundary_cell = _nf(_NOTCH_H_SUB, 2 * _NOTCH_RUNWAY)
    on_local_cell = _nf(_NOTCH_H_SUB, local)
    assert on_boundary_cell == 5
    assert on_local_cell == 10

    # the old reading under-provisions the physical standoff by 2x
    assert on_boundary_cell * local == pytest.approx(0.5 * physical)
    assert on_local_cell * local == pytest.approx(physical)
