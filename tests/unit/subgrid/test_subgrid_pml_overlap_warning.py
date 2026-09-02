"""Subgrid PML-overlap warning uses the grid builder's coordinate frame (#466).

The grid builder places the absorber OUTSIDE the physical domain (extra pad
cells; ``rfx/grid.py``), but the pre-#466 warning tested ``z_lo <
pml_thickness`` / ``z_hi > domain_z - pml_thickness`` — the opposite frame
(absorber inside the first/last N domain cells) — so it fired on INTERIOR
subgrids that were entirely clear of the absorber. The frame fix removes that
false-positive class only. TOUCHING an absorbing face (z_lo <= 0 or z_hi >=
domain_z) still warns: the SAT interface then sits directly against the CPML
interface, and the authoritative static validator flags the same geometry as
``subgrid_overlaps_absorber`` (pinned by
test_production_boundary_terminated_rejects_refined_face_touching_cpml, which
stays green — PR #473 review finding). Frame fix only: the subgrid lane is
EXPERIMENTAL (3-D SBP-SAT falsified, PR #90) and no validation claims are
added.
"""
import warnings

import pytest

from rfx.api import Simulation


def _sim():
    return Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.012), dx=1e-3,
                      boundary="cpml", cpml_layers=8, mode="3d")


def test_interior_subgrid_does_not_warn():
    """An INTERIOR subgrid (strictly inside the physical domain) is clear of
    the absorber, which lives in the padding outside — the pre-#466 frame
    warned here (0.002 < 8 mm "thickness" and 0.010 > 12-8 mm), which was
    the false positive this fix removes."""
    sim = _sim()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overlap warning -> test failure
        try:
            sim.add_refinement((0.002, 0.010), ratio=2)
        except Warning as w:  # pragma: no cover - explicit failure message
            if "absorber" in str(w) or "PML" in str(w):
                pytest.fail(f"false-positive absorber warning: {w}")
            raise


def test_touching_an_absorbing_face_warns():
    """z_lo == 0.0 puts the subgrid boundary directly against the absorbing
    face's CPML interface -> warns (matches the static validator's
    subgrid_overlaps_absorber verdict for the same geometry)."""
    sim = _sim()
    with pytest.warns(UserWarning, match="overlaps PML"):
        sim.add_refinement((0.0, 0.006), ratio=2)


def test_subgrid_into_padding_warns():
    """A z_range extending below z=0 leaves the physical domain and enters
    the absorbing padding on the z_lo face -> the warning must fire."""
    sim = _sim()
    with pytest.warns(UserWarning, match="overlaps PML"):
        sim.add_refinement((-0.002, 0.006), ratio=2)


def test_pec_z_faces_do_not_warn_even_outside():
    """With non-absorbing z faces there is no absorber to overlap: the frame
    fix keys the warning to the per-face boundary kind, not to cpml_layers
    alone."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.012), dx=1e-3,
                     boundary={"x": "cpml", "y": "cpml", "z": "pec"},
                     cpml_layers=8, mode="3d")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            sim.add_refinement((-0.002, 0.006), ratio=2)
        except Warning as w:  # pragma: no cover
            if "absorber" in str(w) or "PML" in str(w):
                pytest.fail(f"absorber warning on PEC z faces: {w}")
            raise
