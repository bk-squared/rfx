"""Subgrid PML-overlap warning uses the grid builder's coordinate frame (#466).

The grid builder places the absorber OUTSIDE the physical domain (extra pad
cells; ``rfx/grid.py``), but the pre-#466 warning tested ``z_lo <
pml_thickness`` / ``z_hi > domain_z - pml_thickness`` — the opposite frame
(absorber inside the first/last N domain cells) — so it fired on subgrids
that were entirely clear of the absorber. Frame fix only: the subgrid lane is
EXPERIMENTAL (3-D SBP-SAT falsified, PR #90) and no new adjacency/validation
semantics are added here.
"""
import warnings

import pytest

from rfx.api import Simulation


def _sim():
    return Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.012), dx=1e-3,
                      boundary="cpml", cpml_layers=8, mode="3d")


def test_full_domain_subgrid_does_not_warn():
    """A subgrid spanning the FULL physical domain is clear of the absorber
    (which lives in the padding outside) — the pre-#466 frame warned here
    (0 < 8 mm and 12 > 12-8 mm), which was the false positive."""
    sim = _sim()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overlap warning -> test failure
        try:
            sim.add_refinement((0.0, 0.012), ratio=2)
        except Warning as w:  # pragma: no cover - explicit failure message
            if "absorber" in str(w) or "PML" in str(w):
                pytest.fail(f"false-positive absorber warning: {w}")
            raise


def test_subgrid_into_padding_warns():
    """A z_range extending below z=0 leaves the physical domain and enters
    the absorbing padding on the z_lo face -> the warning must fire."""
    sim = _sim()
    with pytest.warns(UserWarning, match="absorber"):
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
