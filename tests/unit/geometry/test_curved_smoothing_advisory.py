"""A smoothed curved high-contrast dielectric is announced, and only when smoothing runs (rfx #1632).

The smoother keeps only the diagonal of the inverse-permittivity tensor; on a curved
eps_r 38 rod the resonance was measured further off than without smoothing at every
mesh. Until the fix, a run that smooths such a body says so; a run that does not
smooth must not.
"""
import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.geometry import Box, Cylinder, Sphere
from rfx.geometry.smoothing import CurvedSmoothingWarning

ROD = dict(center=(0.01, 0.01, 0.01), radius=0.004, height=0.008, axis="z")


def _sim(entries, boundary="pec", graded=False):
    kw = dict(dx_profile=np.full(20, 1e-3)) if graded else {}
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.001, boundary=boundary, **kw)
    for i, (shape, eps_r) in enumerate(entries):
        sim.add_material(f"m{i}", eps_r=eps_r)
        sim.add(shape, material=f"m{i}")
    sim.add_source((0.003, 0.011, 0.01), "ez")
    return sim


def _count(sim, **run_kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=2, compute_s_params=False, **run_kw)
    return sum(issubclass(w.category, CurvedSmoothingWarning)
               or "curved high-contrast" in str(w.message) for w in caught)


@pytest.mark.parametrize("smoothing", [True, "kottke_pec"])
@pytest.mark.parametrize("shape", [Cylinder(**ROD), Sphere(center=(0.01, 0.01, 0.01), radius=0.004)])
def test_smoothed_curved_high_contrast_is_announced(shape, smoothing):
    assert _count(_sim([(shape, 38.0)]), skip_preflight=True, subpixel_smoothing=smoothing) == 1


def test_graded_smoothing_is_announced():
    assert _count(_sim([(Cylinder(**ROD), 38.0)], graded=True),
                  skip_preflight=True, subpixel_smoothing=True) == 1


def test_a_hole_in_a_high_contrast_host_is_announced():
    host = Box((0.002, 0.002, 0.002), (0.018, 0.018, 0.018))
    assert _count(_sim([(host, 38.0), (Cylinder(**ROD), 1.0)]),
                  skip_preflight=True, subpixel_smoothing=True) == 1


@pytest.mark.parametrize("entries", [
    [(Box((0.006, 0.006, 0.006), (0.014, 0.014, 0.014)), 38.0)],   # grid-aligned: not affected
    [(Cylinder(**ROD), 9.9)],                                        # just under the threshold
])
def test_not_announced_outside_the_class(entries):
    assert _count(_sim(entries), skip_preflight=True, subpixel_smoothing=True) == 0


def test_no_announcement_without_smoothing_even_through_preflight():
    """The always-on preflight seam check builds the smoothed pairs too; it must stay silent."""
    sim = _sim([(Cylinder(**ROD), 38.0)], boundary="cpml")
    assert _count(sim, subpixel_smoothing=False) == 0


def test_a_metal_post_in_a_high_permittivity_board_is_not_announced():
    """A PEC via or post is a conductor, not a dielectric: no advisory."""
    sim = _sim([(Box((0.002, 0.002, 0.002), (0.018, 0.018, 0.018)), 10.2)])
    sim.add(Cylinder(**ROD), material="pec")
    assert _count(sim, skip_preflight=True, subpixel_smoothing=True) == 0
