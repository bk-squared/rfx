"""A smoothed curved high-contrast dielectric is announced (rfx #1632).

The smoother keeps only the diagonal of the inverse-permittivity tensor; on a
curved eps_r 38 rod the resonance was measured further off than without
smoothing at every mesh. Until the fix, a run that hits this case says so.
"""
import warnings

import pytest

from rfx import Simulation
from rfx.geometry import Box, Cylinder, Sphere
from rfx.geometry.smoothing import CurvedSmoothingWarning, smoothed_shape_pairs


def _sim(shape, eps_r):
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.001, boundary="pec")
    sim.add_material("body", eps_r=eps_r)
    sim.add(shape, material="body")
    return sim


def _warned(sim):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        smoothed_shape_pairs(sim, sim._build_grid())
    return [w for w in caught if issubclass(w.category, CurvedSmoothingWarning)]


@pytest.mark.parametrize("shape", [
    Cylinder(center=(0.01, 0.01, 0.01), radius=0.004, height=0.008, axis="z"),
    Sphere(center=(0.01, 0.01, 0.01), radius=0.004),
])
def test_curved_high_contrast_is_announced(shape):
    assert len(_warned(_sim(shape, 38.0))) == 1


@pytest.mark.parametrize("shape, eps_r", [
    (Box((0.006, 0.006, 0.006), (0.014, 0.014, 0.014)), 38.0),   # grid-aligned: not affected
    (Cylinder(center=(0.01, 0.01, 0.01), radius=0.004, height=0.008, axis="z"), 4.0),  # low contrast
])
def test_not_announced_outside_the_class(shape, eps_r):
    assert _warned(_sim(shape, eps_r)) == []


def test_a_run_with_smoothing_announces_it():
    sim = _sim(Cylinder(center=(0.01, 0.01, 0.01), radius=0.004, height=0.008, axis="z"), 38.0)
    sim.add_source((0.005, 0.01, 0.01), "ez")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=2, compute_s_params=False, skip_preflight=True, subpixel_smoothing=True)
    assert any(issubclass(w.category, CurvedSmoothingWarning) for w in caught)
