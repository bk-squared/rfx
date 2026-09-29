"""A pulsed PEC box contains a folded 1 pF capacitor on one E edge.

With a homogeneous dielectric, smoothing and no smoothing must give the
same probe fields with that capacitor present. In the issue's box with a
2.4 mm dielectric cube, both smoothing modes gave identical traces with
and without the capacitor before #1263.
"""

import sys

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation


COMPONENTS = ("ex", "ey", "ez")
# Allow one float32 epsilon per timestep over the homogeneous 240-step run;
# the inverse-permittivity update orders its coefficient arithmetic differently.
ROUND_OFF = 240 * np.finfo(np.float32).eps


def _run(smoothing, capacitor, *, component="ez", homogeneous=True):
    if homogeneous:
        domain = (0.009, 0.010, 0.011)
        drive = (0.003, 0.004, 0.005)
        load = (0.006, 0.006, 0.004)
        # Keep every smoothing sample inside the dielectric, including the
        # boundary cells. A shape is required to enter Stage 1's eps branch.
        dielectric = Box((-0.002,) * 3, (0.013,) * 3)
        steps = 240
    else:
        domain = (0.015,) * 3
        drive = (0.007,) * 3
        load = (0.010, 0.007, 0.007)
        dielectric = Box((0.0023,) * 3, (0.0047,) * 3)
        steps = 600
    sim = Simulation(freq_max=10e9, domain=domain, dx=1e-3, boundary="pec")
    sim.add_material("diel", eps_r=4.0)
    sim.add(dielectric, material="diel")
    sim.add_port(position=drive, component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    if capacitor:
        sim.add_lumped_rlc(position=load, component=component, C=1e-12,
                           topology="parallel")
    for probe_component in COMPONENTS if homogeneous else ("ez",):
        sim.add_probe(load, probe_component)
    return np.asarray(sim.run(n_steps=steps, skip_preflight=True,
                              subpixel_smoothing=smoothing).time_series)


@pytest.fixture(scope="module", params=COMPONENTS)
def homogeneous_reference(request):
    component = request.param
    without = _run(False, False)
    with_capacitor = _run(False, True, component=component)
    return component, without, with_capacitor


@pytest.mark.parametrize("smoothing", [True, "kottke_pec"])
def test_homogeneous_capacitor_matches_unsmoothed(smoothing, homogeneous_reference):
    component, without, reference = homogeneous_reference
    actual = _run(smoothing, True, component=component)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    peak = float(np.max(np.abs(reference)))
    assert peak > 0.0
    # The unsmoothed comparison is an independent run through the ordinary
    # Yee material update. Check the probe sees the capacitor in that run.
    effect = float(np.max(np.abs(reference - without)) / np.max(np.abs(without)))
    assert effect > ROUND_OFF
    error = float(np.max(np.abs(actual - reference)) / peak)
    print(f"homogeneous {component} {smoothing}: normalized_error={error:.9g}, "
          f"capacitor_effect={effect:.9g}", file=sys.stderr)
    # All three E probes are compared, including the two transverse to C.
    assert error < ROUND_OFF, (component, smoothing, error)


@pytest.mark.parametrize("smoothing", [True, "kottke_pec"])
def test_capacitor_changes_probe_with_dielectric_cube(smoothing):
    without = _run(smoothing, False, homogeneous=False)
    with_capacitor = _run(smoothing, True, homogeneous=False)
    assert np.isfinite(with_capacitor).all()
    peak = float(np.max(np.abs(without)))
    assert peak > 0.0
    effect = float(np.max(np.abs(with_capacitor - without)) / peak)
    print(f"dielectric cube {smoothing}: capacitor_effect={effect:.9g}",
          file=sys.stderr)
    assert effect > ROUND_OFF, (smoothing, effect)
