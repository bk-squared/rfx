"""End-to-end calibration of the coaxial transmission-line reflection method
(broad-E5 redesign). On a real coax line with a matched CPML feed and a
≥~4-cell annulus, the canonical terminations must hit their analytic targets
across the band:

    short   -> Gamma = -1   (|S11| ~ 1, angle ~ 180 deg)
    open    -> Gamma = +1   (|S11| ~ 1)
    matched -> Gamma ~ 0     (|S11| small), and the inferred numerical Z0
               matches the analytic Z_TEM.

These are the validated-envelope targets (short/open |Gamma|=1.00-1.03,
matched 0.02-0.05 at dx=0.375mm); the tolerances reflect that envelope and are
NOT loosened. The method also flags an under-resolved annulus.

Marked slow_physics (FDTD runs); deselected by default.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import (
    coaxial_tem_characteristic_impedance, SMA_PIN_RADIUS, SMA_OUTER_RADIUS,
)

BAND = jnp.asarray([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])


def _run(termination, freq_max=40.0e9, n_steps=5000):
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=freq_max, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim.compute_coaxial_line_reflection(
        termination=termination, n_steps=n_steps, freqs=BAND)


@pytest.mark.slow_physics
def test_short_reflects_minus_one_full_band():
    res = _run("short")
    assert res.status == "passed"
    assert res.annulus_cells >= 3.5
    mag = np.abs(res.s11)
    # lossless short: |Gamma| = 1 across the band (validated 1.00-1.03)
    assert np.all(np.abs(mag - 1.0) < 0.05), mag
    # phase near +-180 deg (Gamma = -1): cos(angle) strongly negative
    assert np.all(np.cos(np.angle(res.s11)) < -0.85), np.degrees(np.angle(res.s11))
    assert np.all(res.recurrence_residual < 0.02), res.recurrence_residual


@pytest.mark.slow_physics
def test_open_reflects_unity_magnitude_full_band():
    res = _run("open")
    assert res.status == "passed"
    mag = np.abs(res.s11)
    assert np.all(np.abs(mag - 1.0) < 0.05), mag
    assert np.all(res.recurrence_residual < 0.02), res.recurrence_residual


@pytest.mark.slow_physics
def test_matched_reflects_near_zero_and_recovers_z0():
    res = _run("matched")
    assert res.status == "passed"
    mag = np.abs(res.s11)
    # matched load -> |Gamma| small (validated 0.02-0.05)
    assert np.all(mag < 0.08), mag
    # inferred numerical Z0 matches analytic Z_TEM within 15%
    z0_an = coaxial_tem_characteristic_impedance(SMA_PIN_RADIUS, SMA_OUTER_RADIUS)
    z0_num = np.real(res.z0_numerical_ohm)
    assert np.all(np.abs(z0_num - z0_an) / z0_an < 0.15), (z0_num, z0_an)


@pytest.mark.slow_physics
def test_resistive_load_reflection_magnitude():
    # known mismatch R=25 ohm on the 48.6 ohm SMA line:
    # |Gamma| = |(25 - 48.6)/(25 + 48.6)| = 0.321 (exact analytic, non-trivial).
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    res = sim.compute_coaxial_line_reflection(
        termination="matched", dut_impedance=25.0, n_steps=5000, freqs=BAND)
    assert res.status == "passed"
    z0 = coaxial_tem_characteristic_impedance(SMA_PIN_RADIUS, SMA_OUTER_RADIUS)
    g_an = abs((25.0 - z0) / (25.0 + z0))
    assert np.all(np.abs(np.abs(res.s11) - g_an) < 0.05), (np.abs(res.s11), g_an)


@pytest.mark.slow_physics
def test_under_resolved_annulus_is_flagged():
    # freq_max=20 GHz -> dx~0.75 mm -> ~1.9-cell annulus (below the >=4 recipe).
    res = _run("short", freq_max=20.0e9, n_steps=1500)
    assert res.annulus_cells < 3.5
    assert res.status == "under_resolved"
