"""Core-solver numerical-dispersion ORACLE (#403 blind spot).

rfx had a preflight guard (`_check_numerical_dispersion`, Taflove Ch.4) that PREDICTS dispersion
from cells-per-wavelength, but no test that MEASURES the solver's actual phase velocity and gates
it against the closed-form Yee dispersion relation. This is that binding oracle.

Analytic Yee (1D plane wave along x):
    (1/(c·dt))² sin²(ω·dt/2) = (1/dx)² sin²(k_x·dx/2)
The discrete wave is SLOWER than c (v_p < c), by a deficit ~1/N² at N cells/wavelength.

Measurement: a +x TFSF plane wave in vacuum; the f0 DFT-phasor phase slope over a line of x-probes
is -k_x (the discrete wavenumber). Measured (CPU): v_p/c err vs Yee 0.01%/0.04% @ N=15/30; the Yee
deficit 0.50%/0.12% converges ~1/N². The N=15 case is DISCRIMINATING — a solver with no/wrong
dispersion would read v_p/c ≈ 1.0, differing from Yee (0.995) by the 0.5% deficit.
Harness: docs/research_notes/experiments/numerical_dispersion_oracle.py
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.grid import Grid

C0 = 299792458.0
F0 = 5e9


def _yee_kx(f0, dt, dx):
    w = 2 * np.pi * f0
    return (2.0 / dx) * np.arcsin((dx / (C0 * dt)) * np.sin(w * dt / 2.0))


def _measure_vp_over_c(dx):
    """Measure v_p/c from the incident plane-wave phase slope; return (measured, yee)."""
    dom = (0.60, 0.12, 0.006)
    grid = Grid(freq_max=10e9, domain=dom, dx=dx, cpml_layers=10)
    dt = grid.dt
    xprobes = np.arange(0.10, 0.40, dx * 4)
    pidx = np.array([grid.position_to_index((float(x), 0.06, 0.003))[0] for x in xprobes], float)
    sim = Simulation(freq_max=10e9, domain=dom, dx=dx, boundary="cpml", cpml_layers=10, mode="3d")
    sim.add_tfsf_source(f0=F0, bandwidth=0.3, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    for x in xprobes:
        sim.add_probe((float(x), 0.06, 0.003), component="ez")
    ns = min(int(0.9 * (2 * 0.40 / C0) / dt), 4000)
    t = jnp.arange(ns) * dt
    kern = jnp.exp(-1j * 2 * jnp.pi * F0 * t) * dt
    ts = sim.forward(eps_override=jnp.ones(grid.shape, jnp.float32), n_steps=ns,
                     checkpoint=False, skip_preflight=True).time_series
    phas = np.asarray(jnp.sum(ts * kern[:, None], axis=0))
    slope = np.polyfit(pidx * dx, np.unwrap(np.angle(phas)), 1)[0]
    kx_meas = abs(slope)
    w = 2 * np.pi * F0
    return w / kx_meas / C0, w / _yee_kx(F0, dt, dx) / C0


@pytest.mark.slow
def test_measured_phase_velocity_matches_yee_dispersion():
    """At N=15 cells/λ the measured v_p matches the analytic Yee value AND is clearly below the
    continuum c — a solver ignoring numerical dispersion would read v_p/c≈1.0 and FAIL this."""
    vp_meas, vp_yee = _measure_vp_over_c(0.004)   # N=15
    # (1) the Yee deficit is real at this resolution (not a rounding-noise-sized effect)
    assert vp_yee < 1.0 - 3e-3, f"Yee deficit too small to discriminate: vp_yee/c={vp_yee:.5f}"
    # (2) measured matches Yee tightly (the leapfrog update obeys its own discrete dispersion)
    assert abs(vp_meas - vp_yee) / vp_yee < 3e-3, f"vp_meas/c={vp_meas:.5f} vs Yee {vp_yee:.5f}"
    # (3) measured tracks the SLOW Yee wave, not the continuum — the discriminating check
    assert vp_meas < 1.0 - 3e-3, f"vp_meas/c={vp_meas:.5f} looks dispersion-free (should be ~{vp_yee:.5f})"


@pytest.mark.slow
def test_numerical_dispersion_converges_with_resolution():
    """The Yee deficit (1 - v_p/c) shrinks as dx halves (~1/N²) and the measurement tracks it."""
    vp15, yee15 = _measure_vp_over_c(0.004)   # N=15
    vp30, yee30 = _measure_vp_over_c(0.002)   # N=30
    assert (1 - yee30) < (1 - yee15), "analytic Yee deficit must shrink with finer dx"
    # measured deficit also shrinks (allow measurement noise, but the trend must hold)
    assert (1 - vp30) < (1 - vp15) + 1e-3, f"measured deficit did not shrink: {1-vp15:.4f}->{1-vp30:.4f}"
    assert abs(vp30 - yee30) / yee30 < 3e-3, f"N=30 vp_meas/c={vp30:.5f} vs Yee {yee30:.5f}"
