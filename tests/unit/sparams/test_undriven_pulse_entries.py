"""S-parameter entries that run no preflight still refuse an undriven pulse (issue 1668).

A GaussianPulse whose bandwidth is given in Hz (tau ~ 1e-20 s) samples to zero
at every time step. ``run()`` and ``forward()`` refuse it through preflight;
these entries run no preflight, so they call the same check
(``rfx.runners._admission.refuse_undriven_drives``) before building any drive
table: through ``admit()`` (coaxial line reflection, coaxial two-port,
coax-MSL transition, waveguide S-matrix, the lumped/wire scan driver). The
MSL S-matrix runs its drives through ``run()``, whose preflight refuses.
Each test gives one entry a bandwidth in Hz and expects the refusal.
"""
from __future__ import annotations

import dataclasses
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse

HZ = GaussianPulse(f0=8e9, bandwidth=4e9)
REFUSAL = "not resolved by dt"


def _coax(depth):
    sim = Simulation(domain=(0.008, 0.008, depth), freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=HZ)
    return sim


def _refused(call, sim):
    """The entry refuses, and the check left no campaign context or other
    assembly state behind (it reads an existing grid or the dx estimate)."""
    before = getattr(sim, "_pf_campaign_ctx", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=REFUSAL):
            call()
    assert getattr(sim, "_pf_campaign_ctx", None) is before


def test_coaxial_line_reflection():
    sim = _coax(0.040)
    _refused(lambda: sim.compute_coaxial_line_reflection(
        termination="short", n_steps=40, freqs=jnp.asarray([8e9])), sim)


def test_coaxial_two_port():
    sim = _coax(0.060)
    _refused(lambda: sim.compute_coaxial_two_port(
        n_steps=40, freqs=jnp.asarray([8e9])), sim)


def test_coax_msl_transition():
    from tests._coax_msl_instrument_fixture import (
        build_instrument_junction, instrument_kwargs,
    )
    sim = build_instrument_junction()
    sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(excitation=HZ)
    _refused(lambda: sim.compute_coax_msl_transition(**instrument_kwargs(20)), sim)


def test_msl_s_matrix():
    sim = Simulation(freq_max=20e9, domain=(0.012, 0.008, 0.0032), dx=2e-4,
                     boundary="cpml", cpml_layers=8, snap="declared")
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0., 0., 0.), (0.012, 0.008, 0.)), material="pec")
    # The trace runs port to port, so no line stub is refused first.
    sim.add(Box((0.002, 0.0034, 0.0008), (0.010, 0.0046, 0.0008)), material="pec")
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    sim._msl_ports[0] = dataclasses.replace(sim._msl_ports[0], waveform=HZ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=REFUSAL):  # run()'s preflight refuses
            sim.compute_msl_s_matrix(freqs=jnp.linspace(2e9, 18e9, 4), num_periods=1.0)


def test_waveguide_s_matrix():
    sim = Simulation(freq_max=12e9, domain=(0.06, 0.02286, 0.01016), dx=1e-3,
                     boundary="cpml", cpml_layers=8)
    sim.add_waveguide_port(0.012, f0=10e9, bandwidth=5e9, direction="+x")
    sim.add_waveguide_port(0.048, f0=10e9, bandwidth=5e9, direction="-x")
    _refused(lambda: sim.compute_waveguide_s_matrix(n_steps=40), sim)


def test_lumped_wire_scan_driver():
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = Simulation(freq_max=10e9, domain=(0.024, 0.012, 0.012), dx=1e-3,
                     boundary="cpml", cpml_layers=8)
    sim.add_port(position=(0.008, 0.006, 0.006), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=5.5e9, bandwidth=4e9))
    sim.add_port(position=(0.016, 0.006, 0.006), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=5.5e9, bandwidth=4e9))
    _refused(lambda: compute_lumped_wire_s_matrix_via_scan(
        sim, np.linspace(2e9, 9e9, 3), n_steps=40), sim)
