"""Preflight guard: TFSF plane-wave + lumped RLC is numerically unstable (footgun surfacing).

A bare add_lumped_rlc(...) driven by a TFSF plane wave has no defined series circuit; the ADE
current-tracking is ill-posed and the fields diverge (measured: ~1e35 by ~250 steps). The
varactor/tunable-load gradient IS validated for the PORT-fed lane (test_lumped_rlc_ad.py); this
guard warns when the element is instead illuminated by a bare plane wave, so the silent NaN is
surfaced. Advisory (warning, not error): a proper PEC-gap RIS unit cell may be stable.
"""
import numpy as np

from rfx import GaussianPulse, Simulation

_CODE = "tfsf_lumped_rlc_unstable"


def _codes(sim):
    return {getattr(i, "code", None) for i in sim.preflight()}


def test_tfsf_plus_lumped_rlc_warns():
    sim = Simulation(freq_max=16e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 20,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add_tfsf_source(f0=8e9, bandwidth=0.6, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    sim.add_lumped_rlc(position=(0.010, 0.010, 0.010), component="ez",
                       R=50.0, C=0.20e-12, topology="series")
    assert _CODE in _codes(sim), "TFSF + lumped RLC should warn about the unstable pairing"


def test_tfsf_alone_no_warning():
    """No false positive: a TFSF plane wave with no lumped element must NOT warn."""
    sim = Simulation(freq_max=16e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 20,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add_tfsf_source(f0=8e9, bandwidth=0.6, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    assert _CODE not in _codes(sim)


def test_lumped_rlc_with_port_no_warning():
    """No false positive: the validated PORT-fed varactor lane must NOT warn."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
                     boundary="cpml", cpml_layers=6)
    sim.add_port(position=(0.0093, 0.0093, 0.0093), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.9))
    sim.add_lumped_rlc(position=(0.0093, 0.0093, 0.0093), component="ez",
                       R=50.0, C=0.20e-12, topology="series")
    assert _CODE not in _codes(sim)
