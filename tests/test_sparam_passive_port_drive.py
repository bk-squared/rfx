"""Regression: scan S-matrix driver vs passive (excite=False) waveform-less ports.

Issue #322 (weekly slow-tests shard red since 2026-07-06): a port registered
with ``excite=False`` and no waveform stores ``waveform=None`` BY DESIGN
(``add_port`` only defaults the waveform when ``excite=True``).  The
production-scan S-matrix driver (``compute_lumped_wire_s_matrix_via_scan``,
PR #258 wire reroute / #214 lumped reroute) drives EVERY sparam-eligible port
in turn via ``_sparam_drive_idx`` — and, before the fix, built the drive-pass
port with ``excitation=None``, crashing in ``jax.vmap(port.excitation)``
(``TypeError: Expected a callable value, got None``) the moment the passive
port's drive pass ran.  This is version-independent (reproduced at the CI's
exact jax 0.6.2), NOT the jax-drift the issue title guessed.

The fix synthesizes the same default excitation ``add_port`` applies for
``excite=True`` ports (``GaussianPulse(f0=freq_max/2, bandwidth=0.8)``) when
the driver selects a waveform-less port.  The gate here is *identity*: a
passive waveform-less port must yield the exact same S-matrix as the same
port registered with that default waveform explicitly (the driver drives by
index and ignores ``excite``), so the synthesized default is provably the
one documented — not merely "something that doesn't crash".

The original crash site (``run(compute_s_params=True)`` on the config-loader
microstrip-thru fixture) stays covered by the slow
``tests/test_config_loader.py::test_full_run_matches_direct``; these tests
keep the guard in the FAST suite for both port families.
"""

from __future__ import annotations

import numpy as np

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan

# Small vacuum geometry (mirrors test_sparam_driver_matches_eager.py).
_DOMAIN = (0.024, 0.012, 0.012)
_DX = 1e-3
_FREQ_MAX = 10e9
_CPML_LAYERS = 8
_FREQS = np.linspace(2e9, 9e9, 7)
_N_STEPS = 500


def _sim():
    return Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, dx=_DX,
                      boundary="cpml", cpml_layers=_CPML_LAYERS)


def _default_waveform():
    """The waveform add_port() defaults to for excite=True ports."""
    return GaussianPulse(f0=_FREQ_MAX / 2, bandwidth=0.8)


# ---------------------------------------------------------------------------
# Lumped 2-port — passive waveform-less port (latent #214-reroute leg)
# ---------------------------------------------------------------------------

def test_driver_lumped_passive_port_matches_explicit_default():
    pos0 = (0.008, 0.006, 0.006)
    pos1 = (0.016, 0.006, 0.006)
    wf0 = GaussianPulse(f0=5.5e9, bandwidth=4e9)

    # Port 1 passive, NO waveform (stores waveform=None) — pre-fix this
    # crashed on drive pass j=1 with jax.vmap(None).
    sim_passive = _sim()
    sim_passive.add_port(position=pos0, component="ez", impedance=50.0,
                         waveform=wf0)
    sim_passive.add_port(position=pos1, component="ez", impedance=50.0,
                         excite=False)
    S_passive, _ = compute_lumped_wire_s_matrix_via_scan(
        sim_passive, _FREQS, n_steps=_N_STEPS)

    # Same ports, port 1 with the documented default waveform explicit.
    sim_explicit = _sim()
    sim_explicit.add_port(position=pos0, component="ez", impedance=50.0,
                          waveform=wf0)
    sim_explicit.add_port(position=pos1, component="ez", impedance=50.0,
                          waveform=_default_waveform())
    S_explicit, _ = compute_lumped_wire_s_matrix_via_scan(
        sim_explicit, _FREQS, n_steps=_N_STEPS)

    assert S_passive.shape == S_explicit.shape == (2, 2, len(_FREQS))
    assert np.all(np.isfinite(S_passive))
    np.testing.assert_array_equal(
        np.asarray(S_passive), np.asarray(S_explicit),
        err_msg="passive waveform-less lumped port must be driven with the "
                "documented add_port default waveform (identity gate)")


# ---------------------------------------------------------------------------
# Wire 2-port — the exact #322 crash shape (config-loader fixture layout)
# ---------------------------------------------------------------------------

def test_driver_wire_passive_port_matches_explicit_default():
    pos0 = (0.008, 0.006, 0.0045)
    pos1 = (0.016, 0.006, 0.0045)
    extent = 0.003
    wf0 = GaussianPulse(f0=5.5e9, bandwidth=4e9)

    sim_passive = _sim()
    sim_passive.add_port(position=pos0, component="ez", impedance=50.0,
                         extent=extent, waveform=wf0)
    sim_passive.add_port(position=pos1, component="ez", impedance=50.0,
                         extent=extent, excite=False)
    S_passive, _ = compute_lumped_wire_s_matrix_via_scan(
        sim_passive, _FREQS, n_steps=_N_STEPS)

    sim_explicit = _sim()
    sim_explicit.add_port(position=pos0, component="ez", impedance=50.0,
                          extent=extent, waveform=wf0)
    sim_explicit.add_port(position=pos1, component="ez", impedance=50.0,
                          extent=extent, waveform=_default_waveform())
    S_explicit, _ = compute_lumped_wire_s_matrix_via_scan(
        sim_explicit, _FREQS, n_steps=_N_STEPS)

    assert S_passive.shape == S_explicit.shape == (2, 2, len(_FREQS))
    assert np.all(np.isfinite(S_passive))
    np.testing.assert_array_equal(
        np.asarray(S_passive), np.asarray(S_explicit),
        err_msg="passive waveform-less wire port must be driven with the "
                "documented add_port default waveform (identity gate)")


def test_run_compute_s_params_wire_passive_port():
    """PUBLIC entry point the weekly lane crashed through:
    run(compute_s_params=True) on a multi-port wire set with a passive
    waveform-less port routes to the scan driver (PR #258 reroute,
    rfx/runners/uniform.py) and must not crash."""
    pos0 = (0.008, 0.006, 0.0045)
    pos1 = (0.016, 0.006, 0.0045)
    extent = 0.003
    wf0 = GaussianPulse(f0=5.5e9, bandwidth=4e9)

    sim = _sim()
    sim.add_port(position=pos0, component="ez", impedance=50.0,
                 extent=extent, waveform=wf0)
    sim.add_port(position=pos1, component="ez", impedance=50.0,
                 extent=extent, excite=False)
    res = sim.run(n_steps=_N_STEPS, compute_s_params=True)
    s = np.asarray(res.s_params)
    assert s.shape[0] == s.shape[1] == 2
    assert np.all(np.isfinite(s))
