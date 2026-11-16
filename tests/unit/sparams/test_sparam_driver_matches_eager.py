"""Mixed lumped/wire S-parameter driver rejection contracts."""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan

# Small shared geometry (≈0.024 × 0.012 × 0.012, dx = 1 mm, 2–9 GHz).
_DOMAIN = (0.024, 0.012, 0.012)
_DX = 1e-3
_FREQ_MAX = 10e9
_CPML_LAYERS = 8
_FREQS = np.linspace(2e9, 9e9, 15)


def _waveform():
    return GaussianPulse(f0=5.5e9, bandwidth=0.8)


def _driver_sim():
    return Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, dx=_DX,
                      boundary="cpml", cpml_layers=_CPML_LAYERS)


# ---------------------------------------------------------------------------
# Mixed lumped + wire set — locked NotImplementedError (#258 behaviour change)
# ---------------------------------------------------------------------------

def _mixed_lumped_wire_sim():
    """One lumped + one wire port (heterogeneous sparam-eligible set)."""
    wf = _waveform()
    sim = _driver_sim()
    sim.add_port(position=(0.008, 0.006, 0.006), component="ez",
                 impedance=50.0, waveform=wf)
    sim.add_port(position=(0.016, 0.006, 0.005), component="ez",
                 impedance=50.0, waveform=wf, extent=0.003)
    return sim


def test_mixed_lumped_wire_driver_raises():
    """The driver rejects heterogeneous lumped+wire sets loudly — their
    off-diagonal wave-decomposition conventions differ
    (rfx/probes/sparam_driver.py mixed-set guard, PR #258). Before #258,
    run(compute_s_params=True) on a mixed set silently returned a wire-only
    diagonal matrix that DROPPED the lumped ports. This locks the guard
    (it shipped without a test — declared-only surfaces rot)."""
    sim = _mixed_lumped_wire_sim()
    with pytest.raises(NotImplementedError, match=r"mixed lumped \+ wire"):
        compute_lumped_wire_s_matrix_via_scan(sim, _FREQS, n_steps=200)


def test_mixed_lumped_wire_run_compute_s_params_raises():
    """The PUBLIC run(compute_s_params=True) path raises on a mixed set too:
    the single-wire fast-path requires ``not lumped_ports``
    (rfx/runners/uniform.py dispatch), so a mixed set routes to the driver
    and hits the same guard — the #258 behaviour change users actually see."""
    sim = _mixed_lumped_wire_sim()
    with pytest.raises(NotImplementedError, match=r"mixed lumped \+ wire"):
        sim.run(n_steps=200, compute_s_params=True)
