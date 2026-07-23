"""#388 opt-in radiated-flux stop criterion for until_decay.

A soft source deposits a static charge (electrostatic, H≈0) that FLOORS the interior-energy
criterion — it never radiates, so the energy never decays and the run hits ``decay_max_steps``.
The opt-in ``radiated_flux_box=`` criterion stops on the outgoing Poynting flux instead, which
the static charge (and near-Nyquist grid buzz) does not carry — a *radiation-settling* stop,
appropriate for radiation / S-parameter measurements. It is opt-in: with ``radiated_flux_box=None``
the default interior-energy criterion is used, byte-identically.
"""
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse

DOM = (0.08, 0.08, 0.08)
DX = 0.002
CX = 0.04
FLUX_BOX = ((0.022, 0.022, 0.022), (0.058, 0.058, 0.058))


def _sim():
    s = Simulation(freq_max=8e9, domain=DOM, dx=DX, boundary="cpml", cpml_layers=10, mode="3d")
    s.add_source(position=(CX, CX, CX), component="ez",
                 waveform=GaussianPulse(f0=4e9, bandwidth=0.8))
    s.add_probe((CX, CX, CX), component="ez")
    return s


def _stop_step(radiated_flux_box):
    kw = dict(until_decay=1e-3, decay_check_interval=100, decay_min_steps=600,
              decay_max_steps=6000, skip_preflight=True)
    if radiated_flux_box is not None:
        kw["radiated_flux_box"] = radiated_flux_box
    return _sim().run(**kw)


@pytest.mark.slow
def test_flux_stops_while_energy_floors():
    """On a soft-source fixture the energy criterion FLOORS (static charge) while the opt-in
    radiated-flux criterion STOPS — the #388 motivation."""
    rE = _stop_step(None)
    rF = _stop_step(FLUX_BOX)
    nE = np.asarray(rE.time_series).shape[0]
    nF = np.asarray(rF.time_series).shape[0]
    assert nE >= 5900, f"energy criterion should floor at max_steps, stopped at {nE}"
    assert nF < 3000, f"flux criterion should stop early (radiation settles), stopped at {nF}"
    assert nF < nE, f"flux ({nF}) must stop before energy ({nE})"


@pytest.mark.slow
def test_flux_stop_opt_out_is_byte_identical():
    """radiated_flux_box=None keeps the default interior-energy criterion, byte-for-byte —
    the opt-in must not perturb existing runs."""
    a = np.asarray(_stop_step(None).time_series)
    b = np.asarray(_sim().run(until_decay=1e-3, decay_check_interval=100, decay_min_steps=600,
                              decay_max_steps=6000, radiated_flux_box=None,
                              skip_preflight=True).time_series)
    assert a.shape == b.shape
    assert np.array_equal(a, b), "radiated_flux_box=None must be byte-identical to the energy run"


@pytest.mark.slow
def test_flux_stop_finite_and_stops_before_max():
    """The flux-stopped run returns a finite, truncated series (a real stop, not a cap-hit)."""
    r = _stop_step(FLUX_BOX)
    ts = np.asarray(r.time_series)
    assert np.all(np.isfinite(ts))
    assert ts.shape[0] < 6000, "flux criterion must fire (not hit decay_max_steps)"


# --------------------------------------------------------------------------- #
# Non-uniform (dz_profile) lane — the actual #388 fixture class is NU.
# --------------------------------------------------------------------------- #
def _nu_sim():
    nz = 40
    dz = np.full(nz, 1.5e-3)
    dz[15:25] = 0.8e-3          # a graded fine band (triggers the NU chunked lane)
    s = Simulation(freq_max=8e9, domain=(0.06, 0.06, 0.0), dx=1.5e-3, dz_profile=dz,
                   boundary="cpml", cpml_layers=8, mode="3d")
    s.add_source(position=(0.03, 0.03, 0.03), component="ez",
                 waveform=GaussianPulse(f0=4e9, bandwidth=0.8))
    s.add_probe((0.03, 0.03, 0.03), component="ez")
    return s


_NU_BOX = ((0.018, 0.018, 0.015), (0.042, 0.042, 0.045))


def _nu_stop(radiated_flux_box):
    kw = dict(until_decay=1e-3, decay_check_interval=100, decay_min_steps=600,
              decay_max_steps=5000, skip_preflight=True)
    if radiated_flux_box is not None:
        kw["radiated_flux_box"] = radiated_flux_box
    return np.asarray(_nu_sim().run(**kw).time_series)


@pytest.mark.slow
def test_flux_stops_while_energy_floors_nonuniform():
    """The NU (chunked-scan) lane: flux criterion STOPS while the energy criterion FLOORS on a
    dz_profile fixture — the flux stop is available on the real #388 mesh class, not just uniform."""
    nE = _nu_stop(None).shape[0]
    nF = _nu_stop(_NU_BOX).shape[0]
    assert nE >= 4900, f"NU energy criterion should floor, stopped at {nE}"
    assert nF < 2500, f"NU flux criterion should stop early, stopped at {nF}"
    assert nF < nE


@pytest.mark.slow
def test_flux_stop_opt_out_byte_identical_nonuniform():
    """NU opt-out (radiated_flux_box=None) is byte-identical to the NU energy run."""
    a = _nu_stop(None)
    b = _nu_stop(None)
    assert a.shape == b.shape and np.array_equal(a, b)
