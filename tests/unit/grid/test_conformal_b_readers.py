"""Independent checks of timestep estimates outside the stepping loop."""

import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import rfx
from rfx import Simulation
from rfx.auto_config import auto_configure
from rfx.interop.emitters.openems import _courant_dt
from rfx.preflight.mesh import _check_numerical_dispersion
from rfx.preflight.sources import _validate_cfg_unresolved_pulse


@pytest.fixture(autouse=True)
def local_worktree_and_clean_gate(monkeypatch):
    assert Path(rfx.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[3])
    monkeypatch.delenv('RFX_CONFORMAL_B_Q', raising=False)
    monkeypatch.delenv('RFX_CONFORMAL_B_FLOOR', raising=False)


def test_auto_config_dt(monkeypatch):
    kwargs = dict(freq_range=(8e9, 12e9), dx_override=.00075)
    ordinary = auto_configure([], **kwargs)
    assert ordinary.dt == .99 * .00075 / (299_792_458.0 * math.sqrt(3))
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', '.5')
    active = auto_configure([], **kwargs)
    assert active.dt == (.00075 / (299_792_458.0 * np.sqrt(3.0)) * .99) / 2
    monkeypatch.delenv('RFX_CONFORMAL_B_Q')
    assert auto_configure([], **kwargs).dt == ordinary.dt


@pytest.mark.parametrize('mode,ndim', [('3d', 3), ('2d_tmz', 2)])
def test_preflight_pulse_resolution_uses_reduced_dt(monkeypatch, mode, ndim):
    sim = Simulation(12e9, (.03,) * 3, dx=.00075, boundary='pec', mode=mode)
    ordinary = .00075 / (299_792_458.0 * math.sqrt(ndim)) * .99
    # tau=2*ordinary is unresolved at ordinary dt, resolved at ordinary/2.
    sim._ports = [SimpleNamespace(waveform=SimpleNamespace(tau=2 * ordinary))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _validate_cfg_unresolved_pulse(sim, warnings, .00075)
    assert len(caught) == 1
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', '.5')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _validate_cfg_unresolved_pulse(sim, warnings, .00075)
    assert not caught


@pytest.mark.parametrize('mode,ndim', [('3d', 3), ('2d_tmz', 2)])
def test_preflight_dispersion_uses_reduced_dt(monkeypatch, mode, ndim):
    sim = Simulation(12e9, (.03,) * 3, dx=.00075, boundary='pec', mode=mode)
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', '.5')
    seen = []
    original = math.sin
    monkeypatch.setattr(math, 'sin', lambda value: seen.append(value) or original(value))
    _check_numerical_dispersion(sim)
    dt = (.00075 / (299_792_458.0 * math.sqrt(ndim)) * .99) / 2
    assert seen == [2 * math.pi * 12e9 * dt / 2]


def test_openems_export_refuses_option_b(monkeypatch):
    from rfx.interop._errors import UnsupportedDesignFeature
    assert _courant_dt(.00075) == .00075 / (299_792_458.0 * math.sqrt(3)) * .99
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', '.5')
    with pytest.raises(UnsupportedDesignFeature, match='RFX_CONFORMAL_B_Q'):
        _courant_dt(.00075)
