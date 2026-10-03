"""Lumped/wire S11 on both lanes, frozen before voltage-unit changes."""
import json
from pathlib import Path

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources import GaussianPulse

LOCK_PROVENANCE = {
    'fixture': 'tests/locks/port_voltage_s11_1266.json',
    'generator': 'same model run with rfx from git archive f43641a1',
    'commit': 'f43641a1', 'date': '2026-10-03', 'run_id': 'local',
    'host': 'macOS, Python 3.11.2, JAX 0.10.2, CPU',
    'pinned_until': '2027-04-01',
}
FREQS = np.array([4e9, 6e9, 8e9])


def reference(lane, kind="wire"):
    key = lane if kind == "wire" else f"lumped_{lane}"
    record = json.loads(Path(__file__).with_name('port_voltage_s11_1266.json').read_text())[key]
    return np.asarray(record['real']) + 1j*np.asarray(record['imag'])


def assert_locked(got, expected):
    assert np.max(np.abs(got-expected)) <= 1e-4*np.max(np.abs(expected))


def measure(lane, kind):
    domain = (.0173, .0147, .0162)
    h = .001
    profiles = {}
    if lane == 'graded':
        profiles = {f'd{a}_profile': np.r_[h, np.full(int(np.ceil((length-2*h)/step)), step), h]
                    for a, length, step in zip('xy', domain, (.0005, .0004))}
        profiles['dz_profile'] = np.full(int(np.ceil(domain[2]/h)), h)
    sim = Simulation(freq_max=20e9, domain=domain, dx=h, cpml_layers=3,
                     boundary=BoundarySpec(x=Boundary('pec', 'cpml'),
                         y=Boundary('cpml', 'cpml'), z=Boundary('cpml', 'cpml')),
                     **profiles)
    sim.add_port((.0063, .0052, .0044), 'ez', impedance=50., extent=.0023 if kind == 'wire' else None,
                 waveform=GaussianPulse(f0=8e9, bandwidth=.9))
    sim.add_probe((.0091, .0073, .0062), 'ez')
    sim.add_flux_monitor(axis='x', coordinate=.0101, freqs=FREQS, name='power')
    dt = float(sim._build_nonuniform_grid().dt) if profiles else .99*h/(299792458*np.sqrt(3))
    if lane == 'graded' and kind == 'lumped':
        result = sim.forward(n_steps=round(8e-10/dt), port_s11_freqs=FREQS,
                             skip_preflight=True)
    else:
        result = sim.run(n_steps=round(8e-10/dt), compute_s_params=True,
                         s_param_freqs=FREQS, skip_preflight=True)
    return np.asarray(result.s_params)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_wire_s11_survives_source_voltage_units(lane):
    got, expected = measure(lane, "wire"), reference(lane)
    print(lane, 'S11=', got, 'peak-relative error=', np.max(abs(got-expected))/np.max(abs(expected)))
    assert_locked(got, expected)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_lumped_s11_survives_source_voltage_units(lane):
    got, expected = measure(lane, "lumped"), reference(lane, "lumped")
    print(lane, 'lumped S11=', got, 'peak-relative error=',
          np.max(abs(got-expected))/np.max(abs(expected)))
    assert_locked(got, expected)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('kind', ['lumped', 'wire'])
def test_lock_rejects_a_one_per_mille_change(lane, kind):
    expected = reference(lane, kind)
    changed = expected.copy()
    changed.flat[0] += 1e-3*np.max(np.abs(expected))
    with pytest.raises(AssertionError):
        assert_locked(changed, expected)
