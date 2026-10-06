"""Band-scoped #1512 admission, before any solver work (including bypass)."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.preflight import line_stub as stub


def line(length=.006, direction='+x'):
    sim = Simulation(domain=(.024, .004, .004), dx=.0005,
                     freq_max=100e9, boundary='cpml', cpml_layers=2)
    plane = .010
    # The device-side segment is deliberately much shorter than the tail.
    lo, hi = ((plane-length, plane+.0002) if direction == '+x'
              else (plane-.0002, plane+length))
    sim.add(Box((lo, .001, .001), (hi, .003, .001)), material='pec')
    sim.add_msl_port((plane, .002, 0), width=.002, height=.001,
                     direction=direction, eps_r_sub=1.)
    return sim


@pytest.mark.parametrize('direction', ['+x', '-x'])
@pytest.mark.parametrize('ratio', [1., 1.5, 1/1.5])
def test_in_band_and_inclusive_near_edges_refuse(direction, ratio):
    sim = line(direction=direction)
    fq = 299792458 / (4*.006)
    with pytest.raises(ValueError, match='open stub.*quarter wave') as error:
        stub.require_no_resonant_line_stub(sim, [fq/ratio])
    assert '6 mm behind the port' in str(error.value)
    assert 'eps_eff=1' in str(error.value)
    assert 'start the strip at the port plane' in str(error.value)


def test_third_odd_resonance_refuses_when_fundamental_is_far_below_band():
    sim = line()
    fq = 299792458 / (4*.006)
    with pytest.raises(ValueError, match='order 3'):
        stub.require_no_resonant_line_stub(sim, [2.99*fq, 3.01*fq])


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('method,frequency_argument', [
    ('run', 's_param_freqs'), ('forward', 'port_s11_freqs'),
    ('_forward_from_materials', 'port_s11_freqs'),
    ('compute_msl_s_matrix', 'freqs'), ('compute_mixed_s_matrix', 'freqs'),
    ('compute_coaxial_line_reflection', 'freqs'),
    ('compute_coaxial_two_port', 'freqs'), ('compute_coax_msl_transition', 'freqs'),
])
def test_every_entry_refuses_before_assembly(method, frequency_argument, skip):
    sim = line()
    kwargs = {frequency_argument: [12e9, 13e9]}
    if method in ('run', 'forward'):
        kwargs['skip_preflight'] = skip
    # Missing calculator/material prerequisites must not hide admission.
    with pytest.raises(ValueError, match='start the strip at the port plane'):
        getattr(sim, method)(**kwargs)


def test_far_tail_is_advisory_and_zero_tail_has_no_finding():
    sim = line(.0001)
    stub.require_no_resonant_line_stub(sim, [10e9, 20e9])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        stub.preflight_line_stubs(sim, warnings)
    assert len(caught) == 1
    assert caught[0].message.code == 'line_stub_behind_port'
    assert 'outside the refusal interval' in str(caught[0].message)
    assert 'odd multiples' in str(caught[0].message)
    assert 'start the strip at the port plane' in str(caught[0].message)
    with warnings.catch_warnings(record=True) as caught:
        stub.preflight_line_stubs(line(0), warnings)
    assert not caught
    assert stub.line_stub_findings(line(0)) == []


def test_port_band_and_fallback_and_nested_calculator_scope():
    sim = line()
    fq = 299792458 / (4*.006)
    assert stub.read_band(sim) == (0., 100e9)
    original = sim._msl_ports[0]
    sim._msl_ports[0] = SimpleNamespace(freqs=np.array([1e9, 2e9]))
    assert stub.read_band(sim) == (1e9, 2e9)
    sim._msl_ports[0] = original

    @stub.line_stub_guard('freqs')
    def inner(self):
        assert stub.read_band(self) == (fq*.1, fq*.2)
        raise RuntimeError('body reached')

    @stub.line_stub_guard('freqs')
    def outer(self, *, freqs):
        inner(self)

    with pytest.raises(RuntimeError, match='body reached'):
        outer(sim, freqs=[fq*.1, fq*.2])
    assert stub.read_band(sim) == (0., 100e9)
    with pytest.raises(ValueError, match='open stub'):
        stub.require_no_resonant_line_stub(sim)


@pytest.mark.parametrize('method', [
    'compute_coaxial_line_reflection', 'compute_coaxial_two_port',
    'compute_coax_msl_transition',
])
def test_coax_declared_pin_tail_is_guarded(method):
    sim = Simulation(domain=(.012, .012, .020), dx=.001,
                     freq_max=20e9, boundary='cpml', cpml_layers=2)
    sim.add(Box((.0055, .0055, .002), (.0065, .0065, .0082)), material='pec')
    sim.add_coaxial_port((.006, .006, .008), face='bottom', pin_length=.004)
    with pytest.raises(ValueError, match='6 mm behind the port'):
        getattr(sim, method)(freqs=[12e9, 13e9])


def test_public_preflight_reports_far_stub():
    report = line(.0001).preflight()
    assert any(f.code == 'line_stub_behind_port' for f in report)


def test_broad_sweep_converts_only_interval_members():
    from scripts.diagnostics.run_msl_broad_e5_sweep import (
        build_simulation, case_geometry_params, enumerate_cases,
    )
    converted = set()
    for case in enumerate_cases():
        geo = case_geometry_params(case)
        sim = build_simulation(case, geo)
        trace = sim._geometry[1].shape
        fq = 299792458 / (4 * geo['port_margin'] * np.sqrt(geo['eps_eff']))
        band = geo['band']
        expected = any(band.freq_lo_hz/1.5 <= order*fq <= 1.5*band.freq_hi_hz
                       for order in range(1, 11, 2))
        from rfx.geometry.csg import declared_bounds
        lower, upper = declared_bounds(trace)
        margin = geo['port_margin'] if expected else 0.
        assert lower[0] == margin
        assert upper[0] == geo['domain_x_m'] - margin
        if expected:
            converted.add((case.substrate_key, case.band_key, case.dx_resolution))
    assert converted == {('ro4003c', 'high', 'sub4'), ('ro4003c', 'high', 'sub6'),
                         ('ro4003c', 'low', 'sub4'), ('teflon', 'high', 'sub4')}


def test_no_declared_conductor_does_not_build_a_grid(monkeypatch):
    sim = Simulation(domain=(.012, .012, .012), dx=.001, freq_max=20e9)
    sim.add_coaxial_port((.006, .006, .004), face='bottom', pin_length=.004)
    def unexpected_grid():
        raise AssertionError('empty signal geometry must not preempt calculator admission')
    monkeypatch.setattr(sim, '_build_realized_grid', unexpected_grid)
    assert stub.line_stub_findings(sim) == []
