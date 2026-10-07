"""Band-scoped #1512 admission, before any solver work (including bypass)."""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.preflight import line_stub as stub


def line_frequency(length=.006):
    """Independent closed form for `line()`: air line, w/h = 2, Hammerstad open end."""
    extension = .412*.001*(1 + .3)*(2 + .264)/((1 - .258)*(2 + .8))
    return 299792458 / (4*(length + extension))


def line(length=.006, direction='+x'):
    # A ground plane and snap='declared' make this a line every entry accepts,
    # so only the stub refusal stands between these tests and a solve.
    sim = Simulation(domain=(.024, .004, .004), dx=.0005, snap='declared',
                     freq_max=100e9, boundary='cpml', cpml_layers=2)
    sim.add(Box((0, 0, 0), (.024, .004, 0)), material='pec')
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
    fq = line_frequency()
    with pytest.raises(ValueError, match='open stub.*quarter wave') as error:
        stub.require_no_resonant_line_stub(sim, [fq/ratio])
    assert '6 mm behind the port' in str(error.value)
    assert 'eps_eff=1' in str(error.value)
    assert 'start the signal strip at that coordinate' in str(error.value)


def test_third_odd_resonance_refuses_when_fundamental_is_far_below_band():
    sim = line()
    fq = line_frequency()
    with pytest.raises(ValueError, match='order 3'):
        stub.require_no_resonant_line_stub(sim, [2.99*fq, 3.01*fq])


def test_fifth_odd_resonance_alone_in_the_window_refuses():
    """Only order 5 lies in [f_lo/1.5, 1.5 f_hi]: 3 f1 is below it, 7 f1 above."""
    sim = line()
    fq = line_frequency()
    band = (4.6*fq, 4.62*fq)
    assert 3*fq < band[0]/1.5 < 5*fq < 1.5*band[1] < 7*fq
    finding, = stub.line_stub_findings(sim)
    assert finding.frequency_hz == pytest.approx(fq, rel=1e-12)
    assert stub.resonant_odd_orders(finding, band) == (5, 5)
    with pytest.raises(ValueError, match='order 5'):
        stub.require_no_resonant_line_stub(sim, band)
    # A read whose window holds no odd order (between 1 and 3) is admitted.
    stub.require_no_resonant_line_stub(sim, (1.6*fq, 1.9*fq))


def reviewer_line(length):
    """The fresh review's solved fixture (eps_r 3.66, h 1 mm, w 2 mm, dx 0.25 mm)."""
    sim = Simulation(domain=(.028, .012, .006), dx=.00025, freq_max=59e9,
                     boundary='cpml', cpml_layers=16, snap='declared')
    sim.add_material('board', eps_r=3.66)
    sim.add(Box((0, 0, 0), (.028, .012, .001)), material='board')
    sim.add(Box((0, 0, 0), (.028, .012, 0)), material='pec')
    sim.add(Box((.004-length, .005, .001), (.024, .007, .001)), material='pec')
    sim.add_msl_port((.004, .006, 0), width=.002, height=.001, direction='+x',
                     eps_r_sub=3.66, name='p1')
    sim.add_msl_port((.024, .006, 0), width=.002, height=.001, direction='-x',
                     eps_r_sub=3.66, name='p2')
    return sim


@pytest.mark.parametrize('length,band,old_refuses,new_refuses', [
    # Solved notch: 29.5 GHz for 0.75 mm, 22.5 GHz for 1.25 mm (review record).
    (.00075, (4e9, 23.5e9), False, False),
    (.00075, (4e9, 39e9), False, True),
    (.00125, (4e9, 23.5e9), False, True),
    (.00125, (4e9, 39e9), True, True),
])
def test_short_stub_decision_counts_the_open_end(length, band, old_refuses, new_refuses):
    """Build only. Old rule: metal length alone; new rule: plus the open-end extension."""
    eps_eff = (3.66 + 1)/2 + (3.66 - 1)/(2*np.sqrt(1 + 12/2))
    extension = .412*.001*(eps_eff + .3)*(2 + .264)/((eps_eff - .258)*(2 + .8))

    def refuses(f1):
        return any(band[0]/1.5 <= order*f1 <= 1.5*band[1] for order in range(1, 400, 2))

    c = 299792458
    assert refuses(c/(4*length*np.sqrt(eps_eff))) is old_refuses
    assert refuses(c/(4*(length + extension)*np.sqrt(eps_eff))) is new_refuses
    sim = reviewer_line(length)
    finding, = stub.line_stub_findings(sim)
    assert finding.port_name == 'p1'
    assert finding.overhang_m == pytest.approx(length, rel=1e-9)
    assert finding.end_extension_m == pytest.approx(extension, rel=1e-12)
    if new_refuses:
        with pytest.raises(ValueError, match='open-end extension'):
            stub.require_no_resonant_line_stub(sim, band)
    else:
        stub.require_no_resonant_line_stub(sim, band)


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('method,frequency_argument', [
    ('run', 's_param_freqs'), ('forward', 'port_s11_freqs'),
    ('_forward_from_materials', 'port_s11_freqs'),
    ('compute_msl_s_matrix', 'freqs'), ('compute_mixed_s_matrix', 'freqs'),
    ('compute_coax_msl_transition', 'freqs'),
])
def test_every_entry_refuses_before_assembly(method, frequency_argument, skip):
    sim = line()
    kwargs = {frequency_argument: [12e9, 13e9]}
    if method in ('run', 'forward'):
        kwargs.pop(frequency_argument)  # Direct MSL solves do not accept lumped S11 requests.
        sim._freq_max = 13e9
        kwargs.update(skip_preflight=skip, n_steps=1)
    if method == '_forward_from_materials':
        kwargs.update(grid=None, materials=None, debye_spec=None, lorentz_spec=None, n_steps=1)
    if method == 'compute_coax_msl_transition':
        kwargs['junction_x'] = .01
        sim.add_coaxial_port((.012, .002, 0), face='bottom')
    if method == 'compute_mixed_s_matrix':
        sim.add_port((.012, .002, .001), 'ez', impedance=50., extent=(0, 0, .001))
    # Valid declarations must reach stub admission before material assembly.
    with pytest.raises(ValueError, match='start the signal strip at that coordinate'):
        getattr(sim, method)(**kwargs)


def test_far_tail_is_advisory_and_zero_tail_has_no_finding():
    sim = line(.0005)
    sim._freq_max = 20e9
    stub.require_no_resonant_line_stub(sim, [10e9, 20e9])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        stub.preflight_line_stubs(sim, warnings)
    assert len(caught) == 1
    assert caught[0].message.code == 'line_stub_behind_port'
    assert 'outside the refusal interval' in str(caught[0].message)
    assert 'odd multiples' in str(caught[0].message)
    assert 'start the signal strip at that coordinate' in str(caught[0].message)
    with warnings.catch_warnings(record=True) as caught:
        stub.preflight_line_stubs(line(0), warnings)
    assert not caught
    assert stub.line_stub_findings(line(0)) == []


def test_fallback_and_nested_calculator_scope():
    sim = line()
    fq = line_frequency()
    assert stub.read_band(sim) == (0., 100e9)

    def inner(self):
        _line_stub_scope = stub.line_stub_admission(self)
        assert stub.read_band(self) == (fq*.1, fq*.2)
        assert stub.read_band(self, [90e9]) == (fq*.1, fq*.2)
        assert stub.read_band(line()) == (0., 100e9)
        raise RuntimeError('body reached')

    def outer(self, *, freqs):
        _line_stub_scope = stub.line_stub_admission(self, freqs)
        inner(self)

    with pytest.raises(RuntimeError, match='body reached'):
        outer(sim, freqs=[fq*.1, fq*.2])
    assert stub.read_band(sim) == (0., 100e9)
    with pytest.raises(ValueError, match='open stub'):
        stub.require_no_resonant_line_stub(sim)


@pytest.mark.parametrize('method', ['compute_coax_msl_transition'])
def test_coax_declared_pin_tail_is_guarded(method):
    sim = Simulation(domain=(.012, .012, .020), dx=.001,
                     freq_max=20e9, boundary='cpml', cpml_layers=2)
    sim.add(Box((.0055, .0055, .002), (.0065, .0065, .0082)), material='pec')
    sim.add_coaxial_port((.006, .006, .008), face='bottom', pin_length=.004)
    sim.add_msl_port((.010, .006, .008), direction='-x', width=.002, height=.001, eps_r_sub=1.)
    kwargs = {'junction_x': .006} if method == 'compute_coax_msl_transition' else {}
    with pytest.raises(ValueError, match='6 mm behind the port'):
        getattr(sim, method)(freqs=[12e9, 13e9], **kwargs)


def test_public_preflight_reports_far_stub():
    report = line(.0005).preflight()
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
        # Independent of rfx: metal length plus the Hammerstad open-end extension.
        e, u, h = geo['eps_eff'], geo['w_m'] / geo['substrate'].h_sub_m, geo['substrate'].h_sub_m
        extension = .412*h*(e + .3)*(u + .264)/((e - .258)*(u + .8))
        fq = 299792458 / (4 * (geo['port_margin'] + extension) * np.sqrt(e))
        band = geo['band']
        expected = any(band.freq_lo_hz/1.5 <= order*fq <= 1.5*band.freq_hi_hz
                       for order in range(1, 400, 2))
        # The predicate on the realized grid agrees: no kept tail is refused.
        stub.require_no_resonant_line_stub(sim, (band.freq_lo_hz, band.freq_hi_hz))
        assert bool(stub.line_stub_findings(sim)) == (not expected)
        from rfx.geometry.csg import declared_bounds
        lower, upper = declared_bounds(trace)
        margin = geo['port_margin'] if expected else 0.
        assert lower[0] == margin
        assert upper[0] == geo['domain_x_m'] - margin
        if expected:
            converted.add((case.substrate_key, case.band_key, case.dx_resolution))
    # The last two entered with the open-end extension (metal length alone: 16.7 and
    # 30.8 GHz, outside 1.5x of the 2-10 and 10-20 GHz reads; effective: 14.9 and 27.1 GHz).
    assert converted == {('ro4003c', 'high', 'sub4'), ('ro4003c', 'high', 'sub6'),
                         ('ro4003c', 'low', 'sub4'), ('teflon', 'high', 'sub4'),
                         ('ro4003c', 'low', 'sub6'), ('teflon', 'high', 'sub6')}


def test_no_declared_conductor_does_not_build_a_grid(monkeypatch):
    sim = Simulation(domain=(.012, .012, .012), dx=.001, freq_max=20e9)
    sim.add_coaxial_port((.006, .006, .004), face='bottom', pin_length=.004)
    def unexpected_grid():
        raise AssertionError('empty signal geometry must not preempt calculator admission')
    monkeypatch.setattr(sim, '_build_realized_grid', unexpected_grid)
    assert stub.line_stub_findings(sim) == []


@pytest.mark.parametrize('differentiable', [False, True])
def test_actual_calculator_band_governs_inner_run_and_forward(differentiable):
    """Two steps exercise real nested entry points, not a decorated test double."""
    from tests.unit.preflight.test_line_stub import make_sim
    sim = make_sim()
    sim._snap = "declared"
    finding, = stub.line_stub_findings(sim)
    requested = np.array([1e9, 2e9])
    assert stub.resonant_odd_orders(finding, (0., sim._freq_max)) is not None
    assert stub.resonant_odd_orders(finding, tuple(requested)) is None
    kwargs = {}
    if differentiable:
        grid = sim._build_realized_grid()
        kwargs['eps_override'] = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0].eps_r
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.compute_msl_s_matrix(freqs=requested, n_steps=2, **kwargs)
    messages = [str(w.message) for w in caught if "The strip continues" in str(w.message)]
    assert messages
    assert all("Read band 1..2 GHz" in text for text in messages)
    assert all("outside the refusal interval" in text for text in messages)
    assert np.asarray(result.S).shape == (1, 1, 2)
    assert stub.read_band(sim) == (0., sim._freq_max)
