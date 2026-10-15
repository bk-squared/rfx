"""Optional inspection must not preempt a lane's declaration refusals."""
import pytest

from rfx import Box, Simulation
from rfx.preflight.line_stub import require_no_resonant_line_stub


def uninspectable():
    from tests.unit.ports.test_msl_reflector_scan_conductors import _base, _NoBBox, Y_C, W_TRACE, H_SUB
    sim = _base()
    entry = type(sim._geometry[0])
    sim._geometry.append(entry(shape=_NoBBox(), material_name='pec'))
    sim.add_msl_port((.0025, Y_C, 0), width=W_TRACE, height=H_SUB,
                     direction='+x', eps_r_sub=2.2, name='p1')
    return sim


def test_uninspectable_shape_reports_advisory_and_preserves_other_checks():
    sim = uninspectable()
    report = sim.preflight(strict=False, check_ntff=False)
    issue, = [i for i in report if i.code == 'line_stub_inspection_unavailable']
    assert issue.severity == 'warning'
    assert 'could not inspect' in str(issue) and '_NoBBox' in str(issue)
    assert any('could NOT evaluate' in str(i) for i in report)
    require_no_resonant_line_stub(sim)


@pytest.mark.parametrize('method', ['run', 'forward'])
def test_uninspectable_shape_defers_to_execution_checks(method, monkeypatch):
    sim = uninspectable()
    def owner(*args, **kwargs):
        raise ValueError('owning lane check reached')
    monkeypatch.setattr(sim, '_auto_preflight', owner)
    with pytest.raises(ValueError, match='owning lane check reached'):
        getattr(sim, method)(skip_preflight=True)


@pytest.mark.parametrize('method,match', [
    ('run', 'not wired'), ('forward', 'not wired'),
    ('_forward_from_materials', 'Conformal PEC'),
    ('compute_msl_s_matrix', "solver='adi'"),
    ('compute_mixed_s_matrix', 'at least one sparam-eligible'),
    ('compute_coaxial_line_reflection', 'constructs the complete line'),
    ('compute_coaxial_two_port', 'constructs the complete line'),
    ('compute_coax_msl_transition', 'exactly one add_coaxial_port'),
])
def test_declaration_refusal_precedes_stub_mesh_planning(method, match, monkeypatch):
    sim = Simulation(domain=(.02, .01, .01), dx=.001, freq_max=10e9,
                     boundary='cpml', cpml_layers=2)
    sim.add(Box((.001, .003, .002), (.018, .005, .002)), material='pec')
    sim.add_msl_port((.005, .004, .001), width=.002, height=.001, eps_r_sub=2.)
    kwargs = {}
    if method in ('run', 'forward') or method.startswith('compute_coaxial'):
        sim.add_coaxial_port((.005, .005, .004), face='bottom')
    if method == 'compute_msl_s_matrix':
        sim._solver = 'adi'
    if method == 'compute_coax_msl_transition':
        kwargs['junction_x'] = .01
    if method == '_forward_from_materials':
        kwargs.update(grid=None, materials=None, debye_spec=None, lorentz_spec=None,
                      n_steps=1, conformal_pec=True)
    def unexpected(*args, **kwargs):
        pytest.fail('declaration refusal must precede mesh planning')
    monkeypatch.setattr(sim, '_auto_configure_mesh', unexpected)
    monkeypatch.setattr(sim, '_build_realized_grid', unexpected)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        getattr(sim, method)(**kwargs)
