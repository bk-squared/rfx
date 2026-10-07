"""Public execution boundaries retain host records under nesting and transforms."""
from concurrent.futures import ThreadPoolExecutor
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Diagnostic
from rfx._diagnostic_context import diagnostic_scope, record_diagnostics
from tests._msl_diagnostic_cases import structure, U


@pytest.mark.parametrize('graded', [False, True])
def test_actual_forward_jit_grad_vmap(graded):
    sim = structure(graded)
    sim.add_probe((16 * U, 6 * U, U), 'ez')
    def forward(scale):
        # The traced override is a field input; geometry and metadata stay concrete.
        shape = sim._build_realized_grid().shape
        result = sim.forward(eps_override=jnp.ones(shape) * scale, n_steps=2,
                             checkpoint=False)
        # Grid is host metadata on the existing uniform ForwardResult. The
        # differentiable wrapper returns the measured leaf and new metadata.
        return result.time_series, result.diagnostics
    result = jax.jit(forward)(jnp.array(3.66))
    assert any(d.code == "msl.normal_resolution" for d in result[1])
    derivative = jax.grad(lambda x: jnp.sum(forward(x)[0]))(3.66)
    assert np.isfinite(float(derivative))
    mapped = jax.jit(jax.vmap(forward))(jnp.array([3.66, 3.66]))
    assert mapped[1] == result[1]
    assert mapped[0].shape[0] == 2
    from rfx._diagnostic_context import _ACTIVE
    assert _ACTIVE.get() is None


def test_actual_simulation_thread_isolation():
    simulations = [structure(name=name) for name in ('thread_first', 'thread_second')]
    def solve(sim):
        return sim.run(n_steps=2, compute_s_params=False).diagnostics
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(solve, simulations))
    for records, name in zip(results, ('thread_first', 'thread_second')):
        family = [d for d in records if d.code.startswith('msl.')]
        assert {d.subject for d in family} == {name}
        assert len(family) == 5


def test_msl_calculator_unions_every_drive_without_duplicates():
    # Manufactured spectra isolate the public calculator's nested-run transport.
    # Real field evolution is covered by the forward and run tests above.
    from tests.unit.autodiff.test_msl_forward_identity import _build_planted_sim, FREQS
    sim, _ = _build_planted_sim(1.)
    backend = sim.run
    seen = []
    def drive(self, **kwargs):
        with diagnostic_scope():
            d = Diagnostic('msl.drive_observation', 'advisory', str(len(seen)),
                           'drive observation', {'drive': len(seen)})
            seen.append(d)
            record_diagnostics((d, d))
            return backend(**kwargs)
    sim.run = MethodType(drive, sim)
    result = sim.compute_msl_s_matrix(freqs=FREQS, n_steps=1, num_periods=1,
                                     enforce_passivity=False)
    assert len(seen) == 2
    assert tuple(d for d in result.diagnostics if d.code == 'msl.drive_observation') == tuple(seen)
    clean = structure().run(n_steps=2, compute_s_params=False, skip_preflight=True)
    assert clean.diagnostics == ()


def test_run_warning_location_is_the_public_caller():
    import inspect
    sim = structure()
    with pytest.warns(UserWarning) as caught:
        line = inspect.currentframe().f_lineno + 1
        sim.run(n_steps=2, compute_s_params=False)
    advisory = [w for w in caught if 'preflight found' in str(w.message)]
    assert len(advisory) == 1
    assert advisory[0].filename == __file__
    assert advisory[0].lineno == line


@pytest.mark.parametrize('entry', ['preflight', 'preflight_sparameters'])
def test_standalone_reports_are_isolated_in_threads(entry):
    simulations = [structure(name=name) for name in ('report_first', 'report_second')]
    def inspect(sim):
        kwargs = {'calculator': 'msl', 'include_general': True} if entry == 'preflight_sparameters' else {}
        return getattr(sim, entry)(**kwargs).diagnostics
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(inspect, simulations))
    for records, name in zip(results, ('report_first', 'report_second')):
        family = [d for d in records if d.code.startswith('msl.')]
        assert {d.subject for d in family} == {name}
        assert len(family) == 5


def test_scoped_coax_gate_accepts_the_existing_plain_issue_list():
    from rfx.preflight._common import PreflightIssue
    from rfx.preflight._impl import run_preflight_gate
    d = Diagnostic('msl.example_refusal', 'refusal', 'port', 'refused', {})
    with pytest.raises(ValueError, match='blocking error') as caught:
        run_preflight_gate([PreflightIssue(d)], context='compute_coax_msl_transition')
    assert caught.value.diagnostics == (d,)


@pytest.mark.parametrize('lane', ['adi', 'subgridded', 'disjoint'])
def test_auxiliary_lanes_attach_their_refusal_and_family_records(lane):
    sim = structure()
    if lane == 'adi':
        sim._solver = 'adi'
    else:
        sim.add_refinement(z_range=(U, 3 * U), ratio=2,
                           topology='stage2_disjoint_3d' if lane == 'disjoint' else 'overlap_z_slab')
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        sim.run(n_steps=2, compute_s_params=False)
    records = caught.value.diagnostics
    assert any(d.severity == 'refusal' for d in records)
    assert {d.code for d in records if d.code.startswith('msl.')} == {
        'msl.lateral_clearance', 'msl.normal_resolution',
        'msl.source_absorber_clearance', 'msl.source_near_field',
    }


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('case,code', [
    ('assembly_unavailable', 'msl.conductor_assembly_unavailable'),
    ('attachment_unavailable', 'msl.conductor_attachment'),
    ('placement_unavailable', 'msl.probe_placement_failed'),
])
def test_family_error_severity_is_a_run_refusal(graded, case, code):
    from tests._msl_diagnostic_cases import cases
    sim = next(sim for name, sim in cases(graded) if name == case)
    with pytest.raises(ValueError) as caught:
        sim.run(n_steps=2, compute_s_params=False)
    assert any(d.code == code and d.severity == 'refusal' for d in caught.value.diagnostics)


@pytest.mark.parametrize('graded', [False, True])
def test_resonant_tail_run_refusal_is_the_structured_stub_finding(graded):
    from dataclasses import replace
    from rfx import Box
    sim = structure(graded, feed=16 * U)
    sim._geometry[-1] = replace(sim._geometry[-1], shape=Box(
        (0, 4 * U, 2 * U), (32 * U, 8 * U, 2 * U)))
    with pytest.raises(ValueError) as caught:
        sim.run(n_steps=2, compute_s_params=False)
    records = caught.value.diagnostics
    assert len(records) == 1
    assert records[0].code == 'msl.line_stub_behind_port'
    assert records[0].severity == 'refusal'
    assert records[0].values['overhang_m'] > 0


@pytest.mark.parametrize('graded', [False, True])
def test_line_stub_realization_run_refusal_retains_the_exception(monkeypatch, graded):
    import rfx.preflight.line_stub as stub
    failure = ValueError('conductor realization failed')
    def unavailable(*args, **kwargs):
        raise failure
    monkeypatch.setattr(stub, 'line_stub_findings', unavailable)
    with pytest.raises(ValueError) as caught:
        structure(graded).run(n_steps=2, compute_s_params=False)
    assert caught.value is failure
    assert len(failure.diagnostics) == 1
    assert failure.diagnostics[0].code == 'msl.line_stub_realization'
    assert failure.diagnostics[0].severity == 'refusal'
    assert failure.diagnostics[0].message == 'conductor realization failed'


def test_run_retains_its_complete_preflight_tuple(monkeypatch):
    sim = structure()
    implementation = sim._preflight_impl
    reports = []
    def inspect(*args, **kwargs):
        report = implementation(*args, **kwargs)
        reports.append(report.diagnostics)
        return report
    monkeypatch.setattr(sim, '_preflight_impl', inspect)
    result = sim.run(n_steps=2, compute_s_params=False)
    assert len(reports) == 1
    assert result.diagnostics == reports[0]


def test_physical_two_port_calculator_unions_both_run_reports(monkeypatch):
    sim = structure(name='left')
    sim.add_msl_port((28 * U, 6 * U, 0), width=4 * U, height=2 * U,
                     direction='-x', name='right', mode='uniform', eps_r_sub=3.66,
                     n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    implementation = sim._preflight_impl
    reports = []
    def inspect(*args, **kwargs):
        report = implementation(*args, **kwargs)
        reports.append(report.diagnostics)
        return report
    monkeypatch.setattr(sim, '_preflight_impl', inspect)
    result = sim.compute_msl_s_matrix(freqs=np.array([3e9]), n_steps=64,
                                     enforce_passivity=False)
    assert len(reports) == 2
    expected = []
    for report in reports:
        for diagnostic in report:
            if diagnostic not in expected:
                expected.append(diagnostic)
    assert result.diagnostics == tuple(expected)
    assert {d.subject for d in result.diagnostics if d.code == 'msl.normal_resolution'} == {'left', 'right'}


@pytest.mark.parametrize('calculator,lane', [
    ('msl', 'adi'), ('msl', 'subgridded'),
    ('mixed', 'graded'), ('mixed', 'adi'), ('mixed', 'subgridded'),
    ('coax_msl', 'graded'), ('coax_msl', 'adi'), ('coax_msl', 'subgridded'),
])
def test_calculator_unsupported_lanes_carry_their_refusal(calculator, lane):
    if calculator == 'mixed':
        from tests.unit.sparams.test_msl_mixed_spatial_collocation import _case
        sim, _ = _case('+x')
        operation = sim.compute_mixed_s_matrix
        kwargs = dict(freqs=np.array([3e9]), n_steps=2, num_periods=1)
    elif calculator == 'coax_msl':
        from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs
        sim = build_instrument_junction()
        operation = sim.compute_coax_msl_transition
        kwargs = instrument_kwargs(n_steps=2)
    else:
        sim = structure()
        operation = sim.compute_msl_s_matrix
        kwargs = dict(freqs=np.array([3e9]), n_steps=2)
    step = float(sim._dx)
    if lane == 'graded':
        count = round(sim._domain[2] / step)
        sim._dz_profile = np.full(count, sim._domain[2] / count)
    elif lane == 'adi':
        sim._solver = 'adi'
    else:
        sim.add_refinement(z_range=(step, 3 * step), ratio=2)
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        operation(**kwargs)
    expected = {'graded': 'uniform', 'adi': 'solver=', 'subgridded': 'SBP-SAT'}[lane]
    assert expected in str(caught.value)
    assert any(d.severity == 'refusal' and d.message == str(caught.value)
               for d in caught.value.diagnostics)
