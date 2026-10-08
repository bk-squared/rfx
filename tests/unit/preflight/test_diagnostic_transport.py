"""Public execution boundaries retain host records under nesting and transforms."""
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Diagnostic
from tests._msl_diagnostic_cases import structure, U


@pytest.mark.parametrize('graded', [False, True])
def test_actual_forward_jit_grad_vmap(graded):
    sim = structure(graded)
    sim.add_probe((16 * U, 6 * U, U), 'ez')
    def forward(scale):
        # The traced override is a field input; geometry and metadata stay concrete.
        shape = sim._build_realized_grid().shape
        result = sim.forward(
            eps_override=jnp.ones(shape) * scale,
            n_steps=2,
            checkpoint=False,
        )
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




def test_msl_calculator_unions_every_drive_without_duplicates():
    # Manufactured spectra isolate the public calculator's nested-run transport.
    # Real field evolution is covered by the forward and run tests above.
    from tests.unit.autodiff.test_msl_forward_identity import _build_planted_sim, FREQS
    sim, _ = _build_planted_sim(1.)
    backend = sim.run
    seen = []
    def drive(self, **kwargs):
        d = Diagnostic('msl.drive_observation', 'advisory', str(len(seen)),
                       'drive observation', {'drive': len(seen)})
        seen.append(d)
        result = backend(
            **kwargs,
        )
        result.diagnostics = (d, d)
        return result
    sim.run = MethodType(drive, sim)
    result = sim.compute_msl_s_matrix(
        freqs=FREQS,
        n_steps=1,
        num_periods=1,
        enforce_passivity=False,
    )
    assert len(seen) == 2
    assert tuple(d for d in result.diagnostics if d.code == 'msl.drive_observation') == tuple(seen)
    clean = structure().run(
        n_steps=2,
        compute_s_params=False,
        skip_preflight=True,
    )
    assert clean.diagnostics == ()


def test_run_warning_location_is_the_public_caller():
    import inspect
    sim = structure()
    with pytest.warns(UserWarning) as caught:
        line = inspect.currentframe().f_lineno + 1
        sim.run(
            n_steps=2,
            compute_s_params=False,
        )
    advisory = [w for w in caught if 'preflight found' in str(w.message)]
    assert len(advisory) == 1
    assert advisory[0].filename == __file__
    assert advisory[0].lineno == line




def test_scoped_coax_gate_accepts_the_existing_plain_issue_list():
    from rfx.preflight._common import PreflightIssue
    from rfx.preflight._impl import run_preflight_gate
    d = Diagnostic('msl.example_refusal', 'refusal', 'port', 'refused', {})
    with pytest.raises(
        ValueError,
        match="blocking error",
    ) as caught:
        run_preflight_gate(
            [PreflightIssue(d)],
            context="compute_coax_msl_transition",
        )
    assert caught.value.diagnostics == (d,)


@pytest.mark.parametrize('lane', ['adi', 'subgridded', 'disjoint'])
def test_auxiliary_lanes_attach_their_refusal_and_family_records(lane):
    sim = structure()
    if lane == 'adi':
        sim._solver = 'adi'
    else:
        sim.add_refinement(
            z_range=(U, 3 * U),
            ratio=2,
            topology="stage2_disjoint_3d" if lane == "disjoint" else "overlap_z_slab",
        )
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        sim.run(
            n_steps=2,
            compute_s_params=False,
        )
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
        sim.run(
            n_steps=2,
            compute_s_params=False,
        )
    assert any(d.code == code and d.severity == 'refusal' for d in caught.value.diagnostics)


@pytest.mark.parametrize('graded', [False, True])
def test_resonant_tail_run_refusal_is_the_structured_stub_finding(graded):
    from dataclasses import replace
    from rfx import Box

    sim = structure(
        graded,
        feed=16 * U,
    )
    sim._geometry[-1] = replace(
        sim._geometry[-1],
        shape=Box((0, 4 * U, 2 * U), (32 * U, 8 * U, 2 * U)),
    )
    with pytest.raises(ValueError) as caught:
        sim.run(
            n_steps=2,
            compute_s_params=False,
        )
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
        structure(graded).run(
            n_steps=2,
            compute_s_params=False,
        )
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
        report = implementation(
            *args,
            **kwargs,
        )
        reports.append(report.diagnostics)
        return report
    monkeypatch.setattr(sim, '_preflight_impl', inspect)
    result = sim.run(
        n_steps=2,
        compute_s_params=False,
    )
    assert len(reports) == 1
    assert result.diagnostics == reports[0]


def test_physical_two_port_calculator_unions_both_run_reports(monkeypatch):
    sim = structure(
        name="left",
    )
    sim.add_msl_port(
        (28 * U, 6 * U, 0),
        width=4 * U,
        height=2 * U,
        direction="-x",
        name="right",
        mode="uniform",
        eps_r_sub=3.66,
        n_probe_offset=3,
        n_probe_spacing=2,
        n_probes=3,
    )
    implementation = sim._preflight_impl
    reports = []
    def inspect(*args, **kwargs):
        report = implementation(
            *args,
            **kwargs,
        )
        reports.append(report.diagnostics)
        return report
    monkeypatch.setattr(sim, '_preflight_impl', inspect)
    result = sim.compute_msl_s_matrix(
        freqs=np.array([3e9]),
        n_steps=64,
        enforce_passivity=False,
    )
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
])
def test_calculator_unsupported_lanes_carry_their_refusal(calculator, lane):
    if calculator == 'mixed':
        from tests.unit.sparams.test_msl_mixed_spatial_collocation import _case
        sim, _ = _case('+x')
        operation = sim.compute_mixed_s_matrix
        kwargs = dict(
            freqs=np.array([3e9]),
            n_steps=2,
            num_periods=1,
        )
    else:
        sim = structure()
        operation = sim.compute_msl_s_matrix
        kwargs = dict(
            freqs=np.array([3e9]),
            n_steps=2,
        )
    step = float(sim._dx)
    if lane == 'graded':
        count = round(sim._domain[2] / step)
        sim._dz_profile = np.full(count, sim._domain[2] / count)
    elif lane == 'adi':
        sim._solver = 'adi'
    else:
        sim.add_refinement(
            z_range=(step, 3 * step),
            ratio=2,
        )
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        operation(
            **kwargs,
        )
    expected = {'graded': 'uniform', 'adi': 'solver=', 'subgridded': 'SBP-SAT'}[lane]
    assert expected in str(caught.value)
    assert any(d.severity == 'refusal' and d.message == str(caught.value)
               for d in caught.value.diagnostics)


def test_mixed_calculator_unions_own_report_and_each_drive(monkeypatch):
    from rfx.preflight._common import PreflightIssue, PreflightReport
    from tests.unit.sparams.test_msl_mixed_spatial_collocation import _case, _backend, FREQS
    sim, target = _case('+x')
    own = Diagnostic('msl.own', 'advisory', 'port', 'own report', {})
    report = PreflightReport([PreflightIssue(own)])
    monkeypatch.setattr(sim, '_auto_preflight', lambda **kwargs: report)
    backend = _backend(target, '+x', {'affine': False}, [])
    seen = []
    def drive(self, *args, **kwargs):
        record = Diagnostic('msl.drive', 'advisory', str(len(seen)), 'drive report', {})
        seen.append(record)
        result = backend(
            self,
            *args,
            **kwargs,
        )
        result['diagnostics'] = (own, record, record)
        return result
    monkeypatch.setattr(sim, '_forward_from_materials', MethodType(drive, sim))
    result, details = sim.compute_mixed_s_matrix(
        freqs=FREQS,
        n_steps=1,
        num_periods=1,
        magnitude_channel="wave",
        enforce_passivity=False,
        return_diagnostics=True,
    )
    assert len(seen) == 2
    assert result.diagnostics == (own, seen[0], seen[1])
    assert details['drive_plan'] == [('lw', 0), ('msl', 0)]


def test_forward_unsupported_lanes_retain_preflight_records():
    for distributed in (False, True):
        sim = structure()
        if not distributed:
            sim.add_refinement(
                z_range=(U, 3 * U),
                ratio=2,
            )
        with pytest.raises(NotImplementedError) as caught:
            sim.forward(
                n_steps=2,
                distributed=distributed,
            )
        records = caught.value.diagnostics
        assert any(d.severity == 'refusal' for d in records)
        assert {d.code for d in records if d.code.startswith('msl.')} == {
            'msl.lateral_clearance', 'msl.normal_resolution',
            'msl.source_absorber_clearance', 'msl.source_near_field',
        }


@pytest.mark.parametrize('prepared', [False, True])
def test_direct_distributed_inputs_keep_the_preflight_report(prepared, monkeypatch):
    from rfx.model import conductors
    from rfx.preflight._common import PreflightIssue, PreflightReport
    sim = structure(True)
    grid = sim._build_nonuniform_grid()
    record = Diagnostic('msl.direct_report', 'advisory', 'port', 'direct report', {})
    report = PreflightReport([PreflightIssue(record)])
    monkeypatch.setattr(sim, '_preflight_impl', lambda **kwargs: report)
    with pytest.warns(
        UserWarning,
        match="direct report",
    ):
        assembly = (
            sim._auto_preflight(
                prepare=True,
            )
            if prepared
            else None
        )
        product = conductors.distributed_solve_inputs(sim, grid, assembly, False, False)
    assert product[-1] == (record,)


def test_gate_refusal_identifies_the_blocking_cause_after_an_advisory_stub():
    from rfx.preflight._common import PreflightIssue
    from rfx.preflight._impl import run_preflight_gate
    earlier = Diagnostic('msl.line_stub_behind_port', 'advisory', 'port',
                         'stub outside the read band', {'frequency_hz': 20e9})
    issues = [
        PreflightIssue(
            earlier,
            severity="warning",
            code="line_stub_behind_port",
        ),
        PreflightIssue(
            "actual blocking configuration",
            severity="error",
            code="other_family",
        ),
    ]
    with (
        pytest.warns(
            UserWarning,
            match="stub outside",
        ),
        pytest.raises(ValueError) as caught,
    ):
        run_preflight_gate(
            issues,
            context="run",
        )
    records = caught.value.diagnostics
    assert records[0] == earlier
    assert any(d.severity == 'refusal' and d.message == 'actual blocking configuration' for d in records)
    assert any(d.code == 'other_family' for d in records)


@pytest.mark.parametrize('topology', ['overlap_z_slab', 'stage2_disjoint_3d'])
def test_research_subgrid_result_keeps_its_report(topology, monkeypatch):
    from tests.contracts.test_subgrid_requires_experimental import _sim
    from rfx.preflight._common import PreflightIssue, PreflightReport
    sim = _sim('research')
    sim._refinement['topology'] = topology
    record = Diagnostic('transport.marker', 'info', 'fixture', 'report retained', {})
    report = PreflightReport([PreflightIssue(record)])
    monkeypatch.setattr(sim, '_preflight_impl', lambda **kwargs: report)
    with pytest.warns(UserWarning):
        result = sim.run(
            n_steps=2,
            compute_s_params=False,
        )
    assert result.diagnostics == (record,)
    assert result.time_series.shape[0] == 2


@pytest.mark.parametrize('frequencies, message', [
    (['not a number'], '#1512: line-port read frequencies must be concrete'),
    ([], '#1512: line-port read frequencies must be finite, nonnegative and nonempty'),
    ([-1.], '#1512: line-port read frequencies must be finite, nonnegative and nonempty'),
    ([float('nan')], '#1512: line-port read frequencies must be finite, nonnegative and nonempty'),
])
def test_line_stub_read_band_refusal_has_a_record(frequencies, message):
    from rfx.preflight.line_stub import read_band
    with pytest.raises(ValueError) as caught:
        read_band(structure(), frequencies)
    assert str(caught.value) == message
    assert caught.value.diagnostics == (
        Diagnostic('uncoded', 'refusal', None, message, {}),)


@pytest.mark.parametrize('kind', ['tfsf', 'waveguide'])
@pytest.mark.parametrize('entry', ['api', 'v2'])
def test_native_fallback_keeps_parent_and_child_reports(kind, entry, monkeypatch):
    from rfx import Result, Simulation
    from rfx.preflight._common import PreflightIssue, PreflightReport
    from rfx.runners.distributed_v2 import run_distributed

    sim = Simulation(
        freq_max=15e9,
        domain=(24e-3, 12e-3, 12e-3),
        dx=1e-3,
        boundary="cpml",
    )
    if kind == 'tfsf':
        sim.add_tfsf_source(
            f0=7.5e9,
            bandwidth=0.5,
        )
    else:
        sim.add_waveguide_port(
            x_position=6e-3,
            y_range=(2e-3, 10e-3),
            z_range=(2e-3, 10e-3),
            mode=(1, 0),
            mode_type="TE",
            direction="+x",
            f0=7.5e9,
            bandwidth=0.5,
        )
    shared = Diagnostic('transport.shared', 'info', None, 'shared report', {})
    parent = Diagnostic('transport.parent', 'info', None, 'parent report', {})
    child = Diagnostic('transport.child', 'info', None, 'child report', {})
    report = PreflightReport([PreflightIssue(shared), PreflightIssue(parent)])
    monkeypatch.setattr(sim, '_preflight_impl', lambda **kwargs: report)
    public_run = sim.run
    payload = np.zeros((2, 0))
    native_result = Result(None, payload, None, None, None,
                           diagnostics=(shared, child, child))
    monkeypatch.setattr(sim, 'run', lambda **kwargs: native_result)
    devices = jax.devices('cpu')[:2]
    assert len(devices) == 2
    with pytest.warns(
        UserWarning,
        match="Falling back to single-device",
    ):
        result = (
            public_run(
                n_steps=2,
                devices=devices,
                compute_s_params=False,
            )
            if entry == "api"
            else run_distributed(
                sim,
                n_steps=2,
                devices=devices,
                diagnostics=report.diagnostics,
            )
        )
    assert result.diagnostics == (shared, parent, child)
    assert result.time_series is payload
    assert native_result.diagnostics == (shared, child, child)
