"""Independent contracts for immutable records, legacy bridges and result transport."""
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from rfx import Diagnostic, MATERIAL_LIBRARY
from rfx._diagnostic_context import diagnostic_scope, diagnostic_result, record_diagnostics
from rfx.api._spec import Result, ForwardResult, MSLSMatrixResult, MixedSMatrixResult, CoaxMSLTransitionResult
from rfx.preflight._common import PreflightIssue, PreflightReport, PreflightWarning, PreflightErrorWarning, PreflightConfigError


ROOT = Path(__file__).resolve().parents[3]


def observation(subject='port'):
    return Diagnostic('msl.example', 'advisory', subject, 'length 2 mm', {'length_m': .002})


def test_frozen_leaf_json_and_public_export():
    values = {'length_m': .002}
    d = Diagnostic('msl.example', 'advisory', 'port', 'length 2 mm', values)
    values['length_m'] = 3
    assert d.values == {'length_m': .002}
    with pytest.raises(FrozenInstanceError):
        d.code = 'changed'
    with pytest.raises(TypeError):
        d.values['length_m'] = 4
    assert json.loads(json.dumps(d.to_dict(), allow_nan=False)) == {
        'code': 'msl.example', 'severity': 'advisory', 'subject': 'port',
        'message': 'length 2 mm', 'values': {'length_m': .002}, 'path': None, 'source': None,
    }
    with pytest.raises(ValueError, match='severity'):
        Diagnostic('x', 'error', None, 'x', {})
    for n in ast.walk(ast.parse((ROOT / 'rfx/diagnostic_records.py').read_text())):
        if isinstance(n, ast.ImportFrom):
            assert not (n.module or '').startswith(('rfx.api', 'rfx.preflight', 'rfx.runners'))


@pytest.mark.parametrize('severity,legacy', [('info','info'),('advisory','warning'),('warning','error'),('refusal','error')])
def test_severity_mapping(severity, legacy):
    d = Diagnostic('x', severity, 'p', 'message', {})
    assert d.legacy_severity == legacy
    assert PreflightWarning(d).severity == legacy


def test_warning_issue_report_bridges_and_refusal():
    d = observation()
    warning = PreflightWarning(d, code='legacy_slug')
    issue = PreflightIssue(warning, code=warning.code, severity=warning.severity)
    assert str(warning) == str(issue) == d.message
    assert issue.code == 'legacy_slug' and issue.diagnostic is d
    unmoved = PreflightIssue('old 7 mm', code='old_code')
    assert unmoved.diagnostic.code == 'old_code' and unmoved.diagnostic.values == {}
    report = PreflightReport([issue, unmoved])
    assert report.diagnostics == (d, unmoved.diagnostic)
    refusal = Diagnostic('msl.refusal', 'refusal', 'p', 'refused', {})
    error = PreflightConfigError(refusal, code='old_refusal')
    assert str(error) == 'refused' and error.diagnostics == (refusal,)
    assert PreflightErrorWarning(refusal).diagnostic is refusal
    with pytest.raises(ValueError) as caught:
        PreflightReport([PreflightIssue(refusal, severity='error')]).raise_for_failure()
    assert caught.value.diagnostics == (refusal,)


def test_record_and_legacy_report_copy_and_pickle_keep_immutable_values():
    import copy
    import pickle
    diagnostic = Diagnostic('msl.example', 'advisory', 'port', 'length 2 mm', {'length_m': .002})
    report = PreflightReport([PreflightIssue(diagnostic, code='legacy_slug')])
    for restored in (copy.deepcopy(report), pickle.loads(pickle.dumps(report))):
        assert restored.diagnostics == (diagnostic,)
        assert restored[0].code == 'legacy_slug'
        assert str(restored[0]) == 'length 2 mm'
        with pytest.raises(TypeError):
            restored.diagnostics[0].values['length_m'] = 3


def test_old_positional_results_and_material_imports():
    from rfx.api._spec import MATERIAL_LIBRARY as old
    assert old is MATERIAL_LIBRARY
    assert Result(None, None, None, None).diagnostics == ()
    assert ForwardResult(jnp.ones(1)).diagnostics == ()
    for cls in (Result, ForwardResult):
        assert cls._fields[-1] == 'diagnostics'
    for cls in (MSLSMatrixResult, MixedSMatrixResult, CoaxMSLTransitionResult):
        assert cls.__dataclass_fields__['diagnostics'].default == ()


@pytest.mark.parametrize('result_kind', ['run', 'forward'])
def test_jit_grad_vmap_static_metadata_without_tracer_leaks(result_kind):
    d = observation()
    def objective(x):
        with diagnostic_scope():
            record_diagnostics((d,))
            result = Result(None, x ** 2, None, None) if result_kind == 'run' else ForwardResult(x ** 2)
            return diagnostic_result(result)
    with jax.checking_leaks():
        result = jax.jit(objective)(jnp.array(3.))
        assert result.diagnostics == (d,)
        assert float(jax.grad(lambda x: objective(x).time_series)(3.)) == 6.
        mapped = jax.jit(jax.vmap(objective))(jnp.array([2., 3.]))
        assert mapped.diagnostics == (d,)
        assert mapped.time_series.tolist() == [4., 9.]
    assert jax.tree_util.tree_leaves(d) == []


def test_nested_drive_union_preserves_changed_observations():
    d1, d2 = observation('first'), observation('second')
    with diagnostic_scope():
        for d in (d1, d1, d2):
            with diagnostic_scope():
                record_diagnostics((d,))
                drive = diagnostic_result(Result(None, None, None, None))
                assert drive.diagnostics == (d,)
        result, inspection = diagnostic_result(Result(None, None, None, None), {'raw': True})
    assert result.diagnostics == (d1, d2)
    assert inspection == {'raw': True}
    with diagnostic_scope():
        assert diagnostic_result(ForwardResult(None)).diagnostics == ()


def test_scopes_are_isolated_in_threads_and_after_refusal():
    from threading import Barrier
    barrier = Barrier(2)
    def run(name):
        with diagnostic_scope():
            record_diagnostics((observation(name),))
            barrier.wait(timeout=10)
            return diagnostic_result(ForwardResult(None)).diagnostics
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(run, ('first', 'second')))
    assert [d.subject for d in first] == ['first']
    assert [d.subject for d in second] == ['second']
    with pytest.raises(ValueError) as caught:
        with diagnostic_scope():
            record_diagnostics(first)
            raise ValueError('unsupported path')
    assert caught.value.diagnostics[0] == first[0]
    assert caught.value.diagnostics[-1].severity == 'refusal'
    with diagnostic_scope():
        assert diagnostic_result(ForwardResult(None)).diagnostics == ()


def test_an_earlier_refusal_level_observation_does_not_hide_a_new_cause():
    earlier = Diagnostic('msl.line_stub_behind_port', 'refusal', 'port',
                         'stub outside this read band', {'frequency_hz': 20e9})
    with pytest.raises(NotImplementedError) as caught:
        with diagnostic_scope():
            record_diagnostics((earlier,))
            raise NotImplementedError('this path does not drive MSL ports')
    assert caught.value.diagnostics[0] == earlier
    assert caught.value.diagnostics[-1].message == 'this path does not drive MSL ports'
    assert caught.value.diagnostics[-1].severity == 'refusal'
