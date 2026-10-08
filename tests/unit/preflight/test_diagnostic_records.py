"""Independent contracts for immutable records, legacy bridges and result transport."""
import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from rfx import Diagnostic, MATERIAL_LIBRARY
from rfx._diagnostic_transport import merge_diagnostics, diagnostic_refusal
from rfx.api._spec import Result, ForwardResult, MSLSMatrixResult, MixedSMatrixResult, CoaxMSLTransitionResult
from rfx.preflight._common import (
    PreflightIssue,
    PreflightReport,
    PreflightWarning,
    PreflightErrorWarning,
    PreflightConfigError,
)


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
    assert json.loads(
        json.dumps(
            d.to_dict(),
            allow_nan=False,
        )
    ) == {
        "code": "msl.example",
        "severity": "advisory",
        "subject": "port",
        "message": "length 2 mm",
        "values": {"length_m": 0.002},
        "path": None,
        "source": None,
    }
    with pytest.raises(
        ValueError,
        match="severity",
    ):
        Diagnostic("x", "error", None, "x", {})
    for n in ast.walk(ast.parse((ROOT / 'rfx/diagnostic_records.py').read_text())):
        if isinstance(n, ast.ImportFrom):
            assert not (n.module or '').startswith(('rfx.api', 'rfx.preflight', 'rfx.runners'))


@pytest.mark.parametrize('severity,legacy', [
    ('info', 'info'), ('advisory', 'warning'), ('warning', 'warning'), ('refusal', 'error'),
])
def test_severity_mapping(severity, legacy):
    d = Diagnostic('x', severity, 'p', 'message', {})
    assert d.legacy_severity == legacy
    assert PreflightWarning(d).severity == legacy


def test_warning_issue_report_bridges_and_refusal():
    d = observation()
    warning = PreflightWarning(
        d,
        code="legacy_slug",
    )
    issue = PreflightIssue(
        warning,
        code=warning.code,
        severity=warning.severity,
    )
    assert str(warning) == str(issue) == d.message
    assert issue.code == 'legacy_slug' and issue.diagnostic is d
    unmoved = PreflightIssue(
        "old 7 mm",
        code="old_code",
    )
    assert unmoved.diagnostic.code == 'old_code' and unmoved.diagnostic.values == {}
    report = PreflightReport([issue, unmoved])
    assert report.diagnostics == (d, unmoved.diagnostic)
    refusal = Diagnostic('msl.refusal', 'refusal', 'p', 'refused', {})
    error = PreflightConfigError(
        refusal,
        code="old_refusal",
    )
    assert str(error) == 'refused' and error.diagnostics == (refusal,)
    assert PreflightErrorWarning(refusal).diagnostic is refusal
    with pytest.raises(ValueError) as caught:
        PreflightReport(
            [
                PreflightIssue(
                    refusal,
                    severity="error",
                )
            ]
        ).raise_for_failure()
    assert caught.value.diagnostics == (refusal,)


def test_record_and_legacy_report_copy_and_pickle_keep_immutable_values():
    import copy
    import pickle
    diagnostic = Diagnostic('msl.example', 'advisory', 'port', 'length 2 mm', {'length_m': .002})
    report = PreflightReport(
        [
            PreflightIssue(
                diagnostic,
                code="legacy_slug",
            )
        ]
    )
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
    for cls in (MSLSMatrixResult, MixedSMatrixResult):
        assert cls.__dataclass_fields__['diagnostics'].default == ()
    assert 'diagnostics' not in CoaxMSLTransitionResult.__dataclass_fields__


@pytest.mark.parametrize('result_kind', ['run', 'forward'])
def test_jit_grad_vmap_static_metadata_without_tracer_leaks(result_kind):
    d = observation()
    def objective(x):
        return (
            Result(
                None,
                x**2,
                None,
                None,
                diagnostics=(d,),
            )
            if result_kind == "run"
            else ForwardResult(
                x**2,
                diagnostics=(d,),
            )
        )

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
    drives = [
        Result(
            None,
            None,
            None,
            None,
            diagnostics=(d,),
        )
        for d in (d1, d1, d2)
    ]
    assert merge_diagnostics((d1,), *(drive.diagnostics for drive in drives)) == (d1, d2)
    assert merge_diagnostics() == ()
    assert ForwardResult(None).diagnostics == ()


def test_an_earlier_advisory_does_not_hide_a_new_refusal():
    earlier = Diagnostic('msl.line_stub_behind_port', 'advisory', 'port',
                         'stub outside this read band', {'frequency_hz': 20e9})
    error = diagnostic_refusal(NotImplementedError('this path does not drive MSL ports'), (earlier,))
    assert error.diagnostics[0] == earlier
    assert error.diagnostics[-1].message == 'this path does not drive MSL ports'
    assert error.diagnostics[-1].severity == 'refusal'


@pytest.mark.parametrize("word", ["critical", "advisory", "note"])
def test_an_unknown_severity_word_is_kept_and_advises(word):
    """Worked before the record existed: the word is kept on the issue and treated as a warning."""
    from rfx.preflight._common import PreflightIssue, PreflightReport, PreflightWarning

    issue = PreflightIssue("x", severity=word)
    assert issue.severity == word and issue.diagnostic.severity == "advisory"
    assert PreflightWarning("x", severity=word).severity == word
    assert PreflightReport([issue]).ok
