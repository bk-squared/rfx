"""Review regressions for record severity, legacy text and serialization."""
import copy
from dataclasses import asdict, replace
import pickle
import warnings

import numpy as np
import pytest

import rfx
from rfx import Diagnostic, Box
from rfx.api._spec import MSLSMatrixResult, MixedSMatrixResult
from rfx.diagnostic_records import from_legacy
from rfx.preflight._common import PreflightConfigError, PreflightIssue
from tests._msl_diagnostic_cases import cases, structure, U


@pytest.mark.parametrize('legacy,level', [('error', 'refusal'), ('warning', 'advisory'), ('info', 'info')])
def test_actual_legacy_severity_controls_the_record(legacy, level):
    record = from_legacy(
        "finding",
        severity=legacy,
    )
    assert record.severity == level
    assert record.legacy_severity == legacy
    original = Diagnostic('msl.example', 'refusal', 'port', 'finding', {'length_m': .001})
    issue = PreflightIssue(
        original,
        severity=legacy,
    )
    assert issue.diagnostic.severity == level
    assert issue.diagnostic.legacy_severity == issue.severity


@pytest.mark.parametrize('graded', [False, True])
def test_every_real_issue_has_matching_severity_and_text(graded):
    for name, sim in cases(graded):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reports = [
                sim.preflight(),
                sim.preflight_sparameters(
                    calculator="msl",
                    include_general=True,
                ),
            ]
        for report in reports:
            for issue in report:
                assert issue.diagnostic.legacy_severity == issue.severity, (name, issue)
                assert issue.diagnostic.message == str(issue), (name, issue)


def test_config_error_issue_keeps_the_error_prefix():
    record = Diagnostic('msl.example', 'advisory', 'port', 'bad geometry', {})
    error = PreflightConfigError(record)
    issue = PreflightIssue(
        "ERROR: " + str(error),
        severity="error",
        diagnostic=error.diagnostic,
    )
    assert issue.diagnostic.message == 'ERROR: bad geometry'
    assert issue.diagnostic.severity == 'refusal'


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_returned_results_have_no_refusal_but_in_band_stub_does(graded, entry):
    sim = structure(
        graded,
        feed=16 * U,
        freq_max=1e9,
    )
    sim._geometry[-1] = replace(
        sim._geometry[-1],
        shape=Box((0, 4 * U, 2 * U), (32 * U, 8 * U, 2 * U)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = getattr(sim, entry)(
            n_steps=2,
            **({"compute_s_params": False} if entry == "run" else {}),
        )
    assert all(record.severity != 'refusal' for record in result.diagnostics)
    stubs = [record for record in result.diagnostics if record.code == 'msl.line_stub_behind_port']
    assert len(stubs) == 1 and stubs[0].severity == 'advisory'
    sim.freq_max = 20e9
    # The guard accepts an explicit read band independently of the solver API.
    from rfx.preflight.line_stub import require_no_resonant_line_stub
    with pytest.raises(ValueError) as caught:
        require_no_resonant_line_stub(sim, np.array([1e9, 20e9]))
    assert caught.value.diagnostics[-1].severity == 'refusal'


@pytest.mark.parametrize('result_kind', ['msl', 'mixed'])
def test_asdict_with_records_and_immutable_roundtrips(result_kind):
    record = Diagnostic('msl.example', 'advisory', 'port', 'length 2 mm', {'length_m': .002})
    a = np.zeros(1)
    result = (
        MSLSMatrixResult(
            a,
            a,
            a,
            a,
            diagnostics=(record,),
        )
        if result_kind == "msl"
        else MixedSMatrixResult(
            a,
            a,
            ("port",),
            ("msl",),
            a,
            diagnostics=(record,),
        )
    )
    assert asdict(result)['diagnostics'] == (asdict(record),)
    for restored in (copy.deepcopy(record), pickle.loads(pickle.dumps(record))):
        assert restored == record and hash(restored) == hash(record)
    for values in (copy.deepcopy(record.values), pickle.loads(pickle.dumps(record.values)), asdict(record)['values']):
        assert values == {'length_m': .002}
        with pytest.raises(TypeError):
            values.update(
                length_m=3,
            )


def test_long_port_name_is_not_dropped_and_diagnostic_is_public():
    name = 'p' * 200
    report = structure(
        name=name,
    ).preflight()
    records = [d for d in report.diagnostics if d.code.startswith('msl.') and 'port_name' in d.values]
    assert records
    assert all(d.subject == d.values['port_name'] == name for d in records)
    assert 'Diagnostic' in rfx.__all__
