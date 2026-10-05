"""The matrix's comparison and generation must remain live."""
import numpy as np
import pytest

from . import comparison
from .builders import BUILDERS
from .generation import generate
from rfx.runners import _admission as admission


@pytest.mark.parametrize('kind', ('exact', 'step', 'accumulated'))
def test_comparison_rejects_a_changed_record(kind):
    with pytest.raises(AssertionError):
        comparison.compare(np.array([1.], dtype=np.float32), np.array([1.1], dtype=np.float32),
                           record='canary', kind=kind, measurements=[])


def test_comparison_rejects_missing_record():
    with pytest.raises(AssertionError, match='record missing'):
        comparison.compare(None, np.ones(1), record='canary', kind='step', measurements=[])


def test_new_admitted_row_requires_a_builder(monkeypatch):
    monkeypatch.setitem(admission._ADMITTED_ON, ('_s0_missing_builder', ''),
                        frozenset(('run_uniform', 'run_nonuniform')))
    with pytest.raises(AssertionError, match='S0 missing builder'):
        generate(BUILDERS)


def test_inactive_equivalence_builder_fails_generation(monkeypatch):
    monkeypatch.setitem(admission.DETECTORS, ('_freq_max', ''), lambda sim: False)
    with pytest.raises(AssertionError, match='S0 inactive builder'):
        generate(BUILDERS)


def test_realized_comparison_excludes_declarations_but_keeps_bounds():
    from .execution import _tree
    a = dict(lane='uniform', declared_length_m=1.1,
             axes=[dict(declared_bounds_m=(0., 1.1), bounds_m=(0., 1.))])
    b = dict(lane='nonuniform', declared_length_m=1.,
             axes=[dict(declared_bounds_m=(0., 1.), bounds_m=(0., 1.))])
    report = dict(failures=[], measurements=[])
    _tree(a, b, 'geometry', 'exact', report)
    assert not report['failures']
    b['axes'][0]['bounds_m'] = (0., 2.)
    _tree(a, b, 'geometry', 'exact', report)
    assert any('bounds_m' in failure for failure in report['failures'])


@pytest.mark.parametrize('helper', ('compare', '_comparison', '_tree'))
def test_matrix_canary_rejects_disabled_layer(monkeypatch, helper):
    from . import execution
    monkeypatch.setattr(execution, helper, lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='disabled'):
        execution._comparison_canary()


def test_ntff_excludes_only_named_kahan_residuals():
    from .execution import _tree
    a = dict(x_lo=np.array([1.]), y_hi=np.array([1e-8]), c_x_lo=np.array([1e10]))
    b = dict(x_lo=np.array([1.]), y_hi=np.array([1e-8]), c_x_lo=np.array([-1e10]))
    report = dict(failures=[], measurements=[])
    _tree(a, b, 'ntff_data', 'accumulated', report)
    assert not report['failures']
    assert {m['peak'] for m in report['measurements']} == {1., 1e-8}
    assert {m['record'] for m in report['measurements']} == {'ntff_data.x_lo', 'ntff_data.y_hi'}


@pytest.mark.parametrize('kind', ('step', 'accumulated'))
def test_small_leaf_five_percent_change_is_not_hidden_by_large_leaf(kind):
    from .execution import _tree
    a = dict(large=np.array([1.]), small=np.array([1e-8]))
    b = dict(large=np.array([1.]), small=np.array([1.05e-8]))
    report = dict(failures=[], measurements=[])
    _tree(a, b, 'record', kind, report)
    assert len(report['failures']) == 1
    assert report['failures'][0].startswith('record.small:')
    small = next(m for m in report['measurements'] if m['record'] == 'record.small')
    assert small['peak'] == 1.05e-8
    assert small['difference'] > small['bar']


@pytest.mark.parametrize('skipped_type', (dict, list))
def test_matrix_canary_rejects_disabled_container_traversal(monkeypatch, skipped_type):
    from . import execution
    original = execution._tree

    def skip_container(a, b, *args):
        if isinstance(a, skipped_type):
            return
        return original(a, b, *args)

    monkeypatch.setattr(execution, '_tree', skip_container)
    with pytest.raises(RuntimeError, match='disabled'):
        execution._comparison_canary()


def test_observer_amplitudes_use_own_peak_and_metadata_is_exact():
    from .execution import _tree
    a = dict(small=dict(accumulator=np.array([1.]), freqs=np.array([5e9])),
             large=dict(accumulator=np.array([1e10]), freqs=np.array([5e9])))
    b = dict(small=dict(accumulator=np.array([1.1]), freqs=np.array([5e9])),
             large=a['large'])
    report = dict(failures=[], measurements=[])
    _tree(a, b, 'dft_planes', 'accumulated', report)
    assert len(report['failures']) == 1
    assert report['failures'][0].startswith('dft_planes.small.accumulator:')
    b['small']['freqs'] = np.array([5e9 + 1])
    report = dict(failures=[], measurements=[])
    _tree(a, b, 'dft_planes', 'accumulated', report)
    assert any(m['record'].endswith('.freqs') and m['bar'] == 0 for m in report['measurements'])
    assert len(report['failures']) == 2


def test_pr_subset_includes_every_cause_and_main_path_refusal():
    from .test_matrix import CELLS, FINDINGS, PR_SUBSET
    causes = {e['cause'] for groups in FINDINGS.values() for entries in groups.values() for e in entries}
    selected_causes = {e['cause'] for identity, groups in FINDINGS.items() if identity in PR_SUBSET
                       for entries in groups.values() for e in entries}
    assert causes <= selected_causes
    assert all(c.steps == 12 for c in CELLS if c.equivalence and c.id in PR_SUBSET and c.id in FINDINGS)
    from .generation import PR_FINDING_CHOICES, PR_REFUSAL_LANES
    assert {c.id for c in CELLS if not c.equivalence and c.refused in PR_REFUSAL_LANES} <= PR_SUBSET
    witnesses = set(PR_FINDING_CHOICES.values())
    assert all(c.id in witnesses for c in CELLS
               if not c.equivalence and c.refused not in PR_REFUSAL_LANES and c.id in PR_SUBSET)


@pytest.mark.parametrize('full', (None, '0', '1'))
def test_full_matrix_requires_explicit_opt_in(monkeypatch, full):
    from types import SimpleNamespace
    from .conftest import pytest_collection_modifyitems
    if full is None:
        monkeypatch.delenv('RFX_S0_FULL', raising=False)
    else:
        monkeypatch.setenv('RFX_S0_FULL', full)
    subset = SimpleNamespace(get_closest_marker=lambda name: None)
    weekly = SimpleNamespace(get_closest_marker=lambda name: True)
    items = [subset, weekly]
    deselected = []
    config = SimpleNamespace(hook=SimpleNamespace(
        pytest_deselected=lambda items: deselected.extend(items)))
    pytest_collection_modifyitems(config, items)
    assert items == ([subset, weekly] if full == '1' else [subset])
    assert deselected == ([] if full == '1' else [weekly])


@pytest.mark.parametrize('stage', ('refused', 'admit'))
def test_refusal_tripwire_covers_query_and_message(monkeypatch, stage):
    from types import SimpleNamespace
    from . import execution
    cell = SimpleNamespace(id='refusal-canary', equivalence=False, row=('_freq_max', ''),
                           refused='run_adi', graded=False, steps=0)
    monkeypatch.setattr(execution, 'build', lambda *args, **kwargs: object())
    monkeypatch.setitem(admission.DETECTORS, cell.row, lambda sim: True)
    monkeypatch.setattr(admission, 'refused', lambda *args: [cell.row])

    def attempt_scan(*args):
        execution.jax.lax.scan(None, None, None)
        raise ValueError('internal assembly error after stepping')

    monkeypatch.setattr(admission, stage, attempt_scan)
    report = execution.execute(cell)
    assert report['refusal_scan_started'] is True
    assert report['failures'] == ['AssertionError: refusal reached a scan']


def test_assembly_fingerprint_preserves_caller_and_message():
    from pathlib import Path
    from .reporting import fingerprint
    root = Path(__file__).resolve().parents[3]
    message = f'ValueError: classified, but {root}/rfx/runners/_admission.py:804 in _adi_homogeneous() passed no pec_sheets/pec_wires collector.'
    expected = message.replace(str(root) + '/', '')
    assert fingerprint(message) == expected
    assert fingerprint(message.replace(':804', ':805')) != expected


def test_shared_cause_does_not_accept_another_cells_fingerprint():
    from .reporting import KnownFinding, assert_record, expand_findings, expectations
    manifest = dict(causes={'assembly': dict(implementations=['rfx/example.py:1'],
        fingerprints={'sheet': dict(value='ValueError: sheet', cells=['a']),
                      'wire': dict(value='ValueError: wire', cells=['b'])}, witnesses={})},
        cells={'a': {'refusal': ['assembly']}, 'b': {'refusal': ['assembly']}})
    findings = expand_findings(manifest)
    known, witnesses, _ = expectations(findings['a']['refusal'])
    with pytest.raises(KnownFinding):
        assert_record(dict(failures=['ValueError: sheet'], measurements=[]), 'refusal', known, witnesses)
    with pytest.raises(RuntimeError, match='ValueError: wire'):
        assert_record(dict(failures=['ValueError: wire'], measurements=[]), 'refusal', known, witnesses)


def test_shared_witness_variant_keeps_record_scope_and_drift_check():
    from .reporting import assert_record, expand_findings, expectations
    manifest = dict(causes={'numeric': dict(implementations=['rfx/example.py:1'],
        fingerprints={'plane': dict(value='dft_planes.p: numeric', records=['observers']),
                      'probe': dict(value='time_series: numeric', records=['probes'])},
        witnesses={'default': {'plane': dict(relative=0.1, relative_bar=1e-4)},
                   'long': {'plane': dict(relative=0.2, relative_bar=1e-4)}})},
        cells={'a': {'observers': ['numeric'], 'probes': ['numeric']},
               'b': {'observers': [dict(cause='numeric', witness='long')]}})
    findings = expand_findings(manifest)
    known, witnesses, _ = expectations(findings['b']['observers'])
    assert known == ['dft_planes.p: numeric']
    assert witnesses == {'dft_planes.p': dict(relative=0.2, relative_bar=1e-4)}
    assert expectations(findings['a']['observers'])[1]['dft_planes.p']['relative'] == 0.1
    assert expectations(findings['a']['probes'])[0] == ['time_series: numeric']
    report = dict(failures=['dft_planes.p: relative diff 0.21'], measurements=[
        dict(record='dft_planes.p', relative=0.21, relative_bar=1e-4, difference=0.21, bar=1e-4)])
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        assert_record(report, 'observers', known, witnesses)
    manifest['cells']['b']['observers'][0]['witness'] = 'typo'
    with pytest.raises(KeyError):
        expand_findings(manifest)
