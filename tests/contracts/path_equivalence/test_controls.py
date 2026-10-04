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
