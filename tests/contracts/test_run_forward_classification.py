"""Hand-written difference/outcome records falsify exception classification."""
import pytest

from tests.contracts import test_run_forward_one_pipeline as contract


@pytest.fixture
def synthetic_consequence(monkeypatch):
    """A consequence pair for the classifier itself: the census has none left
    (PW-oblique D3 and R-rlc D6 are closed), but the rule must keep working."""
    monkeypatch.setattr(contract, 'CONSEQUENCE_PAIRS', (('X-field', 'uniform', 'D0'),))
    monkeypatch.setitem(contract.ID_PATTERNS, 'D0', {'L1': (), 'L2': ()})
    return 'X-field', 'D0'


def test_fixed_id_must_be_removed():
    with pytest.raises(pytest.fail.Exception, match='D4 no longer applies to O-flux/uniform/L2'):
        contract.classify('O-flux', 'uniform', 'L2', [], {'D4': 'flux absent'})


def test_consequences_do_not_explain_witness(synthetic_consequence):
    declaration, ident = synthetic_consequence
    with pytest.raises(pytest.fail.Exception, match='unexplained.*settling_witness/residual'):
        contract.classify(declaration, 'uniform', 'L2', [
            {'path': 'settling_witness/residual', 'bytes_equal': False, 'shape': [[1], [2]]},
        ], {ident: 'field cause'})


@pytest.mark.parametrize('declaration,ident,leaf', [
    ('O-flux', 'D4', 'flux_monitors/0/flux'),
    ('P-waveguide', 'D7', 'waveguide_sparams/0/S'),
])
def test_returned_measurement_value_is_not_missing(declaration, ident, leaf):
    with pytest.raises(pytest.fail.Exception, match='unexplained'):
        contract.classify(declaration, 'uniform', 'L2', [
            {'path': leaf, 'bytes_equal': False, 'shape': [[2], [2]]},
        ], {ident: 'missing field'})


def test_same_shape_witness_value_is_unexplained():
    with pytest.raises(pytest.fail.Exception, match='unexplained.*settling_witness/residual'):
        contract.classify('P-waveguide', 'uniform', 'L2', [
            {'path': 'settling_witness/residual', 'bytes_equal': False, 'shape': [[2], [2]]},
        ], contract.known('P-waveguide', 'uniform', 'L2'))


@pytest.mark.parametrize('declaration,ident,leaf', [
    ('O-flux', 'D4', 'flux_monitors/0/flux'),
    ('P-waveguide', 'D7', 'waveguide_ports/0/voltage'),
])
def test_only_run_side_missing_records_are_explained(declaration, ident, leaf):
    with pytest.raises(pytest.xfail.Exception):
        contract.classify(declaration, 'uniform', 'L2', [
            {'path': leaf, 'run': 'array', 'forward': 'absent'},
        ], {ident: 'run-only record'})
    with pytest.raises(pytest.fail.Exception, match='unexplained'):
        contract.classify(declaration, 'uniform', 'L2', [
            {'path': leaf, 'run': 'absent', 'forward': 'array'},
        ], {ident: 'run-only record'})


def test_witness_shape_change_is_unexplained():
    with pytest.raises(pytest.fail.Exception, match='unexplained.*settling_witness/residual'):
        contract.classify('P-waveguide', 'uniform', 'L2', [
            {'path': 'settling_witness/residual', 'bytes_equal': False, 'shape': [[2], [3]]},
        ], contract.known('P-waveguide', 'uniform', 'L2'))


def test_unmeasured_waveguide_flux_is_unexplained():
    with pytest.raises(pytest.fail.Exception, match='unexplained.*waveguide_port_flux'):
        contract.classify('P-waveguide', 'uniform', 'L2', [
            {'path': 'waveguide_port_flux/0', 'run': 'array', 'forward': 'absent'},
        ], {'D7': 'waveguide records'})


def test_time_series_consequence_still_applies(synthetic_consequence):
    declaration, ident = synthetic_consequence
    with pytest.raises(pytest.xfail.Exception):
        contract.classify(declaration, 'uniform', 'L2', [
            {'path': 'time_series', 'bytes_equal': False, 'shape': [[2], [2]]},
        ], {ident: 'time series consequence'})


def test_per_drive_same_shape_value_is_unexplained():
    with pytest.raises(pytest.fail.Exception, match='unexplained.*settling_witness/per_drive'):
        contract.classify('P-lumped', 'uniform', 'L2', [
            {'path': 'settling_witness/per_drive/0/error_per_bin',
             'bytes_equal': False, 'shape': [[2], [2]]},
        ], contract.known('P-lumped', 'uniform', 'L2'))


@pytest.mark.parametrize('prefix', ['settling_witness', 'settling_witness/runs/0'])
@pytest.mark.parametrize('column', [0, 1])
@pytest.mark.parametrize('key', ['error_per_bin', 'share_per_bin'])
@pytest.mark.parametrize('run,forward', [('array', 'absent'), ('absent', 'array')])
def test_unchecked_per_drive_leaf_is_unexplained(prefix, column, key, run, forward):
    assert 'D5' not in contract.ID_PATTERNS
    ids = contract.known('P-lumped2', 'uniform', 'L2', 'a')
    assert ids == {}
    with pytest.raises(pytest.fail.Exception, match='unexplained.*settling_witness/'):
        contract.classify('P-lumped2', 'uniform', 'L2', [
            {'path': f'{prefix}/per_drive/{column}/{key}', 'run': run, 'forward': forward},
        ], ids)
