"""Hand-written difference/outcome records falsify exception classification."""
import pytest

from tests.contracts import test_run_forward_one_pipeline as contract


def test_d4_fixed_while_d5_open():
    with pytest.raises(pytest.fail.Exception, match='D4 no longer applies to O-flux/uniform/L2'):
        contract.classify('O-flux', 'uniform', 'L2', [
            {'path': 'settling_witness/residual', 'bytes_equal': False, 'shape': [[1], [2]]},
        ], {'D4': 'flux absent', 'D5': 'witness shape'})


@pytest.mark.parametrize('declaration,ident', [('R-rlc', 'D6')])
def test_consequences_do_not_explain_witness(declaration, ident):
    with pytest.raises(pytest.fail.Exception, match='unexplained.*settling_witness/residual'):
        contract.classify(declaration, 'uniform', 'L2', [
            {'path': 'settling_witness/residual', 'bytes_equal': False, 'shape': [[1], [2]]},
        ], {ident: 'field cause'})


@pytest.mark.parametrize('declaration,ident,leaf', [
    ('O-flux', 'D4', 'flux_monitors/0/flux'),
    ('P-waveguide', 'D7', 'waveguide_sparams/0/S'),
    ('P-lumped', 'D8', 's_params'),
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
        ], {'D5': 'witness shape'})


@pytest.mark.parametrize('declaration,ident,leaf', [
    ('O-flux', 'D4', 'flux_monitors/0/flux'),
    ('P-waveguide', 'D7', 'waveguide_ports/0/voltage'),
    ('P-lumped', 'D8', 's_params'),
    ('P-waveguide', 'D5', 'settling_witness/runs/0/residual'),
    ('P-lumped', 'D5', 'settling_witness/per_drive/0/error_per_bin'),
    ('P-lumped2', 'D5', 'settling_witness/per_drive/1/share_per_bin'),
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


def test_witness_shape_change_is_explained():
    with pytest.raises(pytest.xfail.Exception):
        contract.classify('P-waveguide', 'uniform', 'L2', [
            {'path': 'settling_witness/residual', 'bytes_equal': False, 'shape': [[2], [3]]},
        ], {'D5': 'witness shape'})


def test_unmeasured_waveguide_flux_is_unexplained():
    with pytest.raises(pytest.fail.Exception, match='unexplained.*waveguide_port_flux'):
        contract.classify('P-waveguide', 'uniform', 'L2', [
            {'path': 'waveguide_port_flux/0', 'run': 'array', 'forward': 'absent'},
        ], {'D7': 'waveguide records'})


@pytest.mark.parametrize('declaration,ident', [('R-rlc', 'D6')])
def test_time_series_consequence_still_applies(declaration, ident):
    with pytest.raises(pytest.xfail.Exception):
        contract.classify(declaration, 'uniform', 'L2', [
            {'path': 'time_series', 'bytes_equal': False, 'shape': [[2], [2]]},
        ], {ident: 'time series consequence'})


def test_per_drive_same_shape_value_is_unexplained():
    with pytest.raises(pytest.fail.Exception, match='unexplained.*settling_witness/per_drive'):
        contract.classify('P-lumped', 'uniform', 'L2', [
            {'path': 'settling_witness/per_drive/0/error_per_bin',
             'bytes_equal': False, 'shape': [[2], [2]]},
        ], {'D5': 'witness channels'})
