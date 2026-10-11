"""Hand-written freezer/comparison falsifiers; no solver measurements."""
import numpy as np
from tests.contracts.run_forward_census import freeze, compare, compare_l2


def test_one_element_array_change():
    result = compare(freeze(np.array([1., 2., 3.], dtype=np.float32)),
                     freeze(np.array([1., 4., 3.], dtype=np.float32)))
    assert len(result) == 1
    assert result[0]['differing_elements'] == 1
    assert result[0]['max_abs_difference'] == 2.
    assert result[0]['peak'] == 4.


def test_equal_nested_dicts():
    assert compare(freeze({'a': [1, {'b': (None, 3)}]}),
                   freeze({'a': [1, {'b': (None, 3)}]})) == []


def test_closure_change():
    def make(value):
        return lambda: value
    assert compare(freeze(make(1)), freeze(make(2))) == [
        {'path': '/closure/0', 'run': '1', 'forward': '2'}]


def test_unfrozen_is_never_equal():
    assert compare(freeze(object()), freeze(object())) == [
        {'path': '', 'run': "'UNFROZEN:object'", 'forward': "'UNFROZEN:object'", 'unfrozen': True}]


def test_signed_zero_and_shape():
    diff = compare(freeze(np.array([0.], np.float32)), freeze(np.array([-0.], np.float32)))
    assert diff[0]['differing_elements'] == 1
    assert diff[0]['max_abs_difference'] == 0.
    assert len(compare(freeze(np.zeros((1,), np.float32)), freeze(np.zeros((1, 1), np.float32)))) == 1


def test_float32_peak_ulp():
    left = {'x': freeze(np.array([1.], np.float32))}
    right = {'x': freeze(np.array([1.00000011920928955078125], np.float32))}
    result = compare_l2(left, right)
    assert result['observables']['x']['float32_peak_ulps'] == 1.
    assert result['run_only'] == result['forward_only'] == []
