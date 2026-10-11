"""Hand-written freezer/comparison falsifiers; no solver measurements."""
from enum import Enum, IntEnum
from functools import partial

import jax
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


def test_frozensets_by_value():
    assert freeze(frozenset([3, 1, 2])) == ('set', (1, 2, 3))
    assert compare(freeze(frozenset([3, 1, 2])), freeze(frozenset([2, 1, 3]))) == []
    assert compare(freeze(frozenset([1, 2])), freeze(frozenset([1, 3]))) == [
        {'path': '/1/1', 'run': '2', 'forward': '3'}]
    assert freeze({2, 1}) == ('set', (1, 2))


def test_slices_by_value():
    assert freeze(slice(1, 5)) == ('slice', 1, 5, None)
    assert compare(freeze(slice(1, 5)), freeze(slice(1, 6))) == [
        {'path': '/2', 'run': '5', 'forward': '6'}]


def test_partials_by_value():
    def identity(value=0):
        return value
    assert compare(freeze(partial(identity, value=1)), freeze(partial(identity, value=2))) == [
        {'path': '/3/value', 'run': '1', 'forward': '2'}]


def test_cpu_devices_by_value(two_devices):
    devices = two_devices
    assert freeze(devices[0]) == ('device', 'cpu', 0)
    assert freeze(devices[1]) == ('device', 'cpu', 1)
    assert compare(freeze(devices[0]), freeze(devices[1])) == [
        {'path': '/2', 'run': '0', 'forward': '1'}]


def test_enums_by_value():
    class Label(Enum):
        ONE = 'one'
    class Number(IntEnum):
        ONE = 1
    assert freeze(Label.ONE) == ('Label', 'ONE')
    assert freeze(Number.ONE) == ('Number', 'ONE')


def test_observable_errors_remain_one_sided():
    result = compare_l2({'x': freeze(np.array([1.], np.float32))}, {},
                        right_raises={'x': 'raises: ValueError: unavailable'})
    assert result['observables'] == {}
    assert result['run_only'] == ['x']
    assert result['forward_only'] == []
    assert result['one_sided'] == {'x': {'run': 'array', 'forward': 'raises: ValueError: unavailable'}}


def test_one_value_batch_primal_is_frozen_by_value():
    captured = []
    def observe(value):
        captured.append(freeze(value))
        return value
    jax.vmap(observe)(np.array([[1., 3.]], np.float32))
    assert len(captured) == 1
    assert captured[0].dtype == '<f4'
    assert captured[0].shape == (2,)
    assert captured[0].data == b'\x00\x00\x80\x3f\x00\x00\x40\x40'


def test_ad_primal_is_frozen_by_value():
    captured = []
    def observe(value):
        captured.append(freeze(value))
        return value * value
    jax.grad(observe)(jax.numpy.float32(2.))
    assert len(captured) == 1
    assert captured[0].dtype == '<f4'
    assert captured[0].shape == ()
    assert captured[0].data == b'\x00\x00\x00\x40'


def test_record_classes_are_preserved():
    from dataclasses import make_dataclass
    from collections import namedtuple
    First = make_dataclass('First', [('value', int)])
    Second = make_dataclass('Second', [('value', int)])
    assert compare(freeze(First(1)), freeze(Second(1)))[0]['path'] == '/__class__'
    assert freeze(First(1))['__class__'].endswith('.First')
    A = namedtuple('A', 'value')
    B = namedtuple('B', 'value')
    assert compare(freeze(A(1)), freeze(B(1)))[0]['path'] == '/__class__'
    assert freeze(A(1))['__class__'].endswith('.A')


def test_numpy_scalar_dtype_is_preserved():
    assert freeze(np.int32(1)).dtype == '<i4'
    assert freeze(np.int64(1)).dtype == '<i8'
    assert compare(freeze(np.int32(1)), freeze(np.int64(1)))
    assert compare(freeze(np.float32(1)), freeze(np.float64(1)))


def test_callable_identity_and_bound_owner():
    def first():
        return 1
    def second():
        return 1
    assert freeze(first)['module'] == __name__
    assert freeze(first)['qualified_name'] == 'test_callable_identity_and_bound_owner.<locals>.first'
    assert compare(freeze(first), freeze(second))
    second.__qualname__ = first.__qualname__
    second.__module__ = 'different_module'
    assert compare(freeze(first), freeze(second))[0]['path'] == '/module'
    class Parent:
        def method(self):
            return 1
    class Child(Parent):
        pass
    assert freeze(Parent().method)['owner_class'].endswith('.Parent')
    assert compare(freeze(Parent().method), freeze(Child().method))[0]['path'] == '/owner_class'
