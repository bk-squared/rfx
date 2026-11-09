"""Item 5 pole+CPML wrapped-step rows, using the original shared test bodies."""
import pytest

from tests.unit.runners.test_distributed_nu_wrapped_step import (
    test_cpml_two_vs_three_bitwise_and_bars as _two_vs_three,
    test_item6_first_scan_memory as _memory,
)


@pytest.mark.parametrize('name', ['debye'])
def test_cpml_two_vs_three_bitwise_and_bars(name):
    _two_vs_three(name)


@pytest.mark.parametrize('boundary,devices', [('cpml_poles', 3)])
@pytest.mark.parametrize('mode', ('plain', 'grad'))
def test_item6_first_scan_memory(boundary, devices, mode):
    _memory(boundary, devices, mode)
