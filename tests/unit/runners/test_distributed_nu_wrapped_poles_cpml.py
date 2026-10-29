"""Item 5 pole+CPML wrapped-step rows, using the original shared test bodies."""
import pytest

from tests._wrapped_lorentz_cpml import _lorentz_scene, _loss_at_kernel

from tests.unit.runners.test_distributed_nu_wrapped_step import test_cpml_one_slab_invariants as _one_slab, test_cpml_two_vs_three_bitwise_and_bars as _two_vs_three, test_cpml_entries_reach_wrap as _entry, test_item6_first_scan_memory as _memory


def _cpml_one_slab_invariants(name, monkeypatch):
    if name == 'debye':
        _one_slab(name, monkeypatch)
        return
    from tests.unit.runners import test_distributed_nu_wrapped_step as shared
    with monkeypatch.context() as patch:
        patch.setattr(shared, 'cpml_scene', lambda _: _lorentz_scene(name))
        _one_slab(name, patch)
    _loss_at_kernel(name, monkeypatch)


@pytest.mark.parametrize('name', ['debye'])
def test_cpml_two_vs_three_bitwise_and_bars(name):
    _two_vs_three(name)


@pytest.mark.parametrize('name,mode', [(name, mode) for name in ('debye', 'lorentz')
                                    for mode in ('run', 'grad')])
def test_cpml_entries_reach_wrap(name, mode, monkeypatch):
    _entry(name, mode, monkeypatch)


@pytest.mark.parametrize('boundary,devices', [('cpml_poles', 3)])
@pytest.mark.parametrize('mode', ('plain', 'grad'))
def test_item6_first_scan_memory(boundary, devices, mode):
    _memory(boundary, devices, mode)
