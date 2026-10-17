"""Item 5 pole+CPML wrapped-step rows, using the original shared test bodies."""
import numpy as np
import pytest

from tests.unit.runners.test_distributed_nu_wrapped_step import (
    host_devices,
    test_cpml_one_slab_invariants as _one_slab,
    test_cpml_two_vs_three_bitwise_and_bars as _two_vs_three,
    test_cpml_selector as _selector,
    test_cpml_entries_reach_wrap as _entry,
    test_item6_first_scan_memory as _memory,
)


@pytest.mark.parametrize('name', ['debye'])
def test_cpml_one_slab_invariants(name, monkeypatch):
    _one_slab(name, monkeypatch)


@pytest.mark.parametrize('name', ['debye'])
def test_cpml_two_vs_three_bitwise_and_bars(name):
    _two_vs_three(name)


@pytest.mark.parametrize('debye,lorentz', [(True, False), (False, True), (True, True)])
def test_cpml_selector(debye, lorentz):
    _selector(debye, lorentz)


@pytest.mark.parametrize('name,mode', [(name, mode) for name in ('debye', 'lorentz')
                                    for mode in ('run', 'grad')])
def test_cpml_entries_reach_wrap(name, mode, monkeypatch):
    _entry(name, mode, monkeypatch)


def test_wrapped_setup_retains_cpml_with_poles(monkeypatch):
    """The public run entry builds a wrapped CPML step retaining Lorentz ADE."""
    from rfx.runners import _distributed_nu_wrapped as runner
    from tests.unit.runners.test_distributed_edge_averaged_materials import _cpml_1302_box
    original, seen = runner.split, []
    def observe(invariants, carry, *args, **kwargs):
        seen.append(('cpml_x' in invariants, 'lorentz_coeffs' in invariants,
                     'cpml' in carry, 'lorentz' in carry))
        return original(invariants, carry, *args, **kwargs)
    monkeypatch.setattr(runner, 'split', observe)
    sim = _cpml_1302_box('lorentz', 'fwd_graded')
    result = sim.run(n_steps=4, devices=host_devices(2), skip_preflight=True)
    assert np.isfinite(result.time_series).all()
    assert seen and all(all(parts) for parts in seen)



@pytest.mark.parametrize('boundary,devices', [('cpml_poles', 3)])
@pytest.mark.parametrize('mode', ('plain', 'grad'))
def test_item6_first_scan_memory(boundary, devices, mode):
    _memory(boundary, devices, mode)
