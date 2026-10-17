"""Lorentz CPML operand rows, using the shared one-slab invariant body."""
import pytest

from tests.unit.runners.test_distributed_nu_wrapped_poles_cpml import _cpml_one_slab_invariants


@pytest.mark.parametrize('name', ['lorentz', 'lorentz_lossy', 'lorentz_lossy_yz'])
def test_cpml_one_slab_invariants(name, monkeypatch):
    _cpml_one_slab_invariants(name, monkeypatch)
