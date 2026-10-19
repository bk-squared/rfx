"""Explicit one-device selection keeps the pre-existing native record path."""
import jax
import numpy as np
import pytest
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import radiating_model, measure, record, mutation_result
from tests.unit.runners.test_distributed_graded_records_ntff import add_box
from tests.unit.runners.test_distributed_graded_records_planes import add_planes
from tests.unit.runners.test_distributed_graded_records_flux import add_flux


def test_one_device(monkeypatch):
    from rfx.runners._distributed_records import SlabRecords
    monkeypatch.setattr(SlabRecords, '__init__', lambda *a, **kw: pytest.fail('one device reached slab adapter'))
    sim, grid, seam = radiating_model('cpml', 2, grading='z')
    add_planes(sim, grid, seam)
    add_flux(sim, grid, seam)
    add_box(sim)
    sim.add_current_moment_monitor((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 7e-3),
        block_size=2e-3, freqs=[3e9, 7e9])
    one = sim.run(n_steps=12, skip_preflight=True)
    explicit = sim.run(n_steps=12, devices=host_devices(1), skip_preflight=True)
    values = {}
    for key in ('state', 'dft_planes', 'flux_monitors', 'ntff_data', 'current_moment_data'):
        a, b = jax.tree.leaves(getattr(explicit, key)), jax.tree.leaves(getattr(one, key))
        for i, (got, expected) in enumerate(zip(a, b)):
            if isinstance(expected, (jax.Array, np.ndarray)):
                values[f'{key}/{i}'] = measure(got, expected)
            else:
                assert got == expected
    record('one_device', dict(devices=1, boundary='cpml'), values)
    assert all(v['equal'] for v in values.values()), ('one-device byte equality', values)


@pytest.mark.parametrize('mutation', ['plane', 'ntff'])
def test_one_device_mutations(mutation, monkeypatch):
    from rfx import Simulation
    original = Simulation.run
    def wrong(self, **kwargs):
        result = original(self, **kwargs)
        if kwargs.get('devices') is not None:
            if mutation == 'ntff':
                result = result._replace(ntff_data=jax.tree.map(lambda a: 2*a, result.ntff_data))
            else:
                result = result._replace(dft_planes={k: p._replace(accumulator=2*p.accumulator)
                                                    for k, p in result.dft_planes.items()})
        return result
    monkeypatch.setattr(Simulation, 'run', wrong)
    with pytest.raises(AssertionError, match='one-device byte equality') as failure:
        test_one_device(monkeypatch)
    mutation_result('one_device', mutation, failure.value)
