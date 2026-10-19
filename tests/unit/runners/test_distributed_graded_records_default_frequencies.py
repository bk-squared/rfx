"""Default plane/flux bins and records match the one-device public calls."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import measure, record, mutation_result, with_bars


def scene(mesh, kind):
    sim = Simulation(freq_max=15e9, domain=(15.3e-3, 8.7e-3, 10.1e-3), dx=1e-3,
        boundary='pec', **(dict(dx_profile=np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 13), 1e-3])
                          if mesh == 'graded' else {}))
    sim.add_source((5.1e-3, 2.7e-3, 4.3e-3), 'ez', amplitude_kind='field',
        waveform=lambda t: jnp.exp(-((t-1.2e-11)/5e-12)**2))
    sim.add_probe((10.2e-3, 5.6e-3, 6.7e-3), 'ez')
    if kind == 'planes':
        sim.add_dft_plane_probe(axis='y', coordinate=3.1e-3, component='ez', name='default')
    else:
        sim.add_flux_monitor(axis='y', coordinate=3.1e-3, name='default')
    return sim


def compare(one, many, kind, **metadata):
    a, b = ((many.dft_planes['default'], one.dft_planes['default']) if kind == 'planes'
            else (many.flux_monitors['default'], one.flux_monitors['default']))
    names = ('freqs', 'accumulator') if kind == 'planes' else (
        'freqs', 'e1_dft', 'e2_dft', 'h1_dft', 'h2_dft', 'dA', 'dA2')
    values = {name: measure(getattr(a, name), getattr(b, name)) for name in names}
    if one.freqs is None:
        assert many.freqs is None
    else:
        values['result.freqs'] = measure(many.freqs, one.freqs)
    with_bars(values, 'array_equal')
    record('default_frequencies', dict(record=kind, **metadata), values)
    assert max(v['peak'] for name, v in values.items() if name not in ('freqs', 'dA', 'dA2')) > 0
    assert all(v['passed'] for v in values.values()), ('default frequency bar', values)


@pytest.mark.parametrize('mesh,kind', [('graded', 'planes'), ('graded', 'flux'), ('uniform', 'planes')])
@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('n_devices', [2, 3])
def test_default_records(mesh, kind, entry, n_devices):
    devices = host_devices(n_devices)
    sim = scene(mesh, kind)
    one = getattr(sim, entry)(n_steps=24, skip_preflight=True)
    many = getattr(sim, entry)(n_steps=24, skip_preflight=True, devices=devices,
                              **({'distributed': True} if entry == 'forward' else {}))
    jax.block_until_ready(many)
    compare(one, many, kind, mesh=mesh, entry=entry, devices=n_devices)


def test_uniform_run_forward_default_planes():
    sim = scene('uniform', 'planes')
    one = sim.run(n_steps=24, skip_preflight=True)
    forward = sim.forward(n_steps=24, skip_preflight=True)
    jax.block_until_ready(forward)
    compare(one, forward, 'planes', mesh='uniform', entry='run_vs_forward', devices=1)


def bad_grid(mutation):
    def grid(freq_max, n_freqs, **kwargs):
        if mutation == 'folded_grid':
            with jax.ensure_compile_time_eval():
                return jnp.linspace(freq_max / 10, freq_max, n_freqs)
        return jnp.asarray(np.linspace(freq_max / 10, freq_max, n_freqs))
    return grid


@pytest.mark.parametrize('mesh,kind', [('graded', 'planes'), ('graded', 'flux'), ('uniform', 'planes')])
@pytest.mark.parametrize('mutation', ['folded_grid', 'numpy_grid'])
def test_default_record_mutations(mesh, kind, mutation, monkeypatch):
    from rfx import measurement
    from rfx.runners._distributed_records import SlabRecords
    original = SlabRecords.__init__

    def wrong(self, *args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(measurement, 'default_frequency_grid', bad_grid(mutation))
            original(self, *args, **kwargs)

    monkeypatch.setattr(SlabRecords, '__init__', wrong)
    with pytest.raises(AssertionError, match='default frequency bar') as failure:
        test_default_records(mesh, kind, 'run', 2)
    mutation_result('default_frequencies', f'{mesh}/{kind}/{mutation}', failure.value)


@pytest.mark.parametrize('mutation', ['folded_grid', 'numpy_grid'])
def test_uniform_run_forward_default_mutations(mutation, monkeypatch):
    from rfx import measurement
    from rfx.measurement import setup
    original = setup.build_planes

    def wrong(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(measurement, 'default_frequency_grid', bad_grid(mutation))
            return original(*args, **kwargs)

    monkeypatch.setattr(setup, 'build_planes', wrong)
    with pytest.raises(AssertionError, match='default frequency bar') as failure:
        test_uniform_run_forward_default_planes()
    mutation_result('default_frequencies', f'uniform/run_vs_forward/{mutation}', failure.value)
