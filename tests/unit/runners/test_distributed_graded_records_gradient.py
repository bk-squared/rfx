"""G4: public graded eps overrides differentiate through assembled records."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rfx.farfield import compute_far_field_jax
from rfx.probes.probes import flux_spectrum
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import radiating_model, measure, record, mutation_result
from tests.unit.runners.test_distributed_graded_records_ntff import add_box


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('kind', ['flux', 'ntff'])
def test_gradient(n_devices, kind):
    sim, grid, seam = radiating_model('cpml', n_devices)
    if kind == 'ntff':
        add_box(sim)
    else:
        sim.add_flux_monitor(axis='y', coordinate=4e-3, freqs=[3e9], name='flux')
    eps = jnp.full(grid.shape, 1.3, jnp.float32)
    def objective(e, distributed):
        result = sim.forward(eps_override=e, n_steps=16, checkpoint=False, skip_preflight=True,
            **(dict(distributed=True, devices=host_devices(n_devices), checkpoint_every=5)
               if distributed else {}))
        if kind == 'flux':
            return jnp.sum(flux_spectrum(result.flux_monitors['flux']))
        ff = compute_far_field_jax(result.ntff_data, result.ntff_box, result.grid,
                                  jnp.linspace(0, jnp.pi, 5), jnp.linspace(0, 2*jnp.pi, 7))
        return jnp.sum(jnp.abs(ff.E_theta)**2 + jnp.abs(ff.E_phi)**2)
    expected = jax.grad(lambda e: objective(e, False))(eps)
    actual = jax.grad(lambda e: objective(e, True))(eps)
    values = measure(actual, expected)
    record('gradient', dict(devices=n_devices, boundary='cpml', observable=kind), values)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), 'gradient finite'
    assert np.max(np.abs(actual)) > 0 and values['peak'] > 0, 'gradient nonzero'
    assert values['max_abs'] <= 1e-4 * values['peak'], ('gradient bar', values)


@pytest.mark.parametrize('mutation', ['stop_gradient', 'double_gradient'])
def test_gradient_mutations(mutation, monkeypatch):
    from rfx.runners._distributed_records import SlabRecords
    original = SlabRecords.assemble
    def wrong(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        change = jax.lax.stop_gradient if mutation == 'stop_gradient' else lambda a: 2*a-jax.lax.stop_gradient(a)
        result['ntff_data'] = jax.tree.map(change, result['ntff_data'])
        return result
    monkeypatch.setattr(SlabRecords, 'assemble', wrong)
    with pytest.raises(AssertionError, match='gradient (bar|nonzero)') as failure:
        test_gradient(2, 'ntff')
    mutation_result('gradient', mutation, failure.value)
