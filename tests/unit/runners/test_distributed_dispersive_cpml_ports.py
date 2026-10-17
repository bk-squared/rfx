"""Item 5 gradient, traced entry paths, and lumped-port spectrum."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._distributed_dispersive_cpml import check, measure, record, scene

pytestmark = pytest.mark.distributed


def test_debye_cpml_lumped_port_s():
    entry = 'forward'
    sim = scene('debye', cells=False, port=True)
    bins = np.linspace(2e9, 10e9, 20)
    def evaluate(devices=None):
        kw = {} if devices is None else dict(distributed=True, devices=devices)
        return sim.forward(n_steps=400, checkpoint=False, skip_preflight=True,
                           port_s11_freqs=bins, **kw)
    one = np.asarray(evaluate().s_params, np.complex128)
    assert np.isfinite(one).all() and np.any(one)
    for devices in (2, 3):
        multi = np.asarray(evaluate(jax.devices('cpu')[:devices]).s_params, np.complex128)
        assert np.isfinite(multi).all() and one.shape == multi.shape
        relative = float(np.abs(multi - one).max() / np.abs(one).max())
        row = dict(kind='debye', entry='port_' + entry, n=400, devices=devices,
                   relative=relative)
        record(row)
        assert relative <= 1e-6, row


def test_run_lumped_s_retains_one_device_refusal():
    """Item 5 leaves the one-device graded lumped-S run API unchanged."""
    sim = scene('debye', cells=False, port=True)
    for devices in (None, jax.devices('cpu')[:2], jax.devices('cpu')[:3]):
        with pytest.raises(NotImplementedError, match='only for add_port') as exc:
            sim.run(n_steps=400, devices=devices, skip_preflight=True,
                    compute_s_params=True, s_param_freqs=np.linspace(2e9, 10e9, 20))
        record(dict(kind='debye', entry='port_run', devices=1 if devices is None else len(devices),
                    status='refuses', reason=str(exc.value)))
