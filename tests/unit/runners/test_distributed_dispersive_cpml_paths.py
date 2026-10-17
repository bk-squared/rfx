"""Item 5 gradient, traced entry paths, and lumped-port spectrum."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._distributed_dispersive_cpml import check, measure, record, scene

pytestmark = pytest.mark.distributed


@pytest.mark.parametrize('entry', ['vmap', 'remat', 'jit'])
def test_traced_entries(entry, monkeypatch):
    from rfx.runners import _distributed_nu_wrapped as runner
    sim = scene('debye')
    eps = jnp.asarray(sim._assemble_materials_nu(sim._build_nonuniform_grid())[0].eps_r)
    original, calls = runner.build_setup, []
    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'build_setup', observe)
    def forward(e, devices=None):
        kw = {} if devices is None else dict(distributed=True, devices=devices)
        return sim.forward(eps_override=e, n_steps=400, checkpoint=False,
            skip_preflight=True, **({'checkpoint_every': 50} if entry == 'remat' else {}),
            **kw).time_series
    def evaluate(devices=None):
        f = lambda e: forward(e, devices)
        if entry == 'vmap':
            return jax.vmap(f)(eps[None])[0]
        return jax.jit(f)(eps) if entry == 'jit' else f(eps)
    one = np.asarray(evaluate())
    calls.clear()
    multi = np.asarray(evaluate(jax.devices('cpu')[:2]))
    assert calls
    row = dict(kind='debye', entry=entry, n=400, devices=2,
               **measure(one, multi, float(sim._build_nonuniform_grid().dt)))
    record(row)
    check(row)


