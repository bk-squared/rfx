"""Missing device rank must fail before a slab silently loses its x faces."""
import inspect

import pytest

from rfx.runners import _distributed_common as common, distributed_nu as nu, distributed_v2 as uniform


_SHMAPS = sorted({fn for module in (common, nu, uniform) for name, fn in vars(module).items()
                  if name.endswith("_shmap") and inspect.isfunction(fn)},
                 key=lambda fn: fn.__name__)


@pytest.mark.parametrize("helper", _SHMAPS, ids=lambda fn: fn.__name__)
def test_shard_helpers_require_rank_input(helper):
    # Supply every other required parameter; missing rank must fail at binding,
    # before these sentinel values can enter a numerical operation.
    args = {name: None for name, p in inspect.signature(helper).parameters.items()
            if p.default is inspect.Parameter.empty and name != "ranks"}
    with pytest.raises(TypeError, match="required keyword-only argument: 'ranks'"):
        helper(**args)


@pytest.mark.parametrize("graded", [False, True])
def test_cpml_rejects_absent_rank(graded, monkeypatch):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rfx import Simulation

    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2
    jax.clear_caches()
    module = nu if graded else uniform
    calls = []
    for name in ("apply_cpml_e", "apply_cpml_h"):
        original = getattr(module, name)
        def observe(*args, _original=original, _name=name, **kwargs):
            assert kwargs.get("frame") is not None
            owned = kwargs.get("owned")
            assert owned is not None and set(owned) == {"x_lo", "x_hi"}
            for value in owned.values():
                assert not isinstance(value, bool)
                assert isinstance(value, (jax.Array, jax.core.Tracer))
                assert value.shape == () and value.dtype == jnp.bool_
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, observe)
    extra = {"dz_profile": np.array([1., 1., .8, 1.2, 1., 1.]) * .001} if graded else {}
    sim = Simulation(freq_max=15e9, domain=(.012, .008, .006), dx=.001,
                     boundary="cpml", cpml_layers=2, **extra)
    result = sim.run(n_steps=2, devices=devices, skip_preflight=True)
    jax.block_until_ready(result.state)
    assert set(calls) == {"apply_cpml_e", "apply_cpml_h"}
