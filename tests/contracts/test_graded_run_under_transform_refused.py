"""#2241 stopgap: reject transformed graded run before assembly; use forward.

The small 6 x 6 x 2 mm box follows test_port_records_always.model, with a
point source and four steps. The admission expectations and message come
from the #2241 design, independently of the implementation's trace helper.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


REFUSAL = (
    "run() on a graded (non-uniform) mesh called inside a JAX "
    "transformation (jax.jit, vmap, grad, ...) gives a different result "
    "from plain run() (rfx#2241). Call forward() inside the "
    "transformation; forward() under jax.jit is bit-identical to "
    "plain forward() and to plain run()."
)


def scene(*, graded=True):
    sim = Simulation(freq_max=10e9, domain=(.006, .006, .002), dx=.001,
                     boundary="pec", cpml_layers=0,
                     **({"dz_profile": np.array([.0008, .0012])} if graded else {}))
    position = (.003, .003, .0008 if graded else .001)
    sim.add_source(position, "ez", amplitude_kind="field", waveform=lambda t: jnp.cos(t * 2e10))
    sim.add_probe(position, "ez")
    return sim


def transformed(transform, objective):
    if transform == "jit":
        return jax.jit(objective)(1.)
    if transform == "vmap":
        return jax.vmap(objective)(jnp.array([1., 2.]))
    if transform == "grad":
        return jax.grad(objective)(1.)
    if transform == "jvp":
        return jax.jvp(objective, (1.,), (1.,))
    if transform == "checkpoint":
        return jax.checkpoint(objective)(1.)
    assert transform == "scan"
    return jax.lax.scan(lambda carry, x: (carry, objective(x)), 0., jnp.ones(2))


@pytest.mark.parametrize("transform", ["jit", "vmap", "grad", "jvp", "checkpoint", "scan"])
@pytest.mark.parametrize("distributed", [False, True], ids=["single", "distributed_nu"])
@pytest.mark.parametrize("skip_preflight", [False, True])
def test_graded_run_refused_before_assembly(monkeypatch, transform, distributed, skip_preflight):
    sim = scene()
    options = {}
    if distributed:
        devices = jax.devices("cpu")
        if len(devices) < 2:
            pytest.skip("requires two CPU devices")
        options["devices"] = devices[:2]

    def assembly_forbidden(*args, **kwargs):
        pytest.fail("transformed graded run reached preflight/assembly")

    monkeypatch.setattr(sim, "_auto_preflight", assembly_forbidden)

    def objective(scale):
        # Even a closed-over model with no traced inputs must be refused.
        result = sim.run(n_steps=4, skip_preflight=skip_preflight, **options)
        return scale * jnp.sum(jnp.asarray(result.time_series) ** 2)

    with pytest.raises(ValueError) as caught:
        transformed(transform, objective)
    assert type(caught.value) is ValueError
    assert str(caught.value) == REFUSAL


@pytest.mark.parametrize("entry,graded,compiled", [
    ("run", True, False),
    ("forward", True, True),
    ("run", False, True),
], ids=["plain_graded_run", "jitted_graded_forward", "jitted_uniform_run"])
def test_supported_routes_are_not_refused(entry, graded, compiled):
    sim = scene(graded=graded)

    def call():
        return getattr(sim, entry)(n_steps=4, skip_preflight=True).time_series

    values = np.asarray(jax.jit(call)() if compiled else call())
    assert values.shape == (4, 1)
    assert np.isfinite(values).all()
    assert np.any(values != 0)
