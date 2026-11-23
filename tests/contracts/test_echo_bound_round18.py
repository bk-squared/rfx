"""The realized bound has one array tree and no host callback in any context."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contracts.test_echo_bound_round11 import box


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    jax.clear_caches()


def test_concrete_forward_override_and_material_build_no_echo_callback():
    sim = box(jnp.float32(2.2))
    override = jnp.full(sim._build_realized_grid().shape, jnp.float32(2.2))
    result = sim.forward(n_steps=4, eps_override=override, skip_preflight=True)
    reference = sim.run(n_steps=4, skip_preflight=True)
    for key in ('D', 'g', 'n_max', 'tau_max'):
        np.testing.assert_array_equal(result.settling_echo_bound[key], reference.settling_echo_bound[key])
        assert np.shape(result.settling_echo_bound[key]) == ()
    expected = np.nextafter(np.sqrt(np.float32(2.2)), np.float32(np.inf))
    assert result.settling_echo_bound['n_max'] == expected


def test_fixed_declaration_keeps_realized_bound_under_eval_shape():
    sim = box(jnp.float32(2.2))
    def forward():
        return sim.forward(n_steps=4, skip_preflight=True)
    eager, abstract = forward(), jax.eval_shape(forward)
    assert set(abstract.settling_echo_bound) == set(eager.settling_echo_bound) == {'D', 'g', 'n_max', 'tau_max'}
    for key, value in abstract.settling_echo_bound.items():
        assert value.shape == np.shape(eager.settling_echo_bound[key])


@pytest.mark.parametrize('context', ['eager', 'jit', 'vmap', 'eval_shape', 'grad'])
def test_live_echo_bound_has_no_callback(context):
    sim = box(2.2)
    grid = sim._build_realized_grid()
    from rfx.core.yee import init_materials
    from rfx.measurement.echo_bound import realized_echo_bound
    def read(value):
        mats = init_materials(grid.shape)._replace(eps_r=jnp.full(grid.shape, value))
        return realized_echo_bound(grid, mats)
    value = jnp.float32(2.2)
    operation, operand = read, value
    if context == 'jit':
        operation = jax.jit(read)
    elif context == 'vmap':
        operation, operand = jax.vmap(read), jnp.array([2.2, 3.])
    elif context == 'eval_shape':
        abstract = jax.eval_shape(read, value)
        assert set(abstract) == {'D', 'g', 'n_max', 'tau_max'}
    elif context == 'grad':
        operation = jax.grad(lambda v: read(v)['tau_max'])
        assert operation(value) == 0.
    operation(operand)
    lowered = jax.jit(operation).lower(operand).as_text()
    assert lowered.count('pure_callback') == 0, lowered
