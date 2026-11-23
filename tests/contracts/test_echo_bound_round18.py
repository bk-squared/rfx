"""Concrete declarations need no callback; live declarations admit host reads."""
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


def test_concrete_forward_override_and_material_build_no_echo_callback(monkeypatch):
    from rfx.ringdown import _jax

    sim = box(jnp.float32(2.2))
    override = jnp.full(sim._build_realized_grid().shape, jnp.float32(2.2))
    calls = []
    original = _jax.host_echo_index

    def observed(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(_jax, 'host_echo_index', observed)
    result = sim.forward(n_steps=4, eps_override=override, skip_preflight=True)
    jax.block_until_ready(result)
    assert len(calls) == 0, 'concrete material leaves built host_echo_index'
    reference = sim.run(n_steps=4, skip_preflight=True)
    assert result.settling_echo_bound == reference.settling_echo_bound
    assert result.settling_echo_bound['n_max'] == float(np.sqrt(np.float64(np.float32(2.2))))
    assert all(type(value) is float for value in result.settling_echo_bound.values())


def test_fixed_declaration_keeps_realized_bound_under_eval_shape(monkeypatch):
    from rfx.ringdown import _jax

    sim = box(jnp.float32(2.2))
    override = jnp.full(sim._build_realized_grid().shape, jnp.float32(2.2))

    def forbidden(*args):
        pytest.fail('a fixed declaration built host_echo_index')

    monkeypatch.setattr(_jax, 'host_echo_index', forbidden)

    def forward():
        return sim.forward(n_steps=4, eps_override=override, skip_preflight=True)

    eager = forward()
    abstract = jax.eval_shape(forward)
    assert abstract.settling_echo_bound == eager.settling_echo_bound
    assert all(type(value) is float for value in abstract.settling_echo_bound.values())


@pytest.mark.parametrize('operand', ['material', 'override'])
def test_live_echo_index_refuses_without_a_cpu_device(monkeypatch, operand):
    from rfx import _traced_check
    from rfx.backend import backend_class
    from rfx.measurement._echo_forward import echo_override_declaration, forward_echo_bound
    from rfx.ringdown import _jax

    grid = box(2.2)._build_realized_grid()
    monkeypatch.setattr(_traced_check, 'backend_class', lambda: backend_class('gpu'))
    monkeypatch.setenv('JAX_PLATFORMS', 'cuda')

    def no_cpu():
        raise RuntimeError('Unknown backend cpu')

    def forbidden(*args):
        pytest.fail('callback built before host admission')

    monkeypatch.setattr(_traced_check, 'host_cpu_device', no_cpu)
    monkeypatch.setattr(_jax, 'host_echo_index', forbidden)

    def read(value):
        sim = box(value if operand == 'material' else 2.2)
        if operand == 'override':
            sim = echo_override_declaration(sim, dict(eps_override=value))
        return forward_echo_bound(grid, sim, None)[0]

    with pytest.raises(RuntimeError) as caught:
        jax.make_jaxpr(read)(jnp.float32(2.2))
    assert str(caught.value) == (
        "forward: echo delay bound: a host check inside compiled code runs on this "
        "process's CPU device, and JAX has none (JAX_PLATFORMS='cuda'); add cpu, "
        "e.g. JAX_PLATFORMS=cuda,cpu")
