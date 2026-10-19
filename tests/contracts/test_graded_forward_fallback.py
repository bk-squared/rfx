"""A5: the public one-device forward is the independent byte-level oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contracts.run_forward_census import BUILDERS, observables


CASES = [('P-waveguide', 'waveguide ports'), ('PW-A', 'TFSF plane-wave sources')]


def assert_bytes(actual, expected, path):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, path
    assert actual.dtype == expected.dtype, path
    assert np.array_equal(np.frombuffer(actual.tobytes(), dtype=np.uint8),
                          np.frombuffer(expected.tobytes(), dtype=np.uint8)), path


@pytest.mark.parametrize('declaration,reason', CASES)
@pytest.mark.parametrize('skip_preflight', [False, True])
def test_fields_records_s_and_diagnostic(declaration, reason, skip_preflight, two_devices, monkeypatch):
    from rfx.runners import nonuniform

    # ForwardResult publishes probe fields, not the final six field volumes.
    # Observe the runner's returned state as well, without changing its solve.
    states = []
    original = nonuniform.run_nonuniform_path

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(result.state)
        return result

    monkeypatch.setattr(nonuniform, 'run_nonuniform_path', record)
    kwargs = dict(n_steps=80, checkpoint=False, skip_preflight=skip_preflight)
    expected = BUILDERS[declaration]('graded').forward(**kwargs)
    message = (f'Distributed runner does not yet support {reason}. '
               'Falling back to single-device execution with the same arguments.')
    with pytest.warns(UserWarning, match='Falling back to single-device') as caught:
        actual = BUILDERS[declaration]('graded').forward(
            distributed=True, devices=two_devices, **kwargs)
    assert [str(w.message) for w in caught if 'Falling back' in str(w.message)] == [message]
    assert len(states) == 2
    for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        assert_bytes(getattr(states[1], name), getattr(states[0], name), f'state/{name}')
    left, right = observables(actual), observables(expected)
    assert left['raises'] == right['raises']
    assert left['arrays'].keys() == right['arrays'].keys()
    assert 'time_series' in left['arrays']
    assert np.any(np.asarray(expected.time_series) != 0)
    if declaration == 'P-waveguide':
        assert any(p.startswith('waveguide_ports/') for p in left['arrays'])
        assert any(p.startswith('waveguide_sparams/') for p in left['arrays'])
    for path, value in right['arrays'].items():
        other = left['arrays'][path]
        assert (other.shape, other.dtype) == (value.shape, value.dtype), path
        assert np.array_equal(np.frombuffer(other.data, dtype=np.uint8),
                              np.frombuffer(value.data, dtype=np.uint8)), path
    records = [d for d in actual.diagnostics if d.code == 'distributed.one_device_fallback']
    assert len(records) == 1
    diagnostic = records[0]
    assert diagnostic.severity == 'warning'
    assert diagnostic.subject == reason
    assert diagnostic.message == message
    assert dict(diagnostic.values) == {'requested_devices': 2, 'devices_used': 1}
    assert diagnostic.path == 'fwd_distributed_nu'
    assert tuple(d for d in actual.diagnostics if d not in records) == expected.diagnostics


@pytest.mark.parametrize('declaration,reason', CASES)
def test_eps_gradient_bytes(declaration, reason, two_devices):
    sim = BUILDERS[declaration]('graded')
    eps = jnp.ones(sim._build_nonuniform_grid().shape, dtype=jnp.float32)

    def energy(eps_override, distributed):
        kwargs = dict(distributed=True, devices=two_devices) if distributed else {}
        result = sim.forward(eps_override=eps_override, n_steps=80,
                             checkpoint=False, skip_preflight=True, **kwargs)
        return jnp.sum(result.time_series ** 2)

    expected = jax.grad(lambda e: energy(e, False))(eps)
    with pytest.warns(UserWarning, match='Falling back to single-device'):
        actual = jax.grad(lambda e: energy(e, True))(eps)
    assert np.isfinite(expected).all() and np.any(np.asarray(expected) != 0)
    assert_bytes(actual, expected, 'grad/eps_override')


@pytest.mark.parametrize('declaration,reason', CASES)
def test_vmap_bytes(declaration, reason, two_devices):
    sim = BUILDERS[declaration]('graded')
    shape = sim._build_nonuniform_grid().shape
    vacuum = jnp.ones(shape)
    batch = jnp.stack([vacuum, vacuum.at[shape[0] // 2, :, :].set(1.1)])

    def fields(eps, distributed):
        kwargs = dict(distributed=True, devices=two_devices) if distributed else {}
        return sim.forward(eps_override=eps, n_steps=40, checkpoint=False,
                           skip_preflight=True, **kwargs).time_series

    expected = jax.vmap(lambda e: fields(e, False))(batch)
    with pytest.warns(UserWarning, match='Falling back to single-device'):
        actual = jax.vmap(lambda e: fields(e, True))(batch)
    assert_bytes(actual, expected, 'vmap/time_series')


@pytest.mark.parametrize('declaration,reason', CASES)
@pytest.mark.parametrize('options', [{'checkpoint_segments': 2}, {'exchange_interval': 2},
                                     {'mu_r_override': 1.}, {'not_a_forward_option': True}])
def test_single_device_argument_refusals(declaration, reason, options, two_devices):
    kwargs = dict(n_steps=2, skip_preflight=True, **options)
    with pytest.raises((TypeError, ValueError, NotImplementedError)) as expected:
        BUILDERS[declaration]('graded').forward(**kwargs)
    with pytest.warns(UserWarning, match='Falling back to single-device'):
        with pytest.raises(type(expected.value)) as actual:
            BUILDERS[declaration]('graded').forward(distributed=True, devices=two_devices, **kwargs)
    assert str(actual.value) == str(expected.value)
