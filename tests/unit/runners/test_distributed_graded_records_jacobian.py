"""Batched forward-mode tangents preserve the slab axis and physical records."""
import jax
import jax.numpy as jnp
import pytest
from rfx.observables import jacobian_fwd
from tests.unit.runners._graded_records_batching import problem, compare, mutate
from tests.unit.runners._graded_records_scene import mutation_result


@pytest.mark.parametrize('kind', ['planes', 'flux', 'ntff', 'moments'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_batched_jacobian_records(kind, boundary, monkeypatch):
    solve = problem(kind, boundary, monkeypatch)
    alpha = jnp.float32(1.)
    primal, tangent = jax.jvp(lambda a: solve(a, False), (alpha,), (jnp.ones_like(alpha),))
    actual, jacobian = jacobian_fwd(lambda a: solve(a, True), alpha)
    jax.block_until_ready((actual, jacobian))
    # through jax.vmap/jvp the compiled program differs (cross-trace rule, rfx CLAUDE.md)
    failures = []
    for name, observed, expected in (('primal', actual, primal),
            ('tangent', jax.tree.map(lambda a: a[0], jacobian), tangent)):
        try:
            compare(observed, expected, kind=kind, boundary=boundary, transform=f'jacobian/{name}')
        except AssertionError as error:
            failures.append(str(error))
    assert not failures, ('batching bar', failures)


@pytest.mark.parametrize('kind', ['planes', 'flux', 'ntff', 'moments'])
@pytest.mark.parametrize('mutation', ['raw_placement', 'double_records'])
def test_jacobian_mutations(kind, mutation, monkeypatch):
    mutate(monkeypatch, mutation)
    with pytest.raises(Exception, match='(partitioned 2 times|batching (bar|placement axis))') as failure:
        test_batched_jacobian_records(kind, 'cpml', monkeypatch)
    mutation_result('batched_jacobian', f'{kind}/{mutation}', failure.value)
