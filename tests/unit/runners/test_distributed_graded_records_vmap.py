"""A batch axis must not take the slab sharding of graded records."""
import jax
import jax.numpy as jnp
import pytest
from tests.unit.runners._graded_records_batching import problem, compare, mutate
from tests.unit.runners._graded_records_scene import mutation_result


@pytest.mark.parametrize('kind', ['planes', 'flux', 'ntff', 'moments'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('batch', [1, 3])
def test_vmap_records(kind, boundary, batch, monkeypatch):
    solve = problem(kind, boundary, monkeypatch)
    alphas = jnp.linspace(.9, 1.1, batch)
    expected = [solve(a, False) for a in alphas]
    actual = jax.vmap(lambda a: solve(a, True))(alphas)
    jax.block_until_ready(actual)
    # through jax.vmap/jvp the compiled program differs (cross-trace rule, rfx CLAUDE.md)
    failures = []
    for i, reference in enumerate(expected):
        try:
            compare(jax.tree.map(lambda a: a[i], actual), reference, kind=kind,
                    boundary=boundary, transform=f'vmap{batch}/element{i}')
        except AssertionError as error:
            failures.append(str(error))
    assert not failures, ('batching bar', failures)


@pytest.mark.parametrize('kind', ['planes', 'flux', 'ntff', 'moments'])
@pytest.mark.parametrize('mutation', ['raw_placement', 'double_records'])
def test_vmap_mutations(kind, mutation, monkeypatch):
    mutate(monkeypatch, mutation)
    with pytest.raises(Exception, match='(partitioned 2 times|batching (bar|placement axis))') as failure:
        test_vmap_records(kind, 'cpml', 3, monkeypatch)
    mutation_result('vmap', f'{kind}/{mutation}', failure.value)
