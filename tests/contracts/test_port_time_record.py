"""Public named records fail closed on positional and array consumption."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement import PortTimeRecord


def test_names_and_guards():
    samples = jnp.array([[1., 9.], [2., 8.]])
    record = PortTimeRecord('port:4:mode:0', 'lumped', ('I', 'V'), samples)
    np.testing.assert_array_equal(record['V'], [9., 8.])
    np.testing.assert_array_equal(record['I'], [1., 2.])
    for action in (lambda: record[0], lambda: record[:, 0],
                   lambda: np.asarray(record), lambda: iter(record)):
        with pytest.raises(TypeError, match="I.*V"):
            action()
    with pytest.raises(TypeError):
        len(record)
    with pytest.raises(KeyError, match='V_port'):
        record['V_port']
    with pytest.raises(ValueError, match='channels'):
        PortTimeRecord('port:4:mode:0', 'lumped', ('V',), samples)


def test_pytree_preserves_names_under_jit_grad_vmap():
    def make(samples):
        return PortTimeRecord('port:2:mode:0', 'lumped', ('V', 'I'), samples)
    samples = jnp.array([[1., 2.], [3., 4.]])
    record = jax.jit(make)(samples)
    assert (record.id, record.kind, record.channels) == (
        'port:2:mode:0', 'lumped', ('V', 'I'))
    assert len(jax.tree_util.tree_leaves(record)) == 1
    np.testing.assert_array_equal(jax.grad(lambda x: jnp.sum(make(x)['V']))(samples),
                                  [[1., 0.], [1., 0.]])
    batched = jax.vmap(make)(jnp.stack((samples, samples * 2)))
    np.testing.assert_array_equal(batched['I'], [[2., 4.], [4., 8.]])


def test_equivalence_labels_and_pairing_come_from_record():
    from tests.contracts.path_equivalence.execution import _samples, _paired_samples
    a = PortTimeRecord('a', 'wire', ('I', 'V'), np.array([[7., 3.]]))
    b = PortTimeRecord('b', 'lumped', ('V_ref',), np.array([[11.]]))
    got = _samples((a, b), ('_ports', 'lumped_port'), 'fwd_nonuniform')
    assert set(got['a']) == {'V', 'I'}
    np.testing.assert_array_equal(got['a']['V'], [3.])
    left, right = _paired_samples((a, b), (b, a))
    assert left.keys() == right.keys()
    for key in left:
        for channel in left[key]:
            np.testing.assert_array_equal(left[key][channel], right[key][channel])
    missing = PortTimeRecord('a', 'wire', ('I',), np.array([[7.]]))
    left, right = _paired_samples((a,), (missing,))
    assert set(left[0]) - set(right[0]) == {'V'}


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('kind', ['lumped', 'wire'])
def test_owner_channel_set_depends_only_on_kind(graded, kind):
    from rfx import Simulation
    from rfx.measurement.ports import lumped_wire_owner
    sim = Simulation(freq_max=10e9, domain=(.004, .004, .002), dx=.001,
                     **({'dz_profile': np.array([.0008, .0012])} if graded else {}))
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    owner = lumped_wire_owner(grid, 'ez', ((2, 2, 1),), nu=graded,
                              pmc_faces=(), periodic=(False,)*3, port_id=7, kind=kind)
    assert owner.id == 'port:7:mode:0'
    names = tuple(c.name for c in owner.channels)
    expected = ('V', 'I', 'V_port') if kind == 'wire' else ('V', 'I')
    assert names == expected + (() if graded else ('V_ref',))
