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
    assert names == expected + ('V_ref',)


def test_records_hash_and_compare_by_identity():
    samples = jnp.array([[1., 2.]])
    record = PortTimeRecord('port:1:mode:0', 'lumped', ('V', 'I'), samples)
    other = PortTimeRecord('port:1:mode:0', 'lumped', ('V', 'I'), samples)
    assert record == record
    assert record != other
    assert {record: 'first', other: 'second'}[record] == 'first'
    assert hash(record) == object.__hash__(record)


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('entry, reads_s', [('run', False), ('forward', False), ('forward', True)],
                         ids=['run-S-off', 'forward-S-off', 'forward-S-on'])
def test_record_ids_follow_simulation_port_numbers(graded, entry, reads_s):
    import warnings
    from rfx import Simulation
    from rfx.measurement.plan import measurement_plan
    sim = Simulation(freq_max=10e9, domain=(.008, .006, .002), dx=.001, boundary='pec',
                     **({'dz_profile': np.array([.0008, .0012])} if graded else {}))
    sim.add_source((.001, .001, .001), 'ez', waveform=lambda t: jnp.cos(t * 2e10),
                   amplitude_kind='field')
    sim.add_port((.002, .003, 0.), 'ez', impedance=50., excite=False)
    sim.add_port((.006, .003, 0.), 'ez', impedance=50., extent=.002, excite=False)
    kwargs = {'compute_s_params': False} if entry == 'run' else (
        {'port_s11_freqs': np.array([2e9])} if reads_s else {})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = getattr(sim, entry)(n_steps=16, skip_preflight=True, **kwargs)
    expected = {'lumped': 'port:1:mode:0', 'wire': 'port:2:mode:0'}
    assert {r.kind: r.id for r in result.sparam_time_records} == expected
    plan = measurement_plan(sim, n_steps=16, path=('fwd_' if entry == 'forward' else 'run_') +
                            ('nonuniform' if graded else 'uniform'))
    assert {o.kind: o.id for o in plan.owners if o.kind in expected} == expected
