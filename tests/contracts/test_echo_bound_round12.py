"""Material overrides and library sweep members close the unseen-echo gap."""
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box
from tests.contracts.test_echo_bound_closing_rule import shorted_line


C = 299792458.0
RULE = 'the record ends less than 2.05 longest round trips'
N = 6000
DECLARED_EPS = 1.
OVERRIDE_EPS = 4 * DECLARED_EPS
BASE_INDEX = np.sqrt(DECLARED_EPS * 1.)
MEMBER_INDEX = np.sqrt(OVERRIDE_EPS * 1.)


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    jax.clear_caches()


def line():
    sim = shorted_line()
    sim.add(Box((0., 0., 0.), (.340, .003, .002)), material='vacuum')
    sim._geometry.insert(0, sim._geometry.pop())
    sim.add_probe(position=(.010, .001, 0), component='ez')
    sim.add_dft_plane_probe(axis='x', coordinate=.010, component='ez',
                            n_freqs=7, name='read')
    return sim


def records(sim, eps):
    grid = sim._build_realized_grid()
    return sim.forward(eps_override=jnp.full(grid.shape, eps), n_steps=N,
                       skip_preflight=True)


def check_window(dt):
    distance = np.sqrt(.340**2 + .003**2 + .002**2)
    window = (N - round(.5 * N)) * float(dt)
    assert 2.05 * 2 * distance * BASE_INDEX / C <= window
    assert window < 2.05 * 2 * distance * MEMBER_INDEX / C


def test_override_index_closes_plain_and_traced_records(capsys):
    sim = line()
    base = records(sim, jnp.float32(DECLARED_EPS))
    check_window(base.dt)
    assert base.settling_witness['echo_bound']['n_max'] == BASE_INDEX
    assert RULE not in base.settling_witness['reason']
    plain = records(sim, jnp.float32(OVERRIDE_EPS))
    compiled = jax.jit(lambda eps: records(sim, eps))(jnp.float32(OVERRIDE_EPS))
    for label, result in (('plain', plain), ('jit', compiled)):
        row = result.settling_witness
        with capsys.disabled():
            print(f'12r override {label}: n={N}; base_index={BASE_INDEX}; '
                  f'override_index={MEMBER_INDEX}; status={row["status"]}; '
                  f'reason={row["reason"]}', file=sys.stderr, flush=True)
        assert row['echo_bound']['n_max'] == MEMBER_INDEX, row
        assert row['status'] != 'pass', row
        assert 'cannot judge:' in row['reason'] and RULE in row['reason'], row


def test_library_sweep_member_uses_its_own_declaration(capsys):
    from rfx.measurement.sweep_settling import sweep_member_witnesses

    sim = line()
    assert not sim._materials
    result = records(sim, jnp.float32(OVERRIDE_EPS))
    check_window(result.dt)
    series = np.asarray(result.time_series)[None]
    names = tuple(result.dft_planes)
    planes = tuple(result.dft_planes.values())
    accumulators = {name: np.asarray(plane.accumulator)[None]
                    for name, plane in result.dft_planes.items()}
    _, base_rows = sweep_member_witnesses(
        sim, result.grid, series, names, planes, accumulators, param_name='eps_r',
        param_values=np.array([DECLARED_EPS]))
    _, member_rows = sweep_member_witnesses(
        sim, result.grid, series, names, planes, accumulators, param_name='eps_r',
        param_values=np.array([OVERRIDE_EPS]))
    base, = base_rows
    row, = member_rows
    with capsys.disabled():
        print(f'12r sweep: n={N}; base_index={BASE_INDEX}; '
              f'member_index={MEMBER_INDEX}; status={row["status"]}; '
              f'reason={row["reason"]}', file=sys.stderr, flush=True)
    assert base['echo_bound']['n_max'] == BASE_INDEX
    assert RULE not in base['reason']
    assert row['echo_bound']['n_max'] == MEMBER_INDEX, row
    assert row['status'] != 'pass', row
    assert 'cannot judge:' in row['reason'] and RULE in row['reason'], row
    assert not sim._materials


def test_unresolved_library_sweep_material_cannot_be_judged(monkeypatch):
    from rfx.measurement.sweep_settling import sweep_member_witnesses

    sim = line()
    grid = sim._build_realized_grid()
    resolve = sim._resolve_material

    def unresolved(name):
        if name == 'vacuum':
            raise KeyError('missing library vacuum')
        return resolve(name)

    monkeypatch.setattr(sim, '_resolve_material', unresolved)
    dbs, (row,) = sweep_member_witnesses(
        sim, grid, np.zeros((1, 16, 1)), (), (), {}, param_name='eps_r',
        param_values=np.array([OVERRIDE_EPS]))
    assert np.isnan(dbs[0])
    assert row['status'] == 'undetermined', row
    assert "cannot judge: sweep material 'vacuum' cannot be resolved" in row['reason'], row
    assert 'missing library vacuum' in row['reason'], row
