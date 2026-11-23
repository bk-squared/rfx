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
BASE_INDEX = np.nextafter(np.sqrt(np.float32(DECLARED_EPS)), np.float32(np.inf))
MEMBER_INDEX = np.nextafter(np.sqrt(np.float32(OVERRIDE_EPS)), np.float32(np.inf))


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    jax.clear_caches()


def line():
    sim = shorted_line()
    sim.add_material('background', eps_r=DECLARED_EPS)
    sim.add(Box((0., 0., 0.), (.340, .003, .002)), material='background')
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


def test_library_sweep_member_reads_its_own_arrays(capsys, monkeypatch):
    from rfx.vmap_sweep import vmap_material_sweep
    sim = line()
    drive = sim._ports.pop()
    sim.add_source(drive.position, drive.component, waveform=drive.waveform)
    def refuse_fallback(*args, **kwargs):
        raise AssertionError('the member witness must exercise the batched material arrays')
    monkeypatch.setattr('rfx.vmap_sweep._sequential_fallback', refuse_fallback)
    result = vmap_material_sweep(sim, 'background.eps_r', np.array([DECLARED_EPS, OVERRIDE_EPS]), n_steps=N)
    base, row = result.settling_witness
    check_window(result.dt[0])
    with capsys.disabled():
        print(f'19 sweep: base={base}; member={row}', file=sys.stderr, flush=True)
    assert base['echo_bound']['n_max'] == BASE_INDEX
    assert base['status'] == 'pass', base
    assert row['echo_bound']['n_max'] == MEMBER_INDEX, row
    assert row['status'] != 'pass', row
    assert 'cannot judge:' in row['reason'] and RULE in row['reason'], row
