"""A point-sample table cut to x slabs reads the same physics as one device.

A port's V and I (and any point channel) are weighted sums of field samples,
so a structure cut into x slabs must give the same channel values when each
slab sums only the nodes it owns and the slab sums are added once after the
time loop. Expected values always come from ``projector.project`` on the
whole-domain state (the single-device path), never from the slab helpers.

T1: a channel wholly inside one slab is bitwise equal on 1, 2 and 3 slabs.
T2: a channel straddling a seam is within 2 float32 ULP of its own
    magnitude-weighted sum (sum over terms of |w|*|field|).
T3: per-step programs under rank_shard_map on 2 and 3 host devices hold no
    all-reduce, all-gather or partition-id.
T4: traced weights (a differentiated graded mesh) give the single-device
    gradient within 1e-6 relative.
Ghost and alignment rows of every slab state are NaN, so any read of them
shows up as NaN in a compared channel.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement import projector
from rfx.measurement.projector import ComponentTerms, SampleTable, project
from rfx.stepping.slab import Slab, cut
from rfx.stepping.slab_tables import cut_sample_table, reduce_channels, slab_project

COMPONENTS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
COLLECTIVE = re.compile(r'\s(?:all-reduce|all-gather)(?:-start|-done)?\(')


class Fields(NamedTuple):
    ex: object
    ey: object
    ez: object
    hx: object
    hy: object
    hz: object


def whole_state(shape, seed):
    rng = np.random.default_rng(seed)
    return Fields(*(rng.normal(size=shape).astype(np.float32) for _ in COMPONENTS))


def slab_states(state, slab):
    """Host slabs of the whole state; ghost and alignment rows poisoned."""
    stacked = cut(tuple(state), slab, 'state', host=True)
    poisoned = []
    for arr in stacked:
        arr = np.array(arr)
        for r, (lo, hi) in enumerate(slab.owned):
            arr[r, :slab.ghost_width] = np.nan
            arr[r, slab.ghost_width + (hi - lo):] = np.nan
        poisoned.append(arr)
    return Fields(*poisoned)


def terms_by_channel(table):
    """Global (row, component index, flat term index) per channel."""
    rows = {c: [] for c in range(table.n_channels)}
    for t in table.components:
        for n, cid in enumerate(np.asarray(t.channel_ids)):
            rows[int(cid)].append((t.component, int(t.indices[0][n]), int(t.indices[1][n]),
                                   int(t.indices[2][n]), n))
    return rows


def classify(table, slab):
    """Channels wholly inside one slab vs straddling a seam (Slab.owner)."""
    inside, straddle = [], []
    for cid, terms in terms_by_channel(table).items():
        owners = {slab.owner(i) for _, i, _, _, _ in terms}
        (inside if len(owners) <= 1 else straddle).append(cid)
    return inside, straddle


def magnitude_scale(table, state):
    """Per channel sum over its terms of |w|*|field| (float64)."""
    scale = np.zeros(table.n_channels)
    for t in table.components:
        f = np.asarray(getattr(state, t.component), np.float64)
        w = np.abs(np.asarray(t.weights, np.float64))
        vals = np.abs(f[tuple(np.asarray(a) for a in t.indices)])
        np.add.at(scale, np.asarray(t.channel_ids), w * vals)
    return scale


def slab_sum(state, table, slab, *, jit=True):
    local = slab_states(state, slab)
    cut_table = cut_sample_table(table, slab)
    def call(fields, row):
        return jax.jit(lambda f: slab_project(f, row))(fields) if jit else slab_project(fields, row)
    per_slab = [np.asarray(call(Fields(*(a[r] for a in local)), cut_table.row(r)))
                for r in range(slab.n_devices)]
    return np.asarray(reduce_channels(np.stack(per_slab)))


def check_against_whole(state, table, slab, got, expected):
    """T1 bitwise on inside channels, T2 bar on straddling ones."""
    inside, straddle = classify(table, slab)
    got, expected = np.asarray(got, np.float32), np.asarray(expected, np.float32)
    assert np.isfinite(expected).all()
    bad = [c for c in inside
           if got[..., c].tobytes() != expected[..., c].tobytes()]
    assert not bad, {c: (got[..., c], expected[..., c]) for c in bad}
    if straddle:
        scale = magnitude_scale(table, state)[straddle]
        bar = 2 * np.spacing(scale.astype(np.float32))
        err = np.abs(got[..., straddle].astype(np.float64) - expected[..., straddle])
        assert np.all(err <= bar), (straddle, err, bar)
    return inside, straddle


def synthetic_table(nx, ny, nz, seed):
    """Channels on every x row (seam rows, ghost-adjacent rows, the last
    real row next to alignment rows) and on every pair (r, r+1), with
    mixed components and random weights; some channels repeat a node."""
    rng = np.random.default_rng(seed)
    channels = []
    for r in range(nx):
        channels.append([(r, comp) for comp in rng.choice(COMPONENTS, size=3)])
    for r in range(nx - 1):
        channels.append([(r, 'ez'), (r + 1, 'hy'), (r, 'hy'), (r + 1, 'ex')])
    per = {c: ([], [], [], [], []) for c in COMPONENTS}
    for cid, nodes in enumerate(channels):
        for row, comp in nodes:
            i, j, k, ids, w = per[comp]
            i.append(row)
            j.append(int(rng.integers(0, ny)))
            k.append(int(rng.integers(0, nz)))
            ids.append(cid)
            w.append(rng.normal())
    terms = [ComponentTerms(c, tuple(np.asarray(a, np.int32) for a in per[c][:3]),
                            np.asarray(per[c][3], np.int32), np.asarray(per[c][4], np.float32))
             for c in COMPONENTS if per[c][0]]
    ids = np.concatenate([t.channel_ids for t in terms])
    order = np.argsort(ids, kind='stable').astype(np.int32)
    return SampleTable(tuple(terms), len(channels), tuple(map(str, range(len(channels)))),
                       order, ids[order])


# Real port tables from the runners' own port_table call (uniform and graded).
DX_PROFILE = np.array([1, .95, .9, 1, 1.1, 1.05, 1, 1.1, .95, 1]) * 1e-3


def port_scene(graded, kind):
    from rfx import Simulation
    kw = dict(dx_profile=DX_PROFILE) if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.010, .008, .008), dx=.001, boundary='pec', **kw)
    if kind == 'lumped':
        sim.add_port(position=(.004, .004, .004), component='ez', impedance=50.)
        sim.add_port(position=(.006, .003, .004), component='ey', impedance=50.)
    else:
        sim.add_port(position=(.006, .004, .003), component='ez', impedance=50., extent=.002)
        sim.add_port(position=(.007, .003, .004), component='ex', impedance=50., extent=.003)
    return sim


_CAPTURED = {}


def runner_port_table(graded, kind):
    """(grid, specs, kwargs) the runner passed to projector.port_table."""
    key = (graded, kind)
    if key not in _CAPTURED:
        calls = []
        real = projector.port_table

        def spy(grid, specs, **kwargs):
            calls.append((grid, tuple(specs), kwargs))
            return real(grid, specs, **kwargs)

        sim = port_scene(graded, kind)
        projector.port_table = spy
        try:
            if graded:
                sim.forward(n_steps=2, checkpoint=False, skip_preflight=True,
                            port_s11_freqs=np.array([2e9]))
            else:
                sim.run(n_steps=2, compute_s_params=True, s_param_freqs=[2e9],
                        skip_preflight=True)
        finally:
            projector.port_table = real
        assert calls, 'runner did not build a port table'
        _CAPTURED[key] = calls[0]
    return _CAPTURED[key]


def real_table(graded, kind):
    grid, specs, kwargs = runner_port_table(graded, kind)
    with jax.ensure_compile_time_eval():
        return grid, projector.port_table(grid, specs, **kwargs)


@pytest.mark.parametrize('n_slabs', [1, 2, 3])
@pytest.mark.parametrize('nx', [11, 12, 7])
def test_t1_t2_synthetic_table(n_slabs, nx):
    shape = (nx, 5, 4)
    table = synthetic_table(*shape, seed=nx)
    state = whole_state(shape, seed=100 + nx)
    slab = Slab(nx, n_slabs)
    expected = np.asarray(jax.jit(lambda s: project(s, table))(state))
    inside, straddle = check_against_whole(state, table, slab, slab_sum(state, table, slab), expected)
    eager = slab_sum(state, table, slab, jit=False)
    check_against_whole(state, table, slab, eager, np.asarray(project(state, table)))
    assert inside
    if n_slabs > 1:
        assert straddle
        seams = [lo for lo, _ in slab.owned[1:]]
        rows = {i for c in inside for _, i, _, _, _ in terms_by_channel(table)[c]}
        assert all(s in rows and s - 1 in rows for s in seams)
        assert nx - 1 in rows


@pytest.mark.parametrize('n_slabs', [1, 2, 3])
@pytest.mark.parametrize('kind', ['lumped', 'wire'])
@pytest.mark.parametrize('graded', [False, True])
def test_t1_t2_runner_port_tables(graded, kind, n_slabs):
    grid, table = real_table(graded, kind)
    nx = grid.shape[0]
    assert nx % 2 and nx % 3, 'need alignment rows on 2 and 3 slabs'
    state = whole_state(grid.shape, seed=7)
    slab = Slab(nx, n_slabs)
    expected = np.asarray(jax.jit(lambda s: project(s, table))(state))
    inside, straddle = check_against_whole(state, table, slab, slab_sum(state, table, slab), expected)
    assert inside
    if n_slabs == 1:
        assert not straddle


def test_runner_tables_have_both_classes():
    """Neither T1 nor T2 is vacuous on the real tables."""
    seen = {'inside': 0, 'straddle': 0}
    for graded in (False, True):
        for kind in ('lumped', 'wire'):
            grid, table = real_table(graded, kind)
            for n in (2, 3):
                inside, straddle = classify(table, Slab(grid.shape[0], n))
                seen['inside'] += len(inside)
                seen['straddle'] += len(straddle)
    assert seen['inside'] and seen['straddle'], seen


def test_padding_is_inert_and_on_an_owned_row():
    nx = 11
    table = synthetic_table(nx, 5, 4, seed=3)
    for n in (2, 3):
        slab = Slab(nx, n)
        cut_table = cut_sample_table(table, slab)
        for r, (lo, hi) in enumerate(slab.owned):
            row = cut_table.row(r)
            ids = np.concatenate([np.full(len(w), -1) for w in row.weights])
            ids[np.asarray(row.order)] = np.asarray(row.segment_ids)
            weights = np.concatenate([np.asarray(w) for w in row.weights])
            locals_i = np.concatenate([np.asarray(ijk[0]) for ijk in row.indices])
            pad = ids == table.n_channels
            assert np.all(weights[pad] == 0)
            assert np.all((locals_i >= slab.ghost_width) & (locals_i < slab.ghost_width + hi - lo))


def test_t4_traced_graded_weights_gradient():
    grid, specs, kwargs = runner_port_table(True, 'wire')
    state = whole_state(grid.shape, seed=11)
    rng = np.random.default_rng(12)
    with jax.ensure_compile_time_eval():
        n_channels = projector.port_table(grid, specs, **kwargs).n_channels
    coeff = jnp.asarray(rng.normal(size=n_channels), jnp.float32)

    def table_of(dx_arr):
        return projector.port_table(grid._replace(dx_arr=dx_arr), specs, **kwargs)

    def single(dx_arr):
        return jnp.sum(coeff * project(state, table_of(dx_arr)))

    def make_sliced(n):
        slab = Slab(grid.shape[0], n)
        local = slab_states(state, slab)

        def sliced(dx_arr):
            t = cut_sample_table(table_of(dx_arr), slab)
            per = jnp.stack([slab_project(Fields(*(a[r] for a in local)), t.row(r))
                             for r in range(n)])
            return jnp.sum(coeff * reduce_channels(per))
        return sliced

    dx_arr = jnp.asarray(grid.dx_arr)
    reference = np.asarray(jax.grad(single)(dx_arr), np.float64)
    assert np.max(np.abs(reference)) > 0, 'weights do not depend on dx'
    for n in (1, 2, 3):
        got = np.asarray(jax.grad(make_sliced(n))(dx_arr), np.float64)
        assert np.max(np.abs(got - reference)) <= 1e-6 * np.max(np.abs(reference)), (n, got, reference)


# T3: compiled per-step programs on 2 and 3 host devices.
T3_STEPS = 5


def t3_worker(n_devices):
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from rfx.stepping.rank import mesh_ranks, rank_shard_map
    from rfx.stepping.slab import shard_stacked
    devices = jax.devices('cpu')
    assert len(devices) == n_devices, devices
    mesh = Mesh(np.array(devices), ('x',))
    nx = 11
    shape = (nx, 5, 4)
    table = synthetic_table(*shape, seed=nx)
    state = whole_state(shape, seed=100 + nx)
    slab = Slab(nx, n_devices)
    sharding = NamedSharding(mesh, P('x'))
    local = Fields(*(shard_stacked(a, sharding) for a in slab_states(state, slab)))
    cut_table = jax.device_put(cut_sample_table(table, slab), sharding)

    def body(fields, tab, *, rank):
        row = tab.row(0)

        def step(f, _):
            f = Fields(*(a * jnp.float32(1.25) for a in f))
            return f, slab_project(f, row)
        _, record = lax.scan(step, fields, None, length=T3_STEPS)
        return record[None]

    run = jax.jit(rank_shard_map(body, mesh=mesh, in_specs=(P('x'), P('x')),
                                 out_specs=P('x'), check_rep=False))
    ranks = mesh_ranks(mesh)
    per_slab = np.asarray(run(ranks, local, cut_table))
    got = np.asarray(reduce_channels(per_slab))
    f = state
    for n in range(T3_STEPS):
        f = Fields(*(np.asarray(a) * np.float32(1.25) for a in f))
        check_against_whole(f, table, slab, got[n], np.asarray(jax.jit(lambda s: project(s, table))(f)))
    hlo = run.lower(ranks, local, cut_table).compile().as_text()
    print(f'completed t3 {n_devices} {len(COLLECTIVE.findall(hlo))}', flush=True)


@pytest.mark.parametrize('n_devices', [2, 3])
def test_t3_no_collective_or_partition_id(n_devices, tmp_path):
    from tests.contracts.test_distributed_rank_input import assert_clean_dumps
    root = Path(__file__).resolve().parents[3]
    dump = tmp_path / 'hlo'
    dump.mkdir()
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS='cpu',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               XLA_FLAGS=f'--xla_force_host_platform_device_count={n_devices} --xla_dump_to={dump}',
               JAX_ENABLE_COMPILATION_CACHE='false')
    run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--t3', str(n_devices)],
                         cwd=root, env=env, capture_output=True, text=True, timeout=240)
    assert run.returncode == 0, run.stdout + run.stderr
    assert f'completed t3 {n_devices} 0' in run.stdout, run.stdout
    assert_clean_dumps(dump)
    loops = [m for m in sorted(dump.rglob('*after_optimizations*.txt')) if ' while(' in m.read_text()]
    assert loops
    for module in loops:
        assert not COLLECTIVE.search(module.read_text()), module.name


def test_t3_gate_sees_an_in_loop_sum():
    """The collective regex is not vacuous: a per-step psum is caught."""
    from jax import lax
    from jax.sharding import Mesh, PartitionSpec as P
    from rfx.stepping.rank import mesh_ranks, rank_shard_map
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('needs 2 host devices (root conftest)')
    mesh = Mesh(np.array(devices[:2]), ('x',))

    def body(x, *, rank):
        def step(c, _):
            return c * 2, lax.psum(c, 'x')
        return lax.scan(step, x, None, length=3)[1][None]

    run = jax.jit(rank_shard_map(body, mesh=mesh, in_specs=P('x'), out_specs=P('x')))
    x = jnp.ones((2, 3))
    assert COLLECTIVE.search(run.lower(mesh_ranks(mesh), x).compile().as_text())


if __name__ == '__main__':
    assert sys.argv[1] == '--t3', sys.argv
    t3_worker(int(sys.argv[2]))
