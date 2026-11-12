"""Same-buffer equality, signed-zero AD, and explicit host-path witnesses."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rfx.runners import _record_join as join
from tests._multi_device_scene import host_devices
from tests.unit.runners._record_join_scene import KINDS, layout_for, filled
from tests.unit.runners._record_join_frozen import assemble as frozen


def arrays(kind, value):
    if kind == 'planes':
        return [v.accumulator for v in value]
    if kind == 'flux':
        return [a for v in value for a in v[:4]]
    return jax.tree.leaves(value)


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('kind', KINDS)
def test_same_buffer_bits(kind, padded, n):
    layout = layout_for(kind, n, nx=18 + int(padded))
    buffer = filled(layout)
    expected = arrays(kind, frozen(layout, buffer))
    actual = arrays(kind, layout.assemble(buffer))
    for index, (a, b) in enumerate(zip(actual, expected)):
        assert a.sharding.is_fully_replicated
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        if kind == 'moments':
            # Sum zero signs are not specified; the n=2 nonzero bits are.
            a, b = a.copy(), b.copy()
            for value in (a, b):
                value.real[value.real == 0] = 0.
                value.imag[value.imag == 0] = 0.
        if kind == 'moments' and n == 3:
            bar = 9*np.spacing(np.float32(np.max(np.abs(b))))
            assert np.max(np.abs(a-b)) <= bar, ('moment ULP', index, bar)
        else:
            assert a.tobytes() == b.tobytes(), ('record bit equality', kind, index)
    print(dict(kind=kind, devices=n, nx=18 + int(padded), bits='equal or moment ULP'))


@pytest.mark.parametrize('kind', KINDS)
def test_negative_zero_bits(kind):
    layout = layout_for(kind)
    buffer = filled(layout, zeros=True)
    for a, b in zip(arrays(kind, layout.assemble(buffer)), arrays(kind, frozen(layout, buffer))):
        a, b = np.asarray(a), np.asarray(b)
        if kind == 'moments':
            np.testing.assert_array_equal(a, b)
        else:
            assert a.tobytes() == b.tobytes(), ('negative-zero bit equality', kind)


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('nx', [18, 19])
def test_second_axis_chunk_bits(n, nx):
    layout = layout_for('planes', n, nx=nx, single_frequency=True)
    buffer = filled(layout)
    template = layout.templates[0].accumulator
    chunk = join.chunk_shape(template.shape, template.dtype, template.nbytes, 1)
    assert chunk[0] == 1 and chunk[2] < template.shape[2], 'second-axis split not exercised'
    assert np.prod(chunk)*template.dtype.itemsize <= .025*template.nbytes
    actual = np.asarray(layout.assemble(buffer)[0].accumulator)
    expected = np.asarray(frozen(layout, buffer)[0].accumulator)
    assert actual.tobytes() == expected.tobytes(), 'second-axis chunk bit equality'


def test_owned_negative_zero_gradient():
    shape = (40, 4, 8)
    owners = [[(0, (40, 2, 8), (0, rank*2, 0), (40, 2, 8))] for rank in range(2)]
    mesh = Mesh(np.asarray(host_devices(2)), ('x',))
    raw = np.full((2, 40*2*8), -0.0, np.float32)
    data = jax.device_put(raw, NamedSharding(mesh, P('x')))
    weights = np.linspace(.5, 2, np.prod(shape), dtype=np.float32).reshape(shape)

    def loss(x):
        result = join.join_owned(x, owners, shape, x.dtype, mesh=mesh, kind='planes', record_size=raw.nbytes)
        return jnp.sum(result*jnp.asarray(weights))

    def single(x):
        result = jnp.concatenate((x[0].reshape(40, 2, 8), x[1].reshape(40, 2, 8)), axis=1)
        return jnp.sum(result*jnp.asarray(weights))

    actual = np.asarray(jax.grad(loss)(data))
    expected = np.asarray(jax.grad(single)(jax.device_put(raw, host_devices(1)[0])))
    assert actual.tobytes() == expected.tobytes(), 'owned negative-zero gradient'


def test_no_silent_fallback(monkeypatch):
    layout = layout_for('planes')
    def fail(*a, **kw):
        raise RuntimeError('join witness')
    monkeypatch.setattr('rfx.runners._distributed_records.join_owned', fail)
    with pytest.raises(RuntimeError, match='join witness'):
        layout.assemble(layout.initial())


@pytest.mark.parametrize('kind', KINDS)
def test_explicit_host_path(kind, monkeypatch):
    layout = layout_for(kind)
    buffer = jax.tree.map(np.asarray, filled(layout))
    def fail(*a, **kw):
        raise AssertionError('device join reached by multi_process')
    monkeypatch.setattr('rfx.runners._distributed_records.join_owned', fail)
    monkeypatch.setattr('rfx.runners._distributed_ntff.join_owned', fail)
    for a, b in zip(arrays(kind, layout.assemble(buffer, multi_process=True)), arrays(kind, frozen(layout, buffer))):
        np.testing.assert_array_equal(a, b)


def test_empty_snapshot():
    layout = layout_for('snapshots')
    layout.frames = 0
    buffer = layout.initial()
    for a, b in zip(arrays('snapshots', layout.assemble(buffer)), arrays('snapshots', frozen(layout, buffer))):
        assert a.shape == b.shape and a.dtype == b.dtype and a.size == 0


@pytest.mark.parametrize('multi_process', [False, True])
def test_moments_keep_carry_dtype(multi_process):
    layout = layout_for('moments')
    buffer = jnp.real(filled(layout))
    if multi_process:
        buffer = np.asarray(buffer)
    result = layout.assemble(buffer, multi_process=multi_process)
    assert all(a.dtype == buffer.dtype for a in result), 'moment carry dtype'


@pytest.mark.parametrize('multi_process', [False, True])
def test_ntff_x_faces_keep_owner_shape_and_dtype(multi_process):
    layout = layout_for('ntff')
    buffer = jnp.real(filled(layout))
    if multi_process:
        buffer = np.asarray(buffer)
    result = layout.assemble(buffer, multi_process=multi_process)
    for face in (0, 1, 6, 7):
        owner = next(part for part in layout.parts if part[2][face % 6])
        assert result[face].shape == owner[4][face], 'NTFF owner shape'
        assert result[face].dtype == buffer.dtype, 'NTFF owner dtype'


def test_program_cache_reused_across_layout_instances():
    join._join_program.cache_clear()
    first = layout_for('planes', nx=20)
    second = layout_for('planes', nx=20)
    a = first.assemble(first.initial())
    before = join._join_program.cache_info()
    b = second.assemble(second.initial())
    after = join._join_program.cache_info()
    assert after.misses == before.misses and after.hits > before.hits, 'join recompiles the same layout'
    np.testing.assert_array_equal(a[0].accumulator, b[0].accumulator)


def test_runner_flag_selects_host_join(monkeypatch):
    from types import SimpleNamespace
    from jax.experimental import multihost_utils
    from rfx.runners._distributed_records import SlabRecords
    layout = layout_for('ntff')
    carry = filled(layout)
    calls = []
    def gathered(buffer, *, tiled):
        calls.append(tiled)
        return np.asarray(buffer)
    monkeypatch.setattr(multihost_utils, 'process_allgather', gathered)
    records = object.__new__(SlabRecords)
    records.layouts = {'ntff': layout}
    records.ntff_box, records.monitor, records.sim = layout.box, None, SimpleNamespace()
    def fail(*a, **kw):
        raise RuntimeError('device join reached')
    monkeypatch.setattr('rfx.runners._distributed_ntff.join_owned', fail)
    records.assemble({'ntff': carry}, multi_process=True)
    assert calls == [True]
    with pytest.raises(RuntimeError, match='device join reached'):
        records.assemble({'ntff': np.asarray(carry)}, multi_process=False)
