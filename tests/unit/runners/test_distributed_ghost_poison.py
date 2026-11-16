"""Unused inter-slab ghosts cannot influence the wrapped uniform solve."""
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.stepping import slab_wrap, slab_wrap_uniform
from rfx.stepping.sequence import HookPoint


COMPONENTS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
STEPS = 240
pytestmark = pytest.mark.distributed


def scene(boundary):
    """An eighteen-cube TF box crosses both cuts, including a face at row twelve."""
    wall = Boundary('cpml', 'cpml', 0, 0)
    spec = BoundarySpec(x=wall, y=wall, z=wall) if boundary == 'pec' else 'cpml'
    sim = Simulation(freq_max=15e9, domain=((.017 if boundary == 'pec' else .013),) * 3,
                     dx=.001, boundary=spec, cpml_layers=2, precision='float32')
    grid = sim._build_grid()
    assert grid.shape == (18, 18, 18)

    def position(nodes):
        return tuple((n - grid.pad_x_lo) * grid.dx for n in nodes)

    sim.add_material('dielectric', eps_r=2.25)
    sim.add(Box(position((7, 7, 7)), position((10, 10, 10))), material='dielectric')
    sim.add_tfsf_source(f0=7.5e9, bandwidth=.8, closed_box=True, margin=3)
    for nodes in ((8, 8, 8), (9, 10, 7), (12, 8, 10), (13, 7, 9)):
        sim.add_vector_probe(position(nodes))
    sim.add_dft_plane_probe(axis='y', coordinate=position((8, 8, 8))[1],
                            component='ez', freqs=[5e9, 7.5e9, 10e9], name='plane')
    return sim


def _exchange_rows(state, hooks):
    """Execute the wrapper's exchange specification on tagged real/ghost rows."""
    tagged = state._replace(**{name: jnp.ones_like(getattr(state, name)).at[0].set(0).at[-1].set(0)
                               for name in COMPONENTS})
    frame = SimpleNamespace(st=tagged, step_idx=jnp.int32(0))
    for point in HookPoint:
        for hook in hooks.get(point, ()):
            hook(frame)
    return jnp.stack([jnp.stack([jnp.all(getattr(frame.st, name)[row] == 1)
                                for row in (0, -1)]) for name in COMPONENTS])


def _poison_start(carry, exchanged, rank, count):
    """Poison and independently read back every eligible row before either curl."""
    carry = dict(carry)
    checks = []
    for family in ('fdtd', 'cpml'):
        if family not in carry:
            continue
        fields = {}
        for name, value in zip(carry[family]._fields, carry[family]):
            if value.ndim == 0:
                continue
            # X-face psi buffers contain absorber depth only, without slab ghosts.
            if family == 'cpml' and name.split('_')[-1].startswith('x'):
                assert value.shape[0] < carry['fdtd'].ex.shape[0] - 2
                continue
            assert value.shape[0] == carry['fdtd'].ex.shape[0]
            eligible = []
            for side, row in enumerate((0, -1)):
                interior = rank > 0 if side == 0 else rank < count - 1
                sent = exchanged[COMPONENTS.index(name), side] if family == 'fdtd' else False
                eligible.append(interior & ~jnp.asarray(sent))
                value = value.at[row].set(jnp.where(eligible[-1], jnp.nan, value[row]))
            fields[name] = value
            # Read the updated carry, independently of the write expression.
        carry[family] = carry[family]._replace(**fields)
        for name in fields:
            for side, row in enumerate((0, -1)):
                interior = rank > 0 if side == 0 else rank < count - 1
                sent = exchanged[COMPONENTS.index(name), side] if family == 'fdtd' else False
                checks.append(~(interior & ~jnp.asarray(sent)) |
                              jnp.all(jnp.isnan(getattr(carry[family], name)[row])))
    return carry, jnp.stack(checks)


def evaluate(boundary, count, poison):
    captured = []
    original_step = slab_wrap_uniform.make_uniform_step
    original_run = slab_wrap.run_steps

    def instrument(ctx, invariants, *, hooks):
        core = original_step(ctx, invariants, hooks=hooks)

        # The first physical hook is AFTER_H. Wrap the bound callable so the
        # carry is poisoned before either curl, without a product-side seam.
        def step(carry, step_idx, *args):
            exchanged = _exchange_rows(carry['fdtd'], hooks)
            carry, checks = _poison_start(carry, exchanged, jax.lax.axis_index('x'), count)
            final, samples, extras = core(carry, step_idx, *args)
            final['_poison_seen'] = carry['_poison_seen'].at[step_idx].set(checks)
            final['_exchanged'] = exchanged[None]
            return final, samples, extras

        return step

    def run(carry, *args, **kwargs):
        if poison:
            sharding = NamedSharding(carry['fdtd'].ex.sharding.mesh, P('x'))
            checks = 12 + (32 if 'cpml' in carry else 0)
            carry = dict(carry,
                _poison_seen=jnp.zeros((count * STEPS, checks), bool, device=sharding),
                _exchanged=jnp.zeros((count, 6, 2), bool, device=sharding))
        final, samples = original_run(carry, *args, **kwargs)
        captured.append(jax.tree.map(np.asarray, final))
        return final, samples

    with patch.object(slab_wrap, 'run_steps', run), patch.object(
            slab_wrap_uniform, 'make_uniform_step', instrument if poison else original_step):
        result = scene(boundary).run(n_steps=STEPS, devices=jax.devices('cpu')[:count],
                                    skip_preflight=True, compute_s_params=False)
    assert len(captured) == 1
    return result, captured[0]


def _bitwise_finite(want, got, label):
    want, got = np.asarray(want), np.asarray(got)
    assert np.isfinite(want).all(), f'nonfinite reference: {label}'
    assert np.isfinite(got).all(), f'nonfinite real values: {label}'
    assert want.shape == got.shape and want.dtype == got.dtype, label
    assert np.array_equal(np.ascontiguousarray(want).view(np.uint8),
                          np.ascontiguousarray(got).view(np.uint8)), f'bitwise mismatch: {label}'


@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('count', [2, 3])
def test_nonexchanged_ghosts_are_unread(boundary, count):
    if len(jax.devices('cpu')) < count:
        pytest.skip(f'requires {count} forced CPU devices')
    reference, clean = evaluate(boundary, count, False)
    actual, poisoned = evaluate(boundary, count, True)
    seen = poisoned.pop('_poison_seen').reshape(count, STEPS, -1)
    assert seen.all(), f'poison not applied at every step: {np.argwhere(~seen)[:10]}'
    exchanged = poisoned.pop('_exchanged')
    derived = {side: {name for i, name in enumerate(COMPONENTS)
                      if exchanged[1 if side == 'low' else 0, i, column]}
               for column, side in enumerate(('low', 'high'))}
    assert derived == {'high': {'ey', 'ez', 'hx'}, 'low': {'hy', 'hz'}}, derived
    for rank in range(count):
        for side, column in (('low', 0), ('high', 1)):
            interior = rank > 0 if side == 'low' else rank < count - 1
            assert {name for i, name in enumerate(COMPONENTS) if exchanged[rank, i, column]} == (
                derived[side] if interior else set())
    for name in COMPONENTS:
        _bitwise_finite(getattr(reference.state, name), getattr(actual.state, name), name)
    _bitwise_finite(reference.time_series, actual.time_series, 'probe records')
    _bitwise_finite(reference.dft_planes['plane'].accumulator,
                    actual.dft_planes['plane'].accumulator, 'DFT plane')
    clean_leaves, clean_tree = jax.tree.flatten_with_path(clean)
    poisoned_leaves, poisoned_tree = jax.tree.flatten_with_path(poisoned)
    assert clean_tree == poisoned_tree
    for (path, want), (_, got) in zip(clean_leaves, poisoned_leaves):
        family = path[0].key
        if family in ('fdtd', 'cpml') and want.ndim == 3:
            if family == 'fdtd' or not path[1].name.split('_')[-1].startswith('x'):
                want = want.reshape((count, -1) + want.shape[1:])[:, 1:-1]
                got = got.reshape((count, -1) + got.shape[1:])[:, 1:-1]
        _bitwise_finite(want, got, jax.tree_util.keystr(path))
    assert np.any(np.asarray(actual.time_series) != 0)
    jax.clear_caches()
