"""Always-on representation, held-memory and content-key pins (design B8)."""
from dataclasses import replace
import gc
import sys
import time
import tracemalloc

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.measurement.plan import Node, NodeArrays, build_measurement_plan, node_arrays

DX = 1e-3
FREQS = np.array([0.8e9, 1e9], np.float32)


def a1_scene(moment):
    # A 33 x 33 x 20-cell moment slab in 8-cell blocks on a graded mesh.
    dz = np.full(44, DX)
    dz[34:36] = (.9*DX, 1.1*DX)
    sim = Simulation(freq_max=2e9, domain=(56*DX, 56*DX, 44*DX), dx=DX,
                     boundary='cpml', cpml_layers=8, dx_profile=np.full(56, DX),
                     dy_profile=np.full(56, DX), dz_profile=dz)
    sim.add_source((28*DX, 28*DX, 22*DX), 'ez', amplitude_kind='current',
                   waveform=lambda t: jnp.exp(-((t-8e-12)/5e-12)**2))
    if moment:
        sim.add_current_moment_monitor((12*DX,)*3, (44*DX, 44*DX, 32*DX),
                                       block_size=8*DX, freqs=FREQS)
    return sim


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_four_owner_kinds_use_node_arrays(mesh):
    kwargs = {}
    if mesh == 'graded':
        dz = np.full(16, DX)
        dz[5:7] = (.9*DX, 1.1*DX)
        kwargs = dict(dx_profile=np.full(16, DX), dy_profile=np.full(16, DX), dz_profile=dz)
    sim = Simulation(freq_max=2e9, domain=(16*DX,)*3, dx=DX,
                     boundary='cpml', cpml_layers=2, **kwargs)
    lo, hi = (3*DX,)*3, (12*DX,)*3
    sim.add_current_moment_monitor(lo, hi, block_size=3*DX, freqs=FREQS)
    sim.add_ntff_box(lo, hi, freqs=FREQS)
    sim.add_dft_plane_probe(axis='z', coordinate=8*DX, component='ex', freqs=FREQS)
    sim.add_flux_monitor(axis='z', coordinate=8*DX, freqs=FREQS)
    grid = sim._build_realized_grid()
    plan = build_measurement_plan(sim, grid, n_steps=2)
    owners = {o.kind: o for o in plan.owners}
    assert {'current_moment', 'ntff', 'dft_plane', 'flux'} <= owners.keys()
    for kind in ('current_moment', 'ntff', 'dft_plane', 'flux'):
        for channel in owners[kind].channels:
            assert isinstance(channel.nodes, NodeArrays)
            for name, dtype in (('component', np.int8), ('i', np.int32),
                                ('j', np.int32), ('k', np.int32)):
                column = getattr(channel.nodes, name)
                assert column.dtype == dtype and not column.flags.writeable
            assert not channel.nodes.weight.flags.writeable
            if channel.nodes.area is not None:
                assert channel.nodes.area.dtype == np.float64
                assert not channel.nodes.area.flags.writeable
        nodes = owners[kind].channels[0].nodes
        original = tuple(nodes)
        assert nodes == original and original == nodes
        assert hash(nodes) == hash(node_arrays(original))
        assert len(repr(nodes)) < 80
        for change in ({'weight': original[0].weight+1}, {'i': original[0].i+1},
                       {'component': 'hy' if original[0].component != 'hy' else 'ex'}):
            changed = (replace(original[0], **change), *original[1:])
            assert nodes != changed
            assert nodes.content_key() != node_arrays(changed).content_key()
    electric, previous, magnetic, next_e, _, next_h = owners['current_moment'].channels[:6]
    for name in ('component', 'i', 'j', 'k'):
        assert getattr(electric.nodes, name) is getattr(previous.nodes, name)
        assert getattr(electric.nodes, name) is getattr(next_e.nodes, name)
        assert getattr(magnetic.nodes, name) is getattr(next_h.nodes, name)


def test_key_normal_form_and_sequence():
    expected = (Node('ex', 2, 3, 4, float(np.float32(.1)), 2.),
                Node('hz', -1, 5, 6, -2., 3.))
    arrays = NodeArrays(np.array([0, 5]), np.array([2, -1]), np.array([3, 5]),
                        np.array([4, 6]), np.array([.1, -2], np.float32), np.array([2., 3.]))
    assert arrays == expected
    assert hash(arrays) == hash(node_arrays(expected))
    assert node_arrays(arrays) is arrays
    assert arrays[-1] == expected[-1]
    assert arrays[::-1] == expected[::-1]
    assert arrays[:0] == ()
    assert arrays.content_key() is arrays.content_key()
    for node in arrays:
        assert type(node.component) is str
        assert all(type(getattr(node, name)) is int for name in 'ijk')
        assert type(node.weight) is float and type(node.area) is float
    with pytest.raises(IndexError):
        _ = arrays[2]
    with pytest.raises(ValueError):
        arrays.i[0] = 0
    # The accumulator check deliberately excludes area; the monitor check does not.
    other = replace(arrays, area=np.array([4., 3.]))
    assert arrays != other
    assert arrays.content_key(area=False) == other.content_key(area=False)


def test_a1_held_memory(capsys):
    def measure(moment):
        sim = a1_scene(moment)
        grid = sim._build_realized_grid()
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        start = time.perf_counter()
        plan = build_measurement_plan(sim, grid, n_steps=2)
        elapsed = time.perf_counter()-start
        gc.collect()
        held = tracemalloc.get_traced_memory()[0]-before
        tracemalloc.stop()
        if moment:
            m = next(o.source for o in plan.owners if o.kind == 'current_moment')
            cells = (m.i_hi-m.i_lo)*(m.j_hi-m.j_lo)*(m.k_hi-m.k_lo)
        else:
            cells = 0
        return held, elapsed, cells

    # Warm imports and geometry/JAX caches equally before measuring retained plans.
    for moment in (False, True):
        sim = a1_scene(moment)
        grid = sim._build_realized_grid()
        start = time.perf_counter()
        plan = build_measurement_plan(sim, grid, n_steps=2)
        build_time = time.perf_counter()-start
        del plan, sim, grid
    baseline, baseline_time, _ = measure(False)
    held, elapsed, cells = measure(True)
    assert cells == 33*33*20
    per_cell = (held-baseline)/cells
    with capsys.disabled():
        print(f' B8_A1 cells={cells} held_bytes={held-baseline} bytes_per_cell={per_cell:.3f}'
              f' traced_build_s={elapsed:.6f} baseline_s={baseline_time:.6f}'
              f' build_s={build_time:.6f}', file=sys.stderr, flush=True)
    assert per_cell <= 1500, f'A1 retained {per_cell:.3f} bytes per slab cell'
