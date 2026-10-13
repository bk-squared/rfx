"""W1 slice 1: bitwise predictions for the canonical graded slab step."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from rfx import Simulation
from rfx.core.yee import MaterialArrays
from rfx.model.electric_metrics import stage_forward_materials
from rfx.nonuniform import make_current_source, run_nonuniform
from rfx.runners.distributed_nu import build_sharded_nu_grid, run_nonuniform_distributed_pec
from rfx.simulation import SourceSpec, ProbeSpec
from rfx.stepping.slab import cut
from rfx.stepping.slab_wrap import select_step

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


def scene(row='0.2', on=False, steps=12, kind='field'):
    sim = Simulation(freq_max=15e9, domain=(.022, .016, .012), dx=.001,
        boundary='pec', dy_profile=np.array([.001] + [.0009, .0011] * 7 + [.001]))
    grid = sim._build_nonuniform_grid()
    shape = grid.shape
    assert shape == (23, 17, 13)
    mat = MaterialArrays(jnp.ones(shape), jnp.zeros(shape), jnp.ones(shape))
    occ = None
    if row != 'none':
        if row == 'random':
            cells = np.random.default_rng(202).uniform(0, .85, shape).astype(np.float32)
            cells[0, 2:5, 1:4] = .5
            width = -(-shape[0] // 2)
            for seam in range(width, shape[0], width):
                cells[seam-1:seam+1, 1:4, 2:4] = .75
        else:
            cells = np.zeros(shape, np.float32)
            cells[5:7, 6:8, 4:6] = float(row)
        occ = jnp.asarray(cells)
    node = (6 if on else 3, 7, 5)
    sim.add_source((node[0] * .001, .007, .005), 'ez', amplitude_kind=kind)
    source = make_current_source(grid, node, 'ez',
        sim._ports[0].waveform, steps, mat, amplitude_kind=kind)
    probes = [(10, 7, 5, 'ez'), (14, 9, 6, 'ez'), (*node, 'ez'),
              (7, 7, 5, 'hx'), (7, 7, 5, 'hy'), (21, 7, 5, 'ez')]
    return grid, mat, occ, [source], probes


def single(data, steps):
    grid, mat, occ, sources, probes = data
    result = run_nonuniform(grid, mat, steps, cpml_axes='', pec_occupancy=occ,
        sources=sources, probes=probes, pec_faces={f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi')})
    return result['time_series'], result['state']


def several(data, steps, devices, **kw):
    grid, mat, occ, sources, probes = data
    selected = jax.devices('cpu')[:devices]
    mesh = Mesh(np.asarray(selected), ('x',))
    sg = build_sharded_nu_grid(grid, devices)
    result = run_nonuniform_distributed_pec(sg, stage_forward_materials(mat, sg.layout, mesh),
        None, steps, sources=[SourceSpec(*s) for s in sources], probes=[ProbeSpec(*p) for p in probes],
        n_devices=devices, devices=selected,
        sharded_pec_occupancy=cut(occ, sg.layout, 'pec_occupancy', mesh=mesh), **kw)
    return result['time_series'], result['final_state']


def difference(want, got):
    for name, a, b in [('probes', want[0], got[0])] + [
            (f, getattr(want[1], f), getattr(got[1], f)) for f in FIELDS]:
        a, b = np.asarray(a), np.asarray(b)
        unequal = a.view(np.uint32) != b.view(np.uint32)
        if np.any(unequal):
            cell = tuple(int(i) for i in np.argwhere(unequal)[0])
            return f'{name}{cell}: single={a[cell]!r}, slab={b[cell]!r}'
    return None


def assert_bits(want, got):
    assert difference(want, got) is None, difference(want, got)


@pytest.mark.parametrize('row', ['none', '0.2', 'random'])
@pytest.mark.parametrize('kind', ['field', 'current'])
def test_one_slab(row, kind):
    data = scene(row, True, kind=kind)
    assert_bits(single(data, 12), several(data, 12, 1))


@pytest.mark.xfail(strict=True, reason=(
    "W1-P1 falsified: step 3 on 2/3 devices; no occupancy, source off block: "
    "Ey[2,6,5] single=-6.180396e-05 slab=-6.180397e-05; "
    "each occupancy/source row also first differs at step 3 (REPORT.md)"))
@pytest.mark.parametrize('devices', [2, 3])
@pytest.mark.parametrize('steps', [12, 60, 240])
@pytest.mark.parametrize('on', [False, True], ids=['off', 'on'])
@pytest.mark.parametrize('row', ['none', '0.2', '0.5', '0.8', 'random'])
def test_p4_bitwise(row, on, steps, devices):
    data = scene(row, on, steps)
    assert_bits(single(data, steps), several(data, steps, devices))


@pytest.mark.parametrize('feature', ['cpml', 'debye', 'lorentz', 'ports', 'dft', 'flux', 'ntff', 'pmc', 'periodic_x'])
def test_selector(feature):
    assert select_step() == 'wrapped'
    assert select_step(**{feature: True}) == 'legacy'


def test_first_difference():
    """Opt-in localization: increasing n_steps, fields before accumulated probes."""
    import os
    if not os.environ.get('W1_LOCALIZE'):
        pytest.skip('set W1_LOCALIZE for first-difference report')
    for devices in (1, 2, 3):
        for row in ('none', '0.2', '0.5', '0.8', 'random'):
            for on in (False, True):
                for step in range(1, 13):
                    data = scene(row, on, step)
                    delta = difference(single(data, step), several(data, step, devices))
                    if delta:
                        print(f'FIRST devices={devices} row={row} on={on} step={step}: {delta}')
                        break
                else:
                    print(f'FIRST devices={devices} row={row} on={on}: equal through step 12')
