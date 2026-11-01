"""Normal-incidence Floquet splitting obeys empty-cell and reflector physics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.core.yee import init_state, init_materials
from rfx.errors import RfxRefusal
from rfx.floquet import init_floquet_dft, compute_floquet_s_params
from rfx.geometry.rasterize_grid import unpadded_high_walls
from rfx.grid import Grid
from rfx.measurement.floquet import update_floquet_dft
from rfx.stepping.yee import update_h, update_e


def _cell(scene, axis=2):
    """Uniform soft TE sheet, transverse periodicity, normal CPML, 1.5 mm mesh."""
    dx = .0015
    normal = 'xyz'[axis]
    domain = [.015] * 3
    domain[axis] = .090
    grid = Grid(
        freq_max=10e9, domain=tuple(domain), dx=dx, cpml_layers=8,
        cpml_axes=normal, periodic_axes=''.join(a for a in 'xyz' if a != normal),
    )
    dt = float(grid.dt)
    walls = unpadded_high_walls(grid)
    periodic = tuple(a != axis for a in range(3))
    src = grid.axis_pads[axis] + round(.015 / dx)
    sheet = src + round(.045 / dx)
    planes = [src + round(p / dx) for p in (.006, .012, .018)]
    shape = tuple(grid.shape[a] for a in range(3) if a != axis)
    freqs = jnp.arange(3., 4.7501, .25) * 1e9
    ecomp = 'ey' if axis == 0 else 'ex'

    def sl(k):
        index = [slice(None)] * 3
        index[axis] = k
        return tuple(index)

    materials = init_materials(grid.shape)
    if scene == 'slab':
        materials = materials._replace(
            eps_r=materials.eps_r.at[sl(slice(sheet, sheet + round(.012 / dx)))].set(4.)
        )
    cp, cs = init_cpml(grid)
    state = init_state(grid.shape)
    accs = tuple(init_floquet_dft(len(freqs), shape) for _ in planes)
    tau = 1 / (5e9 * .5 * np.pi)
    t0 = 3 * tau

    def step(carry, n):
        state, cs, accs = carry
        state = update_h(state, materials, dt, dx, periodic, high_wall=walls)
        state, cs = apply_cpml_h(state, cp, cs, grid, normal, materials=materials, periodic=periodic)
        state = update_e(state, materials, dt, dx, periodic, high_wall=walls)
        state, cs = apply_cpml_e(state, cp, cs, grid, normal, materials=materials)
        if scene == 'pec':
            state = state._replace(
                ex=state.ex.at[sl(sheet)].set(0.), ey=state.ey.at[sl(sheet)].set(0.)
            )
        arg = (n * dt - t0) / tau
        state = state._replace(**{
            ecomp: getattr(state, ecomp).at[sl(src)].add(-2 * arg * jnp.exp(-arg**2))
        })
        accs = tuple(
            update_floquet_dft(acc, state, k, axis, freqs, dt, n)
            for acc, k in zip(accs, planes)
        )
        return (state, cs, accs), None

    # The pulse has crossed all reflection planes before the record ends.
    _, _, accs = jax.jit(lambda carry: jax.lax.scan(step, carry, jnp.arange(512))[0])(
        (state, cs, accs)
    )
    return np.stack([
        np.asarray(compute_floquet_s_params(accs[0], acc, None, dx, .015, .015, freqs)['S11'])
        for acc in accs
    ])


def test_empty_cell_floor():
    # Vacuum has no physical interface and therefore zero reflected amplitude.
    s11 = _cell('empty')
    assert np.all(np.abs(s11) <= 5e-3), np.abs(s11)


def test_pec_total_reflection():
    # A lossless PEC sheet reflects all incident power: |Gamma| = 1.
    s11 = _cell('pec')
    assert np.all(np.abs(np.abs(s11) - 1) <= 1e-2), np.abs(s11)


@pytest.fixture(scope='module')
def z_slab():
    return _cell('slab')


@pytest.mark.parametrize('axis', [0, 1], ids=['x', 'y'])
def test_slab_orientation(axis, z_slab):
    # Rotating an isotropic dielectric and its TE drive preserves complex S11.
    s11 = _cell('slab', axis)
    assert np.all(np.abs(s11 - z_slab) <= 1e-3), np.abs(s11 - z_slab)


@pytest.mark.parametrize('axis', [0, 1, 2], ids=['x', 'y', 'z'])
def test_port_index_zero_refused(axis):
    state = init_state((3, 3, 3))
    acc = init_floquet_dft(1, (3, 3))
    with pytest.raises(RfxRefusal, match=r'port_index.*at least 1.*0'):
        update_floquet_dft(acc, state, 0, axis, jnp.array([4e9]), 1e-12, 0)
