"""Interior UPML current coefficients and edge-specific pad admission."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation

EPS0 = 8.8541878128e-12


def _upml_sim(boundary='upml'):
    return Simulation(freq_max=8e9, domain=(.008,) * 3, dx=.001,
                      cpml_layers=2, boundary=boundary)


def _solve_witness(sim, lane):
    return getattr(sim, lane)(n_steps=2, skip_preflight=True,
                             **({'compute_s_params': False} if lane == 'run' else {}))


@pytest.mark.parametrize('lane', ['run', 'forward'])
@pytest.mark.parametrize('soft', [False, True])
def test_upml_interior_cell_owned_coefficient(lane, soft):
    sim = _upml_sim()
    sim.add_material('slab', eps_r=4.)
    sim.add(Box((.004, 0, 0), (.008,) * 3), material='slab')
    pos = (.004,) * 3
    if soft:
        sim.add_source(pos, 'ez', waveform=jnp.ones_like, amplitude_kind='current')
    else:
        sim.add_port(pos, 'ez', waveform=jnp.ones_like)
    sim.add_probe(pos, 'ez')
    result = _solve_witness(sim, lane)
    dt = sim._build_grid().dt
    # UPML interior: cell-owned eps=4, sigma_perp=0. This is not the
    # four-cell Yee interface mean 2.5. J=I/dV or port Norton current.
    sigma, current = (0., 1e9) if soft else (20., 20000.)
    reference = (dt / (4 * EPS0)) / (1 + sigma * dt / (2 * 4 * EPS0)) * current
    np.testing.assert_allclose(np.asarray(result.time_series).reshape(-1)[0],
                               reference, rtol=3e-7)


def _pad_sim(kind, axis, side):
    sim = _upml_sim()
    pos = [.004] * 3
    pos[axis] = -.001 if side < 0 else .009
    if kind in ('soft_current', 'soft_field'):
        sim.add_source(tuple(pos), 'ez', waveform=jnp.ones_like,
                       amplitude_kind=kind[5:])
    else:
        sim.add_port(tuple(pos), 'ez', waveform=jnp.ones_like,
                     excite=kind != 'passive', extent=.001 if kind == 'wire' else None)
    return sim


@pytest.mark.parametrize('lane', ['run', 'forward'])
@pytest.mark.parametrize('kind', ['port', 'soft_current', 'wire'])
@pytest.mark.parametrize('axis_side', [(0, -1), (1, 1)])
def test_upml_pad_current_drive_refused(lane, kind, axis_side):
    # An Ez edge in the x or y pad: the UPML update carries that pad's
    # conductivity and the drive's coefficient does not.
    with pytest.raises(ValueError, match='edge .* inside a UPML absorber pad'):
        _solve_witness(_pad_sim(kind, *axis_side), lane)


@pytest.mark.parametrize('lane', ['run', 'forward'])
@pytest.mark.parametrize('kind,axis_side', [
    ('passive', (0, -1)),       # a load only: nothing is injected
    ('soft_field', (1, 1)),     # the field increment is prescribed
    ('port', (2, -1)),          # Ez in the z pad: no perpendicular sigma
    ('soft_current', (2, -1)),
])
def test_upml_pad_admits_what_has_no_coefficient_error(lane, kind, axis_side):
    _solve_witness(_pad_sim(kind, *axis_side), lane)


@pytest.mark.parametrize('component', ['ex', 'ey', 'ez'])
def test_source_coefficient_is_the_upml_kernels_array(component):
    # Tie to the array the UPML E update multiplies the curl by, so a change
    # to that kernel cannot leave the source behind.
    from dataclasses import replace
    from rfx.boundaries.upml import init_upml
    from rfx.core.yee import MaterialArrays
    from rfx.grid import Grid
    from rfx.model.materials import e_update_coefficient_at, with_components
    grid = Grid(freq_max=8e9, domain=(.008,) * 3, dx=.001, cpml_layers=2)
    shape = grid.shape
    eps = jnp.ones(shape).at[shape[0] // 2:, :shape[1] // 2 + 1, :].set(4.)
    sigma = jnp.zeros(shape).at[shape[0] // 2, :, shape[2] // 2:].set(.3)
    materials = with_components(MaterialArrays(eps, sigma, jnp.ones(shape)),
                                grid, periodic=(False,) * 3)
    kernel = init_upml(grid, materials)
    materials = materials._replace(components=replace(
        materials.components, source_upml=True))
    cell = tuple(n // 2 for n in shape)
    got = e_update_coefficient_at(materials, cell, component, grid.dt)
    want = getattr(kernel, f'cb_{component}')[cell]
    np.testing.assert_allclose(float(got), float(want), rtol=3e-7)


@pytest.mark.parametrize('lane', ['run', 'forward'])
def test_cpml_pad_current_matches_plain_operator(lane):
    sim = _upml_sim('cpml')
    pos = (-.001, .004, .004)
    sim.add_port(pos, 'ez', waveform=jnp.ones_like)
    sim.add_probe(pos, 'ez')
    result = _solve_witness(sim, lane)
    dt = sim._build_grid().dt
    reference = dt / (EPS0 + 20 * dt / 2) * 20000
    np.testing.assert_allclose(np.asarray(result.time_series).reshape(-1)[0],
                               reference, rtol=3e-7)


@pytest.mark.parametrize('inverse', [False, True])
def test_tensor_audit_predicate_is_edge_specific(inverse):
    from rfx.core.yee import MaterialArrays
    from rfx.model.materials import with_components
    from rfx.model.source_coefficients import tensor_replaced_edges
    ones = jnp.ones((3,) * 3)
    materials = with_components(MaterialArrays(ones, 0 * ones, ones), None,
                                periodic=(False,) * 3)
    tensors = (ones, ones.at[1, 1, 1].set(.25 if inverse else 4.), ones)
    mask = tensor_replaced_edges(materials, **{
        'aniso_inv_eps' if inverse else 'aniso_eps': tensors})
    assert [int(jnp.sum(part)) for part in mask] == [0, 1, 0]
    assert bool(mask[1][1, 1, 1])
