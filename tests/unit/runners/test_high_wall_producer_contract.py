"""High-wall material formulas and public probe parity across stepping lanes."""
import sys
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.core.yee import component_e_materials, component_h_materials
from rfx.core.yee import MaterialArrays
from rfx.model.materials import realize_components
from tests._multi_device_scene import host_devices


def numpy_wall_cells(array, walls):
    out = np.array(array, copy=True)
    for axis, wall in enumerate(walls):
        if wall:
            last = [slice(None)] * 3
            inside = [slice(None)] * 3
            last[axis], inside[axis] = -1, -2
            out[tuple(last)] = out[tuple(inside)]
    return out


def numpy_back(array, axis):
    return np.take(array, np.maximum(np.arange(array.shape[axis]) - 1, 0), axis=axis)


def numpy_e(array, walls):
    cell = numpy_wall_cells(array, walls)
    out = []
    for a, b in ((1, 2), (0, 2), (0, 1)):
        low_a, low_b = numpy_back(cell, a), numpy_back(cell, b)
        out.append((cell + low_a + low_b + numpy_back(low_a, b)) * np.float32(.25))
    return tuple(out)


def numpy_h(array, walls):
    cell = numpy_wall_cells(array, walls)
    return tuple(np.where(cell == numpy_back(cell, a), cell,
        np.float32(2) / (np.float32(1) / cell + np.float32(1) / numpy_back(cell, a)))
        for a in range(3))


def wall_scene(debye=False, *, graded=False):
    h = .001
    sim = Simulation(freq_max=10e9, domain=(17*h, 6*h, 5*h), dx=h,
        boundary=BoundarySpec(x='pmc', y='pec', z='pec'), cpml_layers=0,
        **({'dx_profile': np.full(17, h)} if graded else {}))
    poles = {'debye_poles': [DebyePole(3., 1 / (2*np.pi*50e9))]} if debye else {}
    sim.add_material('slab', eps_r=4., sigma=.125, mu_r=2., **poles)
    sim.add(Box((10*h, h, h), (17*h, 6*h, 4*h)), material='slab')
    sim.add_source((4*h, 3*h, 2*h), 'ez', waveform=lambda t: jnp.ones_like(t),
                   amplitude_kind='current')
    sim.add_probe((16*h, 2*h, 2*h), 'ez')
    return sim


def assert_components(actual, expected):
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('count,graded', [(2, False), (3, False), (1, True)],
                         ids=['distributed-2', 'distributed-3', 'graded'])
@pytest.mark.parametrize('debye', [False, True], ids=['static', 'debye'])
def test_wall_material_probe_lane_parity(debye, count, graded):
    """A dielectric ending at PMC keeps its near-wall record across lanes."""
    devices = host_devices(count)
    options = dict(n_steps=40, compute_s_params=False, skip_preflight=True)
    reference = wall_scene(debye).run(devices=devices[:1], **options)
    # Endpoint nodes: 17 x 6 x 5 cells, including each terminal node.
    assert reference.grid.shape == (18, 7, 6)
    shape = reference.grid.shape

    def indices(result):
        geometry = result.realized_geometry
        source = geometry.sources[0].edges
        probe = geometry.probes[0].index
        assert source and probe is not None
        # Only y and z have PEC faces in this fixture.
        assert all(0 < edge[a] < result.grid.shape[a] - 1
                   for edge in (*source, probe) for a in (1, 2))
        return f'driven_indices={source} probe_index={probe}'

    reference_indices = indices(reference)
    print(f'{debye=}/reference: {reference_indices}', file=sys.stderr)
    occupied = np.zeros(shape, dtype=bool)
    # Cell centres inside the independently declared slab, before wall images.
    occupied[10:17, 1:6, 1:4] = True
    eps = np.where(occupied, 4., 1.).astype(np.float32)
    sigma = np.where(occupied, .125, 0.).astype(np.float32)
    mu = np.where(occupied, 2., 1.).astype(np.float32)
    cells = MaterialArrays(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sigma),
                           mu_r=jnp.asarray(mu))
    walls = (True, True, True)
    actual_eps, actual_sigma = component_e_materials(cells, high_wall=walls)
    assert_components(actual_eps, numpy_e(eps, walls))
    assert_components(actual_sigma, numpy_e(sigma, walls))
    assert_components(component_h_materials(cells, high_wall=walls), numpy_h(mu, walls))
    print(f'{debye=}/component-formulas: max_error=0 bar=0 mesh={shape}', file=sys.stderr)
    wall = realize_components(cells, reference.grid, periodic=(False,) * 3, high_wall=walls)
    blind = realize_components(cells, reference.grid, periodic=(False,) * 3,
                               high_wall=(False,) * 3)
    contrast = max(float(np.max(np.abs(np.asarray(a)[-1] - np.asarray(b)[-1])))
                   for a, b in zip(wall.eps_update, blind.eps_update))
    print(f'{debye=}/wall-precondition: eps_difference={contrast:.12g} bar=>0', file=sys.stderr)
    assert contrast > 0, 'the high-wall image must change the terminal x row'

    result = wall_scene(debye, graded=graded).run(devices=devices[:count], **options)
    assert result.grid.shape == shape
    a, b = np.asarray(reference.time_series), np.asarray(result.time_series)
    assert a.shape == b.shape == (40, 1)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    # The two grid builders round their float64 Courant calculation differently;
    # the float32 stepping interval must agree, as must the probe records below.
    assert np.float32(result.dt) == np.float32(reference.dt)
    peak = float(np.max(np.abs(a)))
    assert peak > 0, 'the near-wall probe must see the source'
    error = np.abs(b.astype(np.float64) - a.astype(np.float64)).max(axis=1)
    ulp = float(np.spacing(np.float32(peak)))
    label = 'graded' if graded else f'distributed-{count}'
    row = f'{debye=}/{label}'
    print(f'{row}: devices={count} reference_dt={float(reference.dt):.17g} '
          f'compared_dt={float(result.dt):.17g} reference[{reference_indices}] '
          f'compared[{indices(result)}]', file=sys.stderr)
    if graded:
        # Uniform/constant-profile two-kernel bar:
        # tests/unit/nonuniform/test_nonuniform_face_depths.py:91-94.
        residual = float(error.max() / peak)
        bar = options['n_steps'] * 10 * np.finfo(np.float32).eps
        print(f'{row}: max_error={error.max():.12g} peak={peak:.12g} '
              f'residual={residual:.12g} bar={bar:.12g} steps=40 '
              f'mesh={result.grid.shape}', file=sys.stderr)
        assert residual < bar, f'{row}: residual {residual:.12g} >= {bar:.12g}'
    elif debye:
        # Pole cross-trace law: tests/_one_device_parity.py:110;
        # exercised by test_distributed_uniform_forward_media_parity.py:65.
        bar = 1.2e-7 * np.cumsum(np.abs(a.astype(np.float64)).max(axis=1))
        ratio = np.divide(error, bar, out=np.where(error > 0, np.inf, 0.), where=bar > 0)
        worst = int(np.argmax(ratio))
        print(f'{row}: worst_step={worst + 1} expected={a[worst, 0]:.12g} '
              f'measured={b[worst, 0]:.12g} error_at_worst={error[worst]:.12g} '
              f'allowance_at_worst={bar[worst]:.12g} '
              f'max_error={error.max():.12g} peak={peak:.12g} '
              f'peak_ULP={error.max()/ulp:.12g} max_law_ratio={ratio.max():.12g} '
              f'bar=1 steps=40 mesh={result.grid.shape}', file=sys.stderr)
        assert np.all(error <= bar), f'{row}: accumulated-law ratio {ratio.max():.12g} > 1'
    else:
        # One/several: tests/contracts/test_uniform_distributed_main_s.py:84.
        worst = int(np.argmax(error))
        print(f'{row}: worst_step={worst + 1} expected={a[worst, 0]:.12g} '
              f'measured={b[worst, 0]:.12g} absolute_bar={9 * ulp:.12g} '
              f'max_error={error.max():.12g} peak={peak:.12g} '
              f'peak_ULP={error.max()/ulp:.12g} bar=9 steps=40 '
              f'mesh={result.grid.shape}', file=sys.stderr)
        assert np.all(error <= 9 * ulp), f'{row}: {error.max()/ulp:.12g} peak ULP > 9'


def test_periodic_axis_is_never_a_wall():
    """A run-level periodic axis wraps even when the caller's flag says wall.

    ``unpadded_high_walls`` sees only ``grid.periodic_axes``; ``run(...,
    periodic=...)`` can make an unpadded axis periodic. There the edge mean
    must wrap row 0 onto row nx-1 as stored, not onto a replicated row.
    The expectation is numpy: roll on x, last-inside replication on y. It is
    compared at rtol 1e-6: random cells expose float32 summation order (one
    ULP), while a wall read on the periodic axis moves edges by order one.
    """
    rng = np.random.default_rng(1966)
    eps = rng.uniform(1., 5., (5, 4, 3)).astype(np.float32)
    sigma = rng.uniform(0., .5, (5, 4, 3)).astype(np.float32)
    mu = rng.uniform(1., 3., (5, 4, 3)).astype(np.float32)
    from rfx.core.yee import MaterialArrays
    cells = MaterialArrays(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sigma), mu_r=jnp.asarray(mu))
    periodic = (True, False, False)

    def close(actual, expected):
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6, atol=0)

    def back(array, axis):
        return np.roll(array, 1, axis=axis) if periodic[axis] else numpy_back(array, axis)

    def expected_e(array):
        cell = numpy_wall_cells(array, (False, True, False))
        out = []
        for a, b in ((1, 2), (0, 2), (0, 1)):
            low_a, low_b = back(cell, a), back(cell, b)
            out.append((cell + low_a + low_b + back(low_a, b)) * np.float32(.25))
        return tuple(out)

    cell = numpy_wall_cells(mu, (False, True, False))
    expected_h = tuple(np.where(cell == back(cell, a), cell,
        np.float32(2) / (np.float32(1) / cell + np.float32(1) / back(cell, a))) for a in range(3))
    actual_eps, actual_sigma = component_e_materials(cells, periodic, high_wall=(True, True, False))
    close(actual_eps, expected_e(eps))
    close(actual_sigma, expected_e(sigma))
    close(component_h_materials(cells, periodic, high_wall=(True, True, False)), expected_h)
