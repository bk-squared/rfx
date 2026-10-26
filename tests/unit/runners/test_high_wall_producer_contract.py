"""H4/F5: actual lane operands against independent cell-to-edge formulas."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation, _realized
from rfx.boundaries.spec import BoundarySpec
from rfx.core.yee import component_e_materials, component_h_materials
from rfx.geometry.rasterize_grid import unpadded_high_walls
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


def wall_scene(debye=False):
    h = .001
    sim = Simulation(freq_max=10e9, domain=(17*h, 6*h, 5*h), dx=h,
        boundary=BoundarySpec(x='pmc', y='pec', z='pec'), cpml_layers=0)
    poles = {'debye_poles': [DebyePole(3., 1 / (2*np.pi*50e9))]} if debye else {}
    sim.add_material('slab', eps_r=4., sigma=.125, mu_r=2., **poles)
    sim.add(Box((10*h, h, h), (17*h, 6*h, 4*h)), material='slab')
    sim.add_source((4*h, 3*h, 2*h), 'ez', waveform=lambda t: jnp.ones_like(t),
                   amplitude_kind='current')
    sim.add_probe((12*h, 3*h, 2*h), 'ez')
    return sim


def assert_components(actual, expected):
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('devices', [1, 2, 3], ids=['uniform', 'distributed-2', 'distributed-3'])
@pytest.mark.parametrize('debye', [False, True], ids=['static', 'debye'])
def test_lane_uses_wall_materials(monkeypatch, devices, debye):
    run_devices = host_devices(devices)
    import rfx.model.materials as model
    import rfx.materials.debye as ade
    import rfx.runners._distributed_common as common

    sim = wall_scene(debye)
    grid = sim._build_realized_grid()
    cells, pole_spec, *_ = sim._assemble_materials(grid)
    walls = unpadded_high_walls(grid)
    eps = numpy_e(np.asarray(cells.eps_r), walls)
    sigma = numpy_e(np.asarray(cells.sigma), walls)
    mu = numpy_h(np.asarray(cells.mu_r), walls)
    fractions = numpy_e(np.asarray(pole_spec[1][0], dtype=np.float32), walls) if debye else None
    # The independent formula also pins the public S1 and S2 contracts.
    actual_eps, actual_sigma = component_e_materials(cells, high_wall=walls)
    assert_components(actual_eps, eps)
    assert_components(actual_sigma, sigma)
    assert_components(component_h_materials(cells, high_wall=walls), mu)

    realize = model.realize_components
    def observe_realization(*args, **kwargs):
        out = realize(*args, **kwargs)
        # The slab ADE initializer also realizes a one-cell vacuum placeholder.
        # Observe domain/slab operands, whose transverse extent is the domain's.
        if out.eps_update[0].shape[1:] == grid.shape[1:]:
            _realized.ACTIVE.observe('contract.producer', dict(
                eps_e=out.eps_update, sigma_e=out.sigma_update, mu_h=out.mu_update))
        return out
    monkeypatch.setattr(model, 'realize_components', observe_realization)
    build_poles = ade.debye_pole_coeffs
    def observe_poles(poles, dt, shape, weights):
        # Point-source coefficient lookup uses a cropped 2x2x2 neighborhood;
        # the lane contract compares the domain's update coefficients.
        if tuple(shape[1:]) == grid.shape[1:]:
            _realized.ACTIVE.observe('contract.poles', dict(fractions=weights[0]))
        return build_poles(poles, dt, shape, weights)
    monkeypatch.setattr(ade, 'debye_pole_coeffs', observe_poles)
    monkeypatch.setattr(common, 'debye_pole_coeffs', observe_poles)

    with _realized.capture() as captured:
        sim.run(n_steps=4, devices=run_devices, compute_s_params=False,
                skip_preflight=True)
    electric = [r for r in captured.records if 'eps_e' in r]
    magnetic = [r for r in captured.records if 'mu_h' in r]
    pole_records = [r for r in captured.records if 'fractions' in r]
    assert electric and magnetic
    nx = grid.nx
    per = (nx + devices - 1) // devices

    def compare_record(actual, expected, record, *, magnetic=False):
        if np.asarray(actual[0]).shape[0] == nx:
            assert_components(actual, expected)
            return set(range(0, nx, per))
        key = 'h_owned_start' if magnetic else 'owned_start'
        starts = [int(record[key])] if key in record else list(range(0, nx, per))
        matches = []
        for start in starts:
            count = min(per, nx-start)
            matches.append(all(np.array_equal(np.asarray(got)[1:1+count], want[start:start+count])
                               for got, want in zip(actual, expected)))
        assert any(matches), (record['site'], starts, 'no global owned slab matches')
        return {start for start, match in zip(starts, matches) if match}

    electric_starts, magnetic_starts, pole_starts = set(), set(), set()
    for record in electric:
        electric_starts.update(compare_record(record['eps_e'], eps, record))
        compare_record(record['sigma_e'], sigma, record)
    for record in magnetic:
        magnetic_starts.update(compare_record(record['mu_h'], mu, record, magnetic=True))
    assert electric_starts == magnetic_starts == set(range(0, nx, per))
    if devices > 1:
        assert any('owned_start' in r and int(r['owned_start']) <= nx-1
                   < int(r['owned_start']) + int(r['owned_count']) for r in electric)
    if debye:
        assert pole_records
        for record in pole_records:
            pole_starts.update(compare_record(record['fractions'], fractions, record))
        assert pole_starts == set(range(0, nx, per))


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
