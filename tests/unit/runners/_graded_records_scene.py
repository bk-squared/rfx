"""One-device public calls are the oracle for graded slab records."""
import json
import os
from pathlib import Path

import jax
import numpy as np
from tests._multi_device_scene import host_devices
from tests.unit.runners.test_distributed_graded_run import model as base_model

def model(boundary, n_devices):
    host_devices(n_devices)
    sim = base_model(boundary)
    # Both transverse electric components reach each observed plane.
    sim.add_source((8.1e-3, 3.2e-3, 4.3e-3), 'ex', amplitude_kind='field')
    grid = sim._build_nonuniform_grid()
    seam = (grid.shape[0] + n_devices - 1) // n_devices
    return sim, grid, seam


def coordinate(grid, axis, index):
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    coords = coords_from_nonuniform_grid(grid)
    return float(np.asarray(getattr(coords, 'xyz'[axis]))[index])


def measure(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    delta = float(np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128))))
    peak = float(np.max(np.abs(b)))
    return dict(equal=bool(np.array_equal(a, b)), max_abs=delta, peak=peak,
                relative=delta / peak if peak else delta)


def with_bars(values, rule):
    """Keep the measured difference and its declared acceptance bar together."""
    for value in values.values():
        if rule == 'array_equal':
            bar, passed = 0., value['equal']
        elif rule == 'relative_peak':
            bar = 1e-4 * value['peak']
            passed = value['max_abs'] <= bar
        elif rule == 'snapshot_ulp':
            bar = 9 * float(np.spacing(np.float32(value['peak'])))
            passed = value['max_abs'] <= bar
        else:
            raise ValueError(rule)
        value.update(rule=rule, bar=bar, passed=passed)
    return values


def record(kind, metadata, values):
    report = dict(kind=kind, test=os.environ.get("PYTEST_CURRENT_TEST"), **metadata, values=values)
    target = os.environ.get('RFX_GRADED_RECORDS_REPORT')
    if target:
        with Path(target).open('a') as f:
            f.write(json.dumps(report) + '\n')


def pair(sim, n_devices, entry='run', steps=24, **kwargs):
    call = getattr(sim, entry)
    one = call(n_steps=steps, skip_preflight=True, **kwargs)
    many = call(n_steps=steps, skip_preflight=True, devices=host_devices(n_devices),
                **({'distributed': True} if entry == 'forward' else {}), **kwargs)
    jax.block_until_ready(many)
    return one, many


def mutation_result(kind, mutation, failure):
    record('mutation', dict(record=kind, mutation=mutation, status='RED'),
           dict(failure=str(failure)))


def radiating_model(boundary, n_devices, pole=False, grading='x'):
    from rfx import Simulation, Box, DebyePole
    import jax.numpy as jnp
    host_devices(n_devices)
    profile = (np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 13), 1e-3] if grading == 'x'
               else np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 6), 1e-3])
    sim = Simulation(freq_max=15e9, domain=(15e-3, 8e-3, 8e-3), dx=1e-3,
        boundary=boundary, cpml_layers=2, **{f'd{grading}_profile': profile})
    sim.add_source((5.1e-3, 3.2e-3, 4.3e-3), 'ez', amplitude_kind='field',
        waveform=lambda t: jnp.exp(-((t-1.2e-11)/5e-12)**2))
    if pole:
        sim.add_material('ade', eps_r=2.3, debye_poles=[DebyePole(1.7, 1e-11)])
        sim.add(Box((3e-3, 2e-3, 2e-3), (10e-3, 6e-3, 6e-3)), material='ade')
    grid = sim._build_nonuniform_grid()
    return sim, grid, (grid.shape[0] + n_devices - 1) // n_devices
