"""Whole-grid setup ownership at each one-device time loop."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


PATHS = ('bar', 'decay', 'graded', 'graded_decay', 'forward', 'graded_forward', 's', 'rlc')


def cpu_environment(root=None):
    return {**os.environ, 'JAX_PLATFORMS': 'cpu',
            'XLA_FLAGS': '--xla_force_host_platform_device_count=1',
            'PYTHONPATH': str(root or Path(__file__).resolve().parents[3])}


@pytest.mark.parametrize('path', PATHS)
def test_one_device_setup_release(path):
    run = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'pin', path],
                         env=cpu_environment(), text=True, capture_output=True, timeout=240)
    print(run.stdout, end='')
    assert run.returncode == 0, run.stdout + run.stderr


def build_scene(variant='bar'):
    import jax.numpy as jnp
    import numpy as np
    from rfx import Simulation
    shape = (64, 32, 40)
    domain = tuple((n - 1 - 16) * 1e-3 for n in shape)
    options = {}
    if variant.startswith('graded'):
        options['dz_profile'] = np.asarray([.001] + [.0009, .0011] * 11)
    sim = Simulation(freq_max=15e9, domain=domain, dx=1e-3,
                     boundary='cpml', cpml_layers=8, precision='float32', **options)
    grid = sim._build_nonuniform_grid() if options else sim._build_grid()
    assert tuple(grid.shape) == shape
    center = tuple(d / 2 for d in domain)
    if variant == 's':
        for x in (8e-3, 30e-3):
            sim.add_port((x, center[1], center[2]), 'ez', impedance=50.)
    else:
        sim.add_port((8e-3, center[1], center[2]), 'ez', impedance=50.,
                     extent=.003, excite=True)
    if variant == 'rlc':
        from rfx import Box
        sim.add(Box((.020, .003, .005), (.024, .007, .009)), material='pec')
        sim.add_lumped_rlc((.032, .010, .014), 'ex', R=50., L=1e-9, topology='series')
    sim.add_dft_plane_probe(axis='z', coordinate=center[2],
                           freqs=jnp.asarray([3e9, 7.5e9, 12e9], jnp.float32), name='plane')
    sim.add_probe(center, 'ez')
    return sim


def observe(path):
    import gc
    import math
    import weakref
    from dataclasses import fields, is_dataclass
    import jax
    from rfx.stepping import prepared
    import rfx.stepping.uniform_loop as uniform
    import rfx.stepping.graded_loop as graded
    import rfx.model.conductors as products
    observations = []
    kept = {}
    shape = (64, 32, 40)
    original_advance = prepared.advance
    original_solve = products.realized_conductors

    def arrays(value, prefix):
        if isinstance(value, jax.Array):
            if len(value.shape) >= 3 and tuple(value.shape[-3:]) == shape:
                yield prefix, value
        elif is_dataclass(value):
            for field in fields(value):
                yield from arrays(getattr(value, field.name), prefix + '.' + field.name)
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from arrays(item, prefix + '.' + str(key))
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                yield from arrays(item, prefix + '.' + str(i))

    def advance(product, stepper, *args, **kwargs):
        if 'decay' in path:
            names = ['eps_r', 'mu_r']
            if kwargs.get('advisory') is None or kwargs['advisory']['lossy'] is None:
                names.append('sigma')
            kept.clear()
            kept.update((name, weakref.ref(getattr(product.materials, name)))
                        for name in names if getattr(product.materials, name) is not None)
        if path == 'rlc':
            from rfx.model.source_admission import LumpedRLCInputs
            inputs = [value for value in (*args, *kwargs.values())
                      if isinstance(value, LumpedRLCInputs)]
            assert len(inputs) == 1
            expected_ids = {field.name: id(getattr(product, field.name))
                            for field in fields(product)}
            def entry(*a, **k):
                _, snapshot = inputs[0].admission
                assert snapshot is not product
                assert {field.name: id(getattr(snapshot, field.name))
                        for field in fields(snapshot)} == expected_ids
                del snapshot
                return stepper(*a, **k)
            result = original_advance(product, entry, *args, **kwargs)
            assert inputs[0].admission is None
            return result
        return original_advance(product, stepper, *args, **kwargs)

    def solve(*args, **kwargs):
        product = original_solve(*args, **kwargs)
        if path == 's':
            kept.clear()
            for name in ('materials', 'pec_cells', 'pec_edges', 'sheets', 'wires',
                         'sheet_impedance', 'geometry_masks', 'assembly_entries',
                         'assembly', 'sheet_operator'):
                kept.update((label, weakref.ref(value)) for label, value in
                            arrays(getattr(product, name), 'fixture.' + name))
        return product

    def buffers(array):
        if not isinstance(array, jax.Array) or len(array.shape) < 3 or tuple(array.shape[-3:]) != shape:
            return set()
        stride = math.prod(shape) * array.dtype.itemsize
        return {(str(array.device), int(array.unsafe_buffer_pointer()) + i * stride)
                for i in range(math.prod(array.shape[:-3]))}

    def wrap(original, label):
        def scan(body, init, xs=None, length=None, **kwargs):
            if not any(isinstance(a, jax.Array) for a in jax.tree.leaves(init)):
                return original(body, init, xs, length=length, **kwargs)
            closed = jax.make_jaxpr(
                lambda c, x: original(body, c, x, length=length, **kwargs))(init, xs)
            arguments = jax.tree.leaves((init, closed.consts, xs))
            pointers = set().union(*(buffers(a) for a in arguments))
            gc.collect()
            assert all(ref() is not None for ref in kept.values()), (path, 'kept array died')
            expected = set().union(*(buffers(ref()) for ref in kept.values())) - pointers
            outside = set().union(*(buffers(a) for a in jax.live_arrays())) - pointers
            assert outside == expected, (path, label, 'outside', sorted(outside),
                                         'expected', sorted(expected), 'kept', tuple(kept), 'arrays',
                                         [(str(a.dtype), a.shape, int(a.unsafe_buffer_pointer()))
                                          for a in jax.live_arrays() if buffers(a) & outside])
            observations.append((label, len(pointers), len(outside), tuple(kept)))
            return original(body, init, xs, length=length, **kwargs)
        return scan

    prepared.advance = advance
    products.realized_conductors = solve
    if path == 'decay':
        original_scan = jax.lax.scan
        def scan(body, init, xs=None, length=None, **kwargs):
            import inspect
            caller = inspect.currentframe().f_back
            if caller.f_code.co_name == 'run_chunk' and caller.f_globals.get('__name__') == uniform.__name__:
                jax.lax.scan = original_scan
                try:
                    return wrap(original_scan, 'uniform_loop.run_until_decay.run_chunk:jax.lax.scan')(
                        body, init, xs, length=length, **kwargs)
                finally:
                    jax.lax.scan = scan
            return original_scan(body, init, xs, length=length, **kwargs)
        jax.lax.scan = scan
    else:
        module = graded if path.startswith('graded') else uniform
        module.recorded_scan = wrap(module.recorded_scan, module.__name__ + '.recorded_scan')
    sim = build_scene(path)
    if path == 's':
        from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
        result = compute_lumped_wire_s_matrix_via_scan(
            sim, [3e9, 7.5e9, 12e9], n_steps=20, skip_preflight=True)
    elif 'decay' in path:
        result = sim.run(until_decay=1e-3, decay_min_steps=10, decay_check_interval=10,
                         decay_max_steps=40, compute_s_params=False, skip_preflight=True)
    elif 'forward' in path:
        result = sim.forward(n_steps=20, skip_preflight=True)
    else:
        result = sim.run(n_steps=20, compute_s_params=False, skip_preflight=True,
                         **({"conformal_pec": True} if path == 'rlc' else {}))
    jax.block_until_ready(result if path == 's' else result.time_series)
    assert observations, (path, 'no time loop observed')
    print('PIN', path, observations, flush=True)


if __name__ == '__main__':
    observe(sys.argv[2])
