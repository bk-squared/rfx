"""PR F's independent physical masks, metrics and forward S orchestration."""
import gc
import json
import os
import threading
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.unit.runners._msl_multidevice_scene import FREQS, _json


def emit(bar, **values):
    print('MSL_FORWARD ' + json.dumps(_json(dict(bar=bar, **values))), flush=True)


@contextmanager
def sampled_memory():
    """Job cgroup usage, sampled every second and at both endpoints like J0g."""
    current = Path('/sys/fs/cgroup/memory.current')
    peak = [int(current.read_text())]
    done = threading.Event()
    def sample():
        while not done.wait(1):
            peak[0] = max(peak[0], int(current.read_text()))
    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        yield peak
    finally:
        peak[0] = max(peak[0], int(current.read_text()))
        done.set()
        sampler.join()


@pytest.fixture(autouse=True)
def whole_case(request):
    """Reset Linux's process RSS high-water mark before measuring each case."""
    gc.collect()
    jax.clear_caches()
    reset = Path('/proc/self/clear_refs')
    if reset.exists():
        reset.write_text('5')
    started = perf_counter()
    yield
    status = Path('/proc/self/status').read_text()
    peak = int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmHWM:'))) * 1024
    emit('cost', case=request.node.nodeid, seconds=perf_counter() - started,
         peak_bytes=peak, peak_gib=peak / 2**30)
    jax.clear_caches()
    gc.collect()


def parameter_blocks(sim, name):
    """Half-open node-coordinate boxes; no nearest-node rounding of box edges."""
    grid = sim._build_realized_grid()
    assemble = sim._assemble_materials_nu if sim._uses_nonuniform_mesh else sim._assemble_materials
    base = jnp.asarray(assemble(grid, pec_sheets=[], pec_wires=[])[0].eps_r, jnp.float32)
    if name.startswith('T'):
        boxes = [((lo, .0034, 0.), (hi, .0046, .0008)) for lo, hi in
                 ((.0030, .0050), (.0058, .0070), (.0076, .0090), (.0019, .0021))]
        expected = [list(range(23, 33)), list(range(37, 43)), list(range(46, 53)), [18]]
    else:
        boxes = [((lo, .0040, 0.), (hi, .0080, .0008)) for lo, hi in
                 ((.0030, .0040), (.0040, .0044), (.0044, .0048))]
        boxes.append(((.0030, .0019, 0.), (.0048, .0021, .0008)))
        expected = [list(range(23, 28)), list(range(28, 30)), list(range(30, 32)), list(range(23, 32))]
    coords = [np.array([grid.node_of(a, i) for i in range(size)]) for a, size in zip('xyz', grid.shape)]
    masks = []
    for k, (lo, hi) in enumerate(boxes):
        axes = [(c >= low) & (c < high) for c, low, high in zip(coords, lo, hi)]
        mask = axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]
        xs = np.flatnonzero(mask.any(axis=(1, 2))).tolist()
        assert xs == expected[k], (name, k, xs, expected[k])
        ys = range(25, 31) if name.startswith('T') else ([18] if k == 3 else range(28, 48))
        # J0g's full cell lists, independently fixed by the physical boxes.
        expected_cells = list(product(expected[k], ys, range(8, 12)))
        np.testing.assert_array_equal(np.argwhere(mask), expected_cells)
        masks.append(mask.astype(np.float32))
    emit('blocks', scene=name, x_indices=expected,
         cells=[np.argwhere(mask).tolist() for mask in masks])
    masks = jnp.asarray(np.stack(masks))
    return base, masks


def eps_of(base, masks, p):
    return base + jnp.sum(p[:, None, None, None] * masks, axis=0)


def forward_s(sim, eps, length, devices=None, spy=None):
    """D:f77a43 route: shared preparation, forward planes, shared assembly."""
    from rfx.sparams import msl as calc, _msl_stages as stages
    from rfx.sources.msl_port import (
        msl_h_plane_stencil, msl_port_from_entry, msl_probe_x_coords_n,
        msl_ampere_pair, msl_cell, msl_cross_section_span, msl_physical_point,
    )
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    grid = sim._build_realized_grid()
    entries = sim._resolve_msl_probe_entries(grid)
    ports, counts, xs, stencils, _ = stages._prepare_msl_probes(
        entries, grid, msl_h_plane_stencil, msl_port_from_entry, msl_probe_x_coords_n, sim)
    dz, meta, beta, z0, eps_ref, line_eps, refs, trace_k = stages._prepare_msl_geometry(
        calc._msl_cell_profile, entries, eps, FREQS, grid, hammerstad_jensen_z0_eps_eff,
        sim._uses_nonuniform_mesh, jax, msl_ampere_pair, msl_cell,
        msl_cross_section_span, ports, len(entries), sim)
    observers = calc.declaration_variant(sim, _internal_probe_indices=set(sim._internal_probe_indices))
    *_, observers = stages._prepare_msl_witnesses(
        entries, msl_physical_point, len(entries), observers, meta, xs)
    records = []
    for driven in range(len(entries)):
        variant, enames, hnames = stages._prepare_msl_drive(
            calc.GaussianPulse, calc._MSLPortEntry, calc._register_msl_h_planes,
            calc.declaration_variant, driven, entries, FREQS, stencils, jnp,
            ports, len(entries), observers, meta, xs, sim, trace_k)
        variant._time_recorded_dft_planes = set()
        options = {} if devices is None else dict(distributed=True, devices=devices)
        original_forward = variant.forward
        def observed_forward(*args, **kwargs):
            if spy is not None:
                spy(variant, kwargs)
            return original_forward(*args, **kwargs)
        with patch.object(variant, 'forward', side_effect=observed_forward):
            result = variant.forward(eps_override=eps, n_steps=length, **options)
        assert result.dft_planes
        if devices is not None and not isinstance(eps, jax.core.Tracer):
            for plane in result.dft_planes.values():
                assert plane.accumulator.sharding.device_set == set(devices)
        records.append((variant, enames, hnames, result.dft_planes))
    assembled = calc._assemble_msl_plane_records(
        records, entries=entries, freqs_arr=FREQS, grid=grid, msl_ports=ports,
        n_probes_per_port=counts, probe_xs=xs, h_stencils=stencils, dz_arr=dz,
        port_idx_meta=meta, beta0_per_port=beta, z0_hj_per_port=z0,
        eps_r_ref_per_port=eps_ref, line_eps_per_port=line_eps, trace_k_per_port=trace_k)
    return calc.renormalize_power_waves(assembled[0], assembled[-1], refs)


def jacobian(sim, base, masks, length, devices=None):
    """Four explicit pullbacks for J[f,k] = d |S21(f)|² / dp[k]."""
    calls = []
    def spy(variant, options):
        assert not variant._time_recorded_dft_planes
        if devices is not None:
            assert options.get('distributed') is True
            assert options.get('devices') is devices
        calls.append(options)
    def objective(p):
        s = forward_s(sim, eps_of(base, masks, p), length, devices, spy)
        return jnp.abs(s[1, 0]) ** 2
    started = perf_counter()
    with sampled_memory() as peak:
        value, pullback = jax.vjp(objective, jnp.zeros(4, jnp.float32))
        value.block_until_ready()
        reverse_started = perf_counter()
        g0 = pullback(jnp.array([1., 0., 0., 0.], value.dtype))[0]
        g0.block_until_ready()
        g1 = pullback(jnp.array([0., 1., 0., 0.], value.dtype))[0]
        g1.block_until_ready()
        g2 = pullback(jnp.array([0., 0., 1., 0.], value.dtype))[0]
        g2.block_until_ready()
        g3 = pullback(jnp.array([0., 0., 0., 1.], value.dtype))[0]
        g3.block_until_ready()
        result = np.asarray(jnp.stack((g0, g1, g2, g3)))
        reverse_seconds = perf_counter() - reverse_started
    assert len(calls) == len(sim._msl_ports)
    assert np.isfinite(result).all(), result
    assert np.all(np.max(np.abs(result), axis=0) > 0), result
    emit('J', scene=sim._msl_ports[0].direction, n=1 if devices is None else len(devices),
         length=length, J=result, seconds=perf_counter() - started,
         reverse_seconds=reverse_seconds, sampled_peak_bytes=peak[0],
         memory_method='job cgroup memory.current, 1s samples plus endpoints; CPU devices share process memory')
    del pullback
    jax.clear_caches()
    gc.collect()
    return result
