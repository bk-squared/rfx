"""Temporary J0g observation: one device, one attempt, never a bar."""

import json
import os
import re
import subprocess
import sys


def _main():
    import gc
    import threading
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import numpy as np

    from rfx import Box
    from rfx.ringdown import gradient_witness
    from tests.unit.runners._msl_multidevice_scene import FREQS, _json, build_scene

    def emit(kind, **data):
        print('J0G_JSON ' + json.dumps(_json(dict(kind=kind, **data)), allow_nan=False), flush=True)

    assert len(jax.devices()) == 1
    emit('metadata', devices=list(map(str, jax.devices())), freqs_hz=FREQS,
         purpose='J0g only; never a bar', mode='laplace', excite_all=True)
    boxes = {
        'T1A_graded': [(0.0030, 0.0050, 0.0034, 0.0046),
                       (0.0058, 0.0070, 0.0034, 0.0046),
                       (0.0076, 0.0090, 0.0034, 0.0046),
                       (0.0019, 0.0021, 0.0034, 0.0046)],
        'Y1A_graded': [(0.0030, 0.0040, 0.0040, 0.0080),
                       (0.0040, 0.0044, 0.0040, 0.0080),
                       (0.0044, 0.0048, 0.0040, 0.0080),
                       (0.0030, 0.0048, 0.0019, 0.0021)],
    }
    expected = {'T1A_graded': [list(range(23, 33)), list(range(37, 43)),
                              list(range(46, 53)), [18]],
                'Y1A_graded': [list(range(23, 28)), [28, 29], [30, 31]]}
    all_rows, witnesses = {}, {}
    for name, bounds in boxes.items():
        sim = build_scene(name, mode='laplace', excite_all=True).sim
        grid = sim._build_realized_grid()
        materials = sim._assemble_materials_nu(grid, pec_sheets=[], pec_wires=[])[0]
        eps_base = jnp.asarray(materials.eps_r, dtype=jnp.float32)
        assert eps_base.shape == grid.shape
        coords = [np.array([grid.node_of(axis, i) for i in range(n)])
                  for axis, n in zip('xyz', grid.shape)]
        masks, blocks = [], []
        for k, (xlo, xhi, ylo, yhi) in enumerate(bounds):
            mask = np.asarray(Box((xlo, ylo, 0), (xhi, yhi, 0.0008)).mask_on_coords(*coords))
            cells = np.argwhere(mask)
            indices = [np.unique(cells[:, a]).tolist() for a in range(3)]
            if k < len(expected[name]):
                assert indices[0] == expected[name][k], (name, k, indices)
            blocks.append(dict(block=k, x=indices[0], y=indices[1], z=indices[2],
                               cells=cells.tolist()))
            masks.append(jnp.asarray(mask, dtype=jnp.float32))
        emit('blocks', scene=name, grid_shape=grid.shape, eps_base_dtype=str(eps_base.dtype), blocks=blocks)
        rows = {}
        for length in (1050, 1400, 2100):
            emit('start', scene=name, L=length)
            def objective(p):
                eps = eps_base
                for k in range(4):
                    eps = eps + p[k] * masks[k]
                result = sim.compute_msl_s_matrix(freqs=FREQS, n_steps=length, eps_override=eps)
                magnitude = jnp.abs(result.S[1, 0, :])
                return jnp.real(magnitude ** 2), magnitude

            cg = Path('/sys/fs/cgroup')
            peak = [0]
            done = threading.Event()
            def sample():
                while True:
                    peak[0] = max(peak[0], int((cg / 'memory.current').read_text()))
                    if done.wait(1):
                        break
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            started = time.monotonic()
            try:
                value, pullback, magnitude = jax.vjp(objective, jnp.zeros(4, dtype=jnp.float32), has_aux=True)
                # Four explicit unit-cotangent pullbacks; no jacrev or vmap.
                g0 = pullback(jnp.array([1, 0, 0, 0], dtype=value.dtype))[0]
                g0.block_until_ready()
                g1 = pullback(jnp.array([0, 1, 0, 0], dtype=value.dtype))[0]
                g1.block_until_ready()
                g2 = pullback(jnp.array([0, 0, 1, 0], dtype=value.dtype))[0]
                g2.block_until_ready()
                g3 = pullback(jnp.array([0, 0, 0, 1], dtype=value.dtype))[0]
                g3.block_until_ready()
                J = np.asarray(jnp.stack([g0, g1, g2, g3]))
                abs_s21 = np.asarray(magnitude)
                elapsed = time.monotonic() - started
            finally:
                peak[0] = max(peak[0], int((cg / 'memory.current').read_text()))
                done.set()
                sampler.join()
            cgroup_peak = int((cg / 'memory.peak').read_text()) if (cg / 'memory.peak').exists() else None
            row = dict(J=J, column_max=np.max(np.abs(J), axis=0),
                       bin_max=np.max(np.abs(J), axis=1), abs_s21=abs_s21,
                       wall_seconds=elapsed, sampled_peak_bytes=peak[0],
                       memory_method='cgroup memory.current sampled every 1 s plus endpoints (labrun fallback method)',
                       cumulative_cgroup_peak_bytes=cgroup_peak)
            emit('measurement', scene=name, L=length, **row)
            rows[length] = row
            del pullback, value, magnitude, g0, g1, g2, g3
            jax.clear_caches()
            gc.collect()
        all_rows[name] = rows
        witnesses[name] = {}
        for length in (1050, 1400):
            w = gradient_witness(rows[length]['J'], rows[2100]['J'], against='longer_record', bin_axis=0)
            witnesses[name][length] = bool(w.judged and w.value <= 1e-2)
            emit('witness', scene=name, L=length, reference_L=2100,
                 value=w.value, judged=w.judged, ok=w.ok, note=w.note)
    selected = [L for L in (1050, 1400) if all(w[L] for w in witnesses.values())]
    emit('selection', G_lengths=selected + [2100],
         stop_no_pair_passes=not selected,
         stop_zero_column=any(np.any(row['column_max'] == 0) for rows in all_rows.values() for row in rows.values()),
         stop_over_48_GiB=any(rows[2100]['sampled_peak_bytes'] > 48 * 2**30 for rows in all_rows.values()),
         stop_over_2h=any(rows[2100]['wall_seconds'] > 7200 for rows in all_rows.values()),
         conclusions='리더가 채움')


def test_j0g():
    env = os.environ.copy()
    flags = re.sub(r'--xla_force_host_platform_device_count(?:=|\s+)\d+', '', env.get('XLA_FLAGS', ''))
    env['XLA_FLAGS'] = flags + ' --xla_force_host_platform_device_count=1'
    env['JAX_PLATFORMS'] = 'cpu'
    code = 'from tests.unit.runners.test_j0g_throwaway import _main; _main()'
    subprocess.run([sys.executable, '-u', '-c', code], env=env, check=True)
