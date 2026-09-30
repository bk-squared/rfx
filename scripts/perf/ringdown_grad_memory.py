#!/usr/bin/env python3
"""Product memory measurement for #1414; one compiled call per fresh process.

Run the 18-point GPU matrix:
  python scripts/perf/ringdown_grad_memory.py --output /path/results.jsonl
CPU smoke (two fresh processes):
  JAX_PLATFORMS=cpu python scripts/perf/ringdown_grad_memory.py --smoke --output smoke.jsonl

Value returns forward(...).ringdown; grad differentiates sum(abs(s_params)**2)
with respect to a scalar multiplier of the dielectric permittivity. Defaults
for checkpointing and RingdownSpec are unchanged. Peak memory is the device
allocator's lifetime high-water mark, including setup/compilation (no warmup).
CPU memory_stats may be unavailable: null is not zero. Wall time includes
scene construction, compilation and the synchronized call, but excludes the
optional host report, read afterward on value results to record loaded Q.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]


def clean(value):
    """Strict JSON: represent unavailable/nonfinite numbers as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def scene(box, ports):
    import numpy as np
    from rfx import Box, GaussianPulse, Simulation

    # Same cell pitch; larger physical box doubles x and y: exactly 4x cells.
    dx = np.array([1., .8, .6, .9, 1.2, 1.4, 1.2, 1.1, 1., .9, .9, 1.]) * 1e-3
    dy = np.array([1., 1.4, .8, .6, .8, .9, 1., 1.1, 1.2, 1.2, 1.]) * 1e-3
    dz = np.array([.8, .6, 1.4, 1.4, .8]) * 1e-3
    if box == 'large':
        dx, dy = np.tile(dx, 2), np.tile(dy, 2)
    domain = tuple(float(a.sum()) for a in (dx, dy, dz))
    cells = tuple(len(a) for a in (dx, dy, dz))
    shape = tuple(n + 1 for n in cells)  # public profile convention: N cells, N+1 nodes
    sim = Simulation(freq_max=20e9, domain=domain, dx=1e-3, boundary='pec',
                     precision='float32', dx_profile=dx, dy_profile=dy, dz_profile=dz)
    # Material Q=10000 at 12.5 GHz. High port impedance limits resistive loading;
    # actual loaded pole Q is recorded from the public value result's report.
    sigma = 2 * np.pi * 12.5e9 * 8.854187817e-12 * 2.2 / 10000
    sim.add_material('fill', eps_r=2.2, sigma=float(sigma))
    sim.add(Box((0, 0, 0), domain), material='fill')
    pulse = GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5)
    for p, pos in enumerate(((3.3e-3, 3.8e-3, .8e-3), (9.2e-3, 8.2e-3, .8e-3))[:ports]):
        sim.add_port(position=pos, component='ez', impedance=5000., extent=.6e-3,
                     waveform=pulse, excite=p == 0)
    return sim, shape, {'domain_m': domain, 'grid_shape': shape,
                        'cell_shape': cells, 'grid_cells': math.prod(cells),
                        'grid_nodes': math.prod(shape), 'eps_r': 2.2,
                        'sigma_s_per_m': float(sigma), 'port_impedance_ohm': 5000.,
                        'vi_channels': 2 * ports, 'lane': 'graded'}


def measure(args):
    started = time.perf_counter()
    row = dict(window_steps=args.window, n_steps=2 * args.window,
               ports=args.ports, box=args.box, mode=args.mode, pid=os.getpid(),
               peak_bytes_in_use=None, bytes_in_use=None, memory_analysis=None)
    device = None
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from rfx.ringdown import RingdownSpec

        device = jax.devices()[0]
        row.update(jax_version=jax.__version__, backend=jax.default_backend(),
                   device=str(device), device_kind=device.device_kind,
                   preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
                   x64_enabled=bool(jax.config.jax_enable_x64))
        sim, shape, metadata = scene(args.box, args.ports)
        row.update(metadata)

        def forward(p):
            result = sim.forward(n_steps=2 * args.window, skip_preflight=True,
                                 eps_override=jnp.full(shape, 2.2, jnp.float32) * p,
                                 ringdown=RingdownSpec())
            row['realized_grid_shape'] = tuple(result.grid.shape)
            return result.ringdown

        def objective(p):
            return jnp.sum(jnp.abs(forward(p).s_params) ** 2)

        fn = forward if args.mode == 'value' else jax.grad(objective)
        param = jnp.float32(1.)
        t0 = time.perf_counter()
        compiled = jax.jit(fn).lower(param).compile()
        row['compile_wall_s'] = time.perf_counter() - t0
        try:
            analysis = compiled.memory_analysis()
            if analysis is not None:
                row['memory_analysis'] = {
                    name: int(getattr(analysis, name)) for name in (
                        'temp_size_in_bytes', 'argument_size_in_bytes',
                        'output_size_in_bytes', 'alias_size_in_bytes')
                }
        except (AttributeError, NotImplementedError) as exc:
            row['memory_analysis_unavailable'] = str(exc)
        t0 = time.perf_counter()
        result = jax.block_until_ready(compiled(param))
        row['call_wall_s'] = time.perf_counter() - t0
        row['wall_s'] = time.perf_counter() - started
        stats = device.memory_stats() or {}
        row.update({k: stats.get(k) for k in ('peak_bytes_in_use', 'bytes_in_use')})
        observed = result.s_params if args.mode == 'value' else result
        row['result_finite'] = bool(np.all(np.isfinite(np.asarray(observed))))
        if args.mode == 'grad':
            row['gradient'] = float(result)
        row['status'] = 'ok'  # successful execution; see result_finite separately
        if args.mode == 'value':
            try:
                report = result.report  # after the allocator snapshot and wall timer
                row['ringdown_completed'] = report.completed
                row['ringdown_failure'] = report.failure
                row['poles'] = [p._asdict() for p in report.poles]
                row['reported_window_steps'] = report.window_steps
            except Exception as exc:
                row['report_error'] = str(exc)
    except Exception as exc:
        # Preserve the exact exception text, including allocator OOM diagnostics.
        row.update(status=str(exc), error_type=type(exc).__name__,
                   traceback=traceback.format_exc(), wall_s=time.perf_counter() - started)
        if device is not None:
            stats = device.memory_stats() or {}
            row.update({k: stats.get(k) for k in ('peak_bytes_in_use', 'bytes_in_use')})
    return clean(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--window', type=int, default=20000)
    parser.add_argument('--ports', type=int, choices=(1, 2), default=1)
    parser.add_argument('--box', choices=('small', 'large'), default='small')
    parser.add_argument('--mode', choices=('value', 'grad'), default='value')
    args = parser.parse_args()
    if args.window <= 0:
        parser.error('--window must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    if args.worker:
        row = measure(args)
        args.output.write_text(json.dumps(row, allow_nan=False) + '\n')
        return 0 if row['status'] == 'ok' else 1

    # The parent imports no JAX. Each child owns exactly one allocator lifetime.
    points = ([(500, 1, 'small')] if args.smoke else
              [(w, p, 'small') for w in (10000, 20000, 40000, 70000) for p in (1, 2)]
              + [(20000, 1, 'large')])
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    failed = False
    # Retain one log and JSON per process, including fatal allocator aborts.
    artifacts = args.output.with_suffix('.workers')
    artifacts.mkdir(exist_ok=True)
    with args.output.open('w') as output:
        for w, ports, box in points:
            for mode in ('value', 'grad'):
                stem = artifacts / f'w{w}-p{ports}-{box}-{mode}'
                child_json = stem.with_suffix('.json')
                child_json.unlink(missing_ok=True)
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                           '--window', str(w), '--ports', str(ports), '--box', box,
                           '--mode', mode, '--output', str(child_json)]
                t0 = time.perf_counter()
                with stem.with_suffix('.log').open('w') as log:
                    proc = subprocess.run(command, env=env, stdout=log, stderr=log)
                if child_json.exists():
                    row = json.loads(child_json.read_text())
                else:
                    row = dict(window_steps=w, n_steps=2*w, ports=ports, box=box,
                               mode=mode, peak_bytes_in_use=None, bytes_in_use=None,
                               memory_analysis=None, wall_s=time.perf_counter()-t0,
                               status=stem.with_suffix('.log').read_text() or
                               f'process exited with returncode {proc.returncode}')
                row['returncode'] = proc.returncode
                failed |= proc.returncode != 0
                line = json.dumps(row, allow_nan=False)
                output.write(line + '\n')
                output.flush()
                print(line, flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
