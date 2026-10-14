"""Use the per-step 9-ULP cross-trace bar (rfx CLAUDE.md), since R15g replaces
the vacuum-relative gate that collapsed when one CPML made its reference exact.
"""
import gc
import json
from unittest.mock import patch

import jax
import numpy as np

from rfx.runners import distributed_nu, nonuniform

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
WEEKLY_PREFIXES = (*range(1, 51), *range(60, 201, 10), *range(300, 801, 100))
PR_PREFIXES = (200, 400, 800)


def cross_trace(factory, lane, devices, label, *, prefixes):
    """Observe existing returns; leave solver inputs and compiled outputs intact."""
    captured = {}
    single_runner = nonuniform.run_nonuniform_path
    multi_runner = distributed_nu.run_nonuniform_distributed_pec

    def single(*args, **kwargs):
        out = single_runner(*args, **kwargs)
        captured['state'] = {f: np.asarray(getattr(out.state, f)) for f in FIELDS}
        return out

    def multi(*args, **kwargs):
        out = multi_runner(*args, **kwargs)
        grid = args[0]
        values = {}
        for f in FIELDS:
            arr = np.asarray(getattr(out['final_state_sharded'], f))
            slabs = arr.reshape((len(devices), grid.nx_local) + arr.shape[1:])
            owned = slabs[:, grid.ghost_width:grid.ghost_width + grid.nx_per_rank]
            values[f] = owned.reshape((-1,) + arr.shape[1:])[:grid.nx]
        captured['state'] = values
        return out

    def record(n, dev):
        captured.clear()
        sim = factory()
        with jax.default_device(devices[0]):
            if lane == 'run':
                out = sim.run(n_steps=n, skip_preflight=True, devices=dev)
                values = {f: np.asarray(getattr(out.state, f)) for f in FIELDS}
                # Retain the CPML outer tangential-E plane witness.
                for c, f in enumerate(FIELDS[:3]):
                    for axis in range(3):
                        if axis != c:
                            assert not np.any(np.take(values[f], [0, -1], axis=axis)), (label, n, f, axis)
            else:
                kw = {} if dev is None else dict(distributed=True, devices=dev)
                sim.forward(n_steps=n, skip_preflight=True, checkpoint=False, **kw)
                values = captured['state']
        for f, arr in values.items():
            assert arr.dtype == np.float32 and np.isfinite(arr).all(), (label, n, f)
        return values

    steps = tuple(prefixes)
    assert steps and all(n > 0 for n in steps) and tuple(sorted(set(steps))) == steps
    assert {200, 400, 800}.issubset(steps)
    peaks, differences = [], []
    with patch.object(nonuniform, 'run_nonuniform_path', single), patch.object(
            distributed_nu, 'run_nonuniform_distributed_pec', multi):
        for index, n in enumerate(steps):
            one, two = record(n, None), record(n, devices)
            assert all(one[f].shape == two[f].shape for f in FIELDS)
            peaks.append([float(np.max(np.abs(one[f]))) for f in FIELDS])
            differences.append([float(np.max(np.abs(two[f].astype(np.float64) - one[f]))) for f in FIELDS])
            if index % 20 == 19:
                print(f'[{label}] prefix={n}', flush=True)
                jax.clear_caches()
                gc.collect()
    peaks, differences = np.asarray(peaks), np.asarray(differences)
    results = {}
    for length in (200, 400, 800):
        selection = [i for i, n in enumerate(steps) if n <= length]
        peak_e = float(peaks[selection, :3].max())
        peak_h = float(peaks[selection, 3:].max())
        scale_h = max(peak_h, peak_e / 376.730313668)
        spacing = np.spacing(np.asarray([peak_e] * 3 + [scale_h] * 3, dtype=np.float32)).astype(np.float64)
        d = differences[selection] / spacing
        results[length] = dict(P_E=peak_e, P_H=peak_h, H_scale=scale_h,
                               steps=[steps[i] for i in selection], d=d.tolist(), max_d=d.max(axis=0).tolist())
    print('CROSS_TRACE ' + json.dumps(dict(label=label, windows=results)), flush=True)
    for length, result in results.items():
        assert max(result['max_d']) <= 9, (label, length, dict(zip(FIELDS, result['max_d'])))
    return results
