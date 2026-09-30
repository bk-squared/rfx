"""Two-device compiled scan timing, with and without the five-port samples."""
import json
import platform
from pathlib import Path
import time
from types import SimpleNamespace

import jax
import numpy as np

from rfx.probes.sparam_driver import _lumped_recording_probes
from rfx.runners import distributed_v2 as runner
from tests.unit.runners.test_distributed_lumped_port_s import _model

N = 320
entries = {}
original_jax = runner.jax
for label in ('bare', 'record'):
    sim = _model(seam=True, ports=2)
    probes, _ = _lumped_recording_probes(sim._build_grid(), sim._ports)

    def wrap_jit(f, *args, **kwargs):
        entry = jax.jit(f, *args, **kwargs)
        if f.__name__ != '_run_cpml':
            return entry

        def capture(*a, **kw):
            compiled = entry.lower(*a, **kw).compile()
            entries[label] = (compiled, a, kw)
            return compiled(*a, **kw)
        return capture

    runner.jax = SimpleNamespace(**{**vars(jax), 'jit': wrap_jit})
    try:
        result = runner.run_distributed(sim, n_steps=N, devices=jax.devices('cpu')[:2],
                                        _sparam_drive_idx=0,
                                        _sparam_probes=probes if label == 'record' else ())
        jax.block_until_ready(result.time_series)
    finally:
        runner.jax = original_jax

samples = {name: [] for name in entries}
for repeat in range(10):
    for label in (('bare', 'record') if repeat % 2 == 0 else ('record', 'bare')):
        compiled, a, kw = entries[label]
        begin = time.perf_counter()
        result = compiled(*a, **kw)
        jax.block_until_ready(result)
        if repeat:
            samples[label].append((time.perf_counter() - begin) * 1e6 / N)
report = {'python': platform.python_version(), 'platform': platform.platform(),
          'jax': jax.__version__, 'numpy': np.__version__,
          'n_steps': N, 'n_ports': 2, 'n_devices': 2,
          'added_record_bytes_per_device': N * 5 * 4 * 2,
          'timing_us_per_step': samples}
for label, (compiled, _, _) in entries.items():
    memory = compiled.memory_analysis()
    report[label] = {'median_us_per_step': float(np.median(samples[label])),
                     **{name: getattr(memory, name) for name in
                        ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes')}}
report['delta_us_per_step'] = report['record']['median_us_per_step'] - report['bare']['median_us_per_step']
report['delta_percent'] = 100 * report['delta_us_per_step'] / report['bare']['median_us_per_step']
Path(__file__).with_name('benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
