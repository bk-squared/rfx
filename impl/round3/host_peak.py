"""Warm host residue fits, one fresh subprocess per declared measurement point."""

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc


POINTS = ((20000, 592, 8), (60000, 592, 8), (200000, 592, 8),
          (20000, 592, 8), (20000, 1492, 8), (20000, 2992, 8))


def measure(n_win, rank, channels):
    """Measure peak RSS above the warmed, input-allocated high-water mark."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import numpy as np
    from rfx.measurement import matrix_pencil as mp

    size = mp._residue_block_size(n_win, rank, channels)
    rng = np.random.default_rng(2874)
    poles = -rng.uniform(1e-5, 1e-3, rank) + 1j * rng.uniform(.05, 3., rank)
    warm_y = rng.normal(size=(2 * size, channels))
    warm_result = mp._fit_residues(warm_y, poles, 1.)
    del warm_y, warm_result
    y = rng.normal(size=(n_win, channels))
    estimate = mp._residue_memory_bytes(n_win, rank, channels)
    rss_unit = 1 if sys.platform == 'darwin' else 1024
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
    tracemalloc.start()
    start = time.perf_counter()
    result = mp._fit_residues(y, poles, 1.)
    seconds = time.perf_counter() - start
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    final_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
    print(json.dumps(dict(n_win=n_win, r=rank, C=channels, Mb=size,
                          estimate_bytes=estimate, baseline_peak_bytes=baseline,
                          final_peak_bytes=final_peak, rise_bytes=final_peak-baseline,
                          tracemalloc_peak_bytes=traced_peak, seconds=seconds,
                          finite=bool(np.isfinite(result).all()))), flush=True)


def main():
    if len(sys.argv) == 4:
        measure(*(int(value) for value in sys.argv[1:]))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4')
    for point in POINTS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        *(str(value) for value in point)], env=env, check=True)


if __name__ == '__main__':
    main()
