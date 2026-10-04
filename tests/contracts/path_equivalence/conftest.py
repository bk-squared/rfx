"""Bounded subprocess isolation; no module-level XLA or precision switches."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

import pytest


@pytest.fixture(scope='module')
def matrix_worker(tmp_path_factory):
    directory = tmp_path_factory.mktemp('s0-worker')
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=2',
               JAX_PLATFORMS='cpu', JAX_ENABLE_X64='false', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(directory / 'cache'), MPLCONFIGDIR=str(directory / 'mpl'),
               TMPDIR=str(directory))
    root = Path(__file__).resolve().parents[3]
    env['PYTHONPATH'] = str(root)
    pending = queue.Queue()
    with (directory / 'stderr.txt').open('w+') as errors:
        process = subprocess.Popen(
            [sys.executable, '-m', 'tests.contracts.path_equivalence.worker'],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=errors, text=True, bufsize=1, start_new_session=True)

        def read():
            for line in process.stdout:
                pending.put(line)
            pending.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        cache = {}

        def request(cell):
            if cell.id not in cache:
                process.stdin.write(json.dumps({'cell': cell.id}) + '\n')
                process.stdin.flush()
                try:
                    response = pending.get(timeout=120)
                except queue.Empty:
                    pytest.fail(f'S0 worker timed out: {cell.id}')
                assert response is not None, f'S0 worker exited: {errors.name}'
                result = json.loads(response)
                assert 'worker_error' not in result, result.get('worker_error')
                assert result['cell'] == cell.id
                cache[cell.id] = result
            return cache[cell.id]

        try:
            yield request
        finally:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            reader.join(timeout=5)
            process.stdout.close()
            (directory / 'measurements.json').write_text(json.dumps(list(cache.values()), indent=2))


def pytest_configure(config):
    config.addinivalue_line('markers', 's0_weekly: S0 full-matrix cells outside its declared PR subset')


def pytest_collection_modifyitems(config, items):
    # Default to the PR subset even when a gate overrides the slow filter.
    if os.environ.get('RFX_S0_FULL') == '1':
        return
    excluded = [item for item in items if item.get_closest_marker('s0_weekly')]
    items[:] = [item for item in items if not item.get_closest_marker('s0_weekly')]
    config.hook.pytest_deselected(items=excluded)
