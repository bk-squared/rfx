"""Bounded subprocess isolation; no module-level XLA or precision switches."""
import json
import os
from pathlib import Path
import sys

import pytest

from .worker_client import WorkerClient


@pytest.fixture(scope='module')
def matrix_worker(tmp_path_factory):
    directory = tmp_path_factory.mktemp('s0-worker')
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=2',
               JAX_PLATFORMS='cpu', JAX_ENABLE_X64='false', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(directory / 'cache'), MPLCONFIGDIR=str(directory / 'mpl'),
               TMPDIR=str(directory))
    root = Path(__file__).resolve().parents[3]
    env['PYTHONPATH'] = str(root)
    with (directory / 'stderr.txt').open('w+') as errors:
        client = WorkerClient(
            [sys.executable, '-m', 'tests.contracts.path_equivalence.worker'],
            cwd=root, env=env, errors=errors)
        try:
            yield client
        finally:
            try:
                client.close()
            finally:
                (directory / 'measurements.json').write_text(json.dumps(list(client.cache.values()), indent=2))


def pytest_configure(config):
    config.addinivalue_line('markers', 's0_weekly: S0 full-matrix cells outside its declared PR subset')


def pytest_collection_modifyitems(config, items):
    # Default to the PR subset even when a gate overrides the slow filter.
    if os.environ.get('RFX_S0_FULL') == '1':
        return
    excluded = [item for item in items if item.get_closest_marker('s0_weekly')]
    items[:] = [item for item in items if not item.get_closest_marker('s0_weekly')]
    config.hook.pytest_deselected(items=excluded)
