"""Worker recovery using a small Python command, without matrix solves."""
import os
import sys
from types import SimpleNamespace

import pytest

from .worker_client import WorkerClient


COMMAND = '''
import json
import os
import sys
import time

for line in sys.stdin:
    cell = json.loads(line)['cell']
    if cell == 'slow':
        time.sleep(0.75)
    if cell == 'exit':
        sys.exit(0)
    print(json.dumps({'cell': 'wrong' if cell == 'mismatch' else cell,
                      'pid': os.getpid()}), flush=True)
'''


@pytest.fixture
def client(tmp_path):
    with (tmp_path / 'stderr.txt').open('w+') as errors:
        worker = WorkerClient([sys.executable, '-u', '-c', COMMAND],
                              cwd=tmp_path, env=os.environ.copy(), errors=errors,
                              timeout=0.5)
        try:
            yield worker
        finally:
            worker.close()


@pytest.mark.parametrize('failed_cell,message', [('slow', 'timed out'), ('exit', 'exited')])
def test_failed_cell_does_not_poison_next_requests(client, failed_cell, message):
    with pytest.raises(pytest.fail.Exception, match=f'S0 worker {message}:'):
        client(SimpleNamespace(id=failed_cell))
    assert failed_cell not in client.cache
    second = client(SimpleNamespace(id='second'))
    third = client(SimpleNamespace(id='third'))
    assert second['cell'] == 'second'
    assert third['cell'] == 'third'
    assert second['pid'] == third['pid']


def test_cached_results_survive_restart(client):
    cell = SimpleNamespace(id='cached')
    cached = client(cell)
    process, pending, reader = client.process, client.pending, client.reader
    with pytest.raises(pytest.fail.Exception, match='S0 worker timed out: slow'):
        client(SimpleNamespace(id='slow'))
    assert process.poll() is not None
    assert not reader.is_alive()
    assert client(cell) is cached
    assert client.process is None
    result = client(SimpleNamespace(id='after'))
    assert result['cell'] == 'after'
    assert client.pending is not pending
    assert client(cell) is cached


def test_mismatched_cell_still_fails(client):
    with pytest.raises(AssertionError):
        client(SimpleNamespace(id='mismatch'))
    assert 'mismatch' not in client.cache
