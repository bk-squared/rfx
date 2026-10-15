"""Worker recovery using a small Python command, without matrix solves."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from .worker_client import WorkerClient
from .conftest import matrix_worker


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
    if cell in ('mismatch', 'invalid-json'):
        print(json.dumps({'cell': 'wrong'}) if cell == 'mismatch' else 'not json', flush=True)
    print(json.dumps({'cell': cell, 'pid': os.getpid()}), flush=True)
    if cell == 'answer-and-exit':
        sys.exit(0)
'''


@pytest.fixture
def client(tmp_path):
    with (tmp_path / 'stderr.txt').open('w+') as errors:
        worker = WorkerClient([sys.executable, '-u', '-c', COMMAND],
                              cwd=tmp_path, env=os.environ.copy(), errors=errors,
                              timeout=30)
        try:
            yield worker
        finally:
            worker.close()


@pytest.mark.parametrize('failed_cell,message', [('slow', 'timed out'), ('exit', 'exited')])
def test_failed_cell_does_not_poison_next_requests(client, failed_cell, message):
    client(SimpleNamespace(id='ready'))
    client.timeout = 0.5
    with pytest.raises(pytest.fail.Exception, match=f'S0 worker {message}:'):
        client(SimpleNamespace(id=failed_cell))
    assert failed_cell not in client.cache
    client.timeout = 30
    second = client(SimpleNamespace(id='second'))
    third = client(SimpleNamespace(id='third'))
    assert second['cell'] == 'second'
    assert third['cell'] == 'third'
    assert second['pid'] == third['pid']


def test_cached_results_survive_restart(client):
    cell = SimpleNamespace(id='cached')
    cached = client(cell)
    process, pending, reader = client.process, client.pending, client.reader
    client.timeout = 0.5
    with pytest.raises(pytest.fail.Exception, match='S0 worker timed out: slow'):
        client(SimpleNamespace(id='slow'))
    assert process.poll() is not None
    assert not reader.is_alive()
    assert client(cell) is cached
    assert client.process is None
    client.timeout = 30
    result = client(SimpleNamespace(id='after'))
    assert result['cell'] == 'after'
    assert client.pending is not pending
    assert client(cell) is cached


def test_close_tolerates_permission_error_for_exited_worker(client, monkeypatch):
    client(SimpleNamespace(id='answer-and-exit'))
    process = client.process
    process.wait(timeout=30)

    def denied(*args):
        raise PermissionError('exited process group')

    monkeypatch.setattr(os, 'killpg', denied)
    client.close()
    assert client.process is None
    assert process.stdin.closed
    assert process.stdout.closed
    assert not client.reader.is_alive()


def test_close_clears_process_even_on_unexpected_error(client, monkeypatch):
    client(SimpleNamespace(id='answer-and-exit'))
    process = client.process
    process.wait(timeout=30)

    def broken_poll():
        raise RuntimeError('poll failed')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(process, 'poll', broken_poll)
            with pytest.raises(RuntimeError, match='poll failed'):
                client.close()
            assert client.process is None
    finally:
        # Restore the reference only to finish stream/thread cleanup in this test.
        client.process = process
        client.close()


def test_measurements_written_even_if_close_fails(tmp_path, monkeypatch):
    def broken_close(self):
        raise RuntimeError('close failed')

    monkeypatch.setattr(WorkerClient, 'close', broken_close)
    factory = SimpleNamespace(mktemp=lambda name: tmp_path)
    fixture = matrix_worker.__wrapped__(factory)
    worker = next(fixture)
    worker.cache['saved'] = {'cell': 'saved'}
    with pytest.raises(RuntimeError, match='close failed'):
        next(fixture)
    assert json.loads((tmp_path / 'measurements.json').read_text()) == [{'cell': 'saved'}]


@pytest.mark.parametrize('failed_cell,error', [
    ('mismatch', pytest.fail.Exception), ('invalid-json', json.JSONDecodeError),
])
def test_invalid_response_does_not_poison_next_requests(client, failed_cell, error):
    client(SimpleNamespace(id='ready'))
    process, pending, reader = client.process, client.pending, client.reader
    with pytest.raises(error):
        client(SimpleNamespace(id=failed_cell))
    assert failed_cell not in client.cache
    result = client(SimpleNamespace(id='after'))
    assert result['cell'] == 'after'
    assert client.process is not process
    assert client.pending is not pending
    assert process.poll() is not None
    assert not reader.is_alive()


def test_worker_exit_between_requests_does_not_poison_next_requests(client):
    assert client(SimpleNamespace(id='answer-and-exit'))['cell'] == 'answer-and-exit'
    process, reader = client.process, client.reader
    process.wait(timeout=30)
    with pytest.raises(pytest.fail.Exception, match='S0 worker exited:'):
        client(SimpleNamespace(id='failed'))
    assert 'failed' not in client.cache
    assert client.process is None
    assert process.stdin.closed
    assert process.stdout.closed
    assert not reader.is_alive()
    result = client(SimpleNamespace(id='after'))
    assert result['cell'] == 'after'
    assert client.process is not process
    client.close()
    client.close()
    assert client.process is None
