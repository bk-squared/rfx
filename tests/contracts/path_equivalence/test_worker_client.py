"""Worker recovery using a small Python command, without matrix solves."""
import json
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
