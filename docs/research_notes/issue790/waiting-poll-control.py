import inspect
import pytest
from rfx.experiments import worker

source = inspect.getsource(worker._cancel_requested)
assert 'timeout=0,' in source
# Explicit in-memory regression: let cancellation polling wait six seconds
# for SQLite while the computation has only a one-second deadline.
source = source.replace('timeout=0,', 'timeout=6,')
exec(compile(source, '<waiting-sqlite-poll-mutation>', 'exec'), worker.__dict__)
code = pytest.main(['tests/studio/test_worker_process_supervision.py::test_sqlite_lock_cannot_delay_stopping_native_computation', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider'])
assert code == 1, f'expected native deadline regression failure, got {code}'
print('Expected waiting-poll mutation failure confirmed; source files unchanged.')
