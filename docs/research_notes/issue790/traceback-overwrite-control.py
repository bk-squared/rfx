import inspect
import pytest
from rfx.experiments import worker

# Explicit in-memory mutation of the new implementation, not a baseline checkout.
source = inspect.getsource(worker._write_traceback_artifact)
assert 'if not path.exists():' in source
source = source.replace('if not path.exists():', 'if True:')
exec(compile(source, '<traceback-overwrite-mutation>', 'exec'), worker.__dict__)
code = pytest.main(['tests/studio/test_worker_process_supervision.py::test_terminal_commit_retry_preserves_registered_traceback_bytes', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider'])
assert code == 1, f'expected regression test failure, got {code}'
print('Expected overwrite-mutation failure confirmed; source files unchanged.')
