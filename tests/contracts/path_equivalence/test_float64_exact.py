"""B7: exact float64 records on the synthetic B1 CPML box."""
import pytest

from .generation import Cell


@pytest.mark.parametrize('steps, entry', [
    (steps, entry)
    for entry in ('run', 'fwd')
    for steps in (12, 36)
    if steps != 12
])
def test_float64_exact(entry, steps, matrix_worker):
    cell = Cell(('_float64', ''), entry+'_uniform', entry+'_nonuniform',
                False, True, None, steps)
    report = matrix_worker(cell)
    assert not report['failures'], report['failures']
    assert report['measurements']
    differences = [(m['record'], m['difference'], m['relative'])
                   for m in report['measurements'] if m['difference'] != 0]
    assert not differences, f'B7 {cell.id}: nonzero records {differences}'
