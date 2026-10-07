"""Generated S0 records: full weekly matrix and the declared PR subset."""
from pathlib import Path

import pytest

from .builders import BUILDERS
from .generation import generate, pr_subset, record_groups
from .reporting import KnownFinding, assert_record, expectations, group, load_findings

CELLS = generate(BUILDERS)
_MANIFEST = Path(__file__).with_name('findings.json')
FINDINGS = load_findings(_MANIFEST)


PR_SUBSET = pr_subset(CELLS, FINDINGS)


def parameters():
    for cell in CELLS:
        for record in record_groups(cell):
            finding = FINDINGS.get(cell.id, {}).get(record)
            marks = []
            if cell.id not in PR_SUBSET:
                marks.extend((pytest.mark.slow, pytest.mark.s0_weekly))
            if finding:
                marks.append(pytest.mark.xfail(
                    strict=True, raises=KnownFinding,
                    reason='S0 finding: ' + expectations(finding)[2]))
            yield pytest.param(cell, record, marks=marks, id=f'{cell.id}/{record}')


@pytest.mark.parametrize('cell,record', tuple(parameters()))
def test_path_equivalence(cell, record, matrix_worker):
    report = matrix_worker(cell)
    if not cell.equivalence:
        # This assertion is outside KnownFinding: an xfail cannot hide stepping.
        assert report.get('refusal_scan_started') is False, 'refusal reached a scan'
    unexamined = [f for f in report['failures'] if group(f) not in (*record_groups(cell), 'execution')]
    assert not unexamined, unexamined
    finding = FINDINGS.get(cell.id, {}).get(record, [])
    known, witnesses, _ = expectations(finding)
    assert_record(report, record, known, witnesses)
