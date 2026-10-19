"""Graded-record diagnostics are silent and opt in to one named JSONL file."""
import json
from tests.unit.runners._graded_records_scene import record, mutation_result


def test_reports_require_explicit_destination(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv('RFX_GRADED_RECORDS_REPORT', raising=False)
    monkeypatch.chdir(tmp_path)
    record('test', {}, {'value': 1})
    mutation_result('test', 'skip', 'expected failure')
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out == ''
    target = tmp_path / 'report.jsonl'
    monkeypatch.setenv('RFX_GRADED_RECORDS_REPORT', str(target))
    record('test', {}, {'value': 1})
    mutation_result('test', 'skip', 'expected failure')
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert rows[0]['values'] == {'value': 1}
    assert rows[1]['status'] == 'RED' and rows[1]['mutation'] == 'skip'
    assert capsys.readouterr().out == ''
