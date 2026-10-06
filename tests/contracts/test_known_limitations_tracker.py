"""Offline contract for every public limitation subject; no solver imports."""
import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'kl_tracker', ROOT / 'scripts/ci/check_known_limitations.py')
kl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kl)


@pytest.mark.reads_docs_for_gate(reason='Every known limitation must have an explicit tracker or accepted reason')
def test_every_limitation_has_a_valid_tracker():
    subjects = kl.entries(kl.PAGE.read_text())
    assert subjects, 'No subjects found'
    problems = []
    for entry in subjects:
        try:
            kl.trackers(entry)
            # Resolve evidence without importing test modules (and thus JAX).
            match = re.search(
                r'(?:at default|diagnostic, does not enter the result): (tests/[^)]+)\)',
                entry.body,
            )
            if match:
                path, *names = match[1].split('::')
                source = ROOT / path
                assert source.is_file(), f'{entry.title}: missing evidence file {path}'
                nodes = ast.parse(source.read_text()).body
                for name in names:
                    node = next((n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                                 and n.name == name), None)
                    assert node is not None, f'{entry.title}: missing evidence test {match[1]}'
                    nodes = node.body
        except ValueError as exc:
            problems.append(str(exc))
    assert not problems, '\n' + '\n'.join(problems)


@pytest.mark.parametrize('body', ['', 'Tracker: TODO-LEADER', 'Tracker: none',
                                  'Tracker: TODO-NEW-ISSUE (silent gradient error)',
                                  'Tracker: none — accepted limitation (documented)',
                                  'Tracker: none — accepted limitation ()',
                                  'Tracker: none — accepted limitation (   )',
                                  'Tracker: #1\nTracker: #2',
                                  '```\nTracker: #1\n```',
                                  '<!-- Tracker: #1 -->'])
def test_missing_or_invalid_tracker_refuses(body):
    entry, = kl.entries('# Page\n\n## Subject\n' + body + '\n')
    with pytest.raises(ValueError):
        kl.trackers(entry)


def test_all_leaf_subjects_count_and_section_headings_do_not():
    page = ('# Page\nintro\n## Standalone\nTracker: #1\n'
            '## Group\n### First\nTracker: #2, #3\n'
            '### Second\nTracker: none — accepted limitation (refused at default: tests/synthetic.py::test_refuses)\n'
            '## What this page is not\nfooter\n')
    subjects = kl.entries(page)
    assert [e.title for e in subjects] == ['Standalone', 'First', 'Second']
    assert [kl.trackers(e) for e in subjects] == [{1}, {2, 3}, set()]


@pytest.mark.parametrize('subject', ['**Legacy subject.**', '- **Legacy subject:**'])
def test_legacy_bold_subject_cannot_evade_tracker_contract(subject):
    entry, = kl.entries('# Page\n## Section\n' + subject + '\nAvoid this.\n')
    with pytest.raises(ValueError):
        kl.trackers(entry)


@pytest.mark.parametrize('reason', [
    'refused at default', 'warns at default', 'diagnostic, does not enter the result',
])
@pytest.mark.parametrize('reference,error', [
    ('tests/contracts/no_such_evidence.py::test_missing', 'missing evidence file'),
    ('tests/contracts/test_known_limitations_tracker.py::test_missing', 'missing evidence test'),
])
def test_all_acceptance_forms_require_existing_evidence(tmp_path, monkeypatch, reason, reference, error):
    page = tmp_path / 'synthetic.md'
    page.write_text(f'## Subject\nTracker: none — accepted limitation ({reason}: {reference})\n')
    monkeypatch.setattr(kl, 'PAGE', page)
    with pytest.raises(AssertionError, match=error):
        test_every_limitation_has_a_valid_tracker()
