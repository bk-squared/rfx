"""A test that reads prose is a documentation check, and a required lane runs none.

Required checks compare numbers with a tolerance, never text (PI, 2026-09-24),
and a documentation mismatch never holds a merge (PI, 2026-09-22). Four of the
nine red ``main`` runs of 2026-09-22/24 were a notes table compared with a
script's output whose last printed digit followed the OpenBLAS kernel (#1262).
So a test that reads a note, a guide or a README is marked ``docs_consistency``;
the required gates deselect that marker and the non-required docs-consistency
workflow runs it.

``tests/_prose_reads.py`` enforces it at run time: it fails a test without the
marker that opened prose. These tests show that the plugin is wired into every
session this repository runs, that the required gates deselect the marker, and
-- by running planted tests through the plugin in a separate interpreter --
that it catches each way a test reads prose and passes what it should pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from tests import _prose_reads

REPO = Path(__file__).resolve().parents[2]
PR_TESTS = REPO / ".github" / "workflows" / "pr-tests.yml"
DOCS_WORKFLOW = REPO / ".github" / "workflows" / "docs-consistency.yml"

#: A prose file every checkout carries, read-only here.
NOTE = REPO / "docs" / "agent" / "working-on-rfx.mdx"

_PLANTED = {
    # The pre-#1262 shape: a note's text against what a script prints.
    "test_note_equals_tables.py": f'''
        from pathlib import Path
        NOTE = Path({str(NOTE)!r})

        def test_note_equals_tables():
            assert NOTE.read_text() == NOTE.read_text()
    ''',
    "test_marked.py": f'''
        import pytest
        from pathlib import Path

        @pytest.mark.docs_consistency
        def test_marked_note_check():
            assert "rfx" in Path({str(NOTE)!r}).read_text()
    ''',
    "test_builtin_open.py": f'''
        def test_reads_through_open():
            with open({str(NOTE)!r}, encoding="utf-8") as fh:
                assert fh.read()
    ''',
    "test_module_level.py": f'''
        from pathlib import Path
        TEXT = Path({str(NOTE)!r}).read_text()

        def test_uses_nothing_from_the_text():
            assert True
    ''',
    "test_cached_fixture.py": f'''
        import pytest
        from pathlib import Path

        @pytest.fixture(scope="module")
        def note():
            return Path({str(NOTE)!r}).read_text()

        @pytest.mark.docs_consistency
        def test_first_user_is_marked(note):
            assert note

        def test_second_user_is_not(note):
            assert note
    ''',
    "test_not_prose.py": f'''
        from pathlib import Path

        def test_reads_configuration_a_record_and_a_readme():
            assert Path({str(REPO / "pyproject.toml")!r}).read_text()
            assert Path({str(REPO / "docs" / "guides" / "sparameter_support_matrix.json")!r}).read_text()
            assert Path({str(REPO / "README.md")!r}).read_text()   # outside docs/: not seen
    ''',
}

#: What the plugin must say about each planted test: failed at teardown, or passed.
_EXPECTED = {
    "test_note_equals_tables": "error",
    "test_marked_note_check": "passed",
    "test_reads_through_open": "error",
    "test_uses_nothing_from_the_text": "error",
    "test_first_user_is_marked": "passed",
    "test_second_user_is_not": "error",
    "test_reads_configuration_a_record_and_a_readme": "passed",
}


def _outcomes(tmp_path: Path) -> dict[str, str]:
    for name, body in _PLANTED.items():
        (tmp_path / name).write_text(textwrap.dedent(body), encoding="utf-8")
    report = tmp_path / "report.xml"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    env.pop("PYTEST_ADDOPTS", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "tests._prose_reads", f"--junitxml={report}", str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert report.is_file(), proc.stdout + proc.stderr
    outcomes: dict[str, str] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        kinds = {child.tag for child in case}
        outcomes[case.get("name")] = ("error" if "error" in kinds else
                                      "failed" if "failure" in kinds else
                                      "skipped" if "skipped" in kinds else "passed")
        if "error" in kinds:
            message = next(c for c in case if c.tag == "error").get("message", "")
            assert "docs_consistency" in message and "reads prose" in message, message
    return outcomes


def test_the_plugin_fails_each_unmarked_way_of_reading_prose(tmp_path: Path) -> None:
    assert _outcomes(tmp_path) == _EXPECTED


def test_the_plugin_is_loaded_by_the_root_conftest(request) -> None:
    """Every session in this repository, required lanes included, runs it."""
    assert request.config.pluginmanager.has_plugin("tests._prose_reads")


def test_the_audit_hook_is_recording_in_this_session(request) -> None:
    """Not only registered: this session's hook sees an open() of a prose file.

    The record is removed again before teardown, so this unmarked test is not
    failed for the read it makes on purpose.
    """
    NOTE.read_bytes()
    seen = _prose_reads._reads.pop(request.node.nodeid, [])
    assert os.path.relpath(NOTE, REPO) in seen, seen


@pytest.mark.parametrize("path", [
    "docs/design_notes/x.md", "docs/public/guide/y.mdx", "docs/guides/z.rst",
])
def test_prose_is_what_it_says(path: str) -> None:
    assert _prose_reads.is_prose(REPO / path)


@pytest.mark.parametrize("path", [
    "docs/guides/sparameter_support_matrix.json", "CHANGELOG.md", "README.md",
    "changelog.d/1.fixed.md", "scripts/ci/DURATIONS.md", "pyproject.toml",
    "validation/crossval/README.md", "/tmp/elsewhere/docs/x.md",
])
def test_everything_else_is_not_prose(path: str) -> None:
    assert not _prose_reads.is_prose(path if path.startswith("/") else REPO / path)


def _pytest_runs(path: Path) -> list[str]:
    runs = []
    for job in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"].values():
        for step in job.get("steps", []):
            run = str(step.get("run", ""))
            if "pytest" in run and '-m "' in run:
                runs.append(run)
    return runs


def test_the_required_gates_deselect_the_marker_and_the_docs_lane_selects_it() -> None:
    """Both pr-tests gates override addopts, so each needs the marker in -m."""
    required = _pytest_runs(PR_TESTS)
    assert len(required) >= 2, required
    for run in required:
        assert "not docs_consistency" in run, run
    docs = _pytest_runs(DOCS_WORKFLOW)
    assert docs and all('-m "docs_consistency and not gpu"' in run for run in docs), docs
    addopts = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "and not docs_consistency'\"" in addopts
