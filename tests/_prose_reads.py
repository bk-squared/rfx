"""tests/_prose_reads.py -- a test that reads prose is a documentation check.

Required checks compare numbers with a tolerance, never text (PI, 2026-09-24),
and a documentation mismatch never holds a merge (PI, 2026-09-22). So a test
that reads a note, a guide or a README carries the ``docs_consistency``
marker: pyproject's default addopts and the required gates in pr-tests.yml
deselect it, and the non-required docs-consistency workflow runs it. Four of
the nine red ``main`` runs of 2026-09-22/24 were a notes table compared with a
script's output whose last digit followed the OpenBLAS kernel (#1262).

This plugin turns that rule into a check. A Python audit hook sees every
``open`` the process makes, and the prose files among them are recorded
against whatever pytest is doing at that moment: running a test (setup, call,
teardown), setting up a fixture, or importing a test module. Prose is a
``.md``, ``.mdx`` or ``.rst`` file under ``docs/``, where the guides, the
public pages and the design notes live. A test without the marker fails at
teardown, naming the files, if it opened one, if a fixture it uses opened one
when it was set up (a cached module- or session-scoped fixture counts for every
test that uses it, not only the first), or if its module opened one while it
was imported. A module that reads prose at import is a documentation module,
so all of its tests carry the marker.

What it cannot see: a file read by a subprocess the test starts (another
interpreter), a file read by a module that an earlier test module imported
first (an import runs once), a file read while a fixture is torn down, and
prose outside ``docs/``: the READMEs, ``scripts/ci/DURATIONS.md``, the
diagnostics' own Markdown. Those are left out because an ``open`` does not say
why a file was read -- the example-fidelity contract hashes every file under
``validation/crossval/``, READMEs included, to prove a build wrote nothing --
and tests that compare them with code are marked by hand.

The root conftest loads this module through ``pytest_plugins``;
``tests/contracts/test_required_tests_read_no_prose.py`` runs planted tests
through it in a subprocess.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

MARKER = "docs_consistency"
REPO = Path(__file__).resolve().parents[1]
_DOCS = str(REPO / "docs") + os.sep
_SUFFIXES = (".md", ".mdx", ".rst")

#: What pytest is doing: a test's or a test module's nodeid, a fixture's
#: ``(baseid, argname)``, or None in between.
_where: object = None
#: ``_where`` -> prose paths (repository-relative) opened while it was current.
_reads: dict[object, list[str]] = {}
_installed = False


def is_prose(path: object) -> bool:
    """A ``.md``/``.mdx``/``.rst`` file under the repository's ``docs/``."""
    try:
        name = os.fsdecode(path)
    except TypeError:          # a file descriptor
        return False
    return name.endswith(_SUFFIXES) and os.path.abspath(name).startswith(_DOCS)


def _audit(event: str, args: tuple) -> None:
    if event != "open" or _where is None:
        return
    if is_prose(args[0]):
        rel = os.path.relpath(os.path.abspath(os.fsdecode(args[0])), REPO)
        _reads.setdefault(_where, []).append(rel)


def pytest_configure(config: pytest.Config) -> None:
    global _installed
    config.addinivalue_line(
        "markers", f"{MARKER}: documentation checked against code, records or "
        "script output; runs only in the non-required docs-consistency workflow")
    if not _installed:            # an audit hook cannot be removed; add it once
        sys.addaudithook(_audit)
        _installed = True


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """Record what a test module opens while it is imported."""
    global _where
    if not isinstance(collector, pytest.Module):
        yield
        return
    previous, _where = _where, collector.nodeid
    try:
        yield
    finally:
        _where = previous


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    """Record what a fixture opens while it is set up, against the fixture."""
    global _where
    previous, _where = _where, (fixturedef.baseid, fixturedef.argname)
    try:
        yield
    finally:
        _where = previous


def _fixture_reads(item) -> list[str]:
    info = getattr(item, "_fixtureinfo", None)
    if info is None:
        return []
    read = []
    for defs in info.name2fixturedefs.values():
        fixturedef = defs[-1]          # the one this item gets; the rest are overridden
        read += _reads.get((fixturedef.baseid, fixturedef.argname), [])
    return read


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _where
    _where = item.nodeid
    try:
        yield
    finally:
        _where = None


@pytest.fixture(autouse=True)
def _no_prose_read_outside_docs_consistency(request):
    """Fail a test without the marker that read prose, or whose module did."""
    yield
    item = request.node
    module = item.getparent(pytest.Module)
    read = _reads.pop(item.nodeid, []) + _fixture_reads(item) + (
        _reads.get(module.nodeid, []) if module is not None else [])
    if read and item.get_closest_marker(MARKER) is None:
        pytest.fail(
            f"this test reads prose ({', '.join(sorted(set(read)))}) but is not "
            f"marked `{MARKER}`. Required checks compare numbers, not text, and "
            "a documentation mismatch never holds a merge (PI, 2026-09-22/24). "
            f"Mark it @pytest.mark.{MARKER} (the whole module if it reads the "
            "file at import) so the docs-consistency workflow runs it instead; "
            "if it pins a number, read the number from its record, not from a note.",
            pytrace=False,
        )
