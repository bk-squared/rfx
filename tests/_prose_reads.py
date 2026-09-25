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
was imported, or if pytest opened one while it generated that test's
parameters. A module that reads prose at import is a documentation module, so
all of its tests carry the marker.

Nothing legitimate opens prose outside those four: not a conftest, not a
session hook, not the collection of a directory. A prose file opened while
none of them is current fails the whole session (fail closed), naming the file.

A few required tests read prose on purpose because the file is part of a gate,
not documentation: the hash of a pre-declaration's frozen sections (acceptance
criteria may not be edited after the result), or the rule that a
known-limitations entry stays while its defect is open. Such a test carries
``@pytest.mark.reads_docs_for_gate(reason="...")`` on the test function itself
-- not on a class or module, where it would cover tests nobody chose -- and a
marker without a non-empty reason fails the test whether or not it reads
anything. ``tests/contracts/test_required_tests_read_no_prose.py`` pins the
set of such tests, with their reasons, and the set of ``docs_consistency``
tests, so a change to either shows in the diff.

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
GATE_MARKER = "reads_docs_for_gate"
REPO = Path(__file__).resolve().parents[1]
_DOCS = str(REPO / "docs") + os.sep
_SUFFIXES = (".md", ".mdx", ".rst")

#: What pytest is doing: a test's or a test module's nodeid, a fixture's
#: ``(baseid, argname)``, a test function being parametrized
#: ``(parent nodeid, name, "params")``, or None in between.
_where: object = None
#: ``_where`` -> prose paths (repository-relative) opened while it was current.
_reads: dict[object, list[str]] = {}
#: Prose opened while ``_where`` was None; any entry fails the session.
_outside: list[str] = []
_installed = False


def is_prose(path: object) -> bool:
    """A ``.md``/``.mdx``/``.rst`` file under the repository's ``docs/``."""
    try:
        name = os.fsdecode(path)
    except TypeError:          # a file descriptor
        return False
    return name.endswith(_SUFFIXES) and os.path.abspath(name).startswith(_DOCS)


def _audit(event: str, args: tuple) -> None:
    if event != "open" or not is_prose(args[0]):
        return
    rel = os.path.relpath(os.path.abspath(os.fsdecode(args[0])), REPO)
    if _where is None:
        _outside.append(rel)
    else:
        _reads.setdefault(_where, []).append(rel)


def pytest_configure(config: pytest.Config) -> None:
    global _installed
    config.addinivalue_line(
        "markers", f"{MARKER}: documentation checked against code, records or "
        "script output; runs only in the non-required docs-consistency workflow")
    config.addinivalue_line(
        "markers", f"{GATE_MARKER}(reason): a required test that reads a file "
        "under docs/ because the file is part of a gate; the reason is mandatory")
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
def pytest_pycollect_makeitem(collector, name, obj):
    """Record what is opened while one test function is parametrized."""
    global _where
    previous, _where = _where, (collector.nodeid, name, "params")
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
    # The one this item gets from each static request; the rest are overridden.
    defs = [d[-1] for d in info.name2fixturedefs.values() if d] if info else []
    # A fixture the test asks for itself (request.getfixturevalue) is not in the
    # static closure; the item's request records every one it resolved.
    defs += list(getattr(getattr(item, "_request", None), "_fixture_defs", {}).values())
    read = []
    for fixturedef in {id(d): d for d in defs}.values():
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
        _reads.get(module.nodeid, []) if module is not None else []) + (
        _reads.get((item.parent.nodeid, getattr(item, "originalname", item.name),
                    "params"), []))
    gates = list(item.iter_markers_with_node(GATE_MARKER))
    if gates:
        node, gate = gates[0]
        if node is not item:
            pytest.fail(f"@pytest.mark.{GATE_MARKER} is on {node.nodeid}; it must "
                        "decorate the one test function that reads the gate's "
                        "file, not a class or module.", pytrace=False)
        reason = gate.kwargs.get("reason", gate.args[0] if gate.args else "")
        if not (isinstance(reason, str) and reason.strip()):
            pytest.fail(f"@pytest.mark.{GATE_MARKER} needs a non-empty reason= "
                        "saying which gate the file belongs to.", pytrace=False)
        return
    if read and item.get_closest_marker(MARKER) is None:
        pytest.fail(
            f"this test reads prose ({', '.join(sorted(set(read)))}) but is not "
            f"marked `{MARKER}`. Required checks compare numbers, not text, and "
            "a documentation mismatch never holds a merge (PI, 2026-09-22/24). "
            f"Mark it @pytest.mark.{MARKER} (the whole module if it reads the "
            "file at import) so the docs-consistency workflow runs it instead; "
            "if it pins a number, read the number from its record, not from a note; "
            f"if the file is part of a gate, mark it @pytest.mark.{GATE_MARKER}"
            '(reason="...").',
            pytrace=False,
        )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Fail closed: prose opened outside any test, fixture or module import."""
    if not _outside:
        return
    message = (
        f"prose was opened outside any test, fixture or test-module import: "
        f"{', '.join(sorted(set(_outside)))}. A conftest, a session hook or a "
        "directory's collection has no reason to read a note or a guide; move "
        f"the read into a test marked `{MARKER}` (tests/_prose_reads.py).")
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message, red=True)
    else:
        print(message, file=sys.__stderr__)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
