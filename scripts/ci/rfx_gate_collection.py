"""Persist post-deselection nodeids before any tests execute, including on xdist."""
import json
import os
from pathlib import Path


_skipped = []


def pytest_sessionstart(session):
    _skipped.clear()


def pytest_collectreport(report):
    if report.skipped:
        _skipped.append([report.nodeid, str(report.longrepr)])


def pytest_collection_finish(session):
    directory = Path(os.environ["RFX_GATE_COLLECTION_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{os.getpid()}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps([item.nodeid for item in session.items]), encoding="utf-8")
    temporary.replace(target)
    # Collection skips have JUnit cases but are not runnable session.items.
    # Keep their evidence separate so they never satisfy retry coverage.
    target.with_suffix(".skips").write_text(json.dumps(_skipped), encoding="utf-8")
