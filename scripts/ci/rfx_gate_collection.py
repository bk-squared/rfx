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


def pytest_runtest_teardown(item, nextitem):
    """Drop JAX's compiled programs when a test module ends.

    JAX keeps every compiled program of a module alive after the module's last test, so a
    stage-2 worker that runs many modules grows without bound: on a 454-file selection eight
    workers reached the 64 GiB container limit (62 GiB of process memory, 101,168 mappings in
    one process) and the gate was restarted twice; with this hook the same selection passed at
    43 GiB and 31,337 mappings (issue #1528). Nothing is cleared inside a module.
    """
    if nextitem is None or getattr(nextitem, "module", None) is not getattr(item, "module", None):
        import gc

        import jax

        jax.clear_caches()
        gc.collect()
