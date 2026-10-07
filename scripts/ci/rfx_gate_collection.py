"""Persist post-deselection nodeids before any tests execute, including on xdist."""
import json
import os
from pathlib import Path


def pytest_collection_finish(session):
    directory = Path(os.environ["RFX_GATE_COLLECTION_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{os.getpid()}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps([item.nodeid for item in session.items]), encoding="utf-8")
    temporary.replace(target)
