"""Committed waveguide fixtures contain no private absolute paths.

The fixture glob must find the recorded lanes, including the eleven fixtures
with recapture notes. Fixtures and their provenance remain unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from _fixture_recapture import RECAPTURE_NOTE_KEY

FIXTURES = sorted(REPO.glob("tests/fixtures/waveguide_*/*.json"))
PRIVATE_PREFIXES = ("/root/", "/home/", "~/", "/Users/")


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _with_note(paths):
    return [p for p in paths if RECAPTURE_NOTE_KEY in json.loads(p.read_text())]


def test_the_glob_finds_the_lanes() -> None:
    assert FIXTURES, "no waveguide fixtures found -- the glob is broken"
    noted = {p.name for p in _with_note(FIXTURES)}
    # five uniform magnitude + five uniform phase + the graded-dy lane
    assert len(noted) == 11, sorted(noted)
    assert "waveguide_wr90_nu_flux_broad_e5_envelope.json" in noted


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_no_private_absolute_path_anywhere(path: Path) -> None:
    for s in _strings(json.loads(path.read_text())):
        for prefix in PRIVATE_PREFIXES:
            assert not s.startswith(prefix), f"{path.name}: {s[:80]!r}"
            assert f" {prefix}" not in s, f"{path.name}: {s[:120]!r}"
