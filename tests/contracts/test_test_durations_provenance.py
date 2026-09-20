"""`scripts/ci/DURATIONS.md` must describe the `.test_durations` beside it.

The doc is the only record of where the numbers came from and what was done to
them, and it went stale twice: its header claimed 7594 entries and "nothing is
carried from an older file" while the committed file held 10625 entries, 297 of
them carried from the previous one. A provenance note nobody can trust is worse
than none, because it is quoted.

Nothing here reads the split algorithm or predicts a shard time. It checks the
three things that can be checked from the two files alone: the file is
well-formed, the header's counts match it, and the 0.3 s floor retired on
2026-09-18 has not crept back.

Deliberately NOT checked, and why: the carried count cannot be recomputed after
this file merges, because the "older file" it counts against is this one's own
predecessor. Once merged, `git show origin/main:.test_durations` IS this file
and every entry looks carried. It gets a bounds check and a regeneration step
(step 6) instead of a value check -- said plainly so a later reader does not
mistake a bounds check for a verification.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DURATIONS = REPO_ROOT / ".test_durations"
DOC = REPO_ROOT / "scripts" / "ci" / "DURATIONS.md"

# Two distinct phrasings so neither regex can match the other's sentence. If the
# header is reworded, reword these with it; a doc that says neither fails loudly
# rather than silently matching a number that means something else.
ENTRIES_RE = re.compile(r"\*\*([\d,]+) entries in the\s+file\*\*")
CARRIED_RE = re.compile(r"\*\*([\d,]+) of them carried unchanged\*\*")

# The floor was an ADDITIVE 0.3 s on every entry, so while it was applied the
# smallest entry in the file could not be below 0.3 s. Retired 2026-09-18
# (DURATIONS.md, "The 0.3 s floor, retired 2026-09-18").
RETIRED_FLOOR_S = 0.3


def _int(match: re.Match[str]) -> int:
    return int(match.group(1).replace(",", ""))


def _doc_text() -> str:
    assert DOC.is_file(), f"{DOC} is missing — the durations file has no provenance"
    return DOC.read_text(encoding="utf-8")


def _durations() -> dict:
    assert DURATIONS.is_file(), f"{DURATIONS} is missing"
    return json.loads(DURATIONS.read_text(encoding="utf-8"))


def test_the_durations_file_is_well_formed():
    data = _durations()
    assert isinstance(data, dict) and data, "expected a non-empty JSON object of nodeid -> seconds"

    bad_types = [k for k, v in data.items() if not isinstance(v, (int, float)) or isinstance(v, bool)]
    assert not bad_types, f"{len(bad_types)} entries are not numbers, e.g. {bad_types[:3]}"

    negative = {k: v for k, v in data.items() if v < 0}
    assert not negative, (
        f"{len(negative)} entries hold a negative duration, e.g. "
        f"{list(negative.items())[:3]}. pytest-split sums these into a shard cost; a negative "
        "one silently buys a shard free time."
    )

    not_a_nodeid = [k for k in data if "::" not in k]
    assert not not_a_nodeid, (
        f"{len(not_a_nodeid)} keys are not pytest nodeids, e.g. {not_a_nodeid[:3]}"
    )


def test_the_doc_reports_the_entry_count_the_file_has():
    data = _durations()
    matches = ENTRIES_RE.findall(_doc_text())
    assert len(matches) == 1, (
        f"expected exactly one '**N entries in the file**' in {DOC.name}, found {len(matches)}"
    )
    stated = int(matches[0].replace(",", ""))
    assert stated == len(data), (
        f"{DOC.name} says {stated} entries, .test_durations holds {len(data)}. Regenerating "
        "step 6: update the header when the file changes."
    )


def test_the_doc_reports_a_carried_count_that_fits_the_file():
    data = _durations()
    matches = CARRIED_RE.findall(_doc_text())
    assert len(matches) == 1, (
        f"expected exactly one '**N of them carried unchanged**' in {DOC.name}, "
        f"found {len(matches)}"
    )
    carried = int(matches[0].replace(",", ""))
    assert 0 <= carried <= len(data), (
        f"{DOC.name} says {carried} entries carried unchanged, but the file holds "
        f"{len(data)} in total. A carried count is a subset of the entries."
    )


def test_the_retired_floor_has_not_come_back():
    """The floor was additive, so applying it puts a 0.3 s sill under the file."""
    data = _durations()
    smallest = min(data.values())
    assert smallest < RETIRED_FLOOR_S, (
        f"the smallest entry is {smallest} s, at or above the {RETIRED_FLOOR_S} s floor that was "
        "retired on 2026-09-18 — every entry carrying a constant is what that looks like. If a "
        "floor is wanted again, measure the per-test overhead on the lane that needs it and write "
        "the measurement into scripts/ci/DURATIONS.md before re-adding one."
    )
