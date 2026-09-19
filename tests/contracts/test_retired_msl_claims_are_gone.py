"""No shipped page may state a claim about MSL probe clearance that was retired.

Issue #726. Two honesty messages about the same condition contradicted each
other, and the measurement retired BOTH of their headline claims: the guard's
"The V·I-split S11/S21 are unaffected" and preflight's "may read as -5 to -10 dB
instead of 0 dB". The code was fixed first and the PUBLIC GUIDE was not --
``docs/public/guide/sources-ports.mdx`` and ``probes-sparams.mdx`` were still
narrating the old guard output, one of them across a line break where a naive
grep for the sentence missed it.

That is the recurrence this file exists to stop, and it is deliberately a
DOCS scan rather than another message assertion: the message sites are already
pinned by ``tests/unit/ports/test_msl_clearance_diagnostic.py``, and what was
unguarded was prose about them.

The one sentence allowed to contain either claim is the retraction itself, read
from ``rfx.preflight.msl`` so a reworded retraction cannot silently widen the
exemption. A page may also QUOTE a retired figure to say it is retired; such a
line has to carry one of the withdrawal words below, which is what makes the
exemption a statement rather than an oversight.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from rfx.preflight.msl import MSL_PROBE_CLEARANCE_RETRACTION

REPO = Path(__file__).resolve().parents[2]
SCANNED = sorted(
    list((REPO / "docs" / "public" / "guide").glob("*.mdx"))
    + list((REPO / "docs" / "guides").glob("*.md"))
)

# The claims, not the bare words: "are unaffected" is an ordinary English
# phrase that several unrelated entries use correctly.
RETIRED = (
    "S11/S21 are unaffected",
    "S11/S21 are\nunaffected",      # the line break that hid one of them
    "-5 to -10 dB",
    "-5 to -10dB",
)

# A line that withdraws a figure may name it. One of these words has to be
# within the same paragraph for that exemption to apply.
WITHDRAWAL_WORDS = ("withdrawn", "retired", "do not cite", "Neither")


def _paragraphs(text: str):
    for para in re.split(r"\n\s*\n", text):
        yield para


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: p.name)
def test_no_page_states_a_retired_msl_claim(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert SCANNED, "the scan found no pages -- the glob is wrong"
    for para in _paragraphs(text.replace(MSL_PROBE_CLEARANCE_RETRACTION, "")):
        for claim in RETIRED:
            if claim not in para:
                continue
            assert any(w in para for w in WITHDRAWAL_WORDS), (
                f"{path.relative_to(REPO)} states the retired claim "
                f"{claim!r} without withdrawing it:\n\n{para.strip()}\n\n"
                "Either rewrite the paragraph to what the measurement "
                "actually shows (rfx.preflight.msl."
                "MSL_PROBE_CLEARANCE_EFFECT is the single source), or, if "
                "the page is quoting the claim in order to retire it, say "
                "so in the same paragraph."
            )


def test_the_scan_would_catch_the_defect_it_was_written_for() -> None:
    """A scan that matches nothing is not a gate. This reproduces the exact
    two paragraphs that shipped, including the line-broken one, and asserts
    the predicate fires on both."""
    shipped = (
        "not exactly the declared one (issue #752); the V·I-split "
        "S11/S21 are unaffected.\nRefine the mesh before quoting Z0."
    )
    line_broken = (
        "board that was declared. The V·I-split S11/S21 are\n"
        "unaffected; refine the mesh before quoting Z0."
    )
    for sample in (shipped, line_broken):
        assert any(c in sample for c in RETIRED)
        assert not any(w in sample for w in WITHDRAWAL_WORDS)
