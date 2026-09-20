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

Matching collapses whitespace on BOTH sides, so where the line happens to break
inside the sentence does not matter. The first version of this file listed the
two break positions it had actually seen, which pinned the shipped instance
instead of the class: the same sentence broken one word earlier
("S11/S21\\nare unaffected") walked straight through it.

The one sentence allowed to contain either claim is the retraction itself, read
from ``rfx.preflight.msl`` so a reworded retraction cannot silently widen the
exemption. A page may also QUOTE a retired figure to say it is retired; such a
paragraph has to carry one of the withdrawal words below, which is what makes
the exemption a statement rather than an oversight.

Known bound: paragraphs are split on blank lines, so a retired sentence with a
BLANK LINE inside it would escape. That is not prose anyone writes, and
widening the split would merge unrelated paragraphs into one exemption scope,
which is the more likely way to lose a real hit.
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
    "-5 to -10 dB",
    "-5 to -10dB",
)

# A paragraph that withdraws a figure may name it. One of these has to be in
# the same paragraph for that exemption to apply.
WITHDRAWAL_WORDS = ("withdrawn", "retired", "do not cite", "Neither")


def _flat(text: str) -> str:
    """Whitespace-insensitive form: line breaks inside a sentence vanish."""
    return " ".join(text.split())


def offending_paragraphs(text: str) -> list[tuple[str, str]]:
    """Every (claim, paragraph) where a retired claim is stated, not withdrawn.

    The single predicate. Both the per-page test and the self-check below call
    THIS, so a bug in the paragraph split or in the retraction strip fails a
    test instead of being duplicated into one.
    """
    retraction = _flat(MSL_PROBE_CLEARANCE_RETRACTION)
    hits: list[tuple[str, str]] = []
    for para in re.split(r"\n\s*\n", text):
        flat = _flat(para).replace(retraction, "")
        if any(word in flat for word in WITHDRAWAL_WORDS):
            continue
        for claim in RETIRED:
            if _flat(claim) in flat:
                hits.append((claim, para.strip()))
    return hits


def scan(path: Path) -> list[tuple[str, str]]:
    return offending_paragraphs(path.read_text(encoding="utf-8"))


def test_the_scan_covers_the_pages_it_claims_to() -> None:
    assert SCANNED, "the scan found no pages -- the glob is wrong"
    names = {p.name for p in SCANNED}
    for expected in ("sources-ports.mdx", "probes-sparams.mdx",
                     "known_limitations.md", "sparameter_support_matrix.md"):
        assert expected in names, f"{expected} is not being scanned"


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: p.name)
def test_no_page_states_a_retired_msl_claim(path: Path) -> None:
    hits = scan(path)
    assert not hits, (
        f"{path.relative_to(REPO)} states a retired claim without "
        "withdrawing it:\n\n"
        + "\n\n".join(f"[{claim}]\n{para}" for claim, para in hits)
        + "\n\nEither rewrite the paragraph to what the measurement actually "
        "shows (rfx.preflight.msl.MSL_PROBE_CLEARANCE_EFFECT is the single "
        "source), or, if the page is quoting the claim in order to retire it, "
        "say so in the same paragraph."
    )


# Where the line breaks must not matter. The first three are the two shipped
# paragraphs and the break position that walked through the first version of
# this file; the last two are the exemptions, which must NOT be reported.
_BREAK_POSITIONS = [
    pytest.param("the V·I-split S11/S21 are unaffected.\nRefine the mesh.",
                 True, id="shipped_unbroken"),
    pytest.param("The V·I-split S11/S21 are\nunaffected; refine the mesh.",
                 True, id="shipped_broken_before_unaffected"),
    pytest.param("the V·I-split S11/S21\nare unaffected. Refine the mesh.",
                 True, id="broken_one_word_earlier"),
    pytest.param("physical |S11| may read as -5 to\n-10 dB instead of 0 dB.",
                 True, id="figure_broken_mid_range"),
    pytest.param("The `-5 to -10 dB` figure an older message quoted was "
                 "withdrawn on 2026-09-13; do not cite it.",
                 False, id="exempt_withdrawal"),
    pytest.param(MSL_PROBE_CLEARANCE_RETRACTION, False, id="exempt_retraction"),
]


@pytest.mark.parametrize("paragraph,should_fire", _BREAK_POSITIONS)
def test_the_scan_fires_wherever_the_line_happens_to_break(
    tmp_path: Path, paragraph: str, should_fire: bool,
) -> None:
    """The self-check runs the REAL scan over a real file.

    A scan that matches nothing is not a gate, and one that matches only the
    break positions someone happened to see is a gate against one instance.
    """
    page = tmp_path / "sample.mdx"
    page.write_text(
        "# Heading\n\nAn unrelated paragraph.\n\n" + paragraph + "\n",
        encoding="utf-8",
    )
    hits = scan(page)
    assert bool(hits) is should_fire, (paragraph, hits)


def test_an_unrelated_use_of_the_words_is_not_a_hit(tmp_path: Path) -> None:
    """"are unaffected" is ordinary English and several live entries use it
    correctly; the scan matches the CLAIM, not the words."""
    page = tmp_path / "sample.md"
    page.write_text(
        "Moving the target changes the reported RCS; relative comparisons at "
        "a fixed position are unaffected.\n",
        encoding="utf-8",
    )
    assert scan(page) == []
