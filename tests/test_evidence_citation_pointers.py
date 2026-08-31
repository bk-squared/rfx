"""Every ``file.py:LINE`` pointer in the crossval evidence documents must land
on the constant it names.

Why this exists (#812 Phase 0, item 1). ``validation/crossval/manifest.json``
and ``validation/README.md`` are the durable statements of what each crossval
case's evidence *is*. When they delegate a number to a test, they cite it as
``tests/....py:LINE``. That pointer is the reader's only route from the claim to
the assertion that backs it, and a bare line number rots silently: commit
``f88f992`` wrote ``tests/test_patch_canonical_farfield_e4.py:111-116`` into
both documents while, in the same commit, adding six lines to that file's module
docstring -- so the pointer shipped already stale, and line 111 in the shipped
tree is an unrelated comment inside a mechanism note.

The rule this gate enforces is exactly the reader's experience:

1. every cited line must be a module-level constant assignment
   (``NAME = ...``, upper-case), not a comment or a loop header;
2. the citing document must *name* that constant in its own text, so the
   pointer survives the next edit -- a reader who lands on the wrong line
   because something moved can still grep for the name.

It asserts NO physics and changes NO gate. It is a documentation-integrity
contract, in the same family as ``tests/test_rcs_bistatic_caveat_docpin.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# The documents whose pointers are under contract. Both are read by humans as
# the authoritative statement of a crossval case's delegated evidence.
DOCUMENTS = (
    "validation/crossval/manifest.json",
    "validation/README.md",
)

# A pointer looks like ``tests/foo.py:117`` or ``tests/foo.py:117,121,122`` or
# ``tests/foo.py:111-116``.
_CITATION = re.compile(
    r"\b((?:tests|validation|scripts|rfx)/[A-Za-z0-9_/.-]+\.py):(\d+(?:[-,]\d+)*)"
)

# A module-level constant the documents are allowed to point at.
_CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=")

# Deleting every pointer would make this gate vacuously green, so require the
# population it was written for to still be there. Three pointers exist today
# (two in the manifest's cv05 claim_scope, one in the README's cv05 row).
MIN_CITATIONS = 3


def _expand(linespec: str) -> list[int]:
    out: list[int] = []
    for part in linespec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _citations() -> list[tuple[str, str, str, list[int]]]:
    found: list[tuple[str, str, str, list[int]]] = []
    for doc in DOCUMENTS:
        text = (_REPO / doc).read_text(encoding="utf-8")
        for match in _CITATION.finditer(text):
            found.append((doc, text, match.group(1), _expand(match.group(2))))
    return found


def test_the_cited_population_is_still_present():
    """A green gate must mean the pointers are right, not that they are gone."""
    found = _citations()
    assert len(found) >= MIN_CITATIONS, (
        f"only {len(found)} file:line pointers found across {DOCUMENTS}; "
        f"expected at least {MIN_CITATIONS}. If a pointer was legitimately "
        f"removed, lower MIN_CITATIONS in the same commit and say why."
    )


@pytest.mark.parametrize("doc,path,linespec", [
    (doc, path, ",".join(str(n) for n in lines))
    for doc, _text, path, lines in _citations()
])
def test_every_cited_line_is_a_constant_the_document_names(doc, path, linespec):
    text = (_REPO / doc).read_text(encoding="utf-8")
    target = _REPO / path
    assert target.exists(), f"{doc} cites {path}, which does not exist"
    source = target.read_text(encoding="utf-8").splitlines()

    for lineno in _expand(linespec):
        assert 1 <= lineno <= len(source), (
            f"{doc} cites {path}:{lineno}, past end of file ({len(source)} lines)"
        )
        line = source[lineno - 1]
        constant = _CONSTANT.match(line)
        assert constant, (
            f"{doc} cites {path}:{lineno} as the authoritative source for a "
            f"number, but that line is not a constant assignment. It reads:\n"
            f"    {line.rstrip()}\n"
            f"A reader following this pointer lands on the wrong thing."
        )
        name = constant.group(1)
        assert name in text, (
            f"{doc} cites {path}:{lineno}, which defines {name}, but the "
            f"document never names {name}. Cite the constant NAME alongside "
            f"the line so the pointer survives the next edit."
        )
