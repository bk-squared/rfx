"""Doc-contract witness for the RCS bistatic near-field caveat (audit item #2).

`compute_rcs` returns a full bistatic `rcs_dbsm`/`rcs_linear` pattern, but only
the monostatic (backscatter) bin is cross-validated against the exact Mie series
(``tests/test_rcs_mie_fixture.py``). At the auto-placed default NTFF box
(``ntff_offset=1``, deep in the reactive near field) the off-backscatter bins
can be several dB to ~20 dB off (a spurious forward-oblique lobe, ~10 dB high vs
Mie on the committed ka~1 sphere; ``tests/fixtures/rcs_sphere_mie/``).

The audit asked whether that should surface at call time. It was R2-STOPPED to a
doc-pin: ``monostatic_rcs`` is computed at the exact backscatter direction
INDEPENDENT of the observation grid, and every validated monostatic test passes
a FULL observation grid, so there is no call-time signal that separates
"monostatic-only" from "bistatic" intent to key an advisory on without
false-alarming those tests.

This test locks the honesty text (docstrings + public guide) so the doc-pin
cannot be silently stripped. It asserts NO physics and changes NO gate; the
quantitative gate stays in ``tests/test_rcs_mie_fixture.py`` (monostatic) and
``tests/test_rcs_mie_reference_gates.py`` (envelope).
"""

from __future__ import annotations

from pathlib import Path

from rfx.rcs import RCSResult, compute_rcs


def _norm(text: str) -> str:
    # collapse whitespace so line-wrapping in the source can change freely
    return " ".join(text.split()).lower()


def test_compute_rcs_docstring_pins_bistatic_caveat():
    doc = _norm(compute_rcs.__doc__ or "")
    assert "bistatic pattern caveat" in doc
    assert "not validated" in doc
    # names the validated quantity and the exact-Mie anchor
    assert "monostatic_rcs" in doc and "mie" in doc
    # names the near-field cause (default box) ...
    assert "ntff_offset" in doc and "near field" in doc
    # ... and steers users off the falsified "just enlarge ntff_offset" fix
    assert "does not" in doc and "oblique" in doc


def test_rcsresult_docstring_pins_validation_scope():
    doc = _norm(RCSResult.__doc__ or "")
    assert "validation scope" in doc
    assert "monostatic_rcs" in doc and "not validated" in doc
    assert "rcs_dbsm" in doc  # the non-validated field is named


def test_public_guide_pins_bistatic_caveat():
    guide = Path(__file__).resolve().parents[1] / "docs/public/guide/farfield-rcs.md"
    assert guide.exists(), guide
    text = _norm(guide.read_text())
    assert "bistatic pattern is not validated" in text
    assert "monostatic" in text and "mie" in text
    # the guide must keep the honest "enlarging ntff_offset does not fix it" note
    assert "ntff_offset" in text and "does not" in text
