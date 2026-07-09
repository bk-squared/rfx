"""Doc-contract witness for the empty-window gradient caveat (blind-docs audit).

A blind, docs-only agent building an inverse-design loop trusted a
finite-difference-verified gradient whose loss was ~1e-7 — orders of magnitude
too small — because the reflection it was minimizing never landed in
``minimize_reflected_energy``'s late-time split window. AD and FD agreed because
both differentiate the SAME empty window; the docs prescribed the FD check as the
trust ritual but never warned that a passing check cannot detect an empty
observable.

This was R2-STOPPED to a doc-pin (there is no clean call-time discriminator: the
split-window premise depends on the geometry's round-trip time and the run
length, which preflight cannot know). The pin adds (a) the round-trip precondition
to ``minimize_reflected_energy`` + the inverse-design guide, (b) the "FD agreement
validates the machinery, not the physics; check the loss magnitude" caveat to the
autodiff and gradient-behavior guides, and (c) clarifies that ``add_lumped_rlc``
is not a reflection-referenced port (the separate near-miss in the same audit).

This test locks that honesty text so it cannot be silently stripped. It asserts
NO physics and changes NO gate.
"""

from __future__ import annotations

from pathlib import Path

from rfx.optimize_objectives import minimize_reflected_energy

_DOCS = Path(__file__).resolve().parents[1] / "docs/public"


def _norm(text: str) -> str:
    # collapse whitespace so line-wrapping in the source can change freely
    return " ".join(text.split()).lower()


def test_objective_docstring_pins_split_window_precondition():
    doc = _norm(minimize_reflected_energy.__doc__ or "")
    assert "round trip" in doc
    assert "late window" in doc and "empty" in doc
    assert "numerical noise" in doc
    # steers off the false-confidence FD ritual
    assert "finite-difference check does not catch" in doc


def test_inverse_design_guide_pins_split_window_precondition():
    text = _norm((_DOCS / "guide/inverse-design.md").read_text())
    assert "split window must contain the reflection" in text
    assert "round trip" in text
    assert "1e-2" in text and "1e-7" in text


def test_autodiff_guide_pins_fd_necessary_not_sufficient():
    text = _norm((_DOCS / "guide/autodiff-adjoint.mdx").read_text())
    assert "necessary, not sufficient" in text
    assert "autodiff machinery" in text
    assert "numerical noise" in text
    assert "1e-2" in text


def test_gradient_behavior_guide_pins_passing_witness_caveat():
    text = _norm((_DOCS / "guide/gradient-behavior.md").read_text())
    assert "witness is also not sufficient" in text
    assert "window is empty" in text
    assert "magnitude" in text


def test_sources_ports_guide_pins_lumped_rlc_not_a_port():
    text = _norm((_DOCS / "api/sources-ports.mdx").read_text())
    assert "reflection-referenced port" in text
    assert "add_lumped_rlc" in text
    assert "self-interaction" in text
    assert "reference impedance" in text
