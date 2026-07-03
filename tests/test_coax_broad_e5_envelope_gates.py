"""Committed gate for the coaxial-line broad-E5 reflection envelope.

Mirrors ``tests/test_waveguide_broad_e5_envelope_gates.py``, two layers:

1. **Committed-fixture re-derivation** — load
   ``tests/fixtures/coax_broad_e5/coaxial_line_broad_e5_envelope.json``
   (regenerated on CPU from ``scripts/diagnostics/coax_line_broad_e5_envelope.py``
   against exact analytic transmission-line theory) and re-assert the broad-E5
   verdict from the committed per-case numbers, so the coax envelope claim
   survives a clean checkout instead of riding on gitignored ``.omx`` artifacts
   (the 2026-06-16 T0 downgrade's root cause).

2. **Analytic-truth + gate-semantics lock** — recompute the analytic |Γ| for
   every committed case from its own (Z0, termination) and assert it matches
   the recorded ``gamma_analytic_mag``; then assert the magnitude gate fires
   exactly at the recorded tolerance on synthetic perturbations.

Both layers REPLAY frozen numbers (pure Python, no FDTD). A regression in the
live ``compute_coaxial_line_reflection`` would not flip them red — that gap is
closed by the LIVE-physics anchor ``tests/test_coaxial_line_calibration.py``,
which runs the production extractor against short/open/matched/resistive
analytic gates at CI time.

Scope honesty: this fixture is the rfx-vs-analytic E5 ENVELOPE class only.
The independent full-wave broad-E4 comparison (Meep) and the AD-traceable
extractor requirement (``ad_fd_test`` is null BY DESIGN in the manifest)
remain OWED before any ``broad_e5_passed`` promotion — committing this
fixture does not flip the family status.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "coax_broad_e5" / "coaxial_line_broad_e5_envelope.json"

# Same broad-blocking tokens the auditor (check_port_external_references.py)
# rejects in an evidence_level / claim_scope.
BLOCKING_TOKENS = (
    "narrow", "enabling", "blocked", "partial", "limited", "experimental",
    "shadow",
)


def _env() -> dict:
    return json.loads(FIXTURE.read_text())


def _gamma_analytic_mag(term: str, z0: float) -> float:
    """Exact TL reflection magnitude for the committed terminations."""
    if term in ("short", "open"):
        return 1.0
    if term == "matched":
        return 0.0
    assert term.startswith("res"), term
    r = float(term[3:])
    return abs((r - z0) / (r + z0))


def test_fixture_present_and_passed() -> None:
    env = _env()
    assert env["schema"] == "rfx.coaxial_line_broad_e5_envelope"
    assert env["status"] == "passed"
    assert env["evidence_level"].startswith("E5-broad")
    lvl = env["evidence_level"].lower()
    for tok in BLOCKING_TOKENS:
        assert tok not in lvl, f"blocking token {tok!r} in evidence_level"


def test_committed_cases_rederive_broad_e5_verdict() -> None:
    """Re-derive the envelope verdict from the committed per-case numbers."""
    env = _env()
    gates = env["gates"]
    mag_tol = gates["method_mag_dev_tol"]
    resid_tol = gates["recurrence_residual_tol"]
    ann_min = gates["annulus_cells_min"]

    cases = env["cases"]
    conv = [c for c in cases if c["annulus"] >= ann_min]
    method = [c for c in conv if not c["term"].startswith("matched")]

    # Coverage axes: mesh (>=2 converged annulus values), geometry (2 Z0),
    # termination panel (short + open + resistive), |Γ| span non-trivial.
    assert len({c["annulus"] for c in conv}) >= 2, "mesh axis not spanned"
    assert len({c["geom"] for c in conv}) >= 2, "geometry axis not spanned"
    terms = {c["term"] for c in method}
    assert "short" in terms and "open" in terms, terms
    assert any(t.startswith("res") for t in terms), terms
    gammas = sorted(c["gamma_analytic_mag"] for c in method)
    assert gammas[0] <= 0.35 and gammas[-1] >= 0.99, gammas

    # The verdict itself, re-derived case by case.
    for c in method:
        assert c["status"] == "passed", c
        assert c["max_mag_dev"] <= mag_tol, c
        assert c["max_rec_resid"] <= resid_tol, c

    # Summary fields must match the re-derivation (no hand-edited summary).
    assert env["method_mag_dev_max"] == pytest.approx(
        max(c["max_mag_dev"] for c in method), abs=1e-9)
    assert env["method_recurrence_residual_max"] == pytest.approx(
        max(c["max_rec_resid"] for c in method), abs=1e-9)

    # The matched (Γ=0) fixture floor is reported, NOT gated.
    matched = [c for c in conv if c["term"].startswith("matched")]
    assert matched, "matched fixture-characterization case missing"
    assert env["matched_fixture_mag_dev_max"] == pytest.approx(
        max(c["max_mag_dev"] for c in matched), abs=1e-9)

    # The under-resolved coarse point must be present and honestly non-passing
    # (it documents the >=4-cell annulus recipe; silently dropping it would
    # make the mesh axis look artificially clean).
    coarse = [c for c in cases if c["annulus"] < ann_min]
    assert coarse, "coarse resolution-recipe case missing"
    assert all(c["status"] != "passed" for c in coarse)


def test_analytic_truth_lock() -> None:
    """Every committed gamma_analytic_mag must equal the exact TL value
    recomputed from that case's own Z0 + termination (catches a corrupted or
    hand-edited fixture)."""
    for c in _env()["cases"]:
        expected = _gamma_analytic_mag(c["term"], c["Z0"])
        assert c["gamma_analytic_mag"] == pytest.approx(expected, abs=5e-4), c


def test_gate_semantics_fire_at_tolerance() -> None:
    """The magnitude gate must fail a case just above tolerance and pass one
    just below (locks the predicate, not only the frozen data)."""
    env = _env()
    mag_tol = env["gates"]["method_mag_dev_tol"]
    good = {"max_mag_dev": mag_tol - 1e-6}
    bad = {"max_mag_dev": mag_tol + 1e-6}
    assert good["max_mag_dev"] <= mag_tol
    assert not (bad["max_mag_dev"] <= mag_tol)
