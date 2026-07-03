"""Committed gate for the coaxial-line broad-E5 reflection envelope.

Mirrors ``tests/test_waveguide_broad_e5_envelope_gates.py``, two layers:

1. **Committed-fixture re-derivation** — load
   ``tests/fixtures/coax_broad_e5/coaxial_line_broad_e5_envelope.json``
   (regenerated on CPU from ``scripts/diagnostics/coax_line_broad_e5_envelope.py``
   against exact analytic transmission-line theory) and re-assert the broad-E5
   verdict from the committed per-case numbers, so the coax envelope claim
   survives a clean checkout instead of riding on gitignored ``.omx`` artifacts
   (the 2026-06-16 T0 downgrade's root cause).

2. **Analytic-truth lock** — recompute the analytic |Γ| for every committed
   case from its own (Z0, termination) and assert it matches the recorded
   ``gamma_analytic_mag`` (catches a corrupted or hand-edited fixture).

3. **Real-auditor-predicate lock** — drive the ACTUAL
   ``check_port_external_references._envelope_breadth_ok`` predicate against
   the committed fixture (must be broad-valid) and against perturbed copies
   (each must fail-closed), so the fixture stays conformant with the auditor's
   machine-readable breadth schema, not just with this test's own reading.

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

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from check_port_external_references import _envelope_breadth_ok  # type: ignore  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "coax_broad_e5" / "coaxial_line_broad_e5_envelope.json"

# Pinned gate constants: the re-derivation below reads tolerances from the
# fixture, so WITHOUT this pin a silently-loosened fixture gate would keep
# everything green (reviewer finding, PR #256). These are the committed
# producer constants (coax_line_broad_e5_envelope.py MAG_TOL/RES_RESID_TOL).
EXPECTED_GATES = {
    "method_mag_dev_tol": 0.05,
    "recurrence_residual_tol": 0.03,
    "annulus_cells_min": 3.5,
}

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


def test_gate_constants_pinned() -> None:
    """The fixture's gate tolerances must equal the committed producer
    constants — a silently-loosened fixture gate must go red here."""
    assert _env()["gates"] == EXPECTED_GATES
    assert _env()["max_mag_abs_tol"] == EXPECTED_GATES["method_mag_dev_tol"]


def test_committed_cases_rederive_broad_e5_verdict() -> None:
    """Re-derive the envelope verdict from the committed per-case numbers."""
    env = _env()
    gates = env["gates"]
    assert gates == EXPECTED_GATES  # belt-and-braces with the pin test
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


def test_envelope_summary_consistent_with_cases() -> None:
    """The machine-readable envelope_summary (what the auditor gates on) must
    re-derive from the gated method cases — no hand-authored summary."""
    env = _env()
    s = env["envelope_summary"]
    gates = env["gates"]
    conv = [c for c in env["cases"] if c["annulus"] >= gates["annulus_cells_min"]]
    method = [c for c in conv if not c["term"].startswith("matched")]
    assert s["case_count"] == len(method)
    assert s["passed_case_count"] == sum(1 for c in method if c["status"] == "passed")
    assert sorted(s["dx_values_m"]) == sorted({c["dx_m"] for c in method})
    assert sorted(s["geometries"]) == sorted({c["geom"] for c in method})
    assert s["max_mag_abs_diff_across_cases"] == pytest.approx(
        max(c["max_mag_dev"] for c in method), abs=1e-9)
    lo, hi = s["freq_range_hz"]
    assert lo == pytest.approx(4.0e9) and hi == pytest.approx(12.0e9)


def test_real_auditor_predicate_accepts_fixture_and_fails_closed() -> None:
    """Drive the ACTUAL auditor breadth predicate: the committed fixture must
    be broad-valid, and each perturbation must fail-closed (locks the real
    predicate, replacing the earlier tautological version — PR #256 review)."""
    env = _env()
    ok, why = _envelope_breadth_ok(env)
    assert ok, f"auditor rejects the committed fixture: {why}"

    # Each perturbation must flip the verdict.
    p = copy.deepcopy(env)
    del p["envelope_summary"]
    assert not _envelope_breadth_ok(p)[0], "missing summary must fail-closed"

    p = copy.deepcopy(env)
    p["envelope_summary"]["passed_case_count"] -= 1
    assert not _envelope_breadth_ok(p)[0], "a failing case must fail-closed"

    p = copy.deepcopy(env)
    p["envelope_summary"]["dx_values_m"] = p["envelope_summary"]["dx_values_m"][:1]
    assert not _envelope_breadth_ok(p)[0], "single-mesh must not count as broad"

    p = copy.deepcopy(env)
    p["envelope_summary"]["geometries"] = p["envelope_summary"]["geometries"][:1]
    assert not _envelope_breadth_ok(p)[0], "single-geometry must not count as broad"

    p = copy.deepcopy(env)
    p["envelope_summary"]["freq_range_hz"] = [10.0e9, 12.0e9]
    assert not _envelope_breadth_ok(p)[0], "narrow freq span must not count as broad"

    p = copy.deepcopy(env)
    p["envelope_summary"]["max_mag_abs_diff_across_cases"] = p["max_mag_abs_tol"] + 1e-6
    assert not _envelope_breadth_ok(p)[0], "mag diff above tol must fail-closed"
