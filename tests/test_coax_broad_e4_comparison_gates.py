"""Committed gate for the coaxial-line broad-E4 external (Meep) comparison.

Mirrors ``tests/test_coax_broad_e5_envelope_gates.py`` (the analytic envelope)
for the EXTERNAL-solver leg of the coaxial-line reflection lane. The coax family
was T0-downgraded (2026-06-16) because its broad-E4/E5 evidence lived only in
gitignored ``.omx/`` — and the loss was real (the dirs were found empty). The
Meep power-flux reference was regenerated on VESSL (run 369367245835,
2026-07-04, res 3200) and the rfx-vs-Meep comparison committed here so the E4
class survives a clean checkout.

1. **Committed-fixture re-derivation** — load
   ``tests/fixtures/coax_broad_e4/coaxial_line_meep_broad_comparison.json``
   (rfx ``compute_coaxial_line_reflection`` |Γ| vs independent Meep power-flux
   FDTD, short/open full-reflection terminations) and re-assert the broad-E4
   verdict from the committed per-termination numbers.

2. **Real-auditor-predicate lock** — drive the ACTUAL
   ``check_port_external_references._comparison_breadth_ok`` predicate against
   the fixture (must be broad-valid) and against perturbations (fail-closed).

SCOPE / honesty — committing this closes the coax family's Meep-broad-E4
evidence class, but does NOT flip its status: the AD-traceable reflection
extractor requirement (``ad_fd_test`` is null BY DESIGN in the manifest) still
blocks ``broad_e5_passed``. Magnitude only (reference-plane independent for the
lossless line). The matched (|Γ|=0) and resistive loads are covered by the
exact-analytic broad-E5 envelope, not Meep (a 1-cell Meep resistive sheet
merely re-tests rfx's own resistor stamp — R5 trap). The live-physics anchor is
``tests/test_coaxial_line_calibration.py``.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from check_port_external_references import _comparison_breadth_ok  # type: ignore  # noqa: E402

FIXTURE = (
    REPO / "tests" / "fixtures" / "coax_broad_e4"
    / "coaxial_line_meep_broad_comparison.json"
)

# Pinned producer tolerances (build_coaxial_line_meep_broad_comparison.py) — a
# silently-loosened fixture tolerance must go red here.
EXPECTED_MAX_TOL = 0.10
EXPECTED_MEAN_TOL = 0.06
BLOCKING_TOKENS = (
    "narrow", "enabling", "blocked", "partial", "limited", "experimental",
    "shadow", "stub",
)


def _env() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_present_and_passed() -> None:
    env = _env()
    assert env["schema"] == "rfx.coaxial_line_meep_broad_comparison"
    assert env["status"] == "passed"
    assert env["stub"] is False, "must be a REAL Meep run, not the analytic stub"
    assert env["evidence_level"].startswith("E4-broad")
    lvl = env["evidence_level"].lower()
    for tok in BLOCKING_TOKENS:
        assert tok not in lvl, f"blocking token {tok!r} in evidence_level"


def test_gate_tolerances_pinned() -> None:
    env = _env()
    assert env["tolerances"]["max_mag_abs_tol"] == EXPECTED_MAX_TOL
    assert env["tolerances"]["mean_mag_abs_tol"] == EXPECTED_MEAN_TOL


def test_committed_terminations_rederive_broad_e4_verdict() -> None:
    env = _env()
    per = env["per_termination"]
    max_tol = env["tolerances"]["max_mag_abs_tol"]

    # Geometry (termination) axis must span short + open (|Γ|=1 calibration).
    terms = {p["termination"] for p in per}
    assert {"short", "open"} <= terms, terms

    for p in per:
        assert p["rfx_status"] == "passed", p
        assert p["term_status"] == "passed", p
        assert p["max_mag_abs_diff"] <= max_tol, p
        # short/open are full-reflection: rfx |Γ| must land near unity.
        rng = [min(p["rfx_abs_gamma"]), max(p["rfx_abs_gamma"])]
        assert 0.85 <= rng[0] and rng[1] <= 1.15, (p["termination"], rng)

    s = env["summary"]
    assert s["geometry_count"] == len(terms)
    assert s["passed_pair_count"] == len(per)
    assert s["failed_pair_count"] == 0
    assert s["max_mag_abs_diff"] == pytest.approx(
        max(p["max_mag_abs_diff"] for p in per), abs=1e-9)
    assert env["cross_solver_max_mag_abs_diff"] == pytest.approx(
        s["max_mag_abs_diff"], abs=1e-9)


def test_real_auditor_predicate_accepts_and_fails_closed() -> None:
    """Drive the ACTUAL auditor comparison-breadth predicate."""
    env = _env()
    ok, why = _comparison_breadth_ok(env)
    assert ok, f"auditor rejects the committed coax broad-E4 fixture: {why}"

    p = copy.deepcopy(env)
    del p["summary"]
    assert not _comparison_breadth_ok(p)[0], "missing summary must fail-closed"

    p = copy.deepcopy(env)
    p["summary"]["failed_pair_count"] = 1
    assert not _comparison_breadth_ok(p)[0], "a failing pair must fail-closed"

    p = copy.deepcopy(env)
    p["summary"]["geometry_count"] = 1
    assert not _comparison_breadth_ok(p)[0], "single geometry must not be broad"
