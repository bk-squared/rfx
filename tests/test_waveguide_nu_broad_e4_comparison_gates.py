"""Committed gate for the WR-90 NONUNIFORM (graded-dy) flux broad-E4 external
comparison.

Mirrors ``tests/test_waveguide_broad_e5_envelope_gates.py`` (the uniform lane)
for the external-solver leg of the NONUNIFORM waveguide flux lane. The
nonuniform lane already carried a committed broad-E5 *analytic* envelope
(``tests/fixtures/waveguide_nu_broad_e5/``, vs analytic Airy); its one remaining
promotion rung was a broad-E4 EXTERNAL cross-solver check. This locks that
evidence on a clean checkout:

1. **Committed-fixture re-derivation** — load
   ``tests/fixtures/waveguide_nu_broad_e4/waveguide_wr90_nu_flux_broad_e4_comparison.json``
   (rfx NU graded-dy flux vs Palace_r_h2, 5 magnitude pairs over empty /
   PEC-short / slab) and re-assert the broad-E4 verdict from the committed
   per-pair numbers.

2. **Real-auditor-predicate lock** — drive the ACTUAL
   ``check_port_external_references._comparison_breadth_ok`` predicate against
   the fixture (must be broad-valid) and against perturbations (must fail-closed).

SCOPE / honesty — this is the NONUNIFORM lane's external cross-check; the
reference is Palace high-order FEM (the physically-converged reference the
uniform fixture also uses — Meep is non-physical on PEC-short at this
resolution). Magnitude only (cross-solver phase conventions differ 100 deg+).
Both layers replay frozen numbers; the live NU anchor is the np=40
power/reciprocity gate in ``tests/test_waveguide_nu_nontrivial.py``.
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

sys.path.insert(0, str(REPO / "tests"))
from _gate_policy import gate_from_envelope  # type: ignore  # noqa: E402

FIXTURE = (
    REPO / "tests" / "fixtures" / "waveguide_nu_broad_e4"
    / "waveguide_wr90_nu_flux_broad_e4_comparison.json"
)

# What is HARD-PINNED here is the MEASURED envelope, which no policy change may
# move: the absorber fix (#496/#576) took the worst per-pair max from 0.07009 to
# 0.008529 and the worst per-pair mean from 0.0359 to 0.0030, leaving the old
# flat 0.10 / 0.07 12x and 23x loose. Both are PER-PAIR figures — deriving the
# mean tolerance from the summary mean (0.000709) instead produced a gate that
# failed 1 of 5 pairs on its first run.
MEASURED_MAX_ENVELOPE = 0.008529
MEASURED_MEAN_ENVELOPE = 0.0030

# The tolerance itself is DERIVED, not pinned (#576 review F5): restating 0.013 /
# 0.005 as literals here would let this lane silently disagree with the producer
# if the repo-wide multiplier ever moved. Quantum 1000 because the residual is
# milli-scale; every other quantized lane uses 100.
EXPECTED_MAX_TOL = gate_from_envelope(MEASURED_MAX_ENVELOPE, quantum=1000)
EXPECTED_MEAN_TOL = gate_from_envelope(MEASURED_MEAN_ENVELOPE, quantum=1000)
BLOCKING_TOKENS = (
    "narrow", "enabling", "blocked", "partial", "limited", "experimental",
    "shadow",
)


def _env() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_present_and_passed() -> None:
    env = _env()
    assert env["schema"] == "rfx.waveguide_wr90_nu_flux_broad_e4_comparison"
    assert env["status"] == "passed"
    assert env["evidence_level"].startswith("E4-broad")
    lvl = env["evidence_level"].lower()
    for tok in BLOCKING_TOKENS:
        assert tok not in lvl, f"blocking token {tok!r} in evidence_level"
    # It really is the NONUNIFORM lane (graded mesh), not a uniform re-run.
    assert env["mesh"]["kind"] == "nonuniform_dy_profile_ratio"
    assert env["mesh"]["max_min_cell_ratio"] > 1.0, env["mesh"]


def test_gate_tolerances_pinned() -> None:
    """A silently-loosened fixture tolerance must go red here."""
    env = _env()
    assert env["max_mag_abs_tol"] == EXPECTED_MAX_TOL
    assert env["mean_mag_abs_tol"] == EXPECTED_MEAN_TOL


def test_committed_pairs_rederive_broad_e4_verdict() -> None:
    env = _env()
    pairs = env["pairs"]
    max_tol = env["max_mag_abs_tol"]
    mean_tol = env["mean_mag_abs_tol"]

    # Coverage axes: the geometry axis must span empty + pec_short + slab, and
    # both S11 and S21 components must appear.
    geoms = {p["geometry"] for p in pairs}
    assert {"empty", "pec_short", "slab"} <= geoms, geoms
    comps = {p["component"] for p in pairs}
    assert {"S11", "S21"} <= comps, comps

    for p in pairs:
        assert p["status"] == "passed", p
        assert p["max_mag_abs_diff"] <= max_tol, p
        # PASSIVITY, stated rather than implied (#576 review F11). The bounds
        # above constrain agreement with the reference; they do not by
        # themselves say a lossless structure stayed passive. The E4 lane's
        # over-unity |S11| is what started that review, so assert the intent
        # directly: no rfx magnitude may exceed unity by more than the
        # comparison envelope it is being judged against.
        _hi = float(p["rfx_mag_range"][1])
        assert _hi <= 1.0 + max_tol, (
            f"{p['geometry']} {p['component']} rfx |S| reaches {_hi:.6f}, above "
            f"1 + {max_tol} — a lossless structure is not that non-passive; see "
            f"#576 (absorber under-provisioning) before blessing the fixture")
        # The gate is BOTH bounds at once (both enforced in the producer's
        # per-pair status), so quoting the pair is real rather than shorthand.
        # They are 0.013 / 0.005 since #576 derived them from the measured
        # envelope; the 0.10 / 0.07 this comment used to name were the
        # pre-derivation literals.
        assert p["mean_mag_abs_diff"] <= mean_tol, p

    s = env["summary"]
    assert s["geometry_count"] == len(geoms)
    assert s["passed_pair_count"] == len(pairs)
    assert s["failed_pair_count"] == 0
    assert s["max_mag_abs_diff"] == pytest.approx(
        max(p["max_mag_abs_diff"] for p in pairs), abs=1e-9)

    # PEC-short is the sharp |S11|->1 discriminator: rfx NU must land physical
    # (near unity), which is the whole point of the external cross-check.
    ps = next(p for p in pairs if p["geometry"] == "pec_short")
    lo, hi = ps["rfx_mag_range"]
    assert 0.9 <= lo and hi <= 1.1, ps["rfx_mag_range"]


def test_real_auditor_predicate_accepts_and_fails_closed() -> None:
    """Drive the ACTUAL auditor comparison-breadth predicate: the committed
    fixture must be broad-valid, and each perturbation must fail-closed."""
    env = _env()
    ok, why = _comparison_breadth_ok(env)
    assert ok, f"auditor rejects the committed NU broad-E4 fixture: {why}"

    p = copy.deepcopy(env)
    del p["summary"]
    assert not _comparison_breadth_ok(p)[0], "missing summary must fail-closed"

    p = copy.deepcopy(env)
    p["summary"]["failed_pair_count"] = 1
    p["summary"]["passed_pair_count"] = p["summary"]["pair_count"] - 1
    assert not _comparison_breadth_ok(p)[0], "a failing pair must fail-closed"

    p = copy.deepcopy(env)
    p["summary"]["geometries"] = ["empty"]
    p["summary"]["geometry_count"] = 1
    assert not _comparison_breadth_ok(p)[0], "single geometry must not be broad"
