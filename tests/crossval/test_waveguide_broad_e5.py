"""Rectangular-waveguide broad-E5 case: magnitude + phase envelopes, tolerance
governance, broad-E4 external leg, and the LIVE-physics anchor.

One file for the case (tier 3b of the 2026-09 test-corpus reorganisation, see
``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``). Sections,
each formerly its own file:

1. **Magnitude envelope gates** — was ``test_waveguide_broad_e5_envelope_gates.py``.
   (a) Committed-fixture re-derivation: load every band envelope JSON under
   ``tests/fixtures/waveguide_broad_e5/`` (regenerated on gpu-rtx4090, VESSL
   369367242914, all 5 WR bands 20/20 cases pass vs analytic Airy) and
   re-assert the broad-E5 verdict from the committed per-case numbers, so the
   "broad_e5_passed" claim survives a clean checkout instead of riding on
   gitignored ``.omx`` artifacts. (b) Gate-semantics lock: drive the *real*
   ``airy_slab`` reference formula and the ``MAX_TOL`` tolerance from the
   producer with synthetic ideal / perturbed S-parameters. (c) The broad-E4
   external-solver comparison (the external leg of the port-1 close).
2. **Magnitude tolerance envelope (T2.4)** — was
   ``test_waveguide_broad_e5_tolerance_envelope.py``. The framework audit
   flagged ``MAX_TOL=0.05`` / ``noise_floor_baseline=0.0021`` as round
   constants with no derivation. The plan's ``tol = C·(k·dx)² + noise_floor``
   dispersion model was FALSIFIED on the committed sweep (C<0, R²=0.19 — the
   error is dielectric-contrast / slab-interface dominated, not grid
   dispersion); this section LOCKS that finding, requires ``MAX_TOL`` to be a
   bounded measured envelope, and requires the noise floor to be a committed
   empty-guide measurement.
3. **Phase envelope gates (issue #490 Lane 1)** — was
   ``test_waveguide_broad_e5_phase_gates.py``: committed-fixture
   re-derivation of the phase verdict, gate-semantics lock with the corrected
   reference-plane phase transform, the planted-defect falsifier (wrong
   pre-session S21 reference-plane formula must red the gate) and the
   domain-size invariance witness, both replayed from the committed
   ``phase_falsifier_and_domain_invariance.json``.
4. **Phase tolerance envelope** — was
   ``test_waveguide_broad_e5_phase_tolerance_envelope.py``: ``MAX_PHASE_TOL_DEG``
   must envelope every committed case and not be slack (bounded margin), plus
   the 60-degree convention-masking tripwire.
5. **LIVE-physics anchor (T2.3)** — was ``test_waveguide_broad_e5_live_anchor.py``.
   Sections 1-4 are FROZEN replays; a real regression in the production
   ``compute_waveguide_s_matrix`` would NOT flip them red (the rfx_npz is
   gitignored — framework audit, finding "frozen replay"). This section RUNS
   ``compute_waveguide_s_matrix`` at CI time on NON-TRIVIAL geometries
   (PEC-short ``|S11|=1``, empty matched guide ``|S21|≈1``) and checks the
   live output against analytic physics, so an extractor regression turns it
   red. Per the T2.3 design + review: PEC-short uses ``normalize=False`` (the
   two-run ``normalize=True`` has standing-wave node artifacts on strong
   reflectors; bare ``normalize=True`` is the ±10–20 % non-convergent mode);
   the empty guide uses ``normalize='flux'`` (documented-convergent) as the
   S21 witness + passivity check; boundary is CPML on both (``run_until_decay``
   / flux convergence assume an absorbing boundary, #169); the PEC-short test
   asserts the gate margin is tight so the gate is not slack; R5: full
   per-frequency dump on every run.

Every assertion, tolerance, fixture value and parametrisation of the original
files is kept verbatim; only module-level helper names were disambiguated
(``_mag_fixture_files`` / ``_phase_fixture_files``, ``_live_build_sim``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.geometry.csg import Box

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_band_broad_e5_envelope import (  # type: ignore  # noqa: E402
    MAX_TOL,
    _committed_noise_floor,
    airy_slab,
)
from build_waveguide_band_broad_e5_phase_envelope import (  # type: ignore  # noqa: E402
    MAX_PHASE_TOL_DEG, PHASE_MAG_FLOOR, _wrapped_phase_diff_deg,
)

from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER, gate_from_envelope  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "waveguide_broad_e5"
EXPECTED_BANDS = {
    "wr28_kaband",
    "wr62_kuband",
    "wr15_vband",
    "wr340_sband",
    "wr10_wband",
}

# Margin ceiling: MAX_TOL / MAX_PHASE_TOL_DEG may exceed the worst measured
# diff by at most this factor. This is a GOVERNANCE choice (how much slack a
# reviewer tolerates), NOT a physical bound. Magnitude: 0.05 / 0.0414 = 1.21,
# so 1.5 leaves headroom but rejects a slack round-up (e.g. bumping MAX_TOL to
# 0.08 -> 1.93x would breach it). Phase: 15.0 / 11.99 = 1.25, well inside 1.5.
# Issue #528: this is the SAME repo-wide multiplier the quantized-gate cases
# derive from (tests/_gate_policy.py) -- imported rather than restated so a
# relaxation here is visible alongside theirs, not a fresh local literal.
MARGIN_CEIL = ENVELOPE_GATE_MULTIPLIER


# ===========================================================================
# 1. Magnitude envelope gates (formerly test_waveguide_broad_e5_envelope_gates.py)
# ===========================================================================

BROAD_E4 = FIXTURES / "wr90_rectangular_broad_e4_comparison.json"
# Same broad-blocking tokens the auditor (check_port_external_references.py)
# rejects in an evidence_level / claim_scope.
BLOCKING_TOKENS = (
    "narrow", "enabling", "blocked", "partial", "limited", "experimental",
    "shadow", "only",
)


def _mag_fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("waveguide_*_broad_e5_envelope.json"))


def test_all_five_bands_present() -> None:
    """The committed fixture set must cover every promoted WR band."""
    tokens = {
        p.name.replace("waveguide_", "").replace("_broad_e5_envelope.json", "")
        for p in _mag_fixture_files()
    }
    assert tokens == EXPECTED_BANDS, f"committed bands {tokens} != {EXPECTED_BANDS}"


@pytest.mark.parametrize("path", _mag_fixture_files(), ids=lambda p: p.stem)
def test_committed_band_envelope_passes_broad_e5(path: Path) -> None:
    """Re-derive the broad-E5 verdict from the committed per-case numbers."""
    env = json.loads(path.read_text())
    summ = env["envelope_summary"]

    assert env["status"] == "passed", f"{path.name} status={env['status']}"
    assert env["evidence_level"].startswith("E5-broad"), env["evidence_level"]
    # Envelope spans both the mesh and geometry axes (>=2 dx, >=2 eps_r).
    assert len(summ["dx_values_m"]) >= 2, summ["dx_values_m"]
    assert len(summ["eps_r_values"]) >= 2, summ["eps_r_values"]

    # Every case must independently clear the magnitude tolerance, and the
    # JSON-reported aggregate must match a fresh max over the per-case diffs
    # (catches a doctored summary that disagrees with its own cases).
    cases = env["cases"]
    assert len(cases) >= 4, cases
    per_case_max = []
    for c in cases:
        assert c["status"] == "passed", c
        assert c["max_mag_abs_diff"] <= MAX_TOL, c
        per_case_max.append(c["max_mag_abs_diff"])
    assert summ["passed_case_count"] == summ["case_count"] == len(cases)
    assert summ["max_mag_abs_diff_across_cases"] == pytest.approx(
        max(per_case_max), abs=1e-9
    )
    assert summ["max_mag_abs_diff_across_cases"] <= MAX_TOL


def _synthetic_airy(eps_r: float = 2.0, L: float = 3.0e-3, fc_v: float = 9.488e9):
    f = np.linspace(12.4e9, 18.0e9, 11)
    s11, s21 = airy_slab(f, eps_r, L, fc_v)
    return f, s11, s21


def _case_max_diff(rfx_s11, rfx_s21, ref_s11, ref_s21) -> float:
    """Replicate the producer's magnitude-diff gate metric."""
    s11d = np.abs(np.abs(rfx_s11) - np.abs(ref_s11))
    s21d = np.abs(np.abs(rfx_s21) - np.abs(ref_s21))
    return max(float(s11d.max()), float(s21d.max()))


def test_gate_passes_when_rfx_equals_airy() -> None:
    """An ideal slab (rfx == analytic Airy) clears the gate with ~zero diff."""
    _, s11, s21 = _synthetic_airy()
    case_max = _case_max_diff(s11, s21, s11, s21)
    assert case_max <= MAX_TOL
    assert case_max < 1e-12


def test_gate_fails_on_magnitude_perturbation() -> None:
    """A +0.1 |S11| offset (well above MAX_TOL=0.05) must fail the gate."""
    _, s11, s21 = _synthetic_airy()
    perturbed = s11 * 0.0 + (np.abs(s11) + 0.1)  # magnitude bumped by 0.1
    case_max = _case_max_diff(perturbed, s21, s11, s21)
    assert case_max > MAX_TOL
    assert case_max == pytest.approx(0.1, abs=1e-9)


def test_lossless_slab_airy_is_unitary() -> None:
    """Independent witness: the analytic reference itself conserves power."""
    _, s11, s21 = _synthetic_airy()
    unit = np.abs(s11) ** 2 + np.abs(s21) ** 2
    assert np.allclose(unit, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Broad-E4 external-solver comparison (the external leg of the port-1 close)
# ---------------------------------------------------------------------------


def test_broad_e4_comparison_committed_passes() -> None:
    """rfx-vs-external-FDTD comparison across the 3 WR-90 geometries passes."""
    d = json.loads(BROAD_E4.read_text())
    assert d["status"] == "passed", d
    summ = d["summary"]
    assert summ["geometry_count"] == 3, summ
    assert summ["passed_pair_count"] == summ["pair_count"], summ
    tol = d["max_mag_abs_tol"]
    per_pair_max = [p["max_mag_abs_diff"] for p in d["pairs"]]
    for p in d["pairs"]:
        assert p["status"] == "passed", p
        assert p["max_mag_abs_diff"] <= tol, p
    assert summ["max_mag_abs_diff"] == pytest.approx(max(per_pair_max), abs=1e-9)
    # PEC-short magnitude must be essentially exact reflection (the R5 finding:
    # rfx nails |S11|=1, an invalid Meep res-3/4 ref did not -> Palace used).
    pec = [p for p in d["pairs"] if p["geometry"] == "pec_short"]
    assert pec and all(p["max_mag_abs_diff"] < 0.02 for p in pec), pec


def test_broad_e4_comparison_qualifies_for_auditor() -> None:
    """Mirror check_port_external_references.py's broad-E4 acceptance."""
    d = json.loads(BROAD_E4.read_text())
    level = d["evidence_level"].lower()
    scope = d["claim_scope"].lower()
    assert d["status"] == "passed"
    assert d["evidence_level"].startswith("E4")
    assert "broad" in scope
    assert not any(t in level or t in scope for t in BLOCKING_TOKENS), (level, scope)


# ===========================================================================
# 2. Magnitude tolerance envelope, T2.4 (formerly
#    test_waveguide_broad_e5_tolerance_envelope.py)
# ===========================================================================

NOISE_FLOOR_FIXTURE = FIXTURES / "noise_floor_measurement.json"


def _all_case_diffs() -> list[float]:
    diffs: list[float] = []
    for f in sorted(FIXTURES.glob("waveguide_*_broad_e5_envelope.json")):
        for c in json.loads(f.read_text())["cases"]:
            diffs.append(float(c["max_mag_abs_diff"]))
    return diffs


def test_max_tol_is_a_bounded_measured_envelope():
    """MAX_TOL must envelope every committed case AND not be slack."""
    diffs = _all_case_diffs()
    assert diffs, "no committed broad-E5 cases found"
    worst = max(diffs)
    assert worst <= MAX_TOL, (
        f"MAX_TOL={MAX_TOL} is below the worst committed case diff {worst:.4f} — "
        f"it would fail a validated case (a real magnitude regression)."
    )
    assert MAX_TOL <= worst * MARGIN_CEIL, (
        f"MAX_TOL={MAX_TOL} exceeds worst case diff {worst:.4f} × {MARGIN_CEIL} = "
        f"{worst * MARGIN_CEIL:.4f} — the tolerance is slack; a regression could "
        f"hide under the margin. Re-justify or tighten."
    )


def test_noise_floor_is_committed_and_verifiable():
    """The noise floor is a committed empty-guide measurement, not a bare constant."""
    assert NOISE_FLOOR_FIXTURE.exists(), (
        "noise_floor_measurement.json is missing — run "
        "scripts/diagnostics/measure_waveguide_noise_floor.py and commit it."
    )
    data = json.loads(NOISE_FLOOR_FIXTURE.read_text())
    floor = float(data["noise_floor"])
    # On a matched empty guide |S11|=0, |S21|=1 analytically; the residual is the
    # irreducible extractor floor and must be tiny (well under the gate tol).
    assert 0.0 < floor < 0.01, f"empty-guide noise floor {floor:.5f} out of [0, 0.01)"
    assert floor < MAX_TOL, f"noise floor {floor:.5f} >= MAX_TOL {MAX_TOL}"
    # The producer must actually consume the committed measurement.
    assert abs(_committed_noise_floor() - floor) < 1e-12, (
        "the producer's _committed_noise_floor() does not match the committed "
        "measurement — the bare constant was not replaced."
    )
    # The floor is the |S21| transmission residual. |S11| is STRUCTURALLY 0 on a
    # matched guide (the flux extractor clamps |S_ii| to 0 when reflected power
    # <= 0), so it is NOT asserted as a witness — that would be vacuous. We
    # instead require the floor to actually come from the S21 leg.
    assert max(c["max_s21_residual"] for c in data["cases"]) == floor or floor > 0


def test_dispersion_tolerance_model_stays_falsified():
    """LOCK the T2.4 finding: tol ~ C·(k·dx)² + floor does NOT fit the sweep.

    If a future change makes this fit well (positive C, decent R²), the
    dielectric-contrast-dominated finding has changed — update the redesign
    (docs/research_notes/20260617_t2.4_dispersion_tolerance_falsified.md) instead
    of letting a (k·dx)² tolerance silently slip in.
    """
    rows = []
    for f in sorted(FIXTURES.glob("waveguide_*_broad_e5_envelope.json")):
        for c in json.loads(f.read_text())["cases"]:
            cpl = float(c["cells_per_lambda_max_hz"])
            kdx2 = (2.0 * np.pi / cpl) ** 2
            rows.append((kdx2, float(c["max_mag_abs_diff"])))
    x = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    (C, _floor), *_ = np.linalg.lstsq(
        np.vstack([x, np.ones_like(x)]).T, y, rcond=None
    )
    pred = C * x + _floor
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    # Log the fit so a drift toward a physical slope is visible even when the
    # assertion still passes (e.g. C>0 with a mediocre R²).
    print(f"\n[T2.4 dispersion-lock] (k·dx)² fit: C={C:.4f}  R²={r2:.4f} "
          f"(falsified baseline: C=-1.38, R²=0.19)")
    # The model is non-physical here: slope is negative AND the fit is poor.
    assert C < 0 or r2 < 0.5, (
        f"(k·dx)² dispersion model now fits (C={C:.3f}, R²={r2:.3f}) — the "
        f"contrast-dominated finding changed; revisit the T2.4 redesign."
    )


# ===========================================================================
# 3. Phase envelope gates, issue #490 Lane 1 (formerly
#    test_waveguide_broad_e5_phase_gates.py)
# ===========================================================================
#
# The phase envelope JSONs were produced by
# scripts/diagnostics/build_waveguide_band_broad_e5_phase_envelope.py from the
# npz/manifest pair that produced the committed magnitude envelope (restored
# locally for that analysis, since .omx is gitignored and not present in the
# tree; no new FDTD run for the 5-band/20-case sweep, see that script's
# docstring for the full convention note and provenance). The falsifier /
# invariance numbers were produced once by
# scripts/diagnostics/waveguide_phase_falsifier_and_domain_invariance.py and
# are replayed here from the committed JSON, same pattern as the magnitude
# lane's frozen-replay tests.

FALSIFIER_JSON = FIXTURES / "phase_falsifier_and_domain_invariance.json"


def _phase_fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("waveguide_*_broad_e5_phase_envelope.json"))


def test_all_five_bands_present_phase() -> None:
    tokens = {
        p.name.replace("waveguide_", "").replace("_broad_e5_phase_envelope.json", "")
        for p in _phase_fixture_files()
    }
    assert tokens == EXPECTED_BANDS, f"committed phase bands {tokens} != {EXPECTED_BANDS}"


@pytest.mark.parametrize("path", _phase_fixture_files(), ids=lambda p: p.stem)
def test_committed_band_phase_envelope_passes_broad_e5(path: Path) -> None:
    """Re-derive the broad-E5 PHASE verdict from the committed per-case numbers."""
    env = json.loads(path.read_text())
    summ = env["envelope_summary"]

    assert env["status"] == "passed", f"{path.name} status={env['status']}"
    assert env["evidence_level"].startswith("E5-broad")
    assert "phase" in env["evidence_level"]

    cases = env["cases"]
    assert len(cases) >= 4, cases
    per_case_max = []
    for c in cases:
        assert c["status"] == "passed", c
        assert c["max_phase_diff_deg"] <= MAX_PHASE_TOL_DEG, c
        per_case_max.append(c["max_phase_diff_deg"])
    assert summ["passed_case_count"] == summ["case_count"] == len(cases)
    assert summ["max_phase_diff_deg_across_cases"] == pytest.approx(
        max(per_case_max), abs=1e-9
    )
    assert summ["max_phase_diff_deg_across_cases"] <= MAX_PHASE_TOL_DEG
    # Honesty (LOW-b, adversarial review of PR #536: the prior ">= 0" version
    # of this assertion was vacuous -- masked_bin_count is a non-negative
    # count by construction and can never fail it). The 5 committed bands
    # were deliberately designed (see run_waveguide_band_broad_e5_flux_sweep.py
    # comments) to avoid Fabry-Perot nulls, and measurement confirms ZERO
    # masked bins across all 20 cases -- pin that fact so a future change
    # that silently starts masking real bins is caught, not muted.
    assert summ["total_masked_bins"] == 0, (
        f"{path.name}: expected 0 masked bins (bands are designed to avoid "
        f"Fabry-Perot nulls) but got {summ['total_masked_bins']} -- masking "
        f"is now hiding real bins; verify PHASE_MAG_FLOOR is not silently "
        f"swallowing a regression before loosening this assertion"
    )


def _synthetic_case(eps_r: float = 2.0, L: float = 3.0e-3, fc_v: float = 9.488e9,
                     d_left: float = 0.0195, d_right: float = 0.0195):
    f = np.linspace(12.4e9, 18.0e9, 11)
    s11_e, s21_e = airy_slab(f, eps_r, L, fc_v)
    beta_v = (2 * np.pi * f / 299_792_458.0) * np.sqrt(1.0 - (fc_v / f) ** 2)
    s11_ref = s11_e * np.exp(-2j * beta_v * d_left)
    s21_ref = s21_e * np.exp(-1j * beta_v * (d_left + d_right))
    return f, s11_ref, s21_ref


def test_phase_gate_passes_when_rfx_equals_airy() -> None:
    """An ideal case (rfx == corrected Airy reference) clears the gate."""
    _, s11_ref, s21_ref = _synthetic_case()
    d11 = _wrapped_phase_diff_deg(s11_ref, s11_ref)
    d21 = _wrapped_phase_diff_deg(s21_ref, s21_ref)
    case_max = max(float(d11.max()), float(d21.max()))
    assert case_max <= MAX_PHASE_TOL_DEG
    assert case_max < 1e-9


def test_phase_gate_fails_on_20_degree_perturbation() -> None:
    """A +20 degree S21 phase offset (above MAX_PHASE_TOL_DEG=15) must fail."""
    _, s11_ref, s21_ref = _synthetic_case()
    perturbed_s21 = s21_ref * np.exp(1j * np.radians(20.0))
    d21 = _wrapped_phase_diff_deg(perturbed_s21, s21_ref)
    assert d21.max() > MAX_PHASE_TOL_DEG
    assert d21.max() == pytest.approx(20.0, abs=1e-6)


def test_phase_reference_is_unitary_consistent_with_magnitude() -> None:
    """Independent witness: phase-corrected Airy reference still conserves power."""
    _, s11_ref, s21_ref = _synthetic_case()
    unit = np.abs(s11_ref) ** 2 + np.abs(s21_ref) ** 2
    assert np.allclose(unit, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Falsifier + domain-size invariance (workspace gate mandate)
# ---------------------------------------------------------------------------


def test_falsifier_json_exists_and_reds_the_gate() -> None:
    """The planted-defect falsifier (wrong pre-session S21 ref formula) must red.

    This is a REAL bug this session found while deriving the phase
    convention (the placeholder ``exp(+1j*beta_v*slab_length_m)`` term that
    shipped, unexercised, in the magnitude-only envelope builder), not a
    synthetic perturbation -- see
    ``scripts/diagnostics/waveguide_phase_falsifier_and_domain_invariance.py``.
    """
    assert FALSIFIER_JSON.exists(), (
        "phase_falsifier_and_domain_invariance.json is missing -- run "
        "scripts/diagnostics/waveguide_phase_falsifier_and_domain_invariance.py "
        "and commit it"
    )
    data = json.loads(FALSIFIER_JSON.read_text())
    fals = data["falsifier"]
    assert fals["n_cases"] == 20, fals["n_cases"]
    assert fals["gate_reds"] is True, (
        f"falsifier worst-case diff {fals['worst_case_phase_diff_deg']:.2f} deg "
        f"did NOT exceed the gate tolerance {fals['gate_tol_deg']} deg -- the "
        f"planted defect is not being caught"
    )
    # Non-vacuous: the wrong formula must be WILDLY off (near the +-180 deg
    # convention-mismatch signature this whole lane exists to catch), not
    # just barely over the tolerance line.
    assert fals["worst_case_phase_diff_deg"] > 100.0, (
        f"falsifier residual {fals['worst_case_phase_diff_deg']:.2f} deg is "
        f"not the expected ~170-180 deg convention-mismatch signature"
    )


def test_domain_invariance_witness_does_not_flip_verdict() -> None:
    """Growing the WR-340 domain by +100mm must not flip the phase-gate verdict.

    Fresh FDTD run (the only new simulation Lane 1 performs) with the ports/
    reference planes pushed out symmetrically, CPML layers/dx/slab held
    fixed -- a benign domain growth. A spurious CPML-standing-wave phase
    artifact tied to the specific domain size would move under this growth;
    the physical Airy-vs-rfx residual should not.
    """
    assert FALSIFIER_JSON.exists()
    data = json.loads(FALSIFIER_JSON.read_text())
    inv = data["domain_invariance"]
    print(f"\n[domain invariance] baseline={inv['baseline_max_phase_diff_deg']:.3f} deg "
          f"({inv['baseline_verdict']})  grown={inv['grown_max_phase_diff_deg']:.3f} deg "
          f"({inv['grown_verdict']})")
    assert inv["baseline_verdict"] == "passed", inv
    assert inv["grown_verdict"] == "passed", inv
    assert not inv["verdict_flipped"], (
        f"domain growth {inv['baseline_domain_m']}m -> {inv['grown_domain_m']}m "
        f"flipped the phase-gate verdict -- the residual is domain-size-"
        f"dependent, not a stable physical/discretization effect"
    )
    # The grown-domain residual should stay in the same ballpark as baseline
    # (not exactly equal -- more CPML clearance can shift the sub-degree
    # residual slightly -- but a real invariance failure would move it by
    # many degrees, not a fraction of one).
    assert abs(inv["grown_max_phase_diff_deg"] - inv["baseline_max_phase_diff_deg"]) < 5.0, (
        "grown-domain residual moved by >5 deg -- investigate before trusting "
        "either number"
    )


# ===========================================================================
# 4. Phase tolerance envelope (formerly
#    test_waveguide_broad_e5_phase_tolerance_envelope.py)
# ===========================================================================


def _all_case_phase_diffs() -> list[float]:
    diffs: list[float] = []
    for f in sorted(FIXTURES.glob("waveguide_*_broad_e5_phase_envelope.json")):
        for c in json.loads(f.read_text())["cases"]:
            diffs.append(float(c["max_phase_diff_deg"]))
    return diffs


def test_max_phase_tol_is_a_bounded_measured_envelope():
    diffs = _all_case_phase_diffs()
    assert diffs, "no committed broad-E5 phase cases found"
    worst = max(diffs)
    assert worst <= MAX_PHASE_TOL_DEG, (
        f"MAX_PHASE_TOL_DEG={MAX_PHASE_TOL_DEG} is below the worst committed "
        f"case phase diff {worst:.3f} deg -- it would fail a validated case."
    )
    assert MAX_PHASE_TOL_DEG <= worst * MARGIN_CEIL, (
        f"MAX_PHASE_TOL_DEG={MAX_PHASE_TOL_DEG} exceeds worst case diff "
        f"{worst:.3f} deg x {MARGIN_CEIL} = {worst * MARGIN_CEIL:.3f} deg -- "
        f"the tolerance is slack; re-justify or tighten."
    )


def test_phase_residual_is_far_from_the_old_convention_masking_scale():
    """Tripwire: the worst-case phase residual must sit far below the OLD
    cv11 gate's 60-degree window, which existed specifically to mask a
    convention bug (not a physical residual) rather than measure one.

    M5 (adversarial review of PR #536): an earlier version of this docstring
    additionally claimed phase and magnitude residuals "share one mechanism"
    (interface discretization/staircasing). That claim is UNSUPPORTED and is
    retracted here rather than repeated: measured across the 20 committed
    cases, magnitude diff and phase diff are essentially uncorrelated
    (Pearson r = -0.010) and their worst-case orderings invert (the worst
    magnitude case is WR-15 dx=32um eps_r=4; the worst phase case is WR-28
    dx=100um eps_r=2 -- different cases entirely). The 60-degree tripwire
    below is kept on its own merits (it is far above the measured worst case
    and would catch a reintroduced convention bug), not because of a shared
    mechanism with magnitude.
    """
    diffs = _all_case_phase_diffs()
    worst = max(diffs)
    # 60 deg would suggest a genuinely different (larger) error mechanism is
    # active for phase than for magnitude (cf. the OLD unresolved cv11 gate,
    # which needed 60 deg precisely because it was masking a convention bug,
    # not a physical residual). Our worst case (11.99 deg) sits far below
    # that, consistent with "phase and magnitude residuals share one
    # mechanism", not "phase has its own unexplained larger error".
    assert worst < 60.0, (
        f"worst-case phase diff {worst:.2f} deg approaches the old cv11 "
        f"convention-masking scale (60 deg) -- re-verify the reference-plane "
        f"convention rather than assume it is real physics"
    )


# ===========================================================================
# 5. LIVE-physics anchor, T2.3 (formerly test_waveguide_broad_e5_live_anchor.py)
# ===========================================================================

# Canonical 40 mm x 20 mm guide, TE10 cutoff 3.75 GHz (matches the validation
# battery so the live numbers are directly comparable). Single-mode band.
DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
BAND_HZ = (5.0e9, 7.0e9)
N_FREQS = 6


def _live_build_sim(freqs_hz, *, pec_short_x=None):
    """Two-port WR-style guide; optional full-cross-section PEC short.

    Compact local builder (mirrors the validation battery's ``_build_sim``) so
    this live anchor is self-contained.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=max(float(freqs[-1]), f0),
        domain=DOMAIN,
        boundary="cpml",
        cpml_layers=10,
    )
    if pec_short_x is not None:
        thickness = 0.002
        sim.add(
            Box((pec_short_x, 0.0, 0.0),
                (pec_short_x + thickness, DOMAIN[1], DOMAIN[2])),
            material="pec",
        )
    port_freqs = jnp.asarray(freqs)
    for x, direction, name in ((PORT_LEFT_X, "+x", "left"),
                               (PORT_RIGHT_X, "-x", "right")):
        sim.add_waveguide_port(
            x, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=port_freqs, f0=f0, bandwidth=bandwidth,
            waveform="modulated_gaussian", n_modes=1, name=name,
        )
    return sim


def _s_matrix(sim, *, normalize, num_periods=40):
    result = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=normalize)
    s = np.asarray(result.s_params)
    idx = {name: i for i, name in enumerate(result.port_names)}
    return s, np.asarray(result.freqs), idx


def _assert_cpml(sim):
    # Cheap constructor guard (echoes the boundary kwarg). NOTE (issue #395):
    # the empty-guide |S11|≈0 in test_live_empty_guide_s21_anchor is NOT an
    # absorbing-boundary witness — on the flux path device==reference makes it
    # ~0 by construction regardless of CPML quality. The real waveguide-lane
    # PML-reflection gate is the single-run test_matched_load_s11_empty_waveguide
    # in the validation battery.
    assert sim._boundary == "cpml", (
        f"live anchor requires a CPML (absorbing) boundary, got {sim._boundary!r}"
    )


def test_live_pec_short_s11_anchor():
    """LIVE compute_waveguide_s_matrix: PEC-short total reflection, |S11|≈1.

    The primary regression witness. Non-trivial (|S11|=1, NOT 0). A real
    extractor regression (ghost-cell contamination, wrong modal V/I integral)
    drops |S11| below the Meep-class 0.99 gate — exactly what the frozen replay
    cannot see.
    """
    freqs = np.linspace(*BAND_HZ, N_FREQS)
    sim = _live_build_sim(freqs, pec_short_x=0.085)
    _assert_cpml(sim)
    s, _, idx = _s_matrix(sim, normalize=False)
    s11 = np.abs(s[idx["left"], idx["left"], :])
    s21 = np.abs(s[idx["right"], idx["left"], :])
    print(f"\n[live pec-short] |S11|={np.array2string(s11, precision=4)}")
    # NOTE: |S21| is NOT asserted here. With normalize=False (single-run wave
    # decomposition) the off-diagonal S21 is convention-dependent — the source
    # spectrum is not cancelled without the two-run normalization, so the raw
    # right-port ratio is ~1 even behind the short. PEC-short's validated,
    # Meep-class quantity is |S11| (battery test_pec_short_s11_magnitude); the
    # live transmission/S21 path is checked separately on the empty guide with
    # normalize='flux'. (R5: the |S21|≈0 expectation was an extraction-convention
    # misdiagnosis, surfaced here; not chased — R2.)
    print(f"[live pec-short] |S21|(normalize=False, NOT asserted)={np.array2string(s21, precision=4)}")
    assert s11.min() >= 0.99, (
        f"LIVE PEC-short min|S11|={s11.min():.4f} < 0.99 — compute_waveguide_s_matrix "
        f"regression (the frozen broad-E5 replay would NOT catch this)"
    )
    # 1.03 matches the battery's validated near-cutoff ceiling (the 5 GHz bin at
    # f/fc=1.33 carries a small over-unity discrete-Yee Z_TE residual).
    assert s11.max() < 1.03, f"LIVE PEC-short max|S11|={s11.max():.4f} non-passive"
    # Gate-tightness witness (non-vacuous): the live healthy values sit close to
    # the 0.99 floor, so the gate catches a regression of ~1%, not a slack one.
    # This is what makes the LIVE anchor discriminating where the frozen replay
    # (a fixed JSON answer key, blind to the live extractor) is not.
    assert s11.min() - 0.99 < 0.02, (
        f"PEC-short gate is slack: healthy min|S11|={s11.min():.4f} is >0.02 above "
        f"the 0.99 floor, so a real regression could hide under it"
    )


def test_live_empty_guide_s21_anchor():
    """LIVE compute_waveguide_s_matrix: empty matched guide transmits, |S21|≈1.

    Secondary witness covering the TRANSMISSION / S21 extraction path (PEC-short
    checks only reflection). ``normalize='flux'`` (documented-convergent), plus a
    live passivity check |S11|²+|S21|² ≤ 1. Empty-guide |S11|≈0 is used only as a
    sanity bound here, NOT as the S11 regression witness (that is PEC-short).
    """
    freqs = np.linspace(*BAND_HZ, N_FREQS)
    sim = _live_build_sim(freqs)
    _assert_cpml(sim)
    s, _, idx = _s_matrix(sim, normalize="flux")
    s11 = np.abs(s[idx["left"], idx["left"], :])
    s21 = np.abs(s[idx["right"], idx["left"], :])
    power = s11**2 + s21**2
    print(f"\n[live empty] |S21|={np.array2string(s21, precision=4)}  (ideal 1)")
    print(f"[live empty] |S11|={np.array2string(s11, precision=4)}  passivity={np.array2string(power, precision=4)}")
    # Tight transmission witness (measured ~0.999; the battery's matched-load
    # gates are ratcheted to their values, so this is too — 0.98 keeps ~2%
    # cross-machine float margin, far tighter than the prior slack 0.9).
    assert s21.min() >= 0.98, (
        f"LIVE empty-guide min|S21|={s21.min():.4f} < 0.98 — transmission "
        f"extraction regression in compute_waveguide_s_matrix"
    )
    # By-construction determinism bound (NOT a CPML witness — issue #395).
    # On the flux path the empty-guide diagonal is
    # |S11|^2 = |F_ref - F_dev| / |F_ref| and the empty device run equals the
    # vacuum reference run bit-for-bit, so |S11| ~ 0 REGARDLESS of boundary
    # quality — a crippled CPML would not move it. This asserts only that the
    # two-run flux cancellation is deterministic. The real waveguide-lane
    # PML-reflection gate is the single-run (normalize=False)
    # test_matched_load_s11_empty_waveguide in the validation battery, which
    # trebles when the CPML is crippled.
    assert s11.max() < 0.05, (
        f"LIVE empty-guide max|S11|={s11.max():.4f} >= 0.05 — the two-run flux "
        f"cancellation is not deterministic (plumbing regression); this is NOT "
        f"a CPML-absorption check (see battery test_matched_load_s11_empty_waveguide)"
    )
    assert power.max() <= 1.05, (
        f"LIVE empty-guide max(|S11|²+|S21|²)={power.max():.4f} > 1.05 — "
        f"non-passive (energy-injection) extractor bug"
    )


# ---------------------------------------------------------------------------
# Unitarity (power closure) of the committed flux-lane envelopes — v1.8 WP3
# ---------------------------------------------------------------------------
#
# Every case of the five uniform band envelopes stores the per-frequency
# |S11|^2 + |S21|^2 extremes of the rfx flux lane as ``cases[i].unitarity_min``
# / ``cases[i].unitarity_max`` (a lossless slab: the ideal is 1.0 at every
# bin). No test read them before this block. The gate is DERIVED from the
# committed numbers through the shared policy (``tests/_gate_policy.py``,
# ``gate_from_envelope``, quantum 1000 = milli-scale residual), never chosen:
#
#   measured worst |u - 1| per fixture, max over its cases and both extremes
#   (the read-out below is reproduced live by ``_worst_unitarity_deviation``):
#
#     fixture                  worst |u-1|   from case          per-fixture gate
#     wr10_wband               1.034737e-4   dx40_slab_er2  (max)   0.001
#     wr15_vband               1.226664e-4   dx25_slab_er2  (max)   0.001
#     wr28_kaband              3.151894e-3   dx50_slab_er4  (min)   0.005
#     wr340_sband              4.434586e-4   dx1500_slab_er4 (max)  0.001
#     wr62_kuband              5.069375e-4   dx200_slab_er4 (min)   0.001
#
#   overall worst = 3.151894e-3 (WR-28, unitarity_min 0.9968481)
#   gate = ceil(3.151894e-3 x ENVELOPE_GATE_MULTIPLIER x 1000) / 1000
#        = ceil(4.727840) / 1000 = 0.005
#
# One gate for all five fixtures (the overall worst), stated here before the
# assert was written. ``test_unitarity_gate_is_derived_from_the_committed_envelopes``
# recomputes it from the JSONs so a hand-moved pin is caught, and caps the
# measured envelope with a literal pinned OUTSIDE the artifacts so a doctored
# JSON that widens its own ``unitarity_*`` cannot re-derive a wider gate.
# Cheap refute (run once, output in the PR body): perturbing one
# ``unitarity_max`` by +0.01 in a scratch copy makes the assert go red.
UNITARITY_ABS_GATE: float = 0.005
UNITARITY_MEASURED_WORST: float = 3.151894e-3   # WR-28 dx50_slab_er4, unitarity_min
_UNITARITY_WORST_CAP: float = 3.2e-3            # literal pinned outside the JSONs


def _worst_unitarity_deviation(env: dict) -> tuple[float, str]:
    """``max_i max(|unitarity_min_i - 1|, |unitarity_max_i - 1|)`` and the case tag."""
    worst, tag = -1.0, ""
    for c in env["cases"]:
        dev = max(abs(float(c["unitarity_min"]) - 1.0), abs(float(c["unitarity_max"]) - 1.0))
        if dev > worst:
            worst, tag = dev, str(c.get("tag", "?"))
    return worst, tag


@pytest.mark.parametrize("path", _mag_fixture_files(), ids=lambda p: p.stem)
def test_committed_band_envelope_is_unitary_within_the_derived_gate(path: Path) -> None:
    """Every case's stored |S11|^2 + |S21|^2 extremes sit within
    ``UNITARITY_ABS_GATE`` of 1 (derivation in the block comment above)."""
    env = json.loads(path.read_text())
    for c in env["cases"]:
        umin, umax = float(c["unitarity_min"]), float(c["unitarity_max"])
        assert umin <= umax, (path.name, c["tag"], umin, umax)
        assert abs(umin - 1.0) <= UNITARITY_ABS_GATE, (
            f"{path.name} case {c['tag']}: unitarity_min={umin:.7f} is "
            f"{abs(umin - 1.0):.3e} from 1 > gate {UNITARITY_ABS_GATE}")
        assert abs(umax - 1.0) <= UNITARITY_ABS_GATE, (
            f"{path.name} case {c['tag']}: unitarity_max={umax:.7f} is "
            f"{abs(umax - 1.0):.3e} from 1 > gate {UNITARITY_ABS_GATE}")
    worst, tag = _worst_unitarity_deviation(env)
    print(f"[{path.stem}] worst |unitarity-1| = {worst:.6e} ({tag}); gate {UNITARITY_ABS_GATE}")


def test_unitarity_gate_is_derived_from_the_committed_envelopes() -> None:
    """The pin equals the shared policy applied to the live overall worst,
    and the live overall worst equals the one the derivation above quotes."""
    per_fixture = {p.stem: _worst_unitarity_deviation(json.loads(p.read_text()))
                   for p in _mag_fixture_files()}
    assert len(per_fixture) == len(EXPECTED_BANDS), per_fixture
    overall = max(w for w, _ in per_fixture.values())
    assert overall == pytest.approx(UNITARITY_MEASURED_WORST, abs=1e-9), per_fixture
    assert overall <= _UNITARITY_WORST_CAP, per_fixture
    assert UNITARITY_ABS_GATE == gate_from_envelope(overall, quantum=1000), (
        UNITARITY_ABS_GATE, overall)
    # The single gate is not looser than any per-fixture derivation would be
    # by more than the quantization of the overall worst: every per-fixture
    # gate is <= the shared one (the WR-28 fixture sets it).
    for stem, (w, _) in per_fixture.items():
        assert gate_from_envelope(w, quantum=1000) <= UNITARITY_ABS_GATE, (stem, w)
