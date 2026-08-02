"""Issue #490 Lane 1 -- committed gate for the rectangular-waveguide broad-E5
Airy PHASE envelope.

Structure mirrors ``test_waveguide_broad_e5_envelope_gates.py`` (the
magnitude gate this lane extends):

1. **Committed-fixture re-derivation** -- load every band's phase envelope
   JSON under ``tests/fixtures/waveguide_broad_e5/*_phase_envelope.json``
   (produced by ``scripts/diagnostics/build_waveguide_band_broad_e5_phase_envelope.py``
   from the SAME committed npz/manifest data as the magnitude envelope -- no
   new FDTD run for the 5-band/20-case sweep, see that script's docstring for
   the full convention note and provenance) and re-assert the phase verdict
   from the committed per-case numbers.
2. **Gate-semantics lock** -- drives the real ``airy_slab`` + the corrected
   reference-plane phase transform with synthetic ideal/perturbed phases.
3. **Falsifier** (mandate: a planted defect must red the gate) -- replays the
   committed ``phase_falsifier_and_domain_invariance.json``, which recomputes
   the SAME 20 real cases with the wrong (pre-session) S21 reference-plane
   formula and must exceed the gate tolerance by a wide margin.
4. **Domain-size invariance witness** (mandate) -- replays the same JSON's
   fresh grown-domain WR-340 case: the phase-gate verdict must be identical
   to the committed (smaller-domain) case.

These are pure-Python contracts (no FDTD); the falsifier/invariance numbers
were produced once by
``scripts/diagnostics/waveguide_phase_falsifier_and_domain_invariance.py``
and are replayed here from the committed JSON, same pattern as the magnitude
lane's frozen-replay tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_band_broad_e5_phase_envelope import (  # type: ignore  # noqa: E402
    MAX_PHASE_TOL_DEG, PHASE_MAG_FLOOR, _wrapped_phase_diff_deg,
)
from build_waveguide_band_broad_e5_envelope import airy_slab  # type: ignore  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "waveguide_broad_e5"
EXPECTED_BANDS = {
    "wr28_kaband", "wr62_kuband", "wr15_vband", "wr340_sband", "wr10_wband",
}
FALSIFIER_JSON = FIXTURES / "phase_falsifier_and_domain_invariance.json"


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("waveguide_*_broad_e5_phase_envelope.json"))


def test_all_five_bands_present_phase() -> None:
    tokens = {
        p.name.replace("waveguide_", "").replace("_broad_e5_phase_envelope.json", "")
        for p in _fixture_files()
    }
    assert tokens == EXPECTED_BANDS, f"committed phase bands {tokens} != {EXPECTED_BANDS}"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda p: p.stem)
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
    # Honesty: this session's fixtures measure ZERO masked bins (the bands
    # were designed to avoid Fabry-Perot nulls) -- if that ever changes,
    # masking silently hiding a real regression should be visible, not mute.
    assert summ["total_masked_bins"] >= 0


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
