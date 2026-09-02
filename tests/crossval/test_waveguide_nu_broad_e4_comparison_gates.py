"""Committed gate for the WR-90 NONUNIFORM (graded-dy) flux broad-E4 external
comparison.

Mirrors ``tests/crossval/test_waveguide_broad_e5_envelope_gates.py`` (the uniform lane)
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

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from check_port_external_references import _comparison_breadth_ok  # type: ignore  # noqa: E402

sys.path.insert(0, str(REPO / "tests"))
from _gate_policy import gate_from_envelope  # type: ignore  # noqa: E402

FIXTURE = (
    REPO / "tests" / "fixtures" / "waveguide_nu_broad_e4"
    / "waveguide_wr90_nu_flux_broad_e4_comparison.json"
)

# The MEASURED envelope, mirroring the producer's own _ENVELOPE_*_PER_PAIR to
# the digit (an earlier revision wrote 0.0030 for the mean where the producer
# says 0.002998 — same quantized gate, two numbers for one quantity, which is
# the drift this lane keeps being reviewed for). The absorber fix (#496/#576)
# took the worst per-pair max from 0.07009 to 0.008529 and the worst per-pair
# mean from 0.0359 to 0.002998, leaving the old flat 0.10 / 0.07 12x and 23x
# loose. Both are PER-PAIR figures — deriving the mean tolerance from the
# summary mean (0.000709) produced a gate that failed 1 of 5 pairs on its first
# run. These are no longer trust-me literals: since schema v2 the fixture carries
# per-bin arrays, so the test below RECOMPUTES them from the artifact.
MEASURED_MAX_ENVELOPE = 0.008529
MEASURED_MEAN_ENVELOPE = 0.002998

# ABSOLUTE ceilings, deliberately pinned OUTSIDE the artifact. The tolerances are
# derived from a measured envelope, and an envelope re-measured by the next
# regeneration lives INSIDE the thing these gates guard — so a regeneration that
# got 10x worse would re-derive a 10x looser gate and stay green. That is the
# dependency-closure trap; these two numbers are the part no regeneration can
# move.
#
# MEASURED sensitivity, not asserted (a bound whose blind band nobody measured
# is a bound nobody can rely on). Mutation: degrade the slab-S11 residual by a
# factor k, re-derive the fixture's tolerances from the degraded envelope, AND
# move the pinned literals above to match — a coherent author, which is the edit
# every consistency-relation instrument in this file loses to:
#
#     k = 1.15  ->  envelope 0.009808, gate 0.015  ->  ALL GREEN (blind)
#     k = 3.0   ->  envelope 0.025586, gate 0.039  ->  RED (ceiling only)
#
# So this ceiling catches degradation of roughly >=1.2x and is blind below that.
# That band is deliberate — float32 and reference-resolution scatter live in it —
# but it is a real limit and should be quoted as one, not treated as "the gate
# cannot be loosened".
ABSOLUTE_MAX_ENVELOPE_CEILING = 0.010
ABSOLUTE_MEAN_ENVELOPE_CEILING = 0.004

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


def _per_bin(p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rfx = np.asarray(p["rfx_mag"], dtype=float)
    ref = np.asarray(p["ref_mag"], dtype=float)
    return rfx, ref, np.abs(rfx - ref)


def test_per_bin_arrays_rederive_every_committed_scalar() -> None:
    """#576 review F4. Before schema v2 every number in this fixture was a
    frozen scalar that could only be PINNED, never re-derived, and no claim
    about WHERE in the band a residual sat was checkable from the artifact at
    all. That is not hypothetical: this PR's review caught me asserting the
    over-unity excess was localized at the low band edge when the producer's
    own spectrum peaks at the high end. With the per-bin arrays committed, the
    scalars become derived quantities and the spectral claims become falsifiable
    on a clean checkout."""
    env = _env()
    assert env["schema_version"] >= 2, (
        "fixture predates the per-bin arrays — regenerate before trusting any "
        "spectral claim about it")
    for p in env["pairs"]:
        rfx, ref, diff = _per_bin(p)
        assert rfx.size == ref.size == p["n_freqs"] > 1, p
        freqs = np.asarray(p["freqs_ghz"], dtype=float)
        assert freqs.size == rfx.size
        assert np.all(np.diff(freqs) > 0), "bins must be ordered"
        assert freqs[0] * 1e9 == pytest.approx(p["freq_lo_hz"], rel=1e-6)
        assert freqs[-1] * 1e9 == pytest.approx(p["freq_hi_hz"], rel=1e-6)
        # Every committed scalar recomputed, not trusted.
        assert diff.max() == pytest.approx(p["max_mag_abs_diff"], abs=1e-8), p
        assert diff.mean() == pytest.approx(p["mean_mag_abs_diff"], abs=1e-8), p
        assert rfx.min() == pytest.approx(p["rfx_mag_range"][0], abs=1e-8)
        assert rfx.max() == pytest.approx(p["rfx_mag_range"][1], abs=1e-8)
        assert ref.min() == pytest.approx(p["ref_mag_range"][0], abs=1e-8)
        assert ref.max() == pytest.approx(p["ref_mag_range"][1], abs=1e-8)


def test_envelope_is_recomputed_from_the_artifact_and_capped_from_outside() -> None:
    """The tolerances are derived, so the derivation's INPUT must be checked
    two ways: it must match what the producer says it measured (catches a
    hand-edited fixture or producer/test drift), and it must sit under an
    absolute ceiling that lives outside the artifact (catches a regeneration
    that degraded and would otherwise re-derive its own looser gate)."""
    env = _env()
    per_pair = [_per_bin(p)[2] for p in env["pairs"]]
    worst_max = max(float(d.max()) for d in per_pair)
    worst_mean = max(float(d.mean()) for d in per_pair)

    assert worst_max == pytest.approx(MEASURED_MAX_ENVELOPE, abs=1e-6), (
        f"artifact's worst per-pair max {worst_max:.6f} != the pinned envelope "
        f"{MEASURED_MAX_ENVELOPE} that the producer derived its gate from")
    assert worst_mean == pytest.approx(MEASURED_MEAN_ENVELOPE, abs=1e-6), (
        f"artifact's worst per-pair mean {worst_mean:.6f} != the pinned "
        f"envelope {MEASURED_MEAN_ENVELOPE}")

    assert worst_max <= ABSOLUTE_MAX_ENVELOPE_CEILING, (
        f"worst per-pair max {worst_max:.6f} exceeds the absolute ceiling "
        f"{ABSOLUTE_MAX_ENVELOPE_CEILING} — do NOT re-derive a looser gate from "
        f"it; find out what degraded (#576)")
    assert worst_mean <= ABSOLUTE_MEAN_ENVELOPE_CEILING, (
        f"worst per-pair mean {worst_mean:.6f} exceeds the absolute ceiling "
        f"{ABSOLUTE_MEAN_ENVELOPE_CEILING}")

    # And the fixture's stored tolerances really are that derivation.
    assert env["max_mag_abs_tol"] == EXPECTED_MAX_TOL
    assert env["mean_mag_abs_tol"] == EXPECTED_MEAN_TOL


def test_passivity_holds_per_bin_not_just_on_the_range() -> None:
    """Per-bin passivity. The range-based check elsewhere in this file can only
    see the extremum; this sees every bin, and reports WHERE a violation sits —
    the information whose absence let a wrong spectral claim stand for a whole
    review round."""
    env = _env()
    tol = env["max_mag_abs_tol"]
    for p in env["pairs"]:
        rfx, _, _ = _per_bin(p)
        over = np.flatnonzero(rfx > 1.0 + tol)
        assert over.size == 0, (
            f"{p['geometry']} {p['component']}: {over.size} of {rfx.size} bins "
            f"exceed 1 + {tol}; worst {rfx.max():.6f} at "
            f"{np.asarray(p['freqs_ghz'])[int(np.argmax(rfx))]:.3f} GHz")


def test_the_empty_s11_pair_is_structurally_vacuous_and_not_counted_as_evidence() -> None:
    """A finding the per-bin arrays surfaced immediately (#576 review F4).

    ``empty``/``S11`` is identically 0.0 in ALL bins, because the empty run is
    the two-run reference: S11 = (total - incident)/incident is exactly zero by
    construction, not by measurement. Its "agreement" with Palace is therefore
    |0 - ref|, which passes against ANY sufficiently small reference and would
    pass just as well if the extractor were broken. So this lane has FOUR
    load-bearing pairs, not five.

    Pinned rather than removed: the pair is legitimate coverage of the
    convention (a nonzero value here would mean the reference subtraction
    changed), and deleting it would hide that. What must not happen is quoting
    "5/5 pairs passed" as five independent checks. If this ever goes nonzero,
    that is a real signal and this test is where it announces itself."""
    env = _env()
    p = next(x for x in env["pairs"]
             if x["geometry"] == "empty" and x["component"] == "S11")
    rfx, ref, _ = _per_bin(p)
    assert np.all(rfx == 0.0), (
        "empty/S11 is no longer identically zero — the two-run reference "
        f"subtraction convention changed (max {rfx.max():.3g}); that is a real "
        "finding, not a fixture nit")
    # And the reference it is 'agreeing' with is itself ~1e-5, i.e. the
    # comparison is vacuous at this tolerance by three orders of magnitude.
    assert ref.max() < 1e-3, ref.max()
    assert float(np.abs(rfx - ref).max()) < 0.1 * env["max_mag_abs_tol"], (
        "if this ever approaches the tolerance, the pair stops being vacuous "
        "and this test's premise needs rewriting")


# WR-90's broad-wall dimension is a WAVEGUIDE STANDARD, not a measurement from
# this run, so pinning it here is a definition rather than a frozen result — and
# it lets the absorber witness below be recomputed entirely from the artifact
# plus standards, with no constant copied out of the producer.
WR90_A_M = 0.02286
C0 = 299_792_458.0
FAR_PORT_LAMBDA_G_FRACTION_FLOOR = 0.5  # #496


def test_absorber_depth_witness_is_gated_not_just_recorded() -> None:
    """#576 review F4 / #496. The fixture grew a `setup` provenance block, and
    NOTHING asserted it — a recorded number no test reads is decoration, which
    is how this lane shipped 0.33 lambda_g for its whole history.

    Three things, in ascending strength:
      1. the DISCIPLINE: the absorber must be >= 0.5 lambda_g at the lowest
         measured frequency. This is the check whose absence caused #576.
      2. the depth is RECOMPUTED from the artifact's own dx and band edge plus
         the WR-90 standard, so a fixture claiming 46 cells at some other dx
         cannot pass by restating its own fraction.
      3. the PHYSICAL witness that the absorber worked: the PEC-short is a
         lossless total reflector, so its |S11| must sit at unity within
         float32 noise. At 20 cells this read 1.019948; the discipline check
         above is only trustworthy to the extent this one agrees with it.
    """
    env = _env()
    setup = env["setup"]

    f_lo = min(float(p["freq_lo_hz"]) for p in env["pairs"])
    f_c = C0 / (2.0 * WR90_A_M)
    assert f_lo > f_c, "band must be above TE10 cutoff for lambda_g to be real"
    lam_g_low = (C0 / f_lo) / np.sqrt(1.0 - (f_c / f_lo) ** 2)

    dx = float(setup["dx_m"])
    layers = int(setup["cpml_layers"])
    fraction = layers * dx / lam_g_low

    assert fraction >= FAR_PORT_LAMBDA_G_FRACTION_FLOOR, (
        f"absorber is {layers} cells = {fraction:.3f} lambda_g at {f_lo / 1e9:.2f} "
        f"GHz, below the {FAR_PORT_LAMBDA_G_FRACTION_FLOOR} far-port discipline "
        f"(#496). This is the check whose absence let 0.33 lambda_g ship")
    # The fixture's own stated fraction must be that same quantity, so a
    # regeneration cannot record a flattering number next to a thin absorber.
    assert float(setup["cpml_fraction_of_lambda_g_low"]) == pytest.approx(
        fraction, rel=2e-3), (setup["cpml_fraction_of_lambda_g_low"], fraction)

    ps = next(p for p in env["pairs"] if p["geometry"] == "pec_short")
    rfx, _, _ = _per_bin(ps)
    excess = float(np.max(np.abs(rfx - 1.0)))
    assert excess < 2e-3, (
        f"PEC-short |S11| departs from unity by {excess:.2e} — a lossless total "
        f"reflector does not do that. At 20 CPML cells this was 2.0e-2; if it has "
        f"regressed, the absorber/window pair is back (#576), and the "
        f"{fraction:.3f} lambda_g recorded above is not sufficient on its own")
