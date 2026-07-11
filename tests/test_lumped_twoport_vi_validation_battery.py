"""Lumped/wire-port V-I extraction + Z0-normalization validation battery.

Validation-campaign lane: locks the POST-PROCESSING layer of the lumped/wire
port family — the V/I DFT accumulation inside the production scan
(``rfx/simulation.py``), the shared pure wave decomposers
(``rfx/probes/probes.py::decompose_lumped_s_matrix`` /
``decompose_wire_s_matrix`` / ``extract_lumped_s11``), the production scan
driver (``rfx/probes/sparam_driver.py``), and the Z0 normalization — via the
NULL-INPUT-ISOLATION method: a 2-port THRU (two matched ports joined by a
uniform line) has a known trivial answer, so every deviation isolates an
extraction/calibration property independent of any DUT physics. The core
FDTD update is already validated elsewhere; this battery does not re-test it.

Honest posture: changes/relaxes NO gate outside this module. The
``tests/test_twoport_wire_port.py`` floor gate (max|S21| > 1e-3), the
``tests/test_sparam_driver_matches_eager.py`` driver-vs-eager atol-2e-3
locks, and the ``tests/test_run_forward_s11_contract.py`` magnitude-only
CPML atol-2e-3 contract are all untouched. This module's own thru gates
were re-baselined 2026-07-10 in the same PR as the issue #308
receive-wave fix — exactly the tripwire protocol the original gates
prescribed (fail LOUDLY on the convention change, re-measure in the same
PR). The S11 floor/alive gates, the run<->forward cross-check, and the
algebraic-identity test are byte-untouched by the re-baseline.

Measured baseline (R5 measure-before-gate; the DIAGONAL |S11|/|S22| lane
was RE-BASELINED 2026-07-11 for the issue #318 live-cell termination fix —
fresh fixture reruns on this CPU box, x64 OFF, complex64 accumulators, the
documented envelope. The |S21| off-diagonal narrative below is the earlier
issue #308 receive-wave baseline, which the #318 fix left in the same
O(0.55-0.61) class — the |S21| kappa deflation is the SEPARATE issue #313
and is NOT part of the #318 ledger):

THRU fixture (wire 2-port, 16 mm air microstrip w/h=5 over a pec_faces
ground, Zc ~ 50 ohm, driver path of PR #258; 9 bins 3-7 GHz, 4000 steps,
~70 s -> slow_physics):
  - post-#318 per-bin |S11| = [0.0555 0.0476 0.0388 0.0327 0.0337 0.0426
    0.0558 0.0708 0.0858], |S22| = [0.0543 0.0470 0.0397 0.0360 0.0391
    0.0484 0.0611 0.0752 0.0891] across 3-7 GHz (max|S11| 0.0858, max|S22|
    0.0891, mean|S11| 0.0515, per-bin min 0.0327 at 4.5 GHz). This
    V-shaped residual is rfx's MEASURED feed-post reflection, NOT an
    extraction artefact and NOT a termination error (issue #318 H_FIXTURE
    re-diagnosis, 2026-07-11, decisive on three independent witnesses):
    the two wire ports are 1 mm vertical feed posts (~0.26 nH each) whose
    series reactances interfere across the 16 mm line, producing a
    reflection null near 4.5 GHz and rising edges toward the band ends.
    Before the #318 live-cell fix the diagonals read max ~0.130/0.132
    (mean 0.076) because the dead extent cell (inside the PEC trace) was
    wrongly counted in the sigma/drive/Z0 fold, giving a Z0*(n_live/n) =
    33.3-ohm termination instead of 50 ohm (that 33.3-ohm reading is the
    historical issue #313 finding). The reflection channel is ALIVE, not
    a dead readout: mismatching a port drives |S11| up as expected (Z0
    witness), and both diagonals stay well above the 0.02 alive floor.
  - |S21| = 0.54606..0.60974 across 3-7 GHz (post-#318 rerun; post-#308
    it was 0.523601..0.667776 — same O(0.55-0.61) class, the live-cell
    fix did not move the transmission class). HISTORY: before the
    issue #308 fix the shipped receive-side b-wave
    b_i = (-V - Z0_i*I)/(2*sqrt(Z0_i)) structurally cancelled the
    arriving wave against the matched receive cell's local Ohm's law
    -V = +Z0_cell*I (measured -V/I = 16.6672+0.0002j ohm vs
    Z0/n_cells = 16.667 ohm), so a matched thru read |S21| =
    0.0025-0.0046 near-null, family-wide (lumped + wire). #308
    role-selects the receive b-wave to the orthogonal channel
    (V - Z0*I) (sign pinned by the DC falsifier, phase item below; the
    first-cut sign (-V + Z0*I) was amended in the same PR — |S21| is
    sign-invariant, these magnitudes did not move), recovering the
    transmission channel. HONESTY LABEL — REGRESSION LOCK, NOT
    validated physics: the recovered channel is correct (mismatch
    witness responds per-bin; b-wave voltage-dominated at the old
    residual scale 0.38-0.89%) but its MAGNITUDE is unvalidated. The
    extractor-independent flux referee (flux monitors bracketing the
    line, 2026-07-10 falsifier battery) measured a raw transmitted
    power fraction of 0.959-0.998 across 3-7 GHz vs port-based
    |S21|^2 = 0.274-0.446 — per-bin ratio (flux fraction)/|S21|^2 =
    2.237..3.541 (mean 2.911), i.e. implied flux-true |S21| =
    0.971-0.997 (lossless closure gap 0.002-0.040), consistent with the
    physics-true expectation |S21| >= ~0.9 for this near-lossless thru.
    The port-based magnitude deficit is a confirmed drive-side
    common-mode scale bias kappa(f) = 1.49..1.86 (frequency-DEPENDENT)
    entering a_j via the source-cell V/I accounting; it is invisible to
    S11. The flux-vs-port transmitted-power delta is an OPEN item
    (issue #313; recorded per-bin above) — do
    NOT cite the 0.52-0.67 band as thru-transmission physics.
  - S21 signed phase deviation vs the analytic line delay
    exp(-j*2*pi*f*L/c): -0.6268..-0.3482 rad across 3-7 GHz (post-#318;
    post-#308 -0.754891..-0.335259) — a smooth delay-like excess (small
    feed-post group delay on top of the 53.4 ps line delay), no pi
    offset. The DC-limit sign is
    RESOLVED by the same-PR amendment (2026-07-10): the first-cut #308
    sign (-V + Z0*I) measured S21(DC) -> -1 on the low-frequency
    falsifier (0.5-2 GHz; the pi sat in the raw cross-port phasors,
    arg(V2/V1) ~= pi - beta*L, from the source-driven cell field sense
    in V = -E*dx, while both port diagonals were internally
    sign-consistent); the amended receive wave b_i =
    (V - Z0*I)/(2*sqrt(Z0)) — a global -1 of the first cut — measures
    S21(DC) -> +1: at 0.5 GHz S21 = +0.57116-0.11059j (arg -0.191 rad,
    heading to 0 as f -> 0; re-verified on this exact geometry in the
    amendment rerun). |S21|, reciprocity, the flux comparisons and both
    diagonals are sign-invariant and did not move (bit-identical to the
    81c1983 rerun).
  - reciprocity max|S21-S12| = 7.53e-3 absolute (rel 0.013581) on the
    recovered O(0.55-0.61) magnitudes (post-#318; IMPROVED from the
    post-#308 1.043e-2 / rel 0.016254 — the live-cell fix made the two
    ports' V/I bookkeeping more symmetric). A meaningful symmetry check
    on a live channel (pre-#308 it was vacuous at the near-null scale).
  - passivity max singular value over the 9 per-freq 2x2 slices =
    0.632587 (post-#318; margin 0.37 to the physical bound 1.0; IMPROVED
    from the post-#308 0.687410 as the smaller diagonal reflection lowers
    the mixed-matrix norm). Still NOT transmission validation (kappa is
    the SEPARATE #313 open item, deflating the port-based magnitudes);
    kept as a strict sub-unity energy-sanity lock.

RUN<->FORWARD cross-check fixture (1-port wire, CPML — byte-for-byte the
``_wire_sim`` of tests/test_run_forward_s11_contract.py, which this
COMPLEMENTS: that committed test gates |S11| magnitude at atol 2e-3; this
battery adds (a) both return SHAPES pinned as an intended contract and
(b) a COMPLEX-value delta gate at the measured float32 envelope):
  - run(compute_s_params=True) returns shape (1, 1, 7) complex64 — the
    full-matrix rank, S[receive, drive, freq], even for 1 port;
    forward(port_s11_freqs=) returns shape (7,) complex64 — a 1-D
    diagonal via extract_lumped_s11. The rank difference is INTENDED
    behavior, pinned here as a contract: PR #258 deliberately kept the
    1-port wire run() path on the JIT main-scan fast-path (preserving
    the run/forward PEC contract) while multi-port sets go through the
    production scan driver; run() reports the S-matrix convention,
    forward() the AD-friendly per-port diagonal. Do not "fix" one to
    match the other.
  - measured max complex |S_run - S_fwd| = 4.27e-7 all-band (2.15e-7
    in-band) vs the committed 2e-3 magnitude gate — the committed band is
    honest with ~4 orders of margin. Small deltas here are the measured
    float32 conditioning envelope (PR #258 finding: the decompose
    diagonal is algebraically identical to extract_lumped_s11) — gate
    them, don't chase them.

Direction sensitivity of the S21 phase gate (claim VERIFIED, in-test):
the gate is on the SIGNED, per-bin wrapped deviation arg(S21) -
(-2*pi*f*L/c), so it constrains sign AND magnitude. It is NOT
conjugation-invariant: conjugating S21 (the 5/5-recurring comparator bug
class, W3.4) flips arg(S21), and on the measured amended-sign data moves
8 of 9 bins outside the band (measured conj devs +2.347..-0.834 rad vs
band [-1.1, -0.1]; only the 7 GHz bin remains inside). The phase test
asserts this discrimination live on the measured data (conj(S21) must
violate the band), so the gate cannot silently degrade into a
conjugation-blind |dev| check. A flip back to the first-cut receive
sign shifts every bin by pi (to ~+2.39..+2.81 rad) and also fails the
band — the DC-witnessed sign is locked, not merely a convention
envelope.

Preflight (quoted verbatim per feedback_never_ignore_preflight; the thru
fixture asserts and prints this at fixture setup):
  "pec_faces={z_lo} creates an INFINITE PEC boundary AND the geometry
  contains finite PEC objects. For antennas or finite-GP structures, the
  pec_faces boundary makes the ground plane cover the entire domain face,
  which changes the physics (cavity vs radiating antenna). If you need a
  finite ground plane, remove pec_faces and use an explicit PEC Box
  instead."
(advisory-only; intended here — the infinite ground plane IS the
microstrip return). The 1-port CPML fixture measures preflight clean
("All checks passed"), asserted in the fast suite. The environment-level
JAX float64->float32 truncation UserWarning is the documented
x64-off accumulator envelope; x64 is deliberately never flipped at module
level in this file.

Fixture-authoring lesson baked into the thru geometry (2026-07-10 lane):
Box PEC rasterization is lower-inclusive/upper-EXCLUSIVE, so the trace
box must overhang BOTH port columns by >= 1 cell — a box ending exactly
at the port-2 x coordinate leaves that column with no PEC overhead and
produces a silently dead thru. Also: wire extent=1.0 mm at dx=0.5 mm
rasterizes to n_cells=3 (endpoint-inclusive), which enters the
Z0_cell = Z0/n_cells off-diagonal normalization.

No network, no external solver; deterministic (fixed geometry, fixed step
counts, rect DFT window). Tighten gates only with a fresh measured
baseline; a failure here marks an extraction/normalization regression (or
an intentional upstream decomposer-convention change, which must update
the measured provenance here in the same PR).
"""

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.probes import (
    decompose_lumped_s_matrix,
    decompose_wire_s_matrix,
    extract_lumped_s11,
)
from rfx.sources.sources import GaussianPulse

C0_M_PER_S = 299792458.0

# ===========================================================================
# THRU fixture geometry (all lengths integer multiples of dx = 0.5 mm)
# ===========================================================================
_THRU_DX_M = 0.5e-3
_THRU_DOMAIN_M = (0.032, 0.020, 0.010)
_THRU_FREQ_MAX_HZ = 10e9
_THRU_CPML_LAYERS = 8
_THRU_H_M = 1.0e-3        # trace height above ground (2 cells)
_THRU_W_M = 5.0e-3        # trace width (10 cells) -> air microstrip Zc ~ 50 ohm
_THRU_X1_M = 0.008        # port 1 x
_THRU_X2_M = 0.024        # port 2 x
_THRU_L_M = _THRU_X2_M - _THRU_X1_M   # 16 mm port-to-port
_THRU_Y_MID_M = _THRU_DOMAIN_M[1] / 2
_THRU_N_STEPS = 4000
_THRU_FREQS_HZ = np.linspace(3e9, 7e9, 9)   # in-band of f0=5 GHz, bw=0.8

_PEC_FACES_ADVISORY_SNIPPET = (
    "pec_faces={z_lo} creates an INFINITE PEC boundary AND the geometry "
    "contains finite PEC objects."
)

# ===========================================================================
# Gate constants (R5: every gate = measured value + honest margin)
# ===========================================================================
# Measured post-#318 (fresh rerun on the fixed branch, 2026-07-11):
# per-bin |S11| = [0.0555 0.0476 0.0388 0.0327 0.0337 0.0426 0.0558 0.0708
# 0.0858] and |S22| = [0.0543 0.0470 0.0397 0.0360 0.0391 0.0484 0.0611
# 0.0752 0.0891] across 3-7 GHz (max|S11| 0.0858, max|S22| 0.0891, mean
# |S11| 0.0515, per-bin min 0.0327 at 4.5 GHz). This V-shaped curve is
# rfx's MEASURED feed-post reflection, not an extraction error: the wire
# ports are two 1 mm vertical feed posts (~0.26 nH each) whose reactances
# interfere across the 16 mm line, giving a reflection null near 4.5 GHz
# and rising edges toward 3 and 7 GHz (H_FIXTURE re-diagnosis, 2026-07-11 —
# three independent witnesses; see the module docstring). Gate 0.12
# (~1.35x max|S22|) on BOTH diagonals: honest cross-machine float margin,
# AND strictly below the pre-#318 dead-cell floor 0.13 (the 33.3-ohm
# termination bug), so a revert of the live-cell fix fails LOUDLY here.
_THRU_S11_FLOOR_MAX = 0.12
# Two-sided (review finding): a DEAD diagonal channel reads ~0, which would
# sail under the upper bound. The measured per-diagonal maxima are
# 0.0858/0.0891 (~4.3x above this lower bound) and the per-bin minimum is
# 0.0327 (the physical feed-post null; ~1.6x above this bound), so
# requiring max|Sii| > 0.02 makes the in-test liveness of both diagonals
# explicit while clearing the measured 4.5 GHz null with margin.
_THRU_S11_ALIVE_MIN = 0.02

# Measured |S21| = 0.54606..0.60974 across 3-7 GHz (post-#318 fresh rerun,
# 2026-07-11; the live-cell fix moved the composed channel slightly from
# the post-#308 0.5236..0.6678 but kept it the same O(0.55-0.61) class).
# Band [0.35, 0.85] KEPT UNCHANGED (issue #313 kappa regression lock — the
# |S21| deflation is the SEPARATE #313 drive-side common-mode scale bias,
# NOT part of the #318 ledger): REGRESSION LOCK on the recovered channel,
# NOT physics. The flux-vs-port transmitted-power delta is OPEN (flux
# referee measured per-bin (flux fraction)/|S21|^2 = 2.237..3.541; implied
# flux-true |S21| = 0.971-0.997 — module docstring; see the PR body /
# follow-up issue). Lower edge catches a collapse back toward the pre-#308
# near-null 0.0025-0.0046; upper edge strictly < 1 (an over-unity
# extraction artefact also fails). Still never weaker than the committed
# max|S21| > 1e-3 floor of test_twoport_wire_port.py.
_THRU_S21_BAND = (0.35, 0.85)

# Measured signed per-bin phase deviation arg(S21) - (-2*pi*f*L/c),
# wrapped: -0.6268..-0.3482 rad (post-#318 fresh rerun, 2026-07-11;
# post-#308 it was -0.7549..-0.3353 — the live-cell fix shifted the
# small feed-post group-delay excess slightly, same sign, same class).
# Band [-1.1, -0.1] rad KEPT UNCHANGED (margins ~0.47/0.25 rad; both
# edges well inside (-pi, pi) so wrapped values stay comparable).
# HONESTY LABEL — REGRESSION LOCK on the deviation VALUES (flux-vs-port
# magnitude delta OPEN, kappa — separate #313, module docstring), but
# the overall SIGN is physics-anchored by the DC witness: the low-f
# falsifier measured S21(DC) -> +1 under this sign (dev -0.049 rad at
# 0.5 GHz, tracking the analytic delay); the first-cut sign measured -1
# and was amended in the #308 PR. The deviation is a smooth feed-post
# group-delay excess, physical, not a convention artefact. Signed on
# purpose: conjugation moves 8/9 measured bins out of band (verified
# live in the test; conj dev is NOT -dev, the analytic reference phase
# differs per bin — measured conj devs +2.360..-0.962 with only the
# 7 GHz bin inside), and a sign flip back moves all 9 bins (by pi).
_THRU_PHASE_DEV_BAND_RAD = (-1.1, -0.1)

# Measured reciprocity max|S21 - S12| = 7.53e-3 (rel 0.013581) on the
# recovered O(0.55-0.61) magnitudes (post-#318 fresh rerun, 2026-07-11 —
# IMPROVED from the post-#308 1.043e-2 / rel 0.016254 as the live-cell
# fix made the two ports' V/I bookkeeping more symmetric). Abs gate
# re-baselined to 1.5e-2 (~2x measured); rel gate kept at 0.10 (scale-
# free, ~7.4x measured). A break catches an asymmetric edit to the
# shared decomposers or per-port Z0/n_cells bookkeeping.
_THRU_RECIP_ABS_MAX = 1.5e-2
_THRU_RECIP_REL_MAX = 0.10

# Measured passivity: max singular value over the 9 per-freq 2x2 slices
# = 0.632587 (post-#318 fresh rerun, 2026-07-11; IMPROVED from the
# post-#308 0.687410 as the smaller diagonal reflection lowers the
# mixed-matrix norm). Gate 0.85 (~1.34x measured, margin 0.22) —
# strictly below the physical bound 1.0, and BELOW the Frobenius
# dominance bound sqrt(2*0.12^2 + 2*0.85^2) ~= 1.214 implied by the
# S11/S21 gates, so this gate is independently bindable. Energy-sanity
# lock; NOT transmission validation (kappa open item — separate #313,
# module docstring).
_THRU_MAX_SINGULAR_VALUE = 0.85

# ===========================================================================
# run<->forward cross-check constants
# ===========================================================================
_XCHK_F0_HZ = 5e9
_XCHK_FREQS_HZ = np.array([1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]) * 1e9
# Measured max complex |S_run - S_fwd| = 4.27e-7 over ALL 7 bins (CPML,
# well-conditioned) — but that is SINGLE-MACHINE provenance at float32
# ulp scale. Gate 5e-5 (~100x measured; review finding, the v173a
# cross-machine-float lesson) — still 40x tighter than the committed
# magnitude-only atol-2e-3 gate it complements, and still fails on any
# pure-phase divergence that gate cannot see.
_XCHK_COMPLEX_DELTA_MAX = 5.0e-5

# Algebraic-identity lock (no FDTD): extract_lumped_s11 vs the decompose
# diagonals on synthetic well-conditioned V/I. The formulas are
# algebraically identical (PR #258 finding: divergence is float32
# conditioning, not formula); in complex64 they may differ only by
# rounding order. Gate 1e-5 relative (float32 eps ~1.2e-7, x ~100 margin).
_IDENTITY_REL_MAX = 1.0e-5


# ===========================================================================
# Fixtures
# ===========================================================================
def _build_thru(pulse: "GaussianPulse | None" = None) -> Simulation:
    """Wire 2-port air-microstrip THRU (2026-07-10 lane M1 fixture, exact).

    Both ports carry excite=True + the same waveform: the production scan
    driver drives each eligible port BY INDEX regardless of ``pe.excite``
    (one at a time; others are matched loads), and an excite=False port
    stores waveform=None, which cannot be driven. 2-port wire set ->
    run(compute_s_params=True) routes through the PR #258 production scan
    driver, the extraction path under test.
    """
    sim = Simulation(
        freq_max=_THRU_FREQ_MAX_HZ,
        domain=_THRU_DOMAIN_M,
        dx=_THRU_DX_M,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=_THRU_CPML_LAYERS,
    )
    # PEC trace one cell thick on top of the wire-port spans. The x-extent
    # overhangs each port column by one cell — Box rasterization is
    # lower-inclusive/upper-EXCLUSIVE, and a box ending exactly at the
    # port-2 x leaves that column without PEC overhead (silently dead thru;
    # module docstring, fixture-authoring lesson).
    sim.add(
        Box((_THRU_X1_M - _THRU_DX_M, _THRU_Y_MID_M - _THRU_W_M / 2, _THRU_H_M),
            (_THRU_X2_M + _THRU_DX_M, _THRU_Y_MID_M + _THRU_W_M / 2,
             _THRU_H_M + _THRU_DX_M)),
        material="pec",
    )
    if pulse is None:
        pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    sim.add_port(position=(_THRU_X1_M, _THRU_Y_MID_M, 0.0), component="ez",
                 impedance=50.0, extent=_THRU_H_M, waveform=pulse,
                 direction="-x")
    sim.add_port(position=(_THRU_X2_M, _THRU_Y_MID_M, 0.0), component="ez",
                 impedance=50.0, extent=_THRU_H_M, waveform=pulse,
                 direction="+x")
    return sim


@pytest.fixture(scope="module")
def thru_smatrix():
    """Run the THRU once (~70 s); quote preflight verbatim; return S(2,2,9)."""
    sim = _build_thru()
    report = sim.preflight()
    issues = [str(i) for i in report]
    # Quote every preflight message verbatim BEFORE reporting numbers
    # (feedback_never_ignore_preflight).
    for msg in issues:
        print(f"\n[thru battery] preflight (verbatim): {msg}")
    # Exact known advisory set (re-pinned 2026-07-11 for issue #319):
    # the intended pec_faces advisory (the infinite ground plane IS the
    # microstrip return) PLUS one wire_port_dead_extent_cells advisory
    # per port — this fixture GENUINELY has its top extent cell inside
    # the PEC trace. Post-#318 the dead cell is EXCLUDED from the
    # sigma/drive/Z0 fold, so each port now terminates at 50 ohm across
    # its 2 live cells (the pre-#318 33.3-ohm Z0*(n_live/n) reading is
    # the historical issue #313 finding). The battery gates below were
    # MEASURED on this exact fixture, dead cell included, so they stay
    # valid as-is. Anything else = fixture drift, stop.
    codes = sorted(getattr(i, "code", None) for i in report)
    assert codes == ["pec_faces_finite_pec",
                     "wire_port_dead_extent_cells",
                     "wire_port_dead_extent_cells"], (
        f"thru fixture preflight drifted from the measured baseline: {issues}")
    assert any(_PEC_FACES_ADVISORY_SNIPPET in m for m in issues)

    result = sim.run(n_steps=_THRU_N_STEPS, compute_s_params=True,
                     s_param_freqs=_THRU_FREQS_HZ)
    S = np.asarray(result.s_params).astype(np.complex128)
    assert S.shape == (2, 2, len(_THRU_FREQS_HZ)), (
        f"driver S-matrix shape {S.shape}, expected (2, 2, 9)")
    assert np.all(np.isfinite(S)), "thru S-matrix contains non-finite entries"
    with np.printoptions(precision=4, suppress=False):
        print(f"[thru battery] |S11|={np.abs(S[0, 0])}")
        print(f"[thru battery] |S21|={np.abs(S[1, 0])}")
    return S


@pytest.fixture(scope="module")
def crosscheck():
    """1-port wire CPML fixture of test_run_forward_s11_contract.py, run
    through BOTH estimators; returns raw (S_run, S_fwd) complex arrays."""

    def _wire_sim():
        sim = Simulation(
            freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
            boundary="cpml", cpml_layers=6,
        )
        sim.add_port(
            position=(0.0093, 0.0093, 0.0093), component="ez", impedance=50.0,
            waveform=GaussianPulse(f0=_XCHK_F0_HZ, bandwidth=0.9),
            extent=0.004,
        )
        return sim

    sim_r = _wire_sim()
    issues = sim_r.preflight()
    for msg in issues:
        print(f"\n[crosscheck] preflight (verbatim): {msg}")
    # Measured baseline: this fixture preflights CLEAN. Gate on
    # error-severity only (review finding): a future advisory-only
    # validator flagging this vanilla fixture should be PRINTED verbatim
    # above, not fail the whole cross-check module fixture.
    errors = [str(i) for i in issues
              if getattr(i, "severity", "error") == "error"]
    assert errors == [], (
        f"1-port cross-check fixture has error-severity preflight "
        f"findings: {errors}")

    r = sim_r.run(n_steps=2000, compute_s_params=True,
                  s_param_freqs=_XCHK_FREQS_HZ)
    fr = _wire_sim().forward(port_s11_freqs=_XCHK_FREQS_HZ)
    return np.asarray(r.s_params), np.asarray(fr.s_params)


# ===========================================================================
# FAST battery (default suite): decomposer identity + run<->forward contract
# ===========================================================================

def test_extract_lumped_s11_is_the_decompose_diagonal():
    """Algebraic-identity lock: three shipped S11 formulas agree (no FDTD).

    extract_lumped_s11 (S11 = (V + Z0*I)/(V - Z0*I)), the
    decompose_lumped_s_matrix diagonal (b/a wave form) and the
    decompose_wire_s_matrix diagonal (Zin = -V/I input-impedance form) are
    algebraically the same map; PR #258 proved observed run/forward
    divergence is float32 CONDITIONING, not formula. This pins the
    identity on synthetic well-conditioned phasors so a formula edit in
    any one of the three fails loudly. This locks the FORMULA identity
    only; the two ENTRY-POINT implementations are cross-checked end-to-end
    in test_run_forward_complex_values_agree_on_cpml.
    """
    rng = np.random.default_rng(20260710)
    n_ports, n_freqs, z0 = 2, 11, 50.0
    # Well-conditioned: V, Z0*I both O(Z0), away from the a=0 guard.
    v = (rng.normal(1.0, 0.3, (n_ports, n_ports, n_freqs))
         + 1j * rng.normal(0.0, 0.3, (n_ports, n_ports, n_freqs))) * z0
    i = (rng.normal(3.0, 0.3, (n_ports, n_ports, n_freqs))
         + 1j * rng.normal(0.0, 0.3, (n_ports, n_ports, n_freqs)))

    s_lumped = np.asarray(decompose_lumped_s_matrix(v, i, [z0, z0]))
    s_wire = np.asarray(decompose_wire_s_matrix(v, i, [z0, z0], [3, 3]))
    for p in range(n_ports):
        s_ref = np.asarray(extract_lumped_s11(v[p, p], i[p, p], z0=z0))
        for name, s_diag in (("lumped", s_lumped[p, p]),
                             ("wire", s_wire[p, p])):
            rel = np.max(np.abs(s_diag - s_ref) / np.abs(s_ref))
            assert rel < _IDENTITY_REL_MAX, (
                f"decompose_{name} diagonal (port {p}) deviates from "
                f"extract_lumped_s11 by rel {rel:.2e} "
                f"(gate {_IDENTITY_REL_MAX}) — the three shipped S11 "
                f"formulas are no longer the same map")


def test_run_forward_shapes_are_the_intended_ranks(crosscheck):
    """Rank contract: run() (1, 1, n_freqs) full-matrix vs forward() (n_freqs,).

    The rank difference is INTENDED behavior, pinned here on purpose
    (rfx-known-issues 'Added 2026-06-21' item 3 asked for a contract test;
    this is it, together with the value gate below): run() always reports
    the S-matrix convention S[receive, drive, freq] — (1, 1, n_freqs) even
    for one port — while forward() returns the AD-friendly per-port
    diagonal, (n_freqs,) for one port, via extract_lumped_s11. PR #258
    deliberately kept the 1-port wire run() path on the JIT main-scan
    fast-path to preserve the run/forward PEC contract; do NOT re-unify
    the ranks (or the paths) without a superseding decision.
    """
    S_run, S_fwd = crosscheck
    n_freqs = len(_XCHK_FREQS_HZ)
    assert S_run.shape == (1, 1, n_freqs), (
        f"run() 1-port S-matrix rank changed: {S_run.shape}, contract is "
        f"(1, 1, {n_freqs}) — full-matrix convention")
    assert S_fwd.shape == (n_freqs,), (
        f"forward() 1-port s_params rank changed: {S_fwd.shape}, contract "
        f"is ({n_freqs},) — 1-D per-port diagonal")
    assert np.iscomplexobj(S_run) and np.iscomplexobj(S_fwd), (
        "s_params must stay complex (magnitude-only returns would break "
        "phase-consuming consumers)")


def test_run_forward_complex_values_agree_on_cpml(crosscheck):
    """COMPLEX-value cross-check at the measured float32 envelope.

    Complements tests/test_run_forward_s11_contract.py::
    test_run_forward_s11_agree_on_well_conditioned_cpml (same fixture,
    magnitude-only, atol 2e-3 — untouched): measured max complex delta is
    4.27e-7 over all 7 bins, gated at 5e-5 (~100x, cross-machine float32
    headroom), which also catches a pure-PHASE divergence the magnitude
    gate cannot see. NOT a tautology (review-verified): run() uses the
    inline decomposition on the runners/uniform.py + rfx/simulation.py
    scan path while forward() uses extract_lumped_s11 in
    rfx/api/_execute.py — distinct code sites compiled as different XLA
    graphs — so this gate catches a regression in EITHER entry-point
    implementation (DFT accumulation, port eligibility, freq handling),
    which the pure-formula identity test cannot. Small deltas at this
    scale are the measured float32 conditioning envelope (PR #258: the
    formulas are algebraically identical) — if this fails marginally,
    re-measure the envelope before touching anything; if it fails
    grossly, one of the two extraction paths regressed.
    """
    S_run, S_fwd = crosscheck
    delta = np.abs(S_run.reshape(-1).astype(np.complex128)
                   - S_fwd.reshape(-1).astype(np.complex128))
    print(f"\n[crosscheck] max complex |S_run - S_fwd| = {delta.max():.3e} "
          f"(measured 4.27e-7, gate {_XCHK_COMPLEX_DELTA_MAX:.0e})")
    assert delta.max() < _XCHK_COMPLEX_DELTA_MAX, (
        f"run() vs forward() complex S11 delta {delta.max():.3e} exceeds "
        f"{_XCHK_COMPLEX_DELTA_MAX:.0e} (measured envelope 4.27e-7) on a "
        f"well-conditioned CPML port")


# ===========================================================================
# slow_physics battery (opt-in: -m slow_physics): the 2-port THRU locks
# ===========================================================================

@pytest.mark.slow_physics
def test_thru_s11_floor(thru_smatrix):
    """Matched-thru reflection floor: max in-band |S11|, |S22| < 0.12.

    Measured post-#318: max|S11| 0.0858, max|S22| 0.0891 (mean|S11|
    0.0515), a V-shaped curve with a null at 4.5 GHz. This is rfx's
    MEASURED feed-post reflection (two 1 mm posts, ~0.26 nH each,
    interfering across the 16 mm line — issue #318 H_FIXTURE
    re-diagnosis), NOT an extraction error. The gate 0.12 stays below the
    pre-#318 dead-cell floor 0.13 (the 33.3-ohm termination bug), so a
    revert of the live-cell fix fails LOUDLY; a break otherwise means the
    diagonal V-I extraction or Z0 normalization moved — not the FDTD core.
    """
    s11 = np.abs(thru_smatrix[0, 0])
    s22 = np.abs(thru_smatrix[1, 1])
    worst = max(s11.max(), s22.max())
    assert worst < _THRU_S11_FLOOR_MAX, (
        f"thru diagonal floor broke: max(|S11|, |S22|) = {worst:.4f} "
        f"(measured 0.086/0.089, gate {_THRU_S11_FLOOR_MAX}; a value near "
        f"the pre-#318 0.13 means the dead-cell live-fold regressed)")
    # Two-sided liveness (review finding): a dead diagonal reads ~0 and
    # would pass the upper bound. Measured maxima 0.0858/0.0891.
    assert s11.max() > _THRU_S11_ALIVE_MIN and s22.max() > _THRU_S11_ALIVE_MIN, (
        f"thru diagonal channel reads dead: max|S11|={s11.max():.4f}, "
        f"max|S22|={s22.max():.4f} (measured 0.086/0.089, alive floor "
        f"{_THRU_S11_ALIVE_MIN})")


@pytest.mark.slow_physics
def test_thru_s21_band_locks_shipped_decomposer_envelope(thru_smatrix):
    """|S21| stays in [0.35, 0.85] — REGRESSION LOCK, NOT validated physics.

    Measured 0.54606..0.60974 post-#318 (post-#308 0.523601..0.667776 —
    the #318 live-cell fix left the transmission class unchanged; the #308
    receive-wave fix earlier recovered this channel from the pre-fix
    structural near-null 0.0025-0.0046, mechanism history in the module
    docstring). Band [0.35, 0.85] KEPT UNCHANGED — the |S21| kappa
    deflation is the SEPARATE issue #313, not part of the #318 ledger.
    HONESTY LABEL: regression lock only; the flux-vs-port transmitted-
    power delta is OPEN — the extractor-independent flux referee measured
    per-bin (flux fraction)/|S21|^2 = 2.237..3.541 (raw flux transmitted
    fraction 0.959-0.998 vs |S21|^2 = 0.274-0.446; implied flux-true
    |S21| = 0.971-0.997), a confirmed frequency-dependent drive-side
    scale bias kappa(f) = 1.49..1.86 (issue #313). Do
    not cite this band as thru-transmission physics; when the kappa item
    lands, |S21| moves toward 0.97-1.0 and this fails LOUDLY —
    re-baseline in the same PR, do not widen the band to keep both
    behaviors green. The lower edge also catches a collapse back to the
    pre-#308 near-null (dead channel class); the upper edge stays
    strictly < 1 (over-unity extraction artefact class).
    """
    s21 = np.abs(thru_smatrix[1, 0])
    lo, hi = _THRU_S21_BAND
    # Per-bin lower edge (review finding): max() would let 8/9 dead bins
    # slip through. Measured per-bin min 0.5461 (~1.6x this floor).
    assert s21.min() > lo, (
        f"|S21| collapsed below the envelope: per-bin min "
        f"{s21.min():.4f} <= {lo} (measured min 0.5461) — the recovered "
        f"channel died in at least one bin (dead probe / dead thru / "
        f"receive-sign regression class)")
    assert s21.max() < hi, (
        f"|S21| = {s21.max():.4f} left the measured envelope (max 0.6097, "
        f"band hi {hi}). If this is the drive-side kappa scale-bias fix "
        f"landing (flux-true |S21| = 0.971-0.997), re-baseline this "
        f"battery in the same PR")


@pytest.mark.slow_physics
def test_thru_s21_phase_band_is_sign_sensitive(thru_smatrix):
    """Signed S21 phase-deviation band + live conjugation discrimination.

    dev(f) = wrap(arg S21 - (-2*pi*f*L/c)) with the analytic ideal-thru
    delay for the 16 mm air line (DFT kernel exp(-j*2*pi*f*t) => e^{+jwt}
    phasors, outgoing wave e^{-j*beta*x}). Measured dev (post-#318,
    amended receive sign) = -0.6268..-0.3482 rad (post-#308
    -0.754891..-0.335259); band [-1.1, -0.1] rad KEPT UNCHANGED — a smooth
    feed-post group-delay excess over the 53.4 ps line delay, no pi
    offset. HONESTY LABEL — REGRESSION LOCK on the deviation values
    (flux-vs-port magnitude delta OPEN, kappa — SEPARATE #313, module
    docstring); the overall SIGN is physics-anchored by the DC witness:
    the low-f falsifier measured S21(DC) -> +1 under the amended sign
    (V - Z0*I) (dev -0.049 rad at 0.5 GHz); the first-cut sign measured
    -1 and was amended in the #308 PR. Sign AND magnitude are gated
    (per-bin, signed) — verified NOT conjugation-invariant: the test also
    asserts conj(S21) violates the band (on the measured data 8/9 bins
    leave it; conj dev = +2.360..-0.962 rad, only the 7 GHz bin stays
    inside), so the W3.4-class conjugation bug cannot pass; a flip back to
    the first-cut receive sign shifts every bin by pi and fails too.
    """
    s21 = thru_smatrix[1, 0]
    expected = np.exp(-1j * 2 * np.pi * _THRU_FREQS_HZ * _THRU_L_M / C0_M_PER_S)
    dev = np.angle(s21 / expected)              # wrapped signed deviation
    lo, hi = _THRU_PHASE_DEV_BAND_RAD
    print(f"\n[thru battery] signed phase dev (rad): {np.round(dev, 3)}")
    assert np.all((dev > lo) & (dev < hi)), (
        f"S21 signed phase deviation left [{lo}, {hi}] rad "
        f"(measured -0.627..-0.348 under the amended receive sign): "
        f"dev = {np.round(dev, 3)}. A receive-sign regression back to "
        f"the first-cut convention shifts this by pi; any deliberate "
        f"sign decision MUST re-baseline this battery in the same PR")

    # Live sign-discrimination witness: a conjugated S21 must FAIL the
    # same band, otherwise this gate has degraded into a |dev| check.
    dev_conj = np.angle(np.conj(s21) / expected)
    assert not np.all((dev_conj > lo) & (dev_conj < hi)), (
        "conj(S21) also satisfies the signed phase band — the gate lost "
        "its direction sensitivity (conjugation-blind)")


@pytest.mark.slow_physics
def test_thru_reciprocity(thru_smatrix):
    """Decomposer-symmetry lock: max|S21 - S12| small on the live channel.

    Measured post-#318: 7.53e-3 absolute (rel 0.013581) on the recovered
    O(0.55-0.61) magnitudes; gates 1.5e-2 / 0.10. IMPROVED from the
    post-#308 1.043e-2 / rel 0.016254 (the live-cell fix made the two
    ports' V/I bookkeeping more symmetric); the abs gate is re-baselined
    tighter (~2x measured), the scale-free REL gate stays at 0.10. A break
    catches an asymmetric edit to the shared decomposers or per-port
    Z0/n_cells bookkeeping. NOT transmission validation while the kappa
    magnitude item is open (SEPARATE #313, module docstring).
    """
    s21 = thru_smatrix[1, 0]
    s12 = thru_smatrix[0, 1]
    abs_dev = np.abs(s21 - s12)
    rel_dev = abs_dev / np.maximum(np.abs(s21), np.abs(s12))
    assert abs_dev.max() < _THRU_RECIP_ABS_MAX, (
        f"reciprocity |S21-S12| = {abs_dev.max():.2e} "
        f"(measured 7.53e-3, gate {_THRU_RECIP_ABS_MAX})")
    assert rel_dev.max() < _THRU_RECIP_REL_MAX, (
        f"reciprocity rel dev = {rel_dev.max():.4f} "
        f"(measured 0.013581, gate {_THRU_RECIP_REL_MAX})")


@pytest.mark.slow_physics
def test_thru_passivity_singular_values(thru_smatrix):
    """Energy sanity: max singular value of every per-freq 2x2 slice < 0.85.

    Measured post-#318: 0.632587 (margin 0.37 to the physical bound 1.0,
    ~1.34x measured headroom to the gate; IMPROVED from the post-#308
    0.687410 as the smaller diagonal reflection lowers the mixed-matrix
    norm). A strict-passivity regression lock that catches an over-unity
    extraction artefact on either channel — and, per the kappa open item
    (SEPARATE #313, module docstring), a drive-side scale-bias fix moving
    |S21| to the flux-implied 0.971-0.997 would push the singular value
    past 0.85 and fail LOUDLY here too, forcing a re-baseline in the same
    PR. Do not cite this as transmission evidence while the port-based
    magnitude is unvalidated.
    """
    sv_max = max(
        np.linalg.svd(thru_smatrix[:, :, k], compute_uv=False)[0]
        for k in range(thru_smatrix.shape[2])
    )
    assert sv_max < _THRU_MAX_SINGULAR_VALUE, (
        f"max singular value {sv_max:.4f} (measured 0.6326 post-#318, "
        f"gate {_THRU_MAX_SINGULAR_VALUE}) — extraction produced "
        f"over-envelope energy on the thru (or the kappa scale-bias fix "
        f"landed: re-baseline this battery in the same PR)")


# ===========================================================================
# DC-limit sign anchor (slow_physics) — the committed form of the low-f
# falsifier that pinned the receive sign (issue #308 amendment round)
# ===========================================================================
# Measured (post-#318 rerun 2026-07-11, amended sign b=(V - Z0*I)):
# wrapped dev arg(S21) - (-2*pi*f*L/c) = -0.0494 rad @ 0.5 GHz, -0.1015
# rad @ 1.0 GHz (post-#308 -0.0236/-0.0536). Band (-0.25, +0.10) — still
# generous vs measurement but decisively pi-DISCRIMINATING: the first-cut
# receive sign (-V + Z0*I) measured S21(DC) -> -1, i.e. dev ~ +3.04..3.09
# rad at these bins (measured flipped), far outside.
_DCA_FREQS_HZ = np.array([0.5e9, 1.0e9])
_DCA_N_STEPS = 12000            # 0.5 GHz bins need the long settle window
_DCA_DEV_BAND_RAD = (-0.25, +0.10)


@pytest.fixture(scope="module")
def dc_anchor_smatrix():
    """Low-frequency THRU run (same geometry, f0=2.5 GHz bw=1.0 pulse)."""
    sim = _build_thru(pulse=GaussianPulse(f0=2.5e9, bandwidth=1.0))
    report = sim.preflight()
    issues = [str(i) for i in report]
    for msg in issues:
        print(f"\n[dc anchor] preflight (verbatim): {msg}")
    # Same exact set as the thru_smatrix fixture above (re-pinned
    # 2026-07-11 for issue #319): pec_faces + one dead-extent-cell
    # advisory per port (#318 — post-fix each port terminates at 50 ohm
    # across its 2 live cells; the pre-fix 33.3-ohm reading is the
    # historical #313 finding; gates measured on this geometry stay
    # valid as-is).
    codes = sorted(getattr(i, "code", None) for i in report)
    assert codes == ["pec_faces_finite_pec",
                     "wire_port_dead_extent_cells",
                     "wire_port_dead_extent_cells"], (
        f"dc-anchor fixture preflight drifted: {issues}")
    assert any(_PEC_FACES_ADVISORY_SNIPPET in m for m in issues)
    result = sim.run(n_steps=_DCA_N_STEPS, compute_s_params=True,
                     s_param_freqs=_DCA_FREQS_HZ)
    return np.asarray(result.s_params).astype(np.complex128)


@pytest.mark.slow_physics
def test_dc_limit_pins_receive_sign(dc_anchor_smatrix):
    """S21(DC) -> +1: the committed, re-runnable form of the sign witness.

    The 3-7 GHz signed phase band locks the sign against silent
    regression, but re-ARBITRATING which sign is physical previously
    required the offline falsifier lane (re-review finding, both lenses).
    This anchors it in-repo: at 0.5-1 GHz the thru's wrapped phase
    deviation vs the analytic line delay must sit near 0 (measured
    -0.049/-0.102 rad post-#318), NOT near +-pi (the first-cut sign's -1
    DC limit). Physics-anchored: this is the DC witness itself, not an
    envelope.
    """
    s21 = dc_anchor_smatrix[1, 0]
    expected = np.exp(-1j * 2 * np.pi * _DCA_FREQS_HZ * _THRU_L_M / C0_M_PER_S)
    dev = np.angle(s21 / expected)
    lo, hi = _DCA_DEV_BAND_RAD
    print(f"\n[dc anchor] |S21|={np.round(np.abs(s21), 4)} "
          f"dev(rad)={np.round(dev, 4)}")
    assert np.all((dev > lo) & (dev < hi)), (
        f"DC-limit sign anchor failed: dev = {np.round(dev, 4)} rad outside "
        f"[{lo}, {hi}] (measured -0.049/-0.102). A pi-scale dev means the "
        f"receive-wave sign regressed to the first-cut convention")
    # pi-discrimination witness: the sign-flipped S21 must leave the band.
    dev_flipped = np.angle(-s21 / expected)
    assert not np.all((dev_flipped > lo) & (dev_flipped < hi)), (
        "sign-flipped S21 also passes the DC anchor band — the anchor "
        "lost its discriminating power")
