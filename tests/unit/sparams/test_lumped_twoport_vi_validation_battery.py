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
``tests/unit/sparams/test_twoport_wire_port.py`` floor gate (max|S21| > 1e-3), the
``tests/unit/sparams/test_sparam_driver_matches_eager.py`` driver-vs-eager atol-2e-3
locks, and the ``tests/unit/sparams/test_run_forward_s11_contract.py`` magnitude-only
CPML atol-2e-3 contract are all untouched. This module's own thru gates
were re-baselined 2026-07-10 in the same PR as the issue #308
receive-wave fix — exactly the tripwire protocol the original gates
prescribed (fail LOUDLY on the convention change, re-measure in the same
PR). The S11 floor/alive gates, the run<->forward cross-check, and the
algebraic-identity test are byte-untouched by the re-baseline.

RE-BASELINE 2026-08-29 (issue #770, whole-port off-diagonal frame): the
off-diagonal S21/S12 lane was re-pinned to the measured-PHYSICAL class —
|S21| = 0.9341..0.9954, reciprocity 2.6678e-4, sv_max 1.003227, phase dev
-0.3516..-0.8125 rad, DC anchor -0.0575/-0.1152 rad with |S21(DC)| ->
0.9999 (provenance: validation/research/issue770_offdiag_adjudication.py,
pre-declared falsifiers in docs/design_notes/
issue770_offdiag_adjudication_predeclaration.md; adjudicated against the
flux referee 0.971-0.997 and the openEMS envelope 0.973-1.034). The
per-cell #308/#313 narrative below is kept VERBATIM as history of the
legacy frame (still live on the v_port=None decomposer path); its |S21|
0.52-0.67 class is refuted as thru physics, exactly as its own honesty
labels always said.

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
``_wire_sim`` of tests/unit/sparams/test_run_forward_s11_contract.py, which this
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

# RE-PINNED 2026-08-29 for issue #770 (whole-port off-diagonal frame;
# measured provenance validation/research/issue770_offdiag_adjudication.py
# --battery-provenance, this exact fixture, shipped driver path):
# |S21| = 0.9341..0.9954 across 3-7 GHz — the frame the pre-declared
# adjudication validated against EXTERNAL physics (flux referee implied
# |S21| 0.971-0.997 on this geometry, openEMS envelope 0.973-1.034;
# per-bin power closure |S11|^2+|S21|^2 = 0.956-0.991 vs the measured
# 0.2-4.0% flux closure gap; the historical per-cell 0.546-0.610 class
# measured a net-through fraction 0.31-0.37 and was REFUTED as physics —
# it was always labeled a regression lock, per the #313 ledger entry).
# Band edges are PHYSICS-derived, not envelope-tuned: lower 0.90 from the
# external-anchor closure window (net-through fraction >= 0.90 with the
# measured diagonal — the F-A2 window of the #770 pre-declaration);
# upper 1.0 + 1e-3 (physical passivity bound + headroom for the same
# systematic, monotone-in-frequency excess gated below by
# _THRU_MAX_SINGULAR_VALUE — 1.0032 at 3 GHz -> 0.9874 at 7 GHz,
# mechanism unidentified, NOT float noise; measured max 0.9954). A
# collapse back to the per-cell 0.55-0.61 class or to the pre-#308
# near-null fails the lower edge loudly.
_THRU_S21_BAND = (0.90, 1.0 + 1e-3)

# Measured signed per-bin phase deviation arg(S21) - (-2*pi*f*L/c),
# wrapped: -0.8125..-0.3516 rad (issue #770 whole-port off-diagonal
# re-measure 2026-08-29, --battery-provenance; post-#318 per-cell it was
# -0.6268..-0.3482, post-#308 -0.7549..-0.3353 — the frame change kept
# the same sign and the same smooth delay-like class, band edges now at
# 3 GHz / 7 GHz monotonically).
# Band [-1.1, -0.1] rad KEPT UNCHANGED (margins ~0.29/0.25 rad; both
# edges well inside (-pi, pi) so wrapped values stay comparable).
# HONESTY LABEL: the deviation VALUES are a regression lock, but the
# overall SIGN is physics-anchored by the DC witness: the low-f
# falsifier measured S21(DC) -> +1 under this sign (#770 whole-port
# re-measure: dev -0.058 rad at 0.5 GHz, tracking the analytic delay;
# the #308-era per-cell channel measured -0.049); the first-cut sign
# measured -1 and was amended in the #308 PR, and the #770 whole-port
# receive sign was re-pinned by the same witness class (flipped channel
# at +3.08/+3.03 rad). The deviation is a smooth feed-post group-delay
# excess, physical, not a convention artefact. Signed on purpose:
# conjugation and a sign flip both move bins out of band (verified live
# in the test; conj dev is NOT -dev, the analytic reference phase
# differs per bin).
_THRU_PHASE_DEV_BAND_RAD = (-1.1, -0.1)

# Measured reciprocity max|S21 - S12| = 2.6678e-4 (rel 2.78e-4) on the
# whole-port frame (issue #770 re-measure 2026-08-29 — IMPROVED 28x
# from the per-cell 7.53e-3 / rel 0.013581, itself improved from the
# post-#308 1.043e-2: the physical incident wave is a drive-only
# constant, so per-column normalization asymmetry collapses). Abs gate
# KEPT at 1.5e-2 and rel at 0.10 (scale-free) — both now carry ~56x /
# ~360x margin; a break catches an asymmetric edit to the shared
# decomposers or per-port Z0 bookkeeping.
_THRU_RECIP_ABS_MAX = 1.5e-2
_THRU_RECIP_REL_MAX = 0.10

# Measured passivity: max singular value over the 9 per-freq 2x2 slices
# = 1.003227 (issue #770 whole-port re-measure 2026-08-29; per-bin
# 1.0032..0.9874, largest at 3 GHz where |S21| = 0.9954). The matrix is
# now NEAR-UNITARY — the physical singular value is ~1 and the 0.32%
# excess is SYSTEMATIC and monotone in frequency (1.0032 at 3 GHz ->
# 0.9874 at 7 GHz); mechanism unidentified. It is NOT float noise: f64
# fields give 1.0032250 vs f32 1.0032275, 4000/8000/16000 steps are
# bit-identical at 1.0032275433727436, and complex128 algebra matches
# complex64 to 16 digits — all three reproduce the excess rather than
# shrinking it. Gate 1.01: ~2x the measured excess over the physical
# bound, strictly below the repo's 1.02 column-power passivity-tolerance
# class (kept as a plausibility anchor, not a causal explanation — the
# binding magnitude gate for this excess is _THRU_S21_BAND's 1.001
# edge). The historical 0.85 gate was bindable only because the
# per-cell frame DEFLATED |S21| to 0.63 (post-#318 sv 0.632587,
# post-#683-flip 0.6934); a strictly-below-1 gate cannot bind a
# physically ~unity singular value. Energy-sanity lock; the
# transmission MAGNITUDE is now separately gated by _THRU_S21_BAND
# against the external-anchor closure window. Follow-up: root cause of
# the monotone excess is not yet isolated (see the drafted follow-up
# issue in the PR body).
_THRU_MAX_SINGULAR_VALUE = 1.01

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

    Complements tests/unit/sparams/test_run_forward_s11_contract.py::
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
@pytest.mark.xfail(
    strict=True,
    reason="issue #683 flip landed 2026-08-29 and this restore FIRED its "
           "pre-declared falsifier (P1 gate 5, "
           "docs/design_notes/issue683_decomposer_flip_predeclaration.md): "
           "the physical whole-port thru diagonal measures max 0.2910 "
           "in-band, OUTSIDE the pre-declared < 0.12 restore class.  The "
           "reading is quantitatively the fixture's own un-de-embedded "
           "feed posts (measured driven Z_in = 43+j27 ohm at 7 GHz; the "
           "symmetric far post's +j27 series reactance alone gives "
           "|Gamma| ~ 27/104 = 0.26), NOT an extraction error — the "
           "known-load harness passed (n*a +0.9990/+0.9960) and the "
           "off-diagonals are bit-identical to base.  Held as a FIRED "
           "falsifier for review sign-off; do not silently re-pin.")
def test_thru_s11_floor(thru_smatrix):
    """Thru diagonal: physical floor gate — restore FIRED, held for review.

    HISTORY.  The pre-#764 physical floor (max in-band |S11|, |S22| <
    0.12; measured 0.0858/0.0891, the 'feed-post V-shape') was a pin on
    the LEGACY per-cell driven diagonal — a frame-mismatched reading
    (#313/#318) that did not track the load.  #764 moved the diagonal to
    the whole-port reflection S_kk = (V_port - Z0*I)/(V_port + Z0*I) and
    keyed an interim envelope (worst 2.8068 on PRE-injection samples) to
    the #683 flip.  The flip landed 2026-08-29 (POST-injection physical
    sampling + drive-reference decomposer recalibration) and this gate
    was restored per its own instruction — and the restore's pre-declared
    < 0.12 class FIRED: the measured physical diagonal is
    |S11| 0.0093-0.2896 / |S22| 0.0176-0.2910 over the 3-7 GHz bins,
    rising with frequency exactly as the ports' measured feed-post
    reactance does (Z_in - Z0 ~ +j27 ohm at 7 GHz -> far-post mismatch
    |Gamma| ~ 0.26).  The 0.09 'feed-post class' expectation was an
    extrapolation from the legacy artifact reading, and the measurement
    refutes it.  Per the STOP discipline the strict-xfail above keeps
    the firing visible instead of widening the gate; the assert below is
    the pre-declared physical floor, unmodified.
    """
    s11 = np.abs(thru_smatrix[0, 0])
    s22 = np.abs(thru_smatrix[1, 1])
    worst = max(s11.max(), s22.max())
    assert worst < 0.12, (
        f"thru diagonal above the pre-declared physical floor: "
        f"max(|S11|, |S22|) = {worst:.4f} (measured physical value "
        f"0.2910 — the fired-falsifier state this xfail documents)")
    # Two-sided liveness (review finding): a dead diagonal reads ~0 and
    # would pass an upper bound.
    assert s11.max() > _THRU_S11_ALIVE_MIN and s22.max() > _THRU_S11_ALIVE_MIN, (
        f"thru diagonal channel reads dead: max|S11|={s11.max():.4f}, "
        f"max|S22|={s22.max():.4f} (alive floor {_THRU_S11_ALIVE_MIN})")


@pytest.mark.slow_physics
def test_thru_s21_band_locks_shipped_decomposer_envelope(thru_smatrix):
    """|S21| stays in [0.90, 1.001] — the #770 measured-PHYSICAL class.

    RE-BASELINED 2026-08-29 in the issue #770 whole-port off-diagonal PR,
    exactly as the previous docstring prescribed ("when the kappa item
    lands, |S21| moves toward 0.97-1.0 and this fails LOUDLY —
    re-baseline in the same PR"): measured 0.9341..0.9954 on this exact
    fixture (--battery-provenance arm of
    validation/research/issue770_offdiag_adjudication.py), inside the
    flux-referee implied 0.971-0.997 / openEMS 0.973-1.034 class after
    the un-de-embedded feed-post reflection (|S11| up to 0.29 at 7 GHz)
    is accounted — per-bin closure |S11|^2+|S21|^2 = 0.956-0.991 vs the
    measured 0.2-4.0% flux closure gap. The #313 kappa deflation was a
    property of the per-cell frame (historical: 0.546-0.610 post-#318,
    0.5236-0.6678 post-#308, structural near-null 0.0025-0.0046
    pre-#308); it now lives only on the legacy v_port=None decomposer
    path. The lower edge catches any collapse back to those classes; the
    upper edge is the passivity bound plus headroom for the systematic,
    frequency-monotone near-unity excess also gated below by
    test_thru_passivity_singular_values (1.0032 at 3 GHz -> 0.9874 at
    7 GHz; mechanism unidentified, NOT float noise).
    """
    s21 = np.abs(thru_smatrix[1, 0])
    lo, hi = _THRU_S21_BAND
    # Per-bin lower edge (review finding): max() would let 8/9 dead bins
    # slip through. Measured per-bin min 0.9341 (7 GHz).
    assert s21.min() > lo, (
        f"|S21| collapsed below the physical band: per-bin min "
        f"{s21.min():.4f} <= {lo} (measured min 0.9341) — dead probe / "
        f"dead thru / receive-sign regression, or a revert to the "
        f"per-cell 0.55-0.61 frame")
    assert s21.max() < hi, (
        f"|S21| = {s21.max():.4f} above the passivity bound {hi} "
        f"(measured max 0.9954) — over-unity extraction artefact")


@pytest.mark.slow_physics
def test_thru_s21_phase_band_is_sign_sensitive(thru_smatrix):
    """Signed S21 phase-deviation band + live conjugation discrimination.

    dev(f) = wrap(arg S21 - (-2*pi*f*L/c)) with the analytic ideal-thru
    delay for the 16 mm air line (DFT kernel exp(-j*2*pi*f*t) => e^{+jwt}
    phasors, outgoing wave e^{-j*beta*x}). Measured dev (issue #770
    whole-port frame, 2026-08-29) = -0.3516..-0.8125 rad monotone across
    3-7 GHz (per-cell historical: post-#318 -0.6268..-0.3482, post-#308
    -0.754891..-0.335259); band [-1.1, -0.1] rad KEPT UNCHANGED — a
    smooth feed-post group-delay excess over the 53.4 ps line delay, no
    pi offset. The overall SIGN is physics-anchored by the DC witness:
    S21(DC) -> +1 (#770 re-measure dev -0.058 rad at 0.5 GHz; #308-era
    -0.049); the first-cut #308 sign measured -1 and was amended, and
    the #770 whole-port receive sign was pinned by the same witness
    class. Sign AND magnitude are gated (per-bin, signed) — the test
    also asserts conj(S21) violates the band, so the W3.4-class
    conjugation bug cannot pass; a receive-sign flip shifts every bin
    by pi and fails too.
    """
    s21 = thru_smatrix[1, 0]
    expected = np.exp(-1j * 2 * np.pi * _THRU_FREQS_HZ * _THRU_L_M / C0_M_PER_S)
    dev = np.angle(s21 / expected)              # wrapped signed deviation
    lo, hi = _THRU_PHASE_DEV_BAND_RAD
    print(f"\n[thru battery] signed phase dev (rad): {np.round(dev, 3)}")
    assert np.all((dev > lo) & (dev < hi)), (
        f"S21 signed phase deviation left [{lo}, {hi}] rad "
        f"(measured -0.352..-0.813 under the #770 whole-port frame): "
        f"dev = {np.round(dev, 3)}. A receive-sign regression shifts "
        f"this by pi; any deliberate sign decision MUST re-baseline "
        f"this battery in the same PR")

    # Live sign-discrimination witness: a conjugated S21 must FAIL the
    # same band, otherwise this gate has degraded into a |dev| check.
    dev_conj = np.angle(np.conj(s21) / expected)
    assert not np.all((dev_conj > lo) & (dev_conj < hi)), (
        "conj(S21) also satisfies the signed phase band — the gate lost "
        "its direction sensitivity (conjugation-blind)")


@pytest.mark.slow_physics
def test_thru_reciprocity(thru_smatrix):
    """Decomposer-symmetry lock: max|S21 - S12| small on the live channel.

    Measured on the #770 whole-port frame (2026-08-29): 2.6678e-4
    absolute (rel 2.78e-4) — IMPROVED 28x from the per-cell 7.53e-3 /
    rel 0.013581 (itself improved from the post-#308 1.043e-2): the
    physical incident wave is a drive-only constant, so the per-column
    normalization asymmetry the per-cell PRE-referenced a_j carried
    collapses. Gates KEPT at 1.5e-2 / 0.10 (~56x / ~360x margin). A
    break catches an asymmetric edit to the shared decomposers or
    per-port Z0 bookkeeping.
    """
    s21 = thru_smatrix[1, 0]
    s12 = thru_smatrix[0, 1]
    abs_dev = np.abs(s21 - s12)
    rel_dev = abs_dev / np.maximum(np.abs(s21), np.abs(s12))
    assert abs_dev.max() < _THRU_RECIP_ABS_MAX, (
        f"reciprocity |S21-S12| = {abs_dev.max():.2e} "
        f"(measured 2.67e-4, gate {_THRU_RECIP_ABS_MAX})")
    assert rel_dev.max() < _THRU_RECIP_REL_MAX, (
        f"reciprocity rel dev = {rel_dev.max():.4f} "
        f"(measured 2.78e-4, gate {_THRU_RECIP_REL_MAX})")


@pytest.mark.slow_physics
def test_thru_passivity_singular_values(thru_smatrix):
    """Energy sanity: max per-freq singular value <= 1 + extraction noise.

    RE-BASELINED 2026-08-29 for issue #770 (whole-port off-diagonal
    frame, measured provenance --battery-provenance): the matrix is now
    NEAR-UNITARY — measured per-bin sv 0.9874..1.0032 (max 1.003227 at
    3 GHz where |S21| = 0.9954); the 0.32% excess over the physical
    bound is SYSTEMATIC and monotone in frequency (1.0032 at 3 GHz ->
    0.9874 at 7 GHz), mechanism unidentified. It is NOT float noise:
    f64 fields give 1.0032250 vs f32 1.0032275, 4000/8000/16000 steps
    are bit-identical at 1.0032275433727436, and complex128 algebra
    matches complex64 to 16 digits — the repo's 1.02 column-power
    ceilings are kept only as a plausibility anchor, not a causal
    explanation. Gate 1.01 (~2x the measured excess, strictly below
    1.02; the binding magnitude gate for this excess is the S21 band's
    1.001 edge, see below). History: the 0.85 gate was bindable only
    while the per-cell frame deflated |S21| (sv 0.632587 post-#318,
    0.6934 post-#683-flip, interim 3.2061 in the keyed-envelope era;
    0.687410 post-#308) — a strictly-below-1 gate cannot bind a
    physically ~unity singular value. Catches over-unity extraction
    artefacts; the transmission magnitude itself is gated by the S21
    band lock.
    """
    sv_max = max(
        np.linalg.svd(thru_smatrix[:, :, k], compute_uv=False)[0]
        for k in range(thru_smatrix.shape[2])
    )
    assert sv_max < _THRU_MAX_SINGULAR_VALUE, (
        f"thru singular value above the passivity bound + noise "
        f"allowance: {sv_max:.4f} (measured 1.003227 on the #770 "
        f"whole-port frame, gate {_THRU_MAX_SINGULAR_VALUE})")


# ===========================================================================
# DC-limit sign anchor (slow_physics) — the committed form of the low-f
# falsifier that pinned the receive sign (issue #308 amendment round)
# ===========================================================================
# Measured (issue #770 whole-port frame re-measure 2026-08-29,
# --battery-provenance): wrapped dev arg(S21) - (-2*pi*f*L/c) = -0.0575
# rad @ 0.5 GHz, -0.1152 rad @ 1.0 GHz with |S21| = 0.9999/0.9997 — the
# DC thru limit S21 -> +1 now holds in MAGNITUDE too (per-cell
# historical: -0.0494/-0.1015 post-#318, -0.0236/-0.0536 post-#308).
# Band (-0.25, +0.10) KEPT — still decisively pi-DISCRIMINATING: the
# flipped receive sign measured dev +3.0841/+3.0264 rad at these bins,
# far outside.
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
    -0.0575/-0.1152 rad on the #770 whole-port frame with |S21| =
    0.9999/0.9997 — the DC magnitude limit holds too; per-cell
    historical -0.049/-0.102 post-#318), NOT near +-pi (the first-cut
    sign's -1 DC limit). Physics-anchored: this is the DC witness
    itself, not an envelope.
    """
    s21 = dc_anchor_smatrix[1, 0]
    expected = np.exp(-1j * 2 * np.pi * _DCA_FREQS_HZ * _THRU_L_M / C0_M_PER_S)
    dev = np.angle(s21 / expected)
    lo, hi = _DCA_DEV_BAND_RAD
    print(f"\n[dc anchor] |S21|={np.round(np.abs(s21), 4)} "
          f"dev(rad)={np.round(dev, 4)}")
    assert np.all((dev > lo) & (dev < hi)), (
        f"DC-limit sign anchor failed: dev = {np.round(dev, 4)} rad outside "
        f"[{lo}, {hi}] (measured -0.0575/-0.1152). A pi-scale dev means "
        f"the receive-wave sign regressed to the first-cut convention")
    # pi-discrimination witness: the sign-flipped S21 must leave the band.
    dev_flipped = np.angle(-s21 / expected)
    assert not np.all((dev_flipped > lo) & (dev_flipped < hi)), (
        "sign-flipped S21 also passes the DC anchor band — the anchor "
        "lost its discriminating power")
